"""On-policy finetuning of FreqFlow on TPU (JAX): the counterpart of train_onpolicy.py.

Each micro-batch, once onpolicy.start_step is reached and a term is on, mirrors the GPU script's three backward
passes, whose gradients add up:
  1. Roll out n_roll trajectories per chip down eval's sampler grid through the band (grid states with
     t > band_t_min), conditioned on classes drawn in proportion to the pool.
  2. MMD (mmd_gamma > 0): at every band state except t=1, one MMD^2 between all chips' rollout states and
     class-matched noised pool targets; its gradient flows back through the rollouts (jax.vjp).
  3. CFG (cfg_gamma > 0): per chip n_band distinct trajectories, one band state each; the exact (or banked)
     empirical posterior gives v*_c and v*_u; the term is mean ||(v_c - v_u) - (v*_c - v*_u)||^2 and the rows
     become the band's FM rows (v*_u and the null label on CFG-dropped rows).
  4. FM: FreqFlow's loss on on-path rows with t in [0, band_t_min) plus the band rows (t in [0, 1) and no band
     rows when CFG is off).
Every term carries train_ldm_discrete's factor 1/2 (LOSS_SCALE). Per-rank work on the GPUs (band selection, the
torch.cat([x, x]) CFG batch, the frequency split's clamp) is done per chip here, so a chip computes what one
GPU rank does.

  python tpu/train_onpolicy.py --config=tpu/tpu_configs/freqflow_onpolicy.py \
      --config.init_path=nnet_ema.npz --config.tpu.data_dir=/mnt/data/imagenet_packed \
      --config.onpolicy.pool_path=/mnt/data/imagenet_latent_pool --config.onpolicy.cfg_gamma=1.0 \
      --config.onpolicy.mmd_gamma=100 --workdir=gs://BUCKET/onpolicy
"""
import functools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo root, for `tpu.*`

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from absl import app, flags  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from tpu import common  # noqa: E402
from tpu.common import NULL_CLASS, latent_sample  # noqa: E402
from tpu.data import BatchSource, PackedFeatures, to_global  # noqa: E402
from tpu.losses import fm_loss_rows  # noqa: E402
from tpu.onpolicy import mmd_band, mmd_noise_floor, rollout  # noqa: E402
from tpu.pool import LatentPool  # noqa: E402
from tpu.sampling import euler_sample, pair, sampler_grid, unpair  # noqa: E402

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'TPU configuration (tpu/tpu_configs/*.py).', lock_config=False)
flags.mark_flags_as_required(['config'])
flags.DEFINE_string('workdir', None, 'Checkpoints and logs (local or gs://). Default: workdir/tpu/<config>/<overrides>.')
flags.DEFINE_bool('distributed', False, 'Call jax.distributed.initialize() (multi-host TPU slices).')

LOSS_SCALE = 0.5  # train_ldm_discrete.py backprops loss.mean() / 2; every term carries the same factor


def plan(config, n_chips):
    """The band, MMD states and per-chip sizes, validated as train_onpolicy.py does."""
    op, mb = config.onpolicy, config.tpu.per_device_batch
    cfg_on, mmd_on = op.cfg_gamma > 0, op.mmd_gamma > 0
    t_grid = sampler_grid(op.num_steps, op.shift)
    band = [k for k, t in enumerate(t_grid) if t > op.band_t_min]
    n_band = round(mb * (1 - op.band_t_min)) if cfg_on else 0
    if cfg_on or mmd_on:
        if not band:
            raise ValueError(f'no sampler grid state has t > band_t_min={op.band_t_min}')
        if cfg_on and not 1 <= n_band <= min(op.n_roll, mb - 1):
            raise ValueError(f'CFG band rows per micro-batch = {n_band} (micro-batch {mb} x {1 - op.band_t_min:g}); '
                             f'need 1 <= n_band <= min(n_roll={op.n_roll}, micro-batch - 1)')
        if mmd_on and len(band) < 2:
            raise ValueError('MMD needs a band state past the initial noise; lower band_t_min or raise num_steps')
    per = 0
    if mmd_on:
        if op.mmd_n_target % (n_chips * op.n_roll):
            raise ValueError(f'mmd_n_target={op.mmd_n_target} must be a multiple of n_chips * n_roll = {n_chips * op.n_roll}')
        if min(op.mmd_sigma_scales) < 1 / 80:
            raise ValueError('mmd_sigma_scales below 1/80 overflow the shifted float32 kernel exp((1 - d2/d2_base)/s)')
        per = op.mmd_n_target // (n_chips * op.n_roll)
    return dict(cfg_on=cfg_on, mmd_on=mmd_on, t_grid=t_grid, band=band, mmd_ks=band[1:], n_band=n_band, per=per)


def make_step(config, model, tx, schedule, mesh, shardings, abs_params, accum, pool, pl, active):
    op, mb, g_chips = config.onpolicy, config.tpu.per_device_batch, mesh.devices.size
    p_uncond = config.dataset.p_uncond
    rate = config.ema_rate
    key0 = jax.random.key(config.seed)
    apply = lambda p, x, ls, y, g: model.apply({'params': p}, x, ls, y, g)
    grad_sh = common.tree_shardings(abs_params, mesh, True)
    t_grid, band, n_band, per = pl['t_grid'], pl['band'], pl['n_band'], pl['per']
    cfg_on, mmd_on = pl['cfg_on'] and active, pl['mmd_on'] and active
    n_roll, n_states = op.n_roll, len(pl['band'])
    gi = jnp.arange(g_chips)[:, None]

    def shard(a):
        """Pin a [chips * rows, ...] array's rows to their chips: indexing and concatenation otherwise let XLA
        replicate the batch, and every chip would compute all of it."""
        return jax.lax.with_sharding_constraint(a, NamedSharding(mesh, P('data', *([None] * (a.ndim - 1)))))

    def micro(params, arrays, moments, labels, key):
        ks = jax.random.split(key, 12)
        z = latent_sample(moments, ks[0])
        y = jnp.where(jax.random.uniform(ks[1], labels.shape) < p_uncond, NULL_CLASS, labels)
        grads = jax.tree.map(jnp.zeros_like, params)
        m = {}
        band_rows = None
        if cfg_on or mmd_on:
            c = shard(pool.sample_classes(arrays, g_chips * n_roll, ks[2]))
            drop = shard(jax.random.uniform(ks[3], (g_chips * n_roll,)) < p_uncond)
            roll = lambda p: rollout(apply, p, c, t_grid, n_states, ks[4], op.cfg_scale, op.guidance_low,
                                     op.guidance_high, g_chips, op.mmd_grad_ckpt, shard)
            if mmd_on:  # gradient through the rollouts
                traj, vjp = jax.vjp(roll, params)
                (mmd, (per_t, ok)), g_traj = jax.value_and_grad(
                    lambda tr: (lambda r: (r[0], (r[1], r[2])))(
                        mmd_band(pool, arrays, tr, pl['mmd_ks'], t_grid, c, per, op.mmd_sigma_scales, ks[5])),
                    has_aux=True)(traj)
                grads = jax.tree.map(lambda a, b: a + LOSS_SCALE * op.mmd_gamma * b, grads, vjp(g_traj)[0])
                m['mmd'], m['mmd_targets_ok'] = mmd, ok.astype(jnp.float32)
                for k, val in zip(pl['mmd_ks'], per_t):
                    m[f'mmd_t{t_grid[k]:.3f}'] = val
                traj = [jax.lax.stop_gradient(s) for s in traj]
            else:
                traj = roll(jax.lax.stop_gradient(params))
            if cfg_on:
                stack = jnp.stack(traj).reshape(n_states, g_chips, n_roll, 4, 32, 32)
                j = jax.vmap(lambda k: jax.random.permutation(k, n_roll)[:n_band])(jax.random.split(ks[6], g_chips))
                kidx = jax.random.randint(ks[7], (g_chips, n_band), 0, n_states)
                q = g_chips * n_band
                x = shard(stack[kidx, gi, j].reshape(q, 4, 32, 32))
                t = shard(jnp.asarray([t_grid[k] for k in band], jnp.float32)[kidx].reshape(q))
                cs = shard(c.reshape(g_chips, n_roll)[gi, j].reshape(q))
                ds = shard(drop.reshape(g_chips, n_roll)[gi, j].reshape(q))
                xq = x.reshape(q, -1)
                e_c, e_u, empty = pool.posterior_means(arrays, xq, t, cs, 0 if op.bank_full else op.bank_k,
                                                       0 if op.bank_full_uncond else op.bank_k_uncond, ks[8])
                tt = t[:, None]
                v_c, v_u = (xq - e_c) / tt, (xq - e_u) / tt
                if op.cfg_target_full and not (op.bank_full and op.bank_full_uncond):
                    f_c, f_u, empty2 = pool.posterior_means(arrays, xq, t, cs)
                    dv_target, empty = (f_u - f_c) / tt, empty | empty2
                else:
                    dv_target = v_c - v_u
                ls = t * 8 - 4

                def cfg_loss(p):
                    v2 = apply(p, shard(pair(x, x, g_chips)), shard(pair(ls, ls, g_chips)),
                               shard(pair(cs, jnp.full_like(cs, NULL_CLASS), g_chips)), g_chips)[1]
                    vc, vu = unpair(v2, g_chips)
                    dv = (vc - vu).reshape(q, -1)
                    return jnp.mean((dv - dv_target) ** 2), jnp.mean(dv ** 2)

                (loss_cfg, dv2), g_cfg = jax.value_and_grad(cfg_loss, has_aux=True)(params)
                grads = jax.tree.map(lambda a, b: a + LOSS_SCALE * op.cfg_gamma * b, grads, g_cfg)
                m.update(cfg=loss_cfg, cfg_dv2_model=dv2, cfg_dv2_target=jnp.mean(dv_target ** 2),
                         posterior_ok=(~empty).astype(jnp.float32))
                v = jnp.where(ds[:, None], v_u, v_c).reshape(q, 4, 32, 32)
                band_rows = tuple(a.reshape(g_chips, n_band, *a.shape[1:])
                                  for a in (x, t, v, jnp.where(ds, NULL_CLASS, cs)))

        # FM on this micro-batch's rows; with CFG on, the first micro_bs - n_band rows of each chip plus its band rows
        n_on, t_hi = (mb - n_band, op.band_t_min) if band_rows is not None else (mb, 1.0)
        zg = z.reshape(g_chips, mb, *z.shape[1:])[:, :n_on]
        yg = y.reshape(g_chips, mb)[:, :n_on]
        t = jax.random.uniform(ks[9], (g_chips, n_on)) * t_hi
        eps = jax.random.normal(ks[10], zg.shape)
        x_t = (1 - t[..., None, None, None]) * zg + t[..., None, None, None] * eps
        v = eps - zg
        if band_rows is not None:
            x_t, t, v, yg = (jnp.concatenate([a, b], axis=1) for a, b in zip((x_t, t, v, yg), band_rows))
        flat = lambda a: shard(a.reshape(g_chips * mb, *a.shape[2:]))
        fm, g_fm = jax.value_and_grad(
            lambda p: fm_loss_rows(apply, p, flat(x_t), flat(t), flat(v), flat(yg), g_chips).mean())(params)
        grads = jax.tree.map(lambda a, b: a + LOSS_SCALE * b, grads, g_fm)
        m['loss'] = fm
        return grads, m

    batch_sh = NamedSharding(mesh, P(None, 'data'))
    pool_sh = jax.tree.map(lambda a: a.sharding, pool.arrays) if pool is not None else None

    @functools.partial(jax.jit, donate_argnums=0, in_shardings=(shardings, batch_sh, batch_sh, pool_sh),
                       out_shardings=(shardings, NamedSharding(mesh, P())))
    def train_step(state, moments, labels, arrays):
        keys = jax.random.split(jax.random.fold_in(key0, state['step']), accum)
        params = state['params']
        constrain = lambda g: jax.lax.with_sharding_constraint(g, grad_sh)

        def body(g_acc, inp):
            g, m = micro(params, arrays, *inp)
            return constrain(jax.tree.map(jnp.add, g_acc, g)), m

        grads, ms = jax.lax.scan(body, constrain(jax.tree.map(jnp.zeros_like, params)), (moments, labels, keys))
        grads = jax.tree.map(lambda g: g / accum, grads)
        updates, opt_state = tx.update(grads, state['opt_state'], params)
        params = optax.apply_updates(params, updates)
        new = {'params': params, 'ema': common.ema_update(state['ema'], params, rate), 'opt_state': opt_state,
               'step': state['step'] + 1}
        metrics = jax.tree.map(jnp.mean, ms)
        metrics['lr'] = schedule(state['step'] + 1)
        return new, metrics

    return train_step


def main(argv):
    config = FLAGS.config
    common.maybe_init_distributed(FLAGS.distributed)
    mesh = common.make_mesh()
    n_chips, t, op = mesh.devices.size, config.tpu, config.onpolicy
    global_batch = config.train.batch_size
    assert global_batch % (n_chips * t.per_device_batch) == 0, (global_batch, n_chips, t.per_device_batch)
    accum = global_batch // (n_chips * t.per_device_batch)
    workdir = common.workdir_for(config_flags.get_config_filename(FLAGS['config']), sys.argv, FLAGS.workdir)
    log = common.Logger(workdir, config.tpu.get('wandb_project', ''), os.path.basename(workdir.rstrip('/')), config)
    log.info(f'{jax.process_count()} hosts, {n_chips} chips ({jax.devices()[0].device_kind}); batch {global_batch} = '
             f'{accum} micro-batches x {n_chips} chips x {t.per_device_batch}; workdir {workdir}')
    log.info(str(config))
    pl = plan(config, n_chips)

    model = common.make_model(config)
    tx, schedule = common.make_optimizer(config)
    ckpt_dir = os.path.join(workdir, 'ckpts')
    manager = common.checkpoint_manager(ckpt_dir if ckpt_dir.startswith('gs://') else os.path.abspath(ckpt_dir))
    resume = manager.latest_step() is not None
    init = None
    if not resume and config.init_path:
        log.info(f'init params and EMA from {config.init_path}')
        init = common.load_init_params(config.init_path, config.nnet.model_args)
    state, shardings, abs_state = common.create_state(model, tx, config, mesh, init)
    step = 0
    if resume:
        state, step = common.restore_checkpoint(manager, abs_state, shardings)
        log.info(f'resumed from step {step}')

    pool = None
    if pl['cfg_on'] or pl['mmd_on']:
        pool = LatentPool(op.pool_path, mesh, op.expert_chunk)
        smallest = int(pool.class_count_np[pool.class_count_np > 0].min())
        for full, k, name in ((op.bank_full, op.bank_k, 'bank_k'), (op.bank_full_uncond, op.bank_k_uncond, 'bank_k_uncond')):
            if not full and not 1 <= k <= (smallest if name == 'bank_k' else pool.N):
                raise ValueError(f'{name}={k} is out of range for this pool')
        log.info(f'pool: {pool.N} latents, {pool.per} rows per chip; band t = '
                 f'{[round(pl["t_grid"][k], 4) for k in pl["band"]]}, {pl["n_band"]} CFG/FM band rows per chip micro-batch')
        if pl['mmd_on']:
            log.info(f'MMD: {n_chips * op.n_roll} rollout rows vs {op.mmd_n_target} class-matched targets '
                     f'({pl["per"]} per rollout) at t = {[round(pl["t_grid"][k], 4) for k in pl["mmd_ks"]]}. '
                     f'Noise floor (two independent target draws):')
            floor = mmd_noise_floor(pool, pool.arrays, pl['mmd_ks'], pl['t_grid'], n_chips * op.n_roll, pl['per'],
                                    op.mmd_sigma_scales, jax.random.key(config.seed + 4))
            for tt, mean, peak, d2 in floor:
                log.info(f'  t={tt:.4f}  mean {mean:.3e}  max|.| {peak:.3e}  d2_base {d2:.1f}')

    data = PackedFeatures(t.data_dir)
    source = BatchSource(data, global_batch, accum, config.seed, t.prefetch)
    source.start(step)
    steps = {a: make_step(config, model, tx, schedule, mesh, shardings, abs_state['params'], accum, pool, pl, a)
             for a in (False, True)}
    on = pl['cfg_on'] or pl['mmd_on']

    apply = lambda p, x, ls, y, g: model.apply({'params': p}, x, ls, y, g)
    grid_y = jnp.repeat(jnp.arange(5) % 1000, 10)
    sample_grid = jax.jit(lambda ema, key: euler_sample(apply, ema, jax.random.normal(key, (50, *config.z_shape)), grid_y))

    log.info(f'Start fitting, step={step}')
    t_last, s_last = time.time(), step
    while step < config.train.n_steps:
        moments, labels = to_global(mesh, source.get())
        active = on and step >= op.start_step
        state, metrics = steps[active](state, moments, labels, pool.arrays if pool is not None else None)
        step += 1
        if step % config.train.log_interval == 0:
            metrics = jax.device_get(metrics)
            for flag in ('posterior_ok', 'mmd_targets_ok'):
                if flag in metrics and metrics.pop(flag) < 1:
                    raise RuntimeError({'posterior_ok': 'a query had an empty candidate set',
                                        'mmd_targets_ok': 'a class had fewer latents than its MMD targets need; '
                                                          'lower mmd_n_target or n_roll'}[flag])
            if 'cfg' in metrics or 'mmd' in metrics:
                metrics['loss_total'] = (metrics['loss'] + op.cfg_gamma * metrics.get('cfg', 0.)
                                         + op.mmd_gamma * metrics.get('mmd', 0.))
            now = time.time()
            metrics['sec_per_step'] = (now - t_last) / (step - s_last)
            t_last, s_last = now, step
            log.metrics(step, metrics)
        if config.train.eval_interval and step % config.train.eval_interval == 0:
            z = jax.device_get(sample_grid(state['ema'], jax.random.key(step)))
            if jax.process_index() == 0:
                os.makedirs(os.path.join(common.local_dir(workdir), 'samples'), exist_ok=True)
                np.savez(os.path.join(common.local_dir(workdir), 'samples', f'{step}.npz'), latents=z,
                         labels=np.asarray(grid_y))
        if step % config.train.save_interval == 0 or step == config.train.n_steps:
            log.info(f'Save checkpoint {step}...')
            common.save_checkpoint(manager, state, step)
            if t.export_npz:
                common.export_torch_npz(state['ema'], config.nnet.model_args,
                                        os.path.join(workdir, 'export', f'{step}_nnet_ema.npz'))
            common.sync_logs(workdir)
    manager.wait_until_finished()
    log.info(f'Finish fitting, step={step}')


if __name__ == '__main__':
    app.run(main)
