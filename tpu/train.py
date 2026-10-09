"""FreqFlow base training on TPU (JAX): the counterpart of train_ldm_discrete.py.

Same model, loss, data law, optimizer, schedule and EMA: z ~ the SD-VAE posterior of the image or its flip, the
label dropped to the null class with p_uncond, t ~ U[0, 1), FreqFlow's loss (velocity MSE + 1/4 frequency-branch
MSE) times 1/2, AdamW with linear warmup, EMA 0.9999. Mixed precision is bfloat16 (TPU) instead of float16.

Every chip holds the parameters; optimizer moments, the EMA and accumulated gradients are sharded over the chips
(and the parameters too with --config.tpu.shard_params=True). Each chip runs tpu.per_device_batch rows per
micro-batch; batch_size / (chips x per_device_batch) micro-batches are accumulated per step.

  python tpu/train.py --config=tpu/tpu_configs/freqflow_l.py --config.tpu.data_dir=/mnt/data/imagenet_packed \
      --workdir=gs://BUCKET/freqflow_l           # add --distributed on multi-host slices
"""
import functools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo root, for `tpu.*`

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from jax.sharding import NamedSharding, PartitionSpec as P
from ml_collections import config_flags

from tpu import common
from tpu.common import NULL_CLASS, latent_sample
from tpu.data import BatchSource, PackedFeatures, to_global
from tpu.losses import fm_loss_rows
from tpu.sampling import euler_sample

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'TPU training configuration (tpu/tpu_configs/*.py).', lock_config=False)
flags.mark_flags_as_required(['config'])
flags.DEFINE_string('workdir', None, 'Checkpoints and logs (local or gs://). Default: workdir/tpu/<config>/<overrides>.')
flags.DEFINE_bool('distributed', False, 'Call jax.distributed.initialize() (multi-host TPU slices).')


def make_train_step(config, model, tx, schedule, mesh, shardings, abs_params, accum):
    groups = mesh.devices.size  # each GPU rank clamps the frequency split to its own micro-batch
    p_uncond = config.dataset.p_uncond if config.dataset.get('cfg', False) else 0.0
    rate = config.get('ema_rate', 0.9999)
    key0 = jax.random.key(config.seed)
    apply = lambda p, x, ls, y, g: model.apply({'params': p}, x, ls, y, g)
    grad_sh = common.tree_shardings(abs_params, mesh, True)  # accumulated gradients: sharded like the moments

    def micro_loss(params, moments, labels, key):
        k_z, k_y, k_t, k_e = jax.random.split(key, 4)
        z = latent_sample(moments, k_z)
        y = jnp.where(jax.random.uniform(k_y, labels.shape) < p_uncond, NULL_CLASS, labels)
        t = jax.random.uniform(k_t, (z.shape[0],))
        eps = jax.random.normal(k_e, z.shape)
        tt = t[:, None, None, None]
        rows = fm_loss_rows(apply, params, (1 - tt) * z + tt * eps, t, eps - z, y, groups)
        return rows.mean()

    batch_sh = NamedSharding(mesh, P(None, 'data'))

    @functools.partial(jax.jit, donate_argnums=0, in_shardings=(shardings, batch_sh, batch_sh),
                       out_shardings=(shardings, NamedSharding(mesh, P())))
    def train_step(state, moments, labels):
        keys = jax.random.split(jax.random.fold_in(key0, state['step']), accum)
        params = state['params']
        constrain = lambda g: jax.lax.with_sharding_constraint(g, grad_sh)

        def body(carry, inp):
            g_acc, l_acc = carry
            m, lab, k = inp
            loss, g = jax.value_and_grad(lambda p: 0.5 * micro_loss(p, m, lab, k))(params)
            return (constrain(jax.tree.map(jnp.add, g_acc, g)), l_acc + loss), None

        zeros = constrain(jax.tree.map(jnp.zeros_like, params))
        (grads, loss), _ = jax.lax.scan(body, (zeros, jnp.zeros(())), (moments, labels, keys))
        grads = jax.tree.map(lambda g: g / accum, grads)
        updates, opt_state = tx.update(grads, state['opt_state'], params)
        params = optax.apply_updates(params, updates)
        new = {'params': params, 'ema': common.ema_update(state['ema'], params, rate), 'opt_state': opt_state,
               'step': state['step'] + 1}
        # 'loss' as train_ldm_discrete logs it (the unscaled mean); 'lr' the next step's, as LambdaLR reports it
        return new, {'loss': 2 * loss / accum, 'lr': schedule(state['step'] + 1)}

    return train_step


def main(argv):
    config = FLAGS.config
    common.maybe_init_distributed(FLAGS.distributed)
    mesh = common.make_mesh()
    n_chips, t = mesh.devices.size, config.tpu
    global_batch = config.train.batch_size
    assert global_batch % (n_chips * t.per_device_batch) == 0, (global_batch, n_chips, t.per_device_batch)
    accum = global_batch // (n_chips * t.per_device_batch)
    workdir = common.workdir_for(config_flags.get_config_filename(FLAGS['config']), sys.argv, FLAGS.workdir)
    log = common.Logger(workdir, config.tpu.get('wandb_project', ''), os.path.basename(workdir.rstrip('/')), config)
    log.info(f'{jax.process_count()} hosts, {n_chips} chips ({jax.devices()[0].device_kind}); batch {global_batch} = '
             f'{accum} micro-batches x {n_chips} chips x {t.per_device_batch}; workdir {workdir}')
    log.info(str(config))

    model = common.make_model(config)
    tx, schedule = common.make_optimizer(config)
    init = common.load_init_params(config.init_path, config.nnet.model_args) if config.get('init_path') else None
    state, shardings, abs_state = common.create_state(model, tx, config, mesh, init)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(abs_state['params']))
    log.info(f'nnet has {n_params} parameters')
    ckpt_dir = os.path.join(workdir, 'ckpts')
    manager = common.checkpoint_manager(ckpt_dir if ckpt_dir.startswith('gs://') else os.path.abspath(ckpt_dir),
                                        t.max_to_keep)
    preemption = common.Preemption()
    step = 0
    if manager.latest_step() is not None:
        state, step = common.restore_checkpoint(manager, abs_state, shardings)
        log.info(f'resumed from step {step}')

    data = PackedFeatures(t.data_dir)
    source = BatchSource(data, global_batch, accum, config.seed, t.prefetch)
    source.start(step)
    train_step = make_train_step(config, model, tx, schedule, mesh, shardings, abs_state['params'], accum)

    apply = lambda p, x, ls, y, g: model.apply({'params': p}, x, ls, y, g)
    grid_y = jnp.repeat(jnp.arange(5) % 1000, 10)

    @jax.jit
    def sample_grid(ema, key):
        z = jax.random.normal(key, (50, *config.z_shape))
        return euler_sample(apply, ema, z, grid_y)  # the sampler's default 20 steps, as train_ldm_discrete

    log.info(f'Start fitting, step={step}, steps per epoch {source.steps_per_epoch}')
    t_last, s_last = time.time(), step
    while step < config.train.n_steps:
        moments, labels = to_global(mesh, source.get())
        state, metrics = train_step(state, moments, labels)
        step += 1
        if step % config.train.log_interval == 0:
            metrics = jax.device_get(metrics)
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
        preempted = preemption(step)
        if step % config.train.save_interval == 0 or step == config.train.n_steps or preempted:
            log.info(f'Save checkpoint {step}' + (' (preemption)...' if preempted else '...'))
            common.save_checkpoint(manager, state, step)
            if t.export_npz and (step % t.export_interval == 0 or step == config.train.n_steps):
                common.export_torch_npz(state['ema'], config.nnet.model_args,
                                        os.path.join(workdir, 'export', f'{step}_nnet_ema.npz'))
            common.sync_logs(workdir)
        if preempted:
            manager.wait_until_finished()
            log.info(f'Preempted: saved step {step}; re-run the same command to resume')
            return
    manager.wait_until_finished()
    log.info(f'Finish fitting, step={step}')


if __name__ == '__main__':
    app.run(main)
