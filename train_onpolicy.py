"""On-policy finetuning of FreqFlow on ImageNet-256 latents. Port of d-flow's
cifar_fm_dagger_conditional_mmd.py to FreqFlow's time convention (t=0 data,
t=1 noise) and sampler.

Each micro-batch, once onpolicy.start_step is reached and a term is on:
  1. Roll out n_roll trajectories per GPU down eval's sampler grid through the
     band (grid states with t > band_t_min). Every trajectory is conditioned on a
     class drawn in proportion to the pool's class frequencies; its CFG drop mask
     is drawn up front and only affects its FM row. With grad when MMD is on.
  2. MMD (mmd_gamma > 0): at every band state except t=1, one MMD^2 over all GPUs'
     rollout states against all targets. The targets are class-matched so they mix
     the batch's classes as the rollouts do: each rollout of class C brings
     mmd_n_target / (n_gpus * n_roll) distinct class-C pool latents.
  3. CFG (cfg_gamma > 0): n_band DISTINCT trajectories, one band state each,
     become the band's FM rows (exact empirical-posterior labels, y = 1000 on
     dropped rows) and the rows of mean || (v_c - v_u) - (v*_c - v*_u) ||^2.
  4. FM: FreqFlow's loss. With CFG on, on-path rows cover t in [0, band_t_min)
     and the band rows fill the rest of the micro-batch; otherwise every row is
     on-path with t in [0, 1).
The MMD and CFG backwards run on the unwrapped model under no_sync and finish
before the FM forward, so the FM backward is the only one that all-reduces.

  accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_onpolicy.py \
      --config=configs/freqflow_onpolicy.py --config.init_path=/path/to/nnet_ema.pth \
      --config.dataset.path=/path/to/imagenet_sd_ema_feature --config.onpolicy.pool_path=/path/to/pool \
      --config.onpolicy.cfg_gamma=1.0 --config.onpolicy.mmd_gamma=100
"""
import builtins
import contextlib
import os
import sys
from pathlib import Path

import accelerate
import einops
import ml_collections
import torch
import wandb
from absl import logging
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import libs.autoencoder
import onpolicy
import sampler
import utils
from datasets import get_dataset

# train_ldm_discrete.py backprops loss.mean() / 2; every term here carries the same
# factor so the gamma-0 step is that step.
LOSS_SCALE = 0.5


def latent_sample(moments):
    """autoencoder.sample without loading the autoencoder: z = s * (mean + std * xi)."""
    mean, logvar = torch.chunk(moments, 2, dim=1)
    std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
    return onpolicy.SCALE_FACTOR * (mean + std * torch.randn_like(mean))


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    rank, world = accelerator.process_index, accelerator.num_processes
    logging.info(f'Process {rank} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    op = config.onpolicy
    accum = config.train.grad_accum
    assert config.train.batch_size % (world * accum) == 0
    micro_bs = config.train.batch_size // (world * accum)

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'onpolicy_{config.dataset.name}',
                   config=config.to_dict(), name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {world} devices, micro-batch {micro_bs} x {accum} accumulation per device')

    dataset = get_dataset(**config.dataset, train=True)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=micro_bs, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    if config.nnet.name == 'freqflow':  # memory options, as train_ldm_discrete.py: same math, up to float rounding
        from libs.ours.blocks_RoPERMSNorm import ConvNeXtBlock, Stage, TransformerBlock
        # a block's recompute reads the TimeDependentParameter weights its model call seeded: every backward here
        # runs before the next model call, and the rollout's checkpoint re-seeds each step
        for m in train_state.nnet.modules():
            if isinstance(m, Stage):
                m.checkpoint_lowres = config.train.get('checkpoint_lowres', False)
            if isinstance(m, (TransformerBlock, ConvNeXtBlock)):
                m.checkpoint = config.train.get('checkpoint_blocks', False)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)
    if train_state.step == 0 and config.init_path:
        logging.info(f'init nnet and nnet_ema from {config.init_path}')
        state_dict = torch.load(config.init_path, map_location='cpu')
        train_state.nnet.load_state_dict(state_dict)
        train_state.nnet_ema.load_state_dict(state_dict)
    raw = accelerator.unwrap_model(nnet)

    autoencoder = None
    if config.train.eval_interval > 0:
        autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
        autoencoder.to(device)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # --- on-policy setup ---
    cfg_on, mmd_on = op.cfg_gamma > 0, op.mmd_gamma > 0
    t_grid = onpolicy.sampler_grid(op.num_steps, op.shift)
    band = [k for k, t in enumerate(t_grid) if t > op.band_t_min]
    mmd_ks = band[1:]  # state 0 is the initial noise: same law as the target, no model dependence
    n_band = round(micro_bs * (1 - op.band_t_min)) if cfg_on else 0
    pool = None
    if cfg_on or mmd_on:
        if not band:
            raise ValueError(f'no sampler grid state has t > band_t_min={op.band_t_min}')
        if cfg_on and not 1 <= n_band <= min(op.n_roll, micro_bs - 1):
            raise ValueError(f'CFG band rows per micro-batch = {n_band} (micro-batch {micro_bs} x '
                             f'{1 - op.band_t_min:g}); need 1 <= n_band <= min(n_roll={op.n_roll}, micro-batch - 1)')
        if mmd_on and not mmd_ks:
            raise ValueError('MMD needs a band state past the initial noise; lower band_t_min or raise num_steps')
        pool = onpolicy.LatentPool(op.pool_path, device, rank, world)
        smallest = int(pool.class_count[pool.classes].min())
        for full, k, name in ((op.bank_full, op.bank_k, 'bank_k'), (op.bank_full_uncond, op.bank_k_uncond, 'bank_k_uncond')):
            if not full and not 1 <= k <= (smallest if name == 'bank_k' else pool.N):
                raise ValueError(f'{name}={k} is out of range for this pool')
        logging.info(f'pool: {pool.N} latents, {pool.classes.numel()} classes, shard rows {pool.lo}:{pool.hi}; '
                     f'band t = {[round(t_grid[k], 4) for k in band]}, {n_band} CFG/FM band rows per micro-batch')

    def make_gen(i):
        return torch.Generator(device=device).manual_seed(config.seed * 1000003 + rank * 1009 + i)

    gen_roll, gen_pick, gen_bank, gen_mmd = (make_gen(i) for i in range(4))
    if mmd_on:
        if op.mmd_n_target % (world * op.n_roll):
            raise ValueError(f'mmd_n_target={op.mmd_n_target} must be a multiple of n_gpus * n_roll = {world * op.n_roll}')
        if min(op.mmd_sigma_scales) < 1 / 80:
            raise ValueError('mmd_sigma_scales below 1/80 overflow the shifted float32 kernel exp((1 - d2/d2_base)/s)')
        per = op.mmd_n_target // (world * op.n_roll)
        floor = onpolicy.mmd_noise_floor(pool, mmd_ks, t_grid, op.n_roll, per, op.mmd_sigma_scales, make_gen(4))
        logging.info(f'MMD: {world * op.n_roll} rollout rows vs {op.mmd_n_target} class-matched targets ({per} per rollout) '
                     f'at t = {[round(t_grid[k], 4) for k in mmd_ks]}. Noise floor (two independent target draws):')
        for t, mean, peak, d2 in floor:
            logging.info(f'  t={t:.4f}  mean {mean:.3e}  max|.| {peak:.3e}  d2_base {d2:.1f}')

    def onpolicy_terms(metrics):
        """Rollout, then the MMD and CFG backwards (no all-reduce). Returns the
        band's FM rows (x_t, t, v, y), or None when CFG is off."""
        c = pool.sample_classes(op.n_roll, gen_roll)
        drop = torch.rand(op.n_roll, device=device, generator=gen_roll) < config.dataset.p_uncond
        with accelerator.no_sync(nnet), torch.set_grad_enabled(mmd_on):
            traj = onpolicy.rollout(raw, c, t_grid, len(band), gen_roll, op.cfg_scale, op.guidance_low,
                                    op.guidance_high, grad_ckpt=op.mmd_grad_ckpt)
            if mmd_on:
                mmd, per_t = onpolicy.mmd_band(pool, traj, mmd_ks, t_grid, c, per, op.mmd_sigma_scales, gen_mmd)
                accelerator.backward(op.mmd_gamma * mmd * LOSS_SCALE / accum)
                metrics['mmd'] = mmd.detach()
                for k, val in zip(mmd_ks, per_t):
                    metrics[f'mmd_t{t_grid[k]:.3f}'] = val.detach()
        if not cfg_on:
            return None

        # one state per trajectory, n_band distinct trajectories
        j = torch.randperm(op.n_roll, device=device, generator=gen_pick)[:n_band]
        k = torch.randint(0, len(band), (n_band,), device=device, generator=gen_pick)
        x = torch.stack([s.detach() for s in traj])[k, j]
        del traj
        t = torch.tensor(t_grid, device=device, dtype=torch.float32)[k]
        c, drop = c[j], drop[j]
        tt = t.view(-1, 1, 1, 1)
        e_c, e_u = pool.posterior_means(x, t, c, 0 if op.bank_full else op.bank_k,
                                        0 if op.bank_full_uncond else op.bank_k_uncond, gen_bank, op.expert_chunk)
        v_c, v_u = (x - e_c) / tt, (x - e_u) / tt
        if op.cfg_target_full and not (op.bank_full and op.bank_full_uncond):
            e_c, e_u = pool.posterior_means(x, t, c, chunk=op.expert_chunk)
            dv_target = (e_u - e_c) / tt
        else:
            dv_target = v_c - v_u

        with accelerator.no_sync(nnet):
            null = torch.full_like(c, onpolicy.NULL_CLASS)
            v2 = raw(torch.cat([x, x]), log_snr=torch.cat([t, t]) * 8 - 4, y=torch.cat([c, null]))[1].float()
            dv = v2[:n_band] - v2[n_band:]
            loss_cfg = (dv - dv_target).pow(2).mean()
            accelerator.backward(op.cfg_gamma * loss_cfg * LOSS_SCALE / accum)
        metrics['cfg'] = loss_cfg.detach()
        metrics['cfg_dv2_model'] = dv.detach().pow(2).mean()
        metrics['cfg_dv2_target'] = dv_target.pow(2).mean()

        y = torch.where(drop, torch.full_like(c, onpolicy.NULL_CLASS), c)
        v = torch.where(drop.view(-1, 1, 1, 1), v_u, v_c)
        return x, t, v, y

    def train_step(batches):
        sums = {}
        optimizer.zero_grad()
        active = (cfg_on or mmd_on) and train_state.step >= op.start_step
        for i, batch in enumerate(batches):
            metrics = {}
            band_rows = onpolicy_terms(metrics) if active else None
            z, y = latent_sample(batch[0].float()), batch[1]
            n_on, t_hi = (micro_bs - n_band, op.band_t_min) if band_rows is not None else (micro_bs, 1.0)
            z, y = z[:n_on], y[:n_on]
            t = torch.rand(n_on, device=device) * t_hi
            eps = torch.randn_like(z)
            x_t = (1 - t.view(-1, 1, 1, 1)) * z + t.view(-1, 1, 1, 1) * eps
            v = eps - z
            if band_rows is not None:
                x_t, t, v, y = (torch.cat([a, b]) for a, b in zip((x_t, t, v, y), band_rows))
            sync = i == len(batches) - 1
            with contextlib.nullcontext() if sync else accelerator.no_sync(nnet):
                loss = onpolicy.fm_loss(nnet, x_t, t, v, y)
                accelerator.backward(loss.mean() * LOSS_SCALE / accum)
            metrics['loss'] = loss.detach().mean()
            for key, val in metrics.items():
                sums[key] = sums.get(key, 0.) + val / len(batches)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        out = {key: accelerator.gather(val.reshape(1)).mean() for key, val in sorted(sums.items())}
        if 'cfg' in out or 'mmd' in out:
            out['loss_total'] = out['loss'] + op.cfg_gamma * out.get('cfg', 0.) + op.mmd_gamma * out.get('mmd', 0.)
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **out)

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batches = [tree_map(lambda x: x.to(device), next(data_generator)) for _ in range(accum)]
        metrics = train_step(batches)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            wandb.log(metrics, step=train_state.step)

        if autoencoder is not None and accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            y = einops.repeat(torch.arange(5, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
            z_init = torch.randn(5 * 10, *config.z_shape, device=device)
            z = sampler.euler_sampler(train_state.nnet_ema, z_init, y)
            with torch.cuda.amp.autocast():
                samples = autoencoder.decode(z.float())
            samples = make_grid(dataset.unpreprocess(samples), 10)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            logging.info(f'Save checkpoint {train_state.step}...')
            if accelerator.is_main_process:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=', 1)  # a path value may hold '=' (workdir/<config>/<hparams>/...)
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
