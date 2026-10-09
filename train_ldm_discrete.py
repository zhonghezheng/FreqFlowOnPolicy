import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import contextlib
import os
import wandb
import libs.autoencoder
import libs.vae
import fid_eval
import numpy as np
from fft2d import Fourier_filter
from sde import multi_scale_targets
import sampler
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)

def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0, scaling_factor):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)

        if scaling_factor == -1:
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        else:
            # https://arxiv.org/pdf/2301.10972
            xn = scaling_factor * stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
            xn = xn / xn.std(axis=(1,2,3), keepdims=True)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


# def LSimple(x0, nnet, schedule, **kwargs):
#     n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
#     eps_pred = nnet(xn, n, **kwargs)
#     return mos(eps - eps_pred)
class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, loss_coeffs=None, y=None):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output  = model(model_input, log_snr=time_input.flatten()* 8 - 4, y=y)

        low,high = Fourier_filter(model_target, 8)

        target = multi_scale_targets(model_target, levels = len(model_output), scale_correction = True)[32]
        freq_target = multi_scale_targets(torch.cat([low, high, model_target],dim=1), levels = len(model_output), scale_correction = True)[16]

        denoising_loss = mean_flat((model_output[1] - target) ** 2) + mean_flat((model_output[0] - freq_target) ** 2)/4.

        return denoising_loss
    
def LSimple(x0, nnet, schedule, loss_coeffs, scaling_factor, **kwargs):
    n, eps, xn = schedule.sample(x0, scaling_factor)  # n in {1, ..., 1000}

    # possible_log_snr = torch.tensor(schedule.snr, device = x0.device).float().log()
    # log_snr = possible_log_snr[n]
    log_snr = (n.float() / 1000) * 8 - 4 # compute from timestep

    prediction = nnet(xn, log_snr = log_snr, **kwargs)
    low,high = Fourier_filter(eps, 8)
    target = multi_scale_targets(eps, levels = len(prediction), scale_correction = True)[32]
    freq_target = multi_scale_targets(torch.cat([low, high, eps],dim=1), levels = len(prediction), scale_correction = True)[16]
    loss = mos(prediction[0] - freq_target)/4. + mos(prediction[1] - target)
    return loss


def train(config):
    # without it, fragmentation runs the large per-GPU batches out of memory; read at the first CUDA allocation
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    # grads as views into DDP's buckets, instead of a second full-size fp32 copy
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(gradient_as_bucket_view=True)])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    accum = config.train.get('grad_accum', 1)  # micro-batches per optimizer step
    assert config.train.batch_size % (accelerator.num_processes * accum) == 0
    mini_batch_size = config.train.batch_size // (accelerator.num_processes * accum)

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset, train=True)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    if config.nnet.name == 'freqflow':  # memory/speed options: same math, up to float rounding
        import torch._functorch.config
        from libs.ours.blocks_RoPERMSNorm import ConvNeXtBlock, Stage, TransformerBlock
        # the partitioner of compiled blocks recomputes cheap ops in backward until saved activations fit the
        # budget (1.0 saves everything)
        torch._functorch.config.activation_memory_budget = config.train.get('activation_memory_budget', 1.0)
        for m in train_state.nnet.modules():
            if isinstance(m, Stage):
                m.checkpoint_lowres = config.train.get('checkpoint_lowres', False)
            if isinstance(m, (TransformerBlock, ConvNeXtBlock)):
                m.checkpoint = config.train.get('checkpoint_blocks', False)
                if config.train.get('compile', False):
                    m.compile()  # block by block: the identical blocks of a stage share one compiled graph
    # the EMA is never backpropagated: wrapping it in DDP would allocate a full set of gradient buckets for it
    train_state.nnet_ema.requires_grad_(False)
    zero = utils.is_zero(train_state.optimizer)
    nnet, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.optimizer, train_dataset_loader,
        device_placement=[True, not zero, True])  # placing a ZeRO optimizer would call state_dict() unconsolidated
    nnet_ema = train_state.nnet_ema
    train_state.ema_update(0)  # restart the EMA from the weights DDP just broadcast from rank 0
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)
    if train_state.step > 0:  # a resumed job must not replay the first job's data order and noise
        accelerate.utils.set_seed(config.seed + train_state.step, device_specific=True)
    ema = None
    if zero:
        # each rank keeps the EMA of only its optimizer shard; the full EMA module stays on the CPU and is
        # assembled there to sample and to save
        ema = utils.ShardedEMA(train_state.nnet, nnet_ema, train_state.optimizer)
        nnet_ema.to('cpu')
        torch.cuda.empty_cache()

    if config.autoencoder.pretrained_path.endswith("ckpt"):
        autoencoder = libs.vae.AutoencoderKL(embed_dim=16, ch_mult=(1,1,2,2,4), ckpt_path=config.autoencoder.pretrained_path)
    else:
        autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    # training on precomputed features only calls autoencoder.sample, which uses no weights: keep the VAE on the
    # CPU then, and bring it over only to decode the sample grid
    features = 'feature' in config.dataset.name and not config.autoencoder.pretrained_path.endswith("ckpt")
    vae_device = 'cpu' if features else device
    autoencoder.to(vae_device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        if config.autoencoder.pretrained_path.endswith("ckpt"):
            return autoencoder.decode(_batch / 0.2325)
        else:
            return autoencoder.decode(_batch)

    def evaluate_fid():
        # every rank samples its share from the full EMA model; the training step's activations are freed by now
        if ema is not None:
            ema.gather(nnet_ema, here=True)
        nnet_ema.to(device)
        autoencoder.to(device)
        torch.cuda.empty_cache()

        def sample_fn(y, z):
            z = sampler.euler_sampler(nnet_ema, z, y, num_steps=config.fid.num_steps, cfg_scale=config.fid.cfg_scale,
                                      guidance_high=config.fid.guidance_high).float()
            return dataset.unpreprocess(torch.cat([decode(c) for c in z.split(50)]))  # the decoder is large

        value = fid_eval.fid(sample_fn, config.fid.n_samples, config.fid.mini_batch_size, dataset.fid_stat,
                             config.z_shape, device, seed=config.seed)
        if zero:
            nnet_ema.to('cpu')
        autoencoder.to(vae_device)
        torch.cuda.empty_cache()
        return value

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')


    def train_step(_batches):
        _metrics = dict()
        optimizer.zero_grad()
        losses = []
        for i, _batch in enumerate(_batches):
            # only the last micro-batch's backward all-reduces
            with contextlib.nullcontext() if i == len(_batches) - 1 else accelerator.no_sync(nnet):
                _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
                loss_fn = SILoss()
                loss = loss_fn(nnet, _z, loss_coeffs=config.loss_coeffs, y=_batch[1])
                accelerator.backward(loss.mean() / 2 / len(_batches))
            losses.append(loss.detach())
        _metrics['loss'] = accelerator.gather(torch.cat(losses)).mean()
        optimizer.step()
        lr_scheduler.step()
        if ema is None:
            train_state.ema_update(config.get('ema_rate', 0.9999))
        else:
            ema.update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    while train_state.step < config.train.n_steps:
        nnet.train()
        batches = [tree_map(lambda x: x.to(device), next(data_generator)) for _ in range(accum)]
        metrics = train_step(batches)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if ema is not None and train_state.step % config.train.eval_interval == 0:
            ema.gather(nnet_ema, here=accelerator.is_main_process)
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            nnet_ema.to(device)
            autoencoder.to(device)
            y = einops.repeat(torch.arange(5, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
            z_init = torch.randn(5 * 10, *config.z_shape, device=device)
            z = sampler.euler_sampler(nnet_ema,z_init, y)
            samples = decode(z.float())
            samples = make_grid(dataset.unpreprocess(samples), 10)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            if zero:
                nnet_ema.to('cpu')
            autoencoder.to(vae_device)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if zero:  # collective: assemble the full optimizer state and EMA on the main process's CPU
                train_state.optimizer.consolidate_state_dict(to=0)
                ema.gather(nnet_ema, here=accelerator.is_main_process)
            if accelerator.is_main_process:
                # write under a name resume() ignores, then rename, so a job killed mid-save
                # never leaves a half-written .ckpt for the next job to resume from
                tmp = os.path.join(config.ckpt_root, f'{train_state.step}.tmp')
                train_state.save(tmp)
                os.replace(tmp, os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            if zero:
                train_state.optimizer._all_state_dicts = []  # drop the consolidated copy once written
            accelerator.wait_for_everyone()
            if 'fid' in config and train_state.step >= config.fid.start:
                logging.info(f'FID-{config.fid.n_samples} of the EMA model at step {train_state.step}...')
                value = evaluate_fid()
                if accelerator.is_main_process:
                    logging.info(f'step {train_state.step}: FID-{config.fid.n_samples} {value:.4f}')
                    wandb.log({'fid': value}, step=train_state.step)
                    with open(os.path.join(config.workdir, 'fid.txt'), 'a') as f:
                        f.write(f'{train_state.step} {value:.4f}\n')
                accelerator.wait_for_everyone()
            
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    fid_log = os.path.join(config.workdir, 'fid.txt')  # also holds the evaluations of earlier, resumed jobs
    if accelerator.is_main_process and os.path.exists(fid_log):
        step_fid = [(int(step), float(value)) for step, value in (line.split() for line in open(fid_log))]
        logging.info(f'step_fid: {step_fid}')
        logging.info(f'step_best: {min(step_fid, key=lambda x: x[1])[0]}')
    accelerator.wait_for_everyone()



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


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
            hparam, val = argv[i].split('=')
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
