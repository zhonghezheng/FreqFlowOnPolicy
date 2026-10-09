import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    elif name == 'freqflow':
        from libs.ours.blocks_RoPERMSNorm import FreqFlow
        return FreqFlow(kwargs["model_args"])
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, zero=False, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        if zero:  # ZeRO-1: each rank keeps the AdamW state of only its shard of the parameters
            from torch.distributed.optim import ZeroRedundancyOptimizer
            return ZeroRedundancyOptimizer(params, optimizer_class=AdamW, **kwargs)
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))
        if is_zero(self.optimizer):
            # ZeroRedundancyOptimizer.load_state_dict also leaves a copy of this rank's shard of the state in the
            # wrapper; only its local optimizer's copy is used
            self.optimizer.state.clear()

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def trainable_parameters(nnet):
    params_decay = []
    params_nodecay = []
    for name, param in nnet.named_parameters():
        if name.endswith(".nodecay_weight") or name.endswith(".nodecay_bias"):
            params_nodecay.append(param)
        else:
            params_decay.append(param)
    print("params_decay", len(params_decay))
    print("params_nodecay", len(params_nodecay))
    params = [
        {'params': params_decay},
        {'params': params_nodecay, 'weight_decay': 0.0}
    ]
    return params


def is_zero(optimizer):
    from torch.distributed.optim import ZeroRedundancyOptimizer
    return isinstance(optimizer, ZeroRedundancyOptimizer)


class ShardedEMA:
    """EMA of only the parameters this rank's ZeroRedundancyOptimizer updates; the ranks' shards are disjoint.

    Replaces a full EMA copy of the model on every GPU. gather() assembles the full EMA into nnet_ema on the
    ranks that pass here=True; it is collective, so every rank must call it.
    """

    def __init__(self, nnet, nnet_ema, optimizer):
        owner = optimizer._param_to_rank  # the partition the optimizer steps; torch has no public accessor
        ema_params = dict(nnet_ema.named_parameters())
        self.rank = optimizer.rank
        self.params = list(nnet.named_parameters())  # same order on every rank
        self.owner = [owner[p] for _, p in self.params]
        mine = [(n, p) for (n, p), r in zip(self.params, self.owner) if r == self.rank]
        self.src = [p for _, p in mine]
        self.ema = [ema_params[n].detach().to(p.device, copy=True) for n, p in mine]

    @torch.no_grad()
    def update(self, rate):
        torch._foreach_mul_(self.ema, rate)
        torch._foreach_add_(self.ema, self.src, alpha=1 - rate)

    @torch.no_grad()
    def gather(self, nnet_ema, here):
        ema_params = dict(nnet_ema.named_parameters())
        mine = iter(self.ema)
        for (name, p), r in zip(self.params, self.owner):
            buf = next(mine) if r == self.rank else torch.empty_like(p)
            torch.distributed.broadcast(buf, src=r)
            if here:
                ema_params[name].copy_(buf)


def initialize_train_state(config, device):

    nnet = get_nnet(**config.nnet)
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')
    nnet.to(device)  # before the optimizer: ZeroRedundancyOptimizer communicates on its parameters' device

    optimizer = get_optimizer(trainable_parameters(nnet), **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.to(device)
    train_state.ema_update(0)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, evenly_sample=False):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    ## TODO: rewrite this function: if n_samples == 50000 and accelerator.num_processes==8 and mini_batch_size == 50:
    if evenly_sample:
        assert n_samples == 50000 # for exenly sampling, we pnly support FID50K (each category generates 50 images)

        bz_list = amortize(n_samples, batch_size)
        for kk in tqdm(range(len(bz_list)), disable=not accelerator.is_main_process, desc='sample2dir'):
            _batch_size = bz_list[kk]
            samples = unpreprocess_fn(sample_fn(mini_batch_size, category = kk, accelerator=accelerator))
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
            if accelerator.is_main_process:
                for sample in samples:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                    idx += 1
    else:
        for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
            samples = unpreprocess_fn(sample_fn(mini_batch_size))
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
            if accelerator.is_main_process:
                for sample in samples:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                    idx += 1



def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
