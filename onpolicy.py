"""Helpers for train_onpolicy.py: the sharded latent pool and its empirical
posterior, the sampler rollout, the MMD estimator and FreqFlow's FM loss.

Time convention is FreqFlow's: t=0 is data, t=1 is noise,
    x_t = (1 - t) x0 + t eps,    v = eps - x0,
so the empirical-posterior velocity is v*(x_t) = (x_t - E[x0 | x_t]) / t, with
    E[x0 | x_t] = sum_i w_i x_i,    w_i ∝ exp(-||x_t - (1-t) x_i||^2 / (2 t^2)).
"""
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from fft2d import Fourier_filter
from sde import multi_scale_targets

SCALE_FACTOR = 0.18215  # libs.autoencoder.get_model's default, applied by autoencoder.sample
NULL_CLASS = 1000       # CFGDataset's empty_token and the sampler's y_null


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_gather_cat(x):
    """Every rank's x concatenated along dim 0, in rank order (equal shapes). No grad."""
    if not _distributed():
        return x
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out)


class _GatherWithGrad(torch.autograd.Function):
    """all_gather whose backward hands each rank the gradient of its own slice
    SUMMED over ranks. Every rank evaluates the same pooled loss, so that sum is
    world_size x the true slice gradient, and DDP's gradient average divides the
    factor back out."""

    @staticmethod
    def forward(ctx, x):
        return all_gather_cat(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        n = grad.shape[0] // dist.get_world_size()
        rank = dist.get_rank()
        return grad[rank * n:(rank + 1) * n]


def gather_with_grad(x):
    return _GatherWithGrad.apply(x) if _distributed() else x


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def fm_loss(model, x_t, t, v, y):
    """train_ldm_discrete.SILoss's per-row loss at given (x_t, t, v): the spatial
    velocity MSE plus 1/4 of the frequency branch's MSE against the 2x-pooled
    [low, high, v] split of the target."""
    out = model(x_t, log_snr=t * 8 - 4, y=y)
    low, high = Fourier_filter(v, 8)
    target = multi_scale_targets(v, levels=len(out), scale_correction=True)[32]
    freq_target = multi_scale_targets(torch.cat([low, high, v], dim=1), levels=len(out), scale_correction=True)[16]
    return mean_flat((out[1] - target) ** 2) + mean_flat((out[0] - freq_target) ** 2) / 4.


def sampler_grid(num_steps, shift):
    """sampler.euler_sampler's time grid, t_0 = 1 (noise) down to t_N = 0."""
    t = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    return (shift * t / (1 + (shift - 1) * t)).tolist()


def distinct_randint(n, k, high, gen, device):
    """[n, k] indices in [0, high), distinct within each row.

    Draws with replacement, then redraws the repeats until none remain. The
    procedure treats every index alike, so each row is a uniform k-subset. With
    k << high (a 1024-row bank of ~2.5M) the first draw almost never repeats.
    """
    if k > high:
        raise ValueError(f"cannot draw {k} distinct indices from {high}")
    idx = torch.randint(0, high, (n, k), device=device, generator=gen)
    while True:
        srt, order = idx.sort(dim=1)
        rep = torch.zeros_like(idx, dtype=torch.bool)
        rep[:, 1:] = srt[:, 1:] == srt[:, :-1]
        if not rep.any():
            return idx
        rep = torch.zeros_like(rep).scatter_(1, order, rep)
        idx[rep] = torch.randint(0, high, (int(rep.sum()),), device=device, generator=gen)


def _accumulate(m, s, w, logits, p):
    """One chunk of a streaming softmax-weighted mean: running max m [Q], running
    sum of exp(logit - m) s [Q], running weighted sum w [Q, D]. Rows whose
    candidates are all -inf so far keep m = -inf and s = w = 0."""
    m_new = torch.maximum(m, logits.max(1).values)
    ref = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
    alpha = torch.exp(m - ref)
    e = torch.exp(logits - ref[:, None])
    return m_new, s * alpha + e.sum(1), w * alpha[:, None] + e @ p


class LatentPool:
    """The pool written by dump_latent_pool.py, sharded across ranks.

    Rank r keeps the r-th contiguous row range on its GPU. The dump stores rows in
    a seeded random order, so every shard is a random subset of the pool holding
    each class at about its pool share. Labels and the per-class index table stay
    global on every rank, so banks can be drawn from the whole pool; only the
    latents are sharded.
    """

    def __init__(self, path, device, rank=0, world=1):
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        latents = np.load(os.path.join(path, 'latents.npy'), mmap_mode='r')
        labels = np.load(os.path.join(path, 'labels.npy'))
        assert latents.shape[0] == labels.shape[0]
        self.device, self.rank, self.world = device, rank, world
        self.N, self.D = latents.shape[0], int(np.prod(latents.shape[1:]))
        self.lo, self.hi = rank * self.N // world, (rank + 1) * self.N // world
        self.shard = torch.from_numpy(np.ascontiguousarray(latents[self.lo:self.hi])).view(
            self.hi - self.lo, self.D).to(device)
        self.shard_sq = self.shard.float().pow(2).sum(1)

        self.labels = torch.from_numpy(labels.astype(np.int64)).to(device)
        n_classes = int(self.labels.max()) + 1
        self.class_count = torch.bincount(self.labels, minlength=n_classes)
        self.classes = (self.class_count > 0).nonzero().squeeze(1)
        # class_idx[g, :class_count[g]] = the global pool rows of class g
        order = torch.argsort(self.labels, stable=True)
        starts = torch.cumsum(self.class_count, 0) - self.class_count
        pos = torch.arange(self.N, device=device) - starts[self.labels[order]]
        self.class_idx = torch.zeros(n_classes, int(self.class_count.max()), dtype=torch.long, device=device)
        self.class_idx[self.labels[order], pos] = order
        # this shard's members of class g: local_order[local_start[g] : local_start[g] + local_count[g]]
        self.local_labels = self.labels[self.lo:self.hi]
        local_count = torch.bincount(self.local_labels, minlength=n_classes)
        self.local_order = torch.argsort(self.local_labels, stable=True)
        self.local_start = (torch.cumsum(local_count, 0) - local_count).tolist()
        self.local_count = local_count.tolist()

    def class_bank(self, c, k, gen):
        """[n, k] global pool rows: k distinct members of each row's class."""
        counts = self.class_count[c]
        if int(counts.min()) < k:
            raise ValueError(f"bank_k={k} exceeds the smallest class in the batch ({int(counts.min())} rows)")
        keys = torch.rand(c.shape[0], self.class_idx.shape[1], device=self.device, generator=gen)
        keys.masked_fill_(torch.arange(keys.shape[1], device=self.device)[None] >= counts[:, None], 2.)
        return self.class_idx[c[:, None], keys.topk(k, dim=1, largest=False).indices]

    def uniform_bank(self, n, k, gen):
        """[n, k] global pool rows: k distinct rows of the whole pool per row."""
        return distinct_randint(n, k, self.N, gen, self.device)

    def _bank_mask(self, idx):
        """[Q, shard rows] bool: which of this shard's rows are in each query's bank."""
        own = (idx >= self.lo) & (idx < self.hi)
        rows = torch.arange(idx.shape[0], device=self.device)[:, None].expand_as(idx)
        mask = torch.zeros(idx.shape[0], self.hi - self.lo, dtype=torch.bool, device=self.device)
        mask[rows[own], idx[own] - self.lo] = True
        return mask

    @torch.no_grad()
    def posterior_means(self, x_t, t, c, bank_k=0, bank_k_uncond=0, gen=None, chunk=16384):
        """(E[x0 | x_t, class c], E[x0 | x_t]) for every row, as two tensors shaped
        like x_t: the empirical posterior over the row's class and over the whole
        pool. bank_k / bank_k_uncond = 0 scores every candidate (exact); K > 0
        scores a per-row bank of K distinct candidates instead.

        Collective: every rank must call it with the same row count. Queries are
        gathered, each rank scores them against its shard, and the per-shard
        softmax partials are merged with a max all-reduce and a sum all-reduce.
        """
        n = x_t.shape[0]
        xq = all_gather_cat(x_t.flatten(1).float())
        tq = all_gather_cat(t.float())
        cq = all_gather_cat(c)
        Q = xq.shape[0]
        mask_c = self._bank_mask(all_gather_cat(self.class_bank(c, bank_k, gen))) if bank_k else None
        mask_u = self._bank_mask(all_gather_cat(self.uniform_bank(n, bank_k_uncond, gen))) if bank_k_uncond else None

        # log w_i up to a per-query constant: (2a x_t.x_i - a^2 ||x_i||^2) / (2t^2), a = 1 - t
        a = (1 - tq)[:, None]
        inv = 1 / (2 * tq.clamp_min(1e-3)[:, None] ** 2)
        state = [(torch.full((Q,), -float('inf'), device=self.device),
                  torch.zeros(Q, device=self.device),
                  torch.zeros(Q, self.D, device=self.device)) for _ in range(2)]
        for s in range(0, self.hi - self.lo, chunk):
            e = min(s + chunk, self.hi - self.lo)
            p = self.shard[s:e].float()
            logits = (2 * a * (xq @ p.T) - a ** 2 * self.shard_sq[None, s:e]) * inv
            in_c = (self.local_labels[None, s:e] == cq[:, None]) if mask_c is None else mask_c[:, s:e]
            state[0] = _accumulate(*state[0], logits.masked_fill(~in_c, -float('inf')), p)
            l_u = logits if mask_u is None else logits.masked_fill(~mask_u[:, s:e], -float('inf'))
            state[1] = _accumulate(*state[1], l_u, p)

        m = torch.stack([state[0][0], state[1][0]])                                  # [2, Q]
        sw = torch.cat([torch.stack([state[0][1], state[1][1]])[..., None],
                        torch.stack([state[0][2], state[1][2]])], dim=-1)            # [2, Q, 1 + D]
        m_all = m.clone()
        if _distributed():
            dist.all_reduce(m_all, op=dist.ReduceOp.MAX)
        sw = sw * torch.exp(m - torch.where(torch.isfinite(m_all), m_all, torch.zeros_like(m_all)))[..., None]
        if _distributed():
            dist.all_reduce(sw)
        if not (sw[..., 0] > 0).all():
            raise RuntimeError("a query has an empty candidate set")
        e_x0 = (sw[..., 1:] / sw[..., :1])[:, self.rank * n:(self.rank + 1) * n]
        return e_x0[0].view_as(x_t), e_x0[1].view_as(x_t)

    def sample_classes(self, n, gen):
        """n class labels distributed like the pool's class frequencies (the label of a
        uniformly drawn pool row), so class-conditioned rollouts mix like the data."""
        return self.labels[torch.randint(0, self.N, (n,), device=self.device, generator=gen)]

    def class_targets(self, c, per, gen):
        """[len(c), per, D] float32 latents from THIS rank's shard: per of each
        row's class, no latent used twice in the call. Shards are disjoint, so
        draws on different ranks never coincide either."""
        out = torch.empty(c.shape[0], per, dtype=torch.long, device=self.device)
        for cls in c.unique().tolist():
            rows = (c == cls).nonzero().squeeze(1)
            need, count = rows.numel() * per, self.local_count[cls]
            if count < need:
                raise ValueError(f"class {cls} has {count} latents on rank {self.rank}'s shard, "
                                 f"{need} needed; lower mmd_n_target or n_roll")
            members = self.local_order[self.local_start[cls]:self.local_start[cls] + count]
            out[rows] = members[torch.randperm(count, device=self.device, generator=gen)[:need]].view(-1, per)
        return self.shard[out.flatten()].float().view(c.shape[0], per, self.D)


def rollout(model, c, t_grid, n_states, gen, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
            grad_ckpt=False):
    """sampler.euler_sampler from t=1, every trajectory conditioned on its class c.

    Returns the first n_states grid states [x(t_grid[0]), ..., x(t_grid[n_states-1])],
    i.e. n_states - 1 Euler steps. Guidance follows the sampler: v_u + s (v_c - v_u)
    when cfg_scale > 1 and guidance_low <= t <= guidance_high. Differentiable when
    grad is enabled; grad_ckpt then recomputes each step in backward. The
    checkpoint wraps the whole model call, which re-seeds the TimeDependentParameter
    weights for its t, so the recompute sees the weights the forward used.
    """
    n = c.shape[0]
    null = torch.full_like(c, NULL_CLASS)

    def velocity(x, t):
        log_snr = torch.full((n,), t * 8 - 4, device=x.device)
        if cfg_scale > 1.0 and guidance_low <= t <= guidance_high:
            v = model(torch.cat([x, x]), log_snr=log_snr.repeat(2), y=torch.cat([c, null]))[1].float()
            v_c, v_u = v.chunk(2)
            return v_u + cfg_scale * (v_c - v_u)
        return model(x, log_snr=log_snr, y=c)[1].float()

    x = torch.randn(n, 4, 32, 32, device=c.device, generator=gen)
    traj = [x]
    for k in range(n_states - 1):
        t, dt = t_grid[k], t_grid[k + 1] - t_grid[k]
        if grad_ckpt and torch.is_grad_enabled():
            v = checkpoint(velocity, x, t, use_reentrant=False)
        else:
            v = velocity(x, t)
        x = x + dt * v
        traj.append(x)
    return traj


def _sq_dists(a, b):
    """[m, n] squared distances via ||a||^2 - 2a.b + ||b||^2, clamped at 0."""
    return (a.pow(2).sum(1)[:, None] - 2. * (a @ b.T) + b.pow(2).sum(1)[None]).clamp_min(0.)


def mmd2_unbiased(x, y, scales):
    """Unbiased MMD^2 between x [m, D] (with grad) and y [n, D] (detached), with
    k(a, b) = mean_l exp(-||a - b||^2 / (scale_l * d2_base)) and d2_base the mean
    squared distance between two distinct y rows. Measuring the bandwidth on the
    data side keeps the kernel independent of the model. Returns (MMD^2, d2_base).

    Unbiased for independent rows. With mmd_band's class-matched targets each rollout
    also meets its own class's block while rollouts never meet a same-class rollout,
    which biases the estimate by about -(K_same - K_diff)/m. In 4096-D near noise every distance is within a few % of d2_base, so at s ~ 0.1
    the kernel values are ~exp(-1/s) and a float32 sum next to a diagonal of ones
    loses them. Each kernel is evaluated as exp(-1/s) * exp(-(d2/d2_base - 1)/s),
    with the diagonals masked out rather than subtracted.
    """
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"MMD^2 needs at least 2 rows per side, got {m} and {n}")
    d2_yy = _sq_dists(y, y)
    d2_base = ((d2_yy.sum() - d2_yy.diagonal().sum()) / (n * (n - 1))).clamp_min(1e-12)
    eye = lambda k: torch.eye(k, dtype=torch.bool, device=x.device)
    r_xx = (_sq_dists(x, x) / d2_base - 1).masked_fill(eye(m), float('inf'))
    r_yy = (d2_yy / d2_base - 1).masked_fill(eye(n), float('inf'))
    r_xy = _sq_dists(x, y) / d2_base - 1
    val = 0.
    for s in scales:
        k_xx, k_yy, k_xy = (torch.exp(-r / s) for r in (r_xx, r_yy, r_xy))
        val = val + math.exp(-1 / s) * (k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2. * k_xy.mean())
    return val / len(scales), d2_base


def mmd_band(pool, traj, ks, t_grid, c, per, scales, gen):
    """Mean over grid states k in ks of one pooled MMD^2 between all ranks' rollout
    states at t_k and the noised targets: per distinct pool latents of each
    trajectory's class C, so the target mixes the batch's classes exactly as the
    rollouts do, freshly drawn at every t_k. Every rollout is compared with every
    target. Returns (mean MMD^2, [MMD^2 per k])."""
    vals = []
    for k in ks:
        t = t_grid[k]
        x = gather_with_grad(traj[k].flatten(1).float())
        x0 = pool.class_targets(c, per, gen).flatten(0, 1)
        eps = torch.randn(x0.shape, device=x0.device, generator=gen)
        y = all_gather_cat((1 - t) * x0 + t * eps)
        vals.append(mmd2_unbiased(x, y, scales)[0])
    return torch.stack(vals).mean(), vals


@torch.no_grad()
def mmd_noise_floor(pool, ks, t_grid, n_roll, per, scales, gen, repeats=5):
    """The estimator's null at the training sizes: MMD^2 between two independent
    draws of the target law (n_roll stand-in rollout rows per rank against per
    class-matched targets each), with no latent shared between or within the two
    sides. Returns [(t, mean, max |MMD^2|, d2_base)] per grid state."""
    out = []
    for k in ks:
        t, vals = t_grid[k], []
        for _ in range(repeats):
            c = pool.sample_classes(n_roll, gen)
            x0 = pool.class_targets(c, per + 1, gen)
            noised = (1 - t) * x0 + t * torch.randn(x0.shape, device=x0.device, generator=gen)
            val, d2_base = mmd2_unbiased(all_gather_cat(noised[:, 0]), all_gather_cat(noised[:, 1:].flatten(0, 1)), scales)
            vals.append(val.item())
        out.append((t, sum(vals) / len(vals), max(abs(v) for v in vals), d2_base.item()))
    return out
