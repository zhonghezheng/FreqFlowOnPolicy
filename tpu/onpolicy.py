"""onpolicy.py's rollout and MMD in JAX. Time convention as FreqFlow: t = 0 data, t = 1 noise."""
import math

import jax
import jax.numpy as jnp

from tpu.pool import HIGHEST
from tpu.sampling import velocity


def rollout(apply, params, c, t_grid, n_states, key, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
            groups=1, remat_steps=False, shard=lambda a: a):
    """sampler.euler_sampler from t = 1 conditioned on c: the first n_states grid states (n_states - 1 Euler
    steps). Differentiable; remat_steps recomputes each step in backward (onpolicy.rollout's grad_ckpt).
    shard pins the trajectories' rows to their chips."""
    x = shard(jax.random.normal(key, (c.shape[0], 4, 32, 32)))
    traj = [x]
    for k in range(n_states - 1):
        t, dt = t_grid[k], t_grid[k + 1] - t_grid[k]
        f = lambda p, x, t=t: velocity(apply, p, x, t, c, cfg_scale, guidance_low, guidance_high, groups)
        x = shard(x + dt * (jax.checkpoint(f) if remat_steps else f)(params, x))
        traj.append(x)
    return traj


def _sq_dists(a, b):
    return jnp.maximum(jnp.sum(a * a, 1)[:, None] - 2. * jnp.dot(a, b.T, precision=HIGHEST)
                       + jnp.sum(b * b, 1)[None], 0.)


def mmd2_unbiased(x, y, scales):
    """onpolicy.mmd2_unbiased: unbiased MMD^2 between x [m, D] and y [n, D] (no gradient), kernel
    mean_l exp(-||a - b||^2 / (scale_l d2_base)) with d2_base the mean distinct-pair distance among y,
    evaluated as exp(-1/s) exp(-(d2/d2_base - 1)/s) with masked diagonals. Returns (MMD^2, d2_base)."""
    y = jax.lax.stop_gradient(y)
    m, n = x.shape[0], y.shape[0]
    d2_yy = _sq_dists(y, y)
    d2_base = jnp.maximum((d2_yy.sum() - jnp.trace(d2_yy)) / (n * (n - 1)), 1e-12)
    r_xx = jnp.where(jnp.eye(m, dtype=bool), jnp.inf, _sq_dists(x, x) / d2_base - 1)
    r_yy = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, d2_yy / d2_base - 1)
    r_xy = _sq_dists(x, y) / d2_base - 1
    val = 0.
    for s in scales:
        k_xx, k_yy, k_xy = (jnp.exp(-r / s) for r in (r_xx, r_yy, r_xy))
        val = val + math.exp(-1 / s) * (k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2. * k_xy.mean())
    return val / len(scales), d2_base


def mmd_band(pool, arrays, traj, ks, t_grid, c, per, scales, key):
    """Mean over grid states k in ks of one MMD^2 between all rollout states at t_k and noised targets: per
    distinct pool latents of each rollout's class, freshly drawn at every t_k. Returns (mean, [per k], ok)."""
    vals, ok = [], jnp.ones((), bool)
    for i, k in enumerate(ks):
        t = t_grid[k]
        k_rows, k_eps = jax.random.split(jax.random.fold_in(key, i))
        rows, ok_k = pool.class_rows(arrays, c, per, k_rows)
        x0 = pool.gather(arrays, rows.reshape(-1))
        y = (1 - t) * x0 + t * jax.random.normal(k_eps, x0.shape)
        vals.append(mmd2_unbiased(traj[k].reshape(c.shape[0], -1).astype(jnp.float32), y, scales)[0])
        ok = ok & ok_k
    return jnp.stack(vals).mean(), vals, ok


def mmd_noise_floor(pool, arrays, ks, t_grid, n_rows, per, scales, key, repeats=5):
    """The estimator's null at the training sizes: MMD^2 between two independent draws of the target law (n_rows
    stand-in rollout rows against per class-matched targets each). Returns [(t, mean, max |MMD^2|, d2_base)]."""
    out = []
    for i, k in enumerate(ks):
        t, vals = t_grid[k], []
        for r in range(repeats):
            kc, kr, ke = jax.random.split(jax.random.fold_in(key, i * repeats + r), 3)
            c = pool.sample_classes(arrays, n_rows, kc)
            rows, _ = pool.class_rows(arrays, c, per + 1, kr)
            x0 = pool.gather(arrays, rows.reshape(-1)).reshape(n_rows, per + 1, -1)
            noised = (1 - t) * x0 + t * jax.random.normal(ke, x0.shape)
            val, d2 = mmd2_unbiased(noised[:, 0], noised[:, 1:].reshape(n_rows * per, -1), scales)
            vals.append(float(val))
        out.append((t, sum(vals) / len(vals), max(abs(v) for v in vals), float(d2)))
    return out
