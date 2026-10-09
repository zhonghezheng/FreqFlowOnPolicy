"""sampler.euler_sampler in JAX: the same shifted time grid, Euler steps and limited-interval guidance."""
import jax
import jax.numpy as jnp
import numpy as np

from tpu.common import NULL_CLASS


def sampler_grid(num_steps, shift=0.3):
    """Times t_0 = 1 (noise) ... t_N = 0, t = shift s / (1 + (shift - 1) s) over a uniform s grid."""
    s = np.linspace(1, 0, num_steps + 1, dtype=np.float64)
    return (shift * s / (1 + (shift - 1) * s)).tolist()


def pair(a, b, groups):
    """Per device group, [a_g; b_g]: the batch each GPU rank builds with torch.cat([a, b])."""
    rest = a.shape[1:]
    return jnp.concatenate([a.reshape(groups, -1, *rest), b.reshape(groups, -1, *rest)], axis=1).reshape(-1, *rest)


def unpair(x, groups):
    rest = x.shape[1:]
    x = x.reshape(groups, 2, -1, *rest)
    return x[:, 0].reshape(-1, *rest), x[:, 1].reshape(-1, *rest)


def velocity(apply, params, x, t, y, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0, groups=1):
    """The sampler's velocity at a python-float time t: v_u + s (v_c - v_u) inside the guidance interval."""
    log_snr = jnp.full((x.shape[0],), t * 8 - 4, jnp.float32)
    if cfg_scale > 1.0 and guidance_low <= t <= guidance_high:
        v = apply(params, pair(x, x, groups), pair(log_snr, log_snr, groups),
                  pair(y, jnp.full_like(y, NULL_CLASS), groups), groups)[1]
        v_c, v_u = unpair(v, groups)
        return v_u + cfg_scale * (v_c - v_u)
    return apply(params, x, log_snr, y, groups)[1]


def euler_sample(apply, params, z, y, num_steps=20, shift=0.3, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
                 groups=1):
    """z [B, C, H, W] noise at t = 1 -> latents at t = 0. One compiled step per distinct (guided or not)
    branch inside a fori_loop over the grid."""
    grid = jnp.asarray(sampler_grid(num_steps, shift), jnp.float32)
    guided = jnp.asarray([cfg_scale > 1.0 and guidance_low <= t <= guidance_high
                          for t in sampler_grid(num_steps, shift)[:-1]])

    def step(i, x):
        t, dt = grid[i], grid[i + 1] - grid[i]
        log_snr = jnp.full((x.shape[0],), t * 8 - 4)

        def plain(x):
            return apply(params, x, log_snr, y, groups)[1]

        def cfg(x):
            v = apply(params, pair(x, x, groups), pair(log_snr, log_snr, groups),
                      pair(y, jnp.full_like(y, NULL_CLASS), groups), groups)[1]
            v_c, v_u = unpair(v, groups)
            return v_u + cfg_scale * (v_c - v_u)

        v = jax.lax.cond(guided[i], cfg, plain, x) if cfg_scale > 1.0 else plain(x)
        return x + dt * v

    return jax.lax.fori_loop(0, num_steps, step, z.astype(jnp.float32))
