"""FreqFlow's flow-matching loss (train_ldm_discrete.SILoss / onpolicy.fm_loss) on the Flax model."""
import jax.numpy as jnp

from tpu.model import fourier_filter, pool2


def fm_loss_rows(apply, params, x_t, t, v, y, clamp_groups=1):
    """Per-row loss at given (x_t, t, v): the velocity MSE plus 1/4 of the frequency branch's MSE against the
    2x-pooled [low, high, v] split of the target. log_snr = 8 t - 4."""
    out0, out1 = apply(params, x_t, t * 8 - 4, y, clamp_groups)
    low, high = fourier_filter(v, 8, clamp_groups)
    freq_target = pool2(jnp.concatenate([low, high, v], axis=1))
    return jnp.mean((out1 - v) ** 2, axis=(1, 2, 3)) + jnp.mean((out0 - freq_target) ** 2, axis=(1, 2, 3)) / 4
