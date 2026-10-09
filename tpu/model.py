"""FreqFlow in Flax: a port of libs/ours/blocks_RoPERMSNorm.FreqFlow (APE positions, TDRMSN norms, no text
conditioning) with the same parameters, initialization and math.

Layout differences from the PyTorch model, none of which changes the function:
  - activations are channels-last internally; the model takes and returns NCHW like the original
  - each stage's in-blocks and out-blocks run under nn.scan, so their parameters are stacked on a leading
    layer axis (convert.py maps them to and from the per-block PyTorch names)
  - matmuls and convolutions run in `dtype` (bfloat16 on TPU) with float32 parameters; norms, the residual
    stream, the time-dependent weights and the frequency split stay float32, as under PyTorch autocast
"""
import math
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

normal02 = nn.initializers.normal(stddev=0.02)  # timm trunc_normal_(std=0.02): its +-2 bounds are 100 sigma


def torch_conv_init(fan_in):
    """PyTorch's default Conv2d initialization for weight and bias: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1 / math.sqrt(fan_in)
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)


# --- frequency split and multi-scale targets (fft2d.py, sde.py) ---

def _gaussian_templates(h, w, d):
    # fft2d.py centers the Gaussians at int((h - 1) / 2), one off from fftshift's zero frequency at h // 2
    x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    d2 = (x - int((h - 1) / 2)) ** 2 + (y - int((w - 1) / 2)) ** 2
    high = 1 - np.exp(-d2 / (2 * (d / 8.) ** 2))
    low = np.exp(-d2 / (2 * (d * 2.) ** 2))
    return low.astype(np.float32), high.astype(np.float32)


def fourier_filter(x, d, groups=1):
    """fft2d.Fourier_filter on NCHW float32: Gaussian low- and high-pass of each channel, each clamped to the
    min and max of its input. PyTorch clamps to the extremes of the tensor each rank passes, its local
    micro-batch; `groups` splits the batch into that many consecutive row groups (one per device) and clamps
    each to its own extremes."""
    b, _, h, w = x.shape
    xg = x.reshape(groups, -1)
    mx = jnp.repeat(xg.max(axis=1), b // groups).reshape(b, 1, 1, 1)
    mn = jnp.repeat(xg.min(axis=1), b // groups).reshape(b, 1, 1, 1)
    low_t, high_t = _gaussian_templates(h, w, d)
    f = jnp.fft.fftshift(jnp.fft.fftn(x, axes=(-2, -1)), axes=(-2, -1))
    back = lambda g: jnp.fft.ifftn(jnp.fft.ifftshift(g, axes=(-2, -1)), axes=(-2, -1)).real
    low, high = back(f * low_t), back(f * high_t)
    return jnp.clip(low, mn, mx), jnp.clip(high, mn, mx)


def pool2(x):
    """sde.even_multi_scale_targets(levels=2, scale_correction=True)[H // 2]: 2x2 sums divided by 2 (not 4)."""
    b, c, h, w = x.shape
    return x.reshape(b, c, h // 2, 2, w // 2, 2).sum(axis=(3, 5)) / 2


# --- time-dependent norms (libs/ours/sigmoid/module.py) ---

class TDP(nn.Module):
    """TimeDependentParameter: per-sample weights param1 + sigmoid(tc * (nodecay_weight + 1) + nodecay_bias)
    * (param0 - param1), tc = log_snr / 4. Returns [B, *shape]."""
    shape: tuple
    init_value: float

    @nn.compact
    def __call__(self, tc):
        init = nn.initializers.constant(self.init_value)
        p0 = self.param('param0', init, self.shape)
        p1 = self.param('param1', init, self.shape)
        nw = self.param('nodecay_weight', nn.initializers.zeros, self.shape)
        nb = self.param('nodecay_bias', nn.initializers.zeros, self.shape)
        tc = tc.reshape((-1,) + (1,) * len(self.shape))
        return p1 + jax.nn.sigmoid(nb + tc * (nw + 1)) * (p0 - p1)


def _per_sample(w, x):
    """[B, d] -> broadcastable against x [B, ..., d]."""
    return w.reshape((w.shape[0],) + (1,) * (x.ndim - 2) + (w.shape[1],))


class TDRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x, tc):
        x = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True)) * self.dim ** -0.5
        return _per_sample(TDP((self.dim,), 1.0, name='scale')(tc), x) * (x / (rms + self.eps))


class TDLayerNorm(nn.Module):
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, tc):
        x = x.astype(jnp.float32)
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.square(x - mean).mean(axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + self.eps)
        weight = TDP((self.dim,), 1.0, name='weight')(tc)
        bias = TDP((self.dim,), 0.0, name='bias')(tc)
        return _per_sample(bias, x) + _per_sample(weight, x) * xn


# --- blocks ---

def dense(features, dtype, name, use_bias=True):
    return nn.Dense(features, use_bias=use_bias, kernel_init=normal02, bias_init=nn.initializers.zeros,
                    dtype=dtype, param_dtype=jnp.float32, name=name)


class MLPBlock(nn.Module):
    dim: int
    hidden: int
    dtype: Any

    @nn.compact
    def __call__(self, x, tc):
        h = TDRMSNorm(self.dim, name='norm')(x, tc)
        h = jax.nn.gelu(dense(self.hidden, self.dtype, 'w0')(h), approximate=False) * dense(self.hidden, self.dtype, 'w1')(h)
        return dense(self.dim, self.dtype, 'w2')(h)


class SelfAttention(nn.Module):
    dim: int
    heads: int
    dtype: Any

    @nn.compact
    def __call__(self, x, tc):
        b, n, _ = x.shape
        hd = self.dim // self.heads
        qkv = dense(3 * self.dim, self.dtype, 'qkv', use_bias=False)(TDRMSNorm(self.dim, name='norm')(x, tc))
        q, k, v = (qkv.reshape(b, n, 3, self.heads, hd)[:, :, i] for i in range(3))  # columns: q heads, k heads, v heads
        o = jax.nn.dot_product_attention(q, k, v)
        return dense(self.dim, self.dtype, 'dense')(o.reshape(b, n, self.dim))


class TransformerBlock(nn.Module):
    sc: Any  # stage config (libs/ours Args)
    dtype: Any

    @nn.compact
    def __call__(self, x, tc):
        x = x + SelfAttention(self.sc.dim, self.sc.num_attention_heads, self.dtype, name='attn')(x, tc)
        return x + MLPBlock(self.sc.dim, self.sc.hidden_dim, self.dtype, name='mlp')(x, tc)


class ConvNeXtBlock(nn.Module):
    sc: Any
    dtype: Any

    @nn.compact
    def __call__(self, x, tc):
        k, d = self.sc.kernel_size, self.sc.dim
        conv = nn.Conv(d, (k, k), padding=k // 2, feature_group_count=d, dtype=self.dtype, param_dtype=jnp.float32,
                       kernel_init=torch_conv_init(k * k), bias_init=torch_conv_init(k * k), name='dwconv')
        x = x + conv(x)
        return x + MLPBlock(d, self.sc.hidden_dim, self.dtype, name='mlp')(x, tc)


def block_cls(sc):
    return {'TransformerBlock': TransformerBlock, 'ConvNeXtBlock': ConvNeXtBlock}[sc.block_type]


class InBlock(nn.Module):
    """Scan body of the first half: the block, emitting its output as a skip."""
    sc: Any
    dtype: Any
    emit: bool = True

    @nn.compact
    def __call__(self, x, tc):
        x = block_cls(self.sc)(self.sc, self.dtype, name='block')(x, tc)
        return x, (x if self.emit else None)


class OutBlock(nn.Module):
    """Scan body of the second half: skip dense over [x, skip], then the block."""
    sc: Any
    dtype: Any
    emit: bool = True

    @nn.compact
    def __call__(self, x, skip, tc):
        # float32 residual stream (under torch autocast the dense makes it float16; bfloat16 would lose too much)
        x = dense(self.sc.dim, self.dtype, 'skip_dense')(jnp.concatenate([x, skip], axis=-1)).astype(jnp.float32)
        x = block_cls(self.sc)(self.sc, self.dtype, name='block')(x, tc)
        return x, (x if self.emit else None)


class LowresBridge(nn.Module):
    """Stage.upsample over the concatenated skips of the lower-resolution stage: LayerNorm, PixelShuffleUpsample,
    LayerNorm. skips: [S, B, h, w, C] -> [B, h * r, w * r, dim]."""
    lowres_dim: int
    dim: int
    ratio: int
    dtype: Any

    @nn.compact
    def __call__(self, skips, tc):
        s, b, h, w, c = skips.shape
        x = jnp.moveaxis(skips, 0, -2).reshape(b, h, w, s * c)  # torch.cat(skips, -1): skip-major channels
        x = TDLayerNorm(self.lowres_dim, name='ln_in')(x, tc)
        r = self.ratio
        x = dense(self.dim * r * r, self.dtype, 'kernel')(x)
        x = x.reshape(b, h, w, r, r, self.dim).transpose(0, 1, 3, 2, 4, 5).reshape(b, h * r, w * r, self.dim)
        return TDLayerNorm(self.dim, name='ln_out')(x, tc)


class Stage(nn.Module):
    sc: Any
    channels: int
    out_channels: int
    num_classes: Optional[int]
    lowres_dim: Optional[int]
    lowres_ratio: int
    dtype: Any
    remat: bool
    return_skips: bool
    lowres_chunks: int = 1

    @nn.compact
    def __call__(self, images, tc, y, lowres_skips=None):
        sc = self.sc
        assert sc.image_input_ratio == 1
        r, cin = sc.input_feature_ratio, images.shape[-1]
        x = nn.Conv(sc.dim, (r, r), strides=(r, r), padding='VALID', dtype=self.dtype, param_dtype=jnp.float32,
                    kernel_init=torch_conv_init(r * r * cin), bias_init=torch_conv_init(r * r * cin),
                    name='input_conv')(images)
        b, h, w, d = x.shape
        pos = self.param('pos_emb', normal02, (sc.max_height * sc.max_width, d))
        x = x + pos.reshape(sc.max_height, sc.max_width, d)[:h, :w]  # idx = row * max_width + col

        if lowres_skips is not None:
            bridge = (nn.remat(LowresBridge) if self.remat else LowresBridge)(
                self.lowres_dim, d, self.lowres_ratio, self.dtype, name='bridge')
            if self.lowres_chunks > 1:  # bound the bridge's backward to one batch chunk at a time
                parts = [bridge(sk, t) for sk, t in zip(jnp.split(lowres_skips, self.lowres_chunks, axis=1),
                                                         jnp.split(tc, self.lowres_chunks))]
                x = x + jnp.concatenate(parts, axis=0)
            else:
                x = x + bridge(lowres_skips, tc)

        transformer = sc.block_type == 'TransformerBlock'
        if transformer:
            x = x.reshape(b, h * w, d)
            if self.num_classes is not None:
                cls = nn.Embed(self.num_classes, d, embedding_init=normal02, name='class_emb')(y)
                x = jnp.concatenate([cls[:, None].astype(x.dtype), x], axis=1)

        half = sc.num_blocks // 2
        scanned = (lambda m: nn.remat(m, prevent_cse=False)) if self.remat else (lambda m: m)
        emit = self.return_skips  # the out-blocks' outputs are only needed as the next stage's skips
        x0 = x
        x, in_skips = nn.scan(scanned(InBlock), variable_axes={'params': 0}, split_rngs={'params': True},
                              in_axes=nn.broadcast, length=half)(sc, self.dtype, True, name='in_blocks')(x, tc)
        x, _ = (nn.remat(InBlock) if self.remat else InBlock)(sc, self.dtype, False, name='mid_block')(x, tc)
        mid = x
        x, out_skips = nn.scan(scanned(OutBlock), variable_axes={'params': 0}, split_rngs={'params': True},
                               in_axes=(0, nn.broadcast), length=half)(sc, self.dtype, emit, name='out_blocks')(
            x, in_skips[::-1], tc)  # skips.pop(): the last in-block feeds the first out-block

        if transformer and self.num_classes is not None:
            x = x[:, 1:]
        if transformer:
            x = x.reshape(b, h, w, d)
        out = TDLayerNorm(d, name='out_norm')(x, tc)
        k = sc.final_kernel_size
        out = nn.Conv(self.out_channels, (k, k), padding=k // 2, dtype=self.dtype, param_dtype=jnp.float32,
                      kernel_init=torch_conv_init(k * k * d), bias_init=torch_conv_init(k * k * d),
                      name='out_conv')(out)
        if not emit:
            return out, None
        skips = jnp.concatenate([x0[None], in_skips, mid[None], out_skips], axis=0)  # [S, B, N, d] or [S, B, h, w, d]
        if transformer:
            if self.num_classes is not None:
                skips = skips[:, :, 1:]
            skips = skips.reshape(skips.shape[0], b, h, w, d)
        return out, skips


class FreqFlow(nn.Module):
    """margs: the `model` Args of configs/freqflow*.py. __call__(x [B, 4, H, W], log_snr [B], y [B]) returns
    [frequency-branch output [B, 3C, H/2, W/2], spatial-branch velocity [B, C, H, W]] in float32, NCHW."""
    margs: Any
    dtype: Any = jnp.bfloat16
    remat: bool = True
    lowres_chunks: int = 1

    @nn.compact
    def __call__(self, x, log_snr, y, clamp_groups=1):
        m = self.margs
        assert m.norm_type == 'TDRMSN' and not hasattr(m, 'pe_type') and not hasattr(m, 'condition_dim')
        tc = log_snr.astype(jnp.float32) / 4.0
        x = x.astype(jnp.float32)
        low, high = fourier_filter(x, 8, clamp_groups)
        nhwc = lambda a: jnp.transpose(a, (0, 2, 3, 1))
        outputs, skips, prev = [], None, None
        n = len(m.stage_configs)
        for i, sc in enumerate(m.stage_configs):
            inp = nhwc(jnp.concatenate([low, high], axis=1)) if i == 0 else nhwc(x)
            lowres_dim = None if prev is None else prev.dim * (prev.num_blocks // 2 * 2 + 2)
            ratio = 1 if prev is None else sc.max_height // prev.max_height
            transformer = sc.block_type == 'TransformerBlock'
            stage = Stage(sc, m.channels, m.channels * 3 if transformer else m.channels,
                          getattr(m, 'num_classes', None) if transformer else None, lowres_dim, ratio,
                          self.dtype, self.remat, return_skips=i < n - 1,
                          lowres_chunks=self.lowres_chunks if i > 0 else 1, name=f'stage{i}')
            out, skips = stage(inp, tc, y, skips)
            outputs.append(jnp.transpose(out, (0, 3, 1, 2)).astype(jnp.float32))
            prev = sc
        return outputs
