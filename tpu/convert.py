"""Convert FreqFlow weights between PyTorch (libs/ours/blocks_RoPERMSNorm.FreqFlow state_dict) and tpu/model.py.

PyTorch-side files are .pth (needs torch) or .npz with the same keys and layouts (no torch needed), so the TPU
environment never imports torch:

  # GPU env: a PyTorch checkpoint -> torch-keyed .npz, to init TPU training (--config.init_path=...npz)
  python tpu/convert.py pth2npz workdir/.../ckpts/500000.ckpt/nnet_ema.pth nnet_ema.npz
  # TPU env: an orbax checkpoint's EMA (or online) weights -> torch-keyed .npz
  python tpu/convert.py export --ckpt_dir gs://bucket/run/ckpts --step 500000 --which ema --out nnet_ema.npz
  # GPU env: .npz -> .pth for eval.py / fid_eval.py / train_onpolicy.py --config.init_path
  python tpu/convert.py npz2pth nnet_ema.npz nnet_ema.pth
"""
import argparse
import os
import sys

import numpy as np


def _spec(margs):
    """[(flax path, layer index or None, torch key, layout)] for every parameter; layout is 'id', 'lin'
    (torch [out, in] <-> flax [in, out]) or 'conv' (torch [O, I, kh, kw] <-> flax [kh, kw, I, O])."""
    out = []
    tdp_names = ('param0', 'param1', 'nodecay_weight', 'nodecay_bias')

    def tdp(path, key, idx=None):
        out.extend((path + (n,), idx, f'{key}.{n}', 'id') for n in tdp_names)

    def ln(path, key, idx=None):
        tdp(path + ('weight',), f'{key}.weight', idx)
        tdp(path + ('bias',), f'{key}.bias', idx)

    def lin(path, key, idx=None, bias=True):
        out.append((path + ('kernel',), idx, f'{key}.weight', 'lin'))
        if bias:
            out.append((path + ('bias',), idx, f'{key}.bias', 'id'))

    def conv(path, key, idx=None):
        out.append((path + ('kernel',), idx, f'{key}.weight', 'conv'))
        out.append((path + ('bias',), idx, f'{key}.bias', 'id'))

    def block(path, key, sc, idx=None):
        tdp(path + ('mlp', 'norm', 'scale'), f'{key}.block2.norm.scale', idx)
        for w in ('w0', 'w1', 'w2'):
            lin(path + ('mlp', w), f'{key}.block2.{w}', idx)
        if sc.block_type == 'TransformerBlock':
            tdp(path + ('attn', 'norm', 'scale'), f'{key}.block1.norm.scale', idx)
            lin(path + ('attn', 'qkv'), f'{key}.block1.query_key_value', idx, bias=False)
            lin(path + ('attn', 'dense'), f'{key}.block1.dense', idx)
        else:
            conv(path + ('dwconv',), f'{key}.block1.1', idx)

    for i, sc in enumerate(margs.stage_configs):
        p, k = (f'stage{i}',), f'stages.{i}'
        conv(p + ('input_conv',), f'{k}.input_layer.1')
        out.append((p + ('pos_emb',), None, f'{k}.input_layer.3.position_embeddings.weight', 'id'))
        if sc.block_type == 'TransformerBlock' and hasattr(margs, 'num_classes'):
            out.append((p + ('class_emb', 'embedding'), None, f'{k}.class_embeds.weight', 'id'))
        if i > 0:
            ln(p + ('bridge', 'ln_in'), f'{k}.upsample.0')
            lin(p + ('bridge', 'kernel'), f'{k}.upsample.1.kernel')
            ln(p + ('bridge', 'ln_out'), f'{k}.upsample.2')
        half = sc.num_blocks // 2
        for j in range(half):
            block(p + ('in_blocks', 'block'), f'{k}.blocks.{j}', sc, j)
            lin(p + ('out_blocks', 'skip_dense'), f'{k}.skip_denses.{j}', j)
            block(p + ('out_blocks', 'block'), f'{k}.blocks.{half + 1 + j}', sc, j)
        block(p + ('mid_block', 'block'), f'{k}.blocks.{half}', sc)
        ln(p + ('out_norm',), f'{k}.output_layer.0')
        conv(p + ('out_conv',), f'{k}.output_layer.2')
    return out


_TO_FLAX = {'id': lambda a: a, 'lin': lambda a: a.T, 'conv': lambda a: a.transpose(2, 3, 1, 0)}
_TO_TORCH = {'id': lambda a: a, 'lin': lambda a: a.T, 'conv': lambda a: a.transpose(3, 2, 0, 1)}


def torch_to_flax(state_dict, margs):
    """{torch key: array} -> flax params (nested dicts of float32 numpy arrays)."""
    spec = _spec(margs)
    missing = {key for _, _, key, _ in spec} - set(state_dict)
    extra = set(state_dict) - {key for _, _, key, _ in spec}
    if missing or extra:
        raise KeyError(f'state_dict mismatch: missing {sorted(missing)[:5]}..., unexpected {sorted(extra)[:5]}...')
    params, stacks = {}, {}
    for path, idx, key, layout in spec:
        a = _TO_FLAX[layout](np.asarray(state_dict[key], dtype=np.float32))
        if idx is None:
            _set(params, path, a)
        else:
            stacks.setdefault(path, {})[idx] = a
    for path, by_idx in stacks.items():
        _set(params, path, np.stack([by_idx[j] for j in range(len(by_idx))]))
    return params


def flax_to_torch(params, margs):
    """flax params -> {torch key: float32 numpy array}."""
    sd = {}
    for path, idx, key, layout in _spec(margs):
        a = np.asarray(_get(params, path), dtype=np.float32)
        sd[key] = np.ascontiguousarray(_TO_TORCH[layout](a if idx is None else a[idx]))
    return sd


def _set(tree, path, value):
    for p in path[:-1]:
        tree = tree.setdefault(p, {})
    tree[path[-1]] = value


def _get(tree, path):
    for p in path:
        tree = tree[p]
    return tree


def load_torch_weights(path):
    """A torch state_dict from .pth (needs torch) or a torch-keyed .npz, as {key: float32 numpy array}."""
    if path.endswith('.npz'):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    import torch
    return {k: v.float().numpy() for k, v in torch.load(path, map_location='cpu').items()}


def save_torch_weights(state_dict, path):
    if path.endswith('.npz'):
        np.savez(path, **state_dict)
    else:
        import torch
        torch.save({k: torch.from_numpy(v) for k, v in state_dict.items()}, path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('pth2npz', 'npz2pth'):
        s = sub.add_parser(name)
        s.add_argument('src')
        s.add_argument('dst')
    s = sub.add_parser('export', help='orbax checkpoint -> torch-keyed .npz')
    s.add_argument('--ckpt_dir', required=True)
    s.add_argument('--step', type=int, default=None, help='default: latest')
    s.add_argument('--which', choices=('ema', 'params'), default='ema')
    s.add_argument('--out', required=True)
    s.add_argument('--config', default=None, help='tpu config the run used (default: read from the checkpoint dir)')
    args = parser.parse_args()

    if args.cmd in ('pth2npz', 'npz2pth'):
        save_torch_weights(load_torch_weights(args.src), args.dst)
        return
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # restore into host memory, leaving the accelerators alone
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tpu import checkpoints  # noqa: E402
    margs, params = checkpoints.load_weights(args.ckpt_dir, args.step, args.which, args.config)
    save_torch_weights(flax_to_torch(params, margs), args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
