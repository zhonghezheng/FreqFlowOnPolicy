"""Extract the ImageNet-256 SD-VAE features that train_ldm_discrete.py reads.

Writes the layout datasets.CachedFolder expects: <out>/<wnid>/<stem>.npz holding
'moments' and 'moments_flip', each [8, 32, 32] float32 (posterior mean and logvar
from FrozenAutoencoderKL.encode_moments), for every training image and its
horizontal flip. Labels come from the sorted wnid folder names, as in training.

Input is a preprocessed uint8 cache [N, 256, 256, 3] (ADM center crop, the same
algorithm as datasets.center_crop_arr) with an index of relative image paths, e.g.
LakonLab's data/cached_latents/train_u8_256.{u8.npy,index.npz}. The flip is taken
on the 256x256 crop, which equals cropping the flipped image.

Sharded and resumable: shard k of S covers rows [k*N//S, (k+1)*N//S), skips files
that already exist, and writes each file atomically, so a killed task can simply
be resubmitted.

    python extract_imagenet_features.py --u8 PREFIX --out OUT --num-shards 16 --shard $SLURM_ARRAY_TASK_ID
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

import libs.autoencoder


def save(path, moments, moments_flip):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, moments=moments, moments_flip=moments_flip)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--u8', required=True, help='prefix of <prefix>.u8.npy and <prefix>.index.npz')
    parser.add_argument('--out', required=True)
    parser.add_argument('--ckpt', default='assets/stable-diffusion/autoencoder_kl_ema.pth')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0, help='only the first N rows of the shard (pilot)')
    args = parser.parse_args()

    images = np.load(args.u8 + '.u8.npy', mmap_mode='r')
    relpaths = np.load(args.u8 + '.index.npz')['relpaths']
    n = len(relpaths)
    assert images.shape == (n, 256, 256, 3), images.shape
    lo, hi = args.shard * n // args.num_shards, (args.shard + 1) * n // args.num_shards
    if args.limit:
        hi = min(hi, lo + args.limit)
    outs = [os.path.join(args.out, os.path.splitext(str(r))[0] + '.npz') for r in relpaths[lo:hi]]
    for d in sorted({os.path.dirname(p) for p in outs}):
        os.makedirs(d, exist_ok=True)
    todo = [i for i, p in enumerate(outs) if not os.path.exists(p)]
    print(f'shard {args.shard}/{args.num_shards}: rows {lo}:{hi}, {len(todo)} to write', flush=True)

    autoencoder = libs.autoencoder.get_model(args.ckpt).cuda()
    pool = ThreadPoolExecutor(8)
    pending = []
    with torch.no_grad():
        for s in tqdm(range(0, len(todo), args.batch), mininterval=30):
            idx = todo[s:s + args.batch]
            x = torch.from_numpy(np.stack([images[lo + i] for i in idx])).cuda()
            x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
            m = autoencoder.encode_moments(torch.cat([x, x.flip(-1)])).float().cpu().numpy()
            b = len(idx)
            pending += [pool.submit(save, outs[i], m[j], m[b + j]) for j, i in enumerate(idx)]
            if len(pending) > 4096:
                for f in pending:
                    f.result()
                pending = []
    for f in pending:
        f.result()
    print(f'shard {args.shard} done', flush=True)


if __name__ == '__main__':
    main()
