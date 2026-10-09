"""Dump the latent pool used by train_onpolicy.py (expert posterior + MMD target).

Reads the ImageNet feature folder that ImageNet256Features trains on (one .npz
per image holding 'moments' and 'moments_flip') and stores, for every image and
its horizontal flip, the SD-VAE posterior MEAN times the scale factor:
z = 0.18215 * mean, i.e. autoencoder.sample without the std * xi noise.

Rows are written in one seeded random order, so a contiguous row range is a
random subset of the pool: train_onpolicy.py gives rank r the r-th range as its
shard, where a class-sorted file would hand each rank a few whole classes.

Output in --out:
  latents.npy  [2N, 4, 32, 32] float16 (~21 GB for ImageNet)
  labels.npy   [2N] int16, the class index CachedFolder assigns (= the training label)
  meta.json

Holds the whole pool in RAM before shuffling (~21 GB), so run it on a compute node:
  python dump_latent_pool.py --features /path/to/imagenet_sd_ema_feature --out /path/to/pool
"""
import argparse
import json
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import CachedFolder
from onpolicy import SCALE_FACTOR


class MeanLatents(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        data = np.load(path)
        return data['moments'][:4], data['moments_flip'][:4], target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='same folder as config.dataset.path')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    samples = CachedFolder(args.features).samples
    n = len(samples)
    latents = np.empty((2 * n, 4, 32, 32), dtype=np.float16)
    labels = np.empty(2 * n, dtype=np.int16)
    loader = DataLoader(MeanLatents(samples), batch_size=256, num_workers=args.workers)
    i = 0
    for mean, mean_flip, target in tqdm(loader, desc='read'):
        b = target.shape[0]
        latents[2 * i:2 * (i + b):2] = (SCALE_FACTOR * mean.float()).numpy()
        latents[2 * i + 1:2 * (i + b):2] = (SCALE_FACTOR * mean_flip.float()).numpy()
        labels[2 * i:2 * (i + b):2] = target.numpy()
        labels[2 * i + 1:2 * (i + b):2] = target.numpy()
        i += b
    assert i == n

    perm = np.random.default_rng(args.seed).permutation(2 * n)
    os.makedirs(args.out, exist_ok=True)
    out = np.lib.format.open_memmap(os.path.join(args.out, 'latents.npy'), mode='w+',
                                    dtype=np.float16, shape=latents.shape)
    for s in tqdm(range(0, 2 * n, 65536), desc='write'):
        out[s:s + 65536] = latents[perm[s:s + 65536]]
    out.flush()
    np.save(os.path.join(args.out, 'labels.npy'), labels[perm])
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(dict(features=os.path.abspath(args.features), n_images=n, n_rows=2 * n,
                       flips=True, scale_factor=SCALE_FACTOR, seed=args.seed,
                       n_classes=int(labels.max()) + 1), f, indent=2)
    print(f'wrote {2 * n} latents ({n} images + flips) to {args.out}')


if __name__ == '__main__':
    main()
