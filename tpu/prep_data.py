"""Pack the ImageNet feature folder into shuffled .npy shards for TPU training.

Input: extract_imagenet_features.py's output, <features>/<wnid>/<stem>.npz holding 'moments' and
'moments_flip' ([8, 32, 32] float32 each: SD-VAE posterior mean and logvar of the image and of its flip).
Labels are the sorted wnid folder index, exactly as datasets.CachedFolder (torchvision DatasetFolder) assigns.

Output in --out, one seeded random order over all images so every shard is a random subset of the classes:
  shard_00000.moments.npy  [n, 8, 32, 32] float32
  shard_00000.flip.npy     [n, 8, 32, 32] float32
  shard_00000.labels.npy   [n] int16
  ...
  meta.json                image count, shard list, wnids, seed
84 GB for ImageNet at 10k images per shard. Shards are written atomically and existing ones skipped, so an
interrupted run resumes. Needs only numpy:

  python tpu/prep_data.py --features /data/users/cat10301/datasets/imagenet_sd_ema_feature \
      --out /data/users/cat10301/datasets/imagenet_sd_ema_feature_packed
  gcloud storage cp -r /data/users/cat10301/datasets/imagenet_sd_ema_feature_packed gs://BUCKET/
The latent pool for train_onpolicy (dump_latent_pool.py's latents.npy / labels.npy / meta.json) is used as is.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np


def list_samples(root):
    """(path, label) in torchvision DatasetFolder order: sorted class folders, sorted files within."""
    classes = sorted(e.name for e in os.scandir(root) if e.is_dir())
    samples = []
    for label, wnid in enumerate(classes):
        for dirpath, _, fnames in sorted(os.walk(os.path.join(root, wnid), followlinks=True)):
            samples += [(os.path.join(dirpath, f), label) for f in sorted(fnames) if f.lower().endswith('.npz')]
    return samples, classes


def _read(path):
    with np.load(path) as d:
        return d['moments'].astype(np.float32), d['moments_flip'].astype(np.float32)


def _save(path, array):
    tmp = path + '.tmp.npy'
    np.save(tmp, array)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    samples, classes = list_samples(args.features)
    n = len(samples)
    order = np.random.default_rng(args.seed).permutation(n)
    os.makedirs(args.out, exist_ok=True)
    shards = []
    with Pool(args.workers) as pool:
        for k, start in enumerate(range(0, n, args.shard_size)):
            rows = order[start:start + args.shard_size]
            name = f'shard_{k:05d}'
            shards.append({'name': name, 'n': int(len(rows))})
            if os.path.exists(os.path.join(args.out, f'{name}.labels.npy')):
                continue
            pairs = pool.map(_read, [samples[i][0] for i in rows], chunksize=64)
            _save(os.path.join(args.out, f'{name}.moments.npy'), np.stack([p[0] for p in pairs]))
            _save(os.path.join(args.out, f'{name}.flip.npy'), np.stack([p[1] for p in pairs]))
            _save(os.path.join(args.out, f'{name}.labels.npy'),
                  np.array([samples[i][1] for i in rows], dtype=np.int16))  # last: marks the shard complete
            print(f'{name}: {len(rows)} images ({start + len(rows)}/{n})', flush=True)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(dict(n_images=n, shard_size=args.shard_size, shards=shards, seed=args.seed,
                       classes=classes, features=os.path.abspath(args.features)), f, indent=1)
    print(f'packed {n} images into {len(shards)} shards in {args.out}')


if __name__ == '__main__':
    main()
