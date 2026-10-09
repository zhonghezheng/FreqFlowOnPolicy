"""FID of class-conditional samples against precomputed reference statistics, during distributed training.

Uses pytorch-fid's Inception pool3 features, the network the reference statistics in assets/fid_stats were
computed with. Images are quantized to 8 bits first, as if saved to PNG and read back, so the result matches
sampling to a folder and running pytorch-fid on it.
"""
import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from pytorch_fid.inception import InceptionV3


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid's calculate_frechet_distance, which passes an argument recent scipy removed from sqrtm."""
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():  # singular product: shift the diagonals
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@torch.no_grad()
def fid(sample_fn, n_samples, mini_batch_size, stats_path, z_shape, device, seed=0, num_classes=1000):
    """FID of sample_fn(y, z) -> images in [0, 1], B x 3 x H x W.

    Collective: each rank samples n_samples / world_size images. Labels cycle through the classes
    (n_samples / num_classes per class) and the noise comes from a fixed seed, so every checkpoint is scored
    on the same labels and noise. Returns the FID on rank 0 and None elsewhere.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed * world + rank)
    total = torch.zeros(2048, dtype=torch.float64, device=device)
    outer = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
    index = torch.arange(rank, n_samples, world, device=device)
    for batch in index.split(mini_batch_size):
        z = torch.randn(len(batch), *z_shape, device=device, generator=generator)
        x = sample_fn(batch % num_classes, z).float()
        x = (x * 255 + 0.5).clamp(0, 255).floor() / 255  # the 8-bit rounding of torchvision's save_image
        feats = inception(x)[0].flatten(1).double()
        total += feats.sum(0)
        outer += feats.T @ feats
    n = torch.tensor(float(len(index)), dtype=torch.float64, device=device)
    for t in (total, outer, n):
        dist.all_reduce(t)
    del inception
    if rank != 0:
        return None
    mu = total / n
    sigma = (outer - n * torch.outer(mu, mu)) / (n - 1)
    ref = np.load(stats_path)
    return frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(), ref['mu'], ref['sigma'])
