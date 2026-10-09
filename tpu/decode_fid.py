"""GPU env: decode tpu/sample.py's latents with the SD autoencoder and compute FID against the reference
statistics, with the same Inception features and 8-bit rounding as fid_eval.py (pytorch-fid's pipeline).

  python tpu/decode_fid.py /tmp/fid/latents_*.npz [--png_dir /tmp/fid/png]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import libs.autoencoder  # noqa: E402
from fid_eval import frechet_distance  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('latents', nargs='+')
    parser.add_argument('--stats', default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz')
    parser.add_argument('--vae', default='assets/stable-diffusion/autoencoder_kl_ema.pth')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--png_dir', default=None)
    args = parser.parse_args()
    from pytorch_fid.inception import InceptionV3

    z = np.concatenate([np.load(p)['latents'] for p in args.latents]).astype(np.float32)
    vae = libs.autoencoder.get_model(args.vae).cuda()
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    total = torch.zeros(2048, dtype=torch.float64, device='cuda')
    outer = torch.zeros(2048, 2048, dtype=torch.float64, device='cuda')
    if args.png_dir:
        from torchvision.utils import save_image
        os.makedirs(args.png_dir, exist_ok=True)
    with torch.no_grad():
        for s in range(0, len(z), args.batch):
            with torch.autocast('cuda', dtype=torch.float16):
                x = vae.decode(torch.from_numpy(z[s:s + args.batch]).cuda())
            x = (0.5 * (x.float() + 1)).clamp(0, 1)
            x = (x * 255 + 0.5).clamp(0, 255).floor() / 255  # torchvision save_image's 8-bit rounding
            if args.png_dir:
                for i, img in enumerate(x):
                    save_image(img, os.path.join(args.png_dir, f'{s + i}.png'))
            f = inception(x)[0].flatten(1).double()
            total += f.sum(0)
            outer += f.T @ f
    n = len(z)
    mu = total / n
    sigma = (outer - n * torch.outer(mu, mu)) / (n - 1)
    ref = np.load(args.stats)
    print(f'FID-{n}: {frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(), ref["mu"], ref["sigma"]):.4f}')


if __name__ == '__main__':
    main()
