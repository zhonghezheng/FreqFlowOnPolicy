"""GPU env: onpolicy.LatentPool's exact posterior means and onpolicy.mmd2_unbiased (value and gradient), for
check_pool.py."""
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, '.')
import onpolicy  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
pool_path, out = sys.argv[1], sys.argv[2]
# torchrun: the pool is sharded over the ranks (one GPU cannot hold its float32 squared-norm pass); every rank
# passes the same queries
dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
pool = onpolicy.LatentPool(pool_path, torch.device('cuda'), rank, world)
lat = np.load(pool_path + '/latents.npy', mmap_mode='r')
g = torch.Generator(device='cuda').manual_seed(0)
Q = 64
rows = torch.randint(0, pool.N, (Q,), device='cuda', generator=g)
c = pool.labels[rows]
t = torch.tensor([1.0, 0.9363, 0.878, 0.8246], device='cuda')[torch.arange(Q, device='cuda') % 4]
x0 = torch.from_numpy(np.stack([np.asarray(lat[r]) for r in rows.tolist()])).cuda().float().view(Q, -1)
x = (1 - t[:, None]) * x0 + t[:, None] * torch.randn(Q, pool.D, device='cuda', generator=g)
x[::5] = torch.randn(len(x[::5]), pool.D, device='cuda', generator=g)  # some queries not built from the pool
e_c, e_u = pool.posterior_means(x.view(Q, 4, 32, 32), t, c)
xm = torch.randn(32, 4096, device='cuda', generator=g, requires_grad=True)
ym = torch.randn(256, 4096, device='cuda', generator=g) * 0.9 + 0.1
val, d2 = onpolicy.mmd2_unbiased(xm, ym, (0.1, 0.25))
val.backward()
if rank == 0:
  np.savez(out, x=x.cpu().numpy(), t=t.cpu().numpy(), c=c.cpu().numpy(), e_c=e_c.flatten(1).cpu().numpy(),
           e_u=e_u.flatten(1).cpu().numpy(), xm=xm.detach().cpu().numpy(), ym=ym.cpu().numpy(),
           mmd=val.item(), d2=d2.item(), mmd_grad=xm.grad.cpu().numpy())
  print('wrote', out)
dist.destroy_process_group()
