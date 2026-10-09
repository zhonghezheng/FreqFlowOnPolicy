"""TPU env: tpu/pool.py's sharded posterior and tpu/onpolicy.py's MMD against dump_pool_reference.py, plus the
banks' and MMD targets' invariants."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '.')
from tpu.common import make_mesh  # noqa: E402
from tpu.onpolicy import mmd2_unbiased  # noqa: E402
from tpu.pool import LatentPool, _bank_key  # noqa: E402

pool_path, ref_file = sys.argv[1], sys.argv[2]
ref = np.load(ref_file)
mesh = make_mesh()
pool = LatentPool(pool_path, mesh)
A = pool.arrays
rel = lambda a, b: float(np.abs(np.asarray(a, np.float64) - b).max() / np.abs(b).max())
x, t, c = jnp.asarray(ref['x']), jnp.asarray(ref['t']), jnp.asarray(ref['c'], jnp.int32)

e_c, e_u, empty = jax.jit(lambda A, x, t, c: pool.posterior_means(A, x, t, c))(A, x, t, c)
print(f'{mesh.devices.size} devices, {pool.per} rows each; exact posterior vs torch: '
      f'E[x0|x_t,c] rel {rel(e_c, ref["e_c"]):.2e}, E[x0|x_t] rel {rel(e_u, ref["e_u"]):.2e}, empty {bool(empty)}')

val, d2 = jax.jit(lambda a, b: mmd2_unbiased(a, b, (0.1, 0.25)))(jnp.asarray(ref['xm']), jnp.asarray(ref['ym']))
grad = jax.jit(jax.grad(lambda a, b: mmd2_unbiased(a, b, (0.1, 0.25))[0]))(jnp.asarray(ref['xm']), jnp.asarray(ref['ym']))
print(f'MMD^2 {float(val):.6e} vs torch {float(ref["mmd"]):.6e} (rel {abs(float(val) - ref["mmd"]) / abs(ref["mmd"]):.1e}), '
      f'd2_base rel {abs(float(d2) - ref["d2"]) / ref["d2"]:.1e}, grad rel {rel(grad, ref["mmd_grad"]):.2e}')

# banks: exactly K candidates per query, of the right class
K = 1024
key = jax.random.key(3)
e_cb, e_ub, empty_b = jax.jit(lambda A, x, t, c, k: pool.posterior_means(A, x, t, c, K, K, k))(A, x, t, c, key)
key_c, key_u = jax.random.split(key)
seeds_c = jax.random.bits(key_c, (x.shape[0],), jnp.uint32)
seeds_u = jax.random.bits(key_u, (x.shape[0],), jnp.uint32)
members = A['class_idx'][c]
kc = jnp.where(jnp.arange(members.shape[1])[None] < A['class_count'][c][:, None], _bank_key(seeds_c[:, None], members),
               jnp.iinfo(jnp.int32).max)
tau_c = jnp.sort(kc, axis=1)[:, K - 1]
n_c = (kc <= tau_c[:, None]).sum(1)
tau_u = pool._uniform_bank_threshold(A, seeds_u, K)
all_rows = jnp.arange(pool.N)
n_u = jax.lax.map(lambda a: (_bank_key(a[0], all_rows) <= a[1]).sum(), (seeds_u, tau_u))
print(f'banks: class-bank sizes {int(n_c.min())}..{int(n_c.max())}, uniform-bank sizes {int(n_u.min())}..{int(n_u.max())} '
      f'(K={K}); banked means differ from exact by rel {rel(e_cb, np.asarray(e_c)):.2f} / {rel(e_ub, np.asarray(e_u)):.2f}; '
      f'empty {bool(empty_b)}')

# MMD targets: distinct, of the row's class; gather returns the pool rows
cr = pool.sample_classes(A, 256, jax.random.key(5))
cr = cr.at[:8].set(cr[0])  # several rows of one class
rows, ok = pool.class_rows(A, cr, 8, jax.random.key(6))
rows_np = np.asarray(rows)
labels = np.load(pool_path + '/labels.npy')
lat = np.load(pool_path + '/latents.npy', mmap_mode='r')
sample = rows_np.reshape(-1)[::97]
got = np.asarray(pool.gather(A, jnp.asarray(sample)))
print(f'class_rows: ok {bool(ok)}, distinct {len(np.unique(rows_np)) == rows_np.size}, '
      f'class-correct {bool((labels[rows_np] == np.asarray(cr)[:, None]).all())}; '
      f'gather exact {np.array_equal(got, np.asarray(lat[sample]).reshape(len(sample), -1).astype(np.float32))}')
