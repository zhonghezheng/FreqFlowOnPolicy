"""dump_latent_pool.py's pool in JAX, sharded over the chips: the empirical posterior of onpolicy.LatentPool and
the class-matched MMD targets.

Chip g keeps the g-th contiguous block of pool rows (latents float16 [rows, 4096], their squared norms and
labels; padding rows carry label -1). Labels and the per-class index table are also replicated, for drawing
classes and targets. All functions are traceable and take the arrays as arguments (pool.arrays), so a jitted
step receives the pool as an input instead of baking 21 GB of constants into the program.
"""
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

HIGHEST = jax.lax.Precision.HIGHEST  # logits are O(1e3); their O(1) differences need full float32 products


def _mix32(x):
    """lowbias32: a bijective uint32 hash."""
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7feb352d)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846ca68b)
    return x ^ (x >> 16)


def _bank_key(seed, row):
    """31-bit pseudo-random key of (query seed, global pool row), recomputable anywhere: a bank of K rows is
    the K rows with the smallest keys among the candidates."""
    return (_mix32(row.astype(jnp.uint32) ^ _mix32(seed)) >> 1).astype(jnp.int32)


def _accumulate(m, s, w, logits, p):
    """onpolicy._accumulate: one chunk of a streaming softmax-weighted mean."""
    m_new = jnp.maximum(m, logits.max(1))
    ref = jnp.where(jnp.isfinite(m_new), m_new, 0.)
    alpha = jnp.exp(m - ref)
    e = jnp.exp(logits - ref[:, None])
    return m_new, s * alpha + e.sum(1), w * alpha[:, None] + jnp.dot(e, p, precision=HIGHEST)


class LatentPool:
    def __init__(self, path, mesh, chunk=16384):
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        lat = np.load(os.path.join(path, 'latents.npy'), mmap_mode='r')
        labels = np.load(os.path.join(path, 'labels.npy')).astype(np.int32)
        assert lat.shape[0] == labels.shape[0]
        self.mesh, self.n_dev = mesh, mesh.devices.size
        self.N, self.D = lat.shape[0], int(np.prod(lat.shape[1:]))
        per_dev = -(-self.N // self.n_dev)
        self.chunk = min(chunk, per_dev)
        self.per = -(-per_dev // self.chunk) * self.chunk  # rows per chip, a multiple of the chunk
        rows = self.per * self.n_dev

        def latents_cb(index):
            lo, hi = index[0].start or 0, index[0].stop or rows
            out = np.zeros((hi - lo, self.D), np.float16)
            if lo < self.N:
                out[:min(hi, self.N) - lo] = np.asarray(lat[lo:min(hi, self.N)]).reshape(-1, self.D)
            return out

        padded = np.full(rows, -1, np.int32)
        padded[:self.N] = labels
        sh_rows, sh_vec = NamedSharding(mesh, P('data', None)), NamedSharding(mesh, P('data'))
        rep = NamedSharding(mesh, P())
        latents = jax.make_array_from_callback((rows, self.D), sh_rows, latents_cb)
        shard_labels = jax.make_array_from_callback((rows,), sh_vec, lambda i: padded[i])
        sq = jax.jit(lambda x: jnp.sum(jnp.square(x.astype(jnp.float32)), axis=1), out_shardings=sh_vec)(latents)

        self.n_classes = int(labels.max()) + 1
        count = np.bincount(labels, minlength=self.n_classes)
        order = np.argsort(labels, kind='stable')
        starts = np.cumsum(count) - count
        class_idx = np.zeros((self.n_classes, count.max()), np.int32)
        class_idx[labels[order], np.arange(self.N) - starts[labels[order]]] = order
        self.class_count_np = count
        self.arrays = {
            'latents': latents, 'sq': sq, 'labels': shard_labels,
            'labels_all': jax.device_put(labels, rep),
            'class_count': jax.device_put(count.astype(np.int32), rep),
            'class_idx': jax.device_put(class_idx, rep),
        }

    # --- the empirical posterior ---

    def posterior_means(self, arrays, xq, tq, cq, bank_k=0, bank_k_uncond=0, key=None):
        """(E[x0 | x_t, class c], E[x0 | x_t], empty) for queries xq [Q, D] at times tq [Q] with classes cq [Q],
        over the row's class and over the whole pool (bank_k / bank_k_uncond > 0: over K distinct rows of
        each instead). Weights w_i ∝ exp(-||x_t - (1 - t) x_i||^2 / (2 t^2)). `empty` flags a query without
        candidates."""
        q = xq.shape[0]
        zeros = jnp.zeros((q,), jnp.int32)
        seeds_c, tau_c, seeds_u, tau_u = jnp.zeros((q,), jnp.uint32), zeros, jnp.zeros((q,), jnp.uint32), zeros
        if bank_k or bank_k_uncond:
            key_c, key_u = jax.random.split(key)
        if bank_k:
            seeds_c = jax.random.bits(key_c, (q,), jnp.uint32)
            members = arrays['class_idx'][cq]
            keys = _bank_key(seeds_c[:, None], members)
            keys = jnp.where(jnp.arange(members.shape[1])[None] < arrays['class_count'][cq][:, None], keys,
                             jnp.iinfo(jnp.int32).max)
            tau_c = jnp.sort(keys, axis=1)[:, bank_k - 1]
        if bank_k_uncond:
            seeds_u = jax.random.bits(key_u, (q,), jnp.uint32)
            tau_u = self._uniform_bank_threshold(arrays, seeds_u, bank_k_uncond)
        fn = jax.shard_map(partial(self._local_posterior, bank_c=bool(bank_k), bank_u=bool(bank_k_uncond)),
                           mesh=self.mesh, in_specs=(P('data', None), P('data'), P('data')) + (P(),) * 7,
                           out_specs=(P(), P(), P()), check_vma=False)
        return fn(arrays['latents'], arrays['sq'], arrays['labels'], xq.astype(jnp.float32), tq.astype(jnp.float32),
                  cq, seeds_c, tau_c, seeds_u, tau_u)

    def _local_posterior(self, lat, sq, lab, xq, tq, cq, seeds_c, tau_c, seeds_u, tau_u, bank_c, bank_u):
        q, ch = xq.shape[0], self.chunk
        row0 = jax.lax.axis_index('data') * self.per
        a = (1 - tq)[:, None]
        inv = 1 / (2 * jnp.maximum(tq, 1e-3)[:, None] ** 2)

        def body(state, s):
            p = jax.lax.dynamic_slice_in_dim(lat, s * ch, ch).astype(jnp.float32)
            psq = jax.lax.dynamic_slice_in_dim(sq, s * ch, ch)
            plab = jax.lax.dynamic_slice_in_dim(lab, s * ch, ch)
            rows = row0 + s * ch + jnp.arange(ch)
            logits = (2 * a * jnp.dot(xq, p.T, precision=HIGHEST) - a ** 2 * psq[None]) * inv
            in_c = plab[None] == cq[:, None]
            in_u = jnp.broadcast_to((plab >= 0)[None], in_c.shape)
            if bank_c:
                in_c = in_c & (_bank_key(seeds_c[:, None], rows[None]) <= tau_c[:, None])
            if bank_u:
                in_u = in_u & (_bank_key(seeds_u[:, None], rows[None]) <= tau_u[:, None])
            return tuple(_accumulate(*st, jnp.where(mask, logits, -jnp.inf), p)
                         for st, mask in zip(state, (in_c, in_u))), None

        init = tuple((jnp.full((q,), -jnp.inf), jnp.zeros((q,)), jnp.zeros((q, self.D))) for _ in range(2))
        state, _ = jax.lax.scan(body, init, jnp.arange(self.per // ch))
        out, empty = [], jnp.zeros((), bool)
        for m, s, w in state:  # merge the chips' softmax partials: max all-reduce, then sum all-reduce
            m_all = jax.lax.pmax(m, 'data')
            scale = jnp.exp(m - jnp.where(jnp.isfinite(m_all), m_all, 0.))
            s_all = jax.lax.psum(s * scale, 'data')
            w_all = jax.lax.psum(w * scale[:, None], 'data')
            empty = empty | jnp.any(s_all <= 0)
            out.append(w_all / jnp.maximum(s_all, 1e-30)[:, None])
        return out[0], out[1], empty

    def _uniform_bank_threshold(self, arrays, seeds, k):
        """Per query, the k-th smallest bank key over all pool rows: each chip's k smallest, then the global."""
        def local(lab, seeds):
            rows = jax.lax.axis_index('data') * self.per + jnp.arange(self.per)
            keys = jnp.where((lab >= 0)[None], _bank_key(seeds[:, None], rows[None]), jnp.iinfo(jnp.int32).max)
            smallest = -jax.lax.top_k(-keys, k)[0]
            both = jax.lax.all_gather(smallest, 'data', axis=1, tiled=True)
            return jnp.sort(both, axis=1)[:, k - 1]
        return jax.shard_map(local, mesh=self.mesh, in_specs=(P('data'), P()), out_specs=P(),
                             check_vma=False)(arrays['labels'], seeds)

    # --- classes and targets ---

    def sample_classes(self, arrays, n, key):
        """Labels of n uniformly drawn pool rows: classes in proportion to the pool."""
        return arrays['labels_all'][jax.random.randint(key, (n,), 0, self.N)]

    def class_rows(self, arrays, c, per, key):
        """[len(c), per] pool rows: per members of each row's class, no row used twice in the call (rows of the
        same class take consecutive slices of one random order of the class). Returns (rows, ok); ok is False
        when a class has fewer members than its rows need."""
        r = c.shape[0]
        idx = jnp.arange(r)
        rank = jnp.sum((c[None] == c[:, None]) & (idx[None] < idx[:, None]), axis=1)  # earlier rows of the class
        count = arrays['class_count'][c]
        members = arrays['class_idx'][c]
        keys = jax.random.uniform(key, (self.n_classes, members.shape[1]))[c]  # shared by rows of a class
        keys = jnp.where(jnp.arange(members.shape[1])[None] < count[:, None], keys, 2.)
        order = jnp.take_along_axis(members, jnp.argsort(keys, axis=1), axis=1)
        take = rank[:, None] * per + jnp.arange(per)[None]
        rows = jnp.take_along_axis(order, jnp.minimum(take, members.shape[1] - 1), axis=1)
        return rows, jnp.all((rank + 1) * per <= count)

    def gather(self, arrays, rows):
        """Pool latents of the global rows [M] as float32 [M, D], replicated: each chip contributes the rows
        it holds, then a sum all-reduce (no chip ever holds more than its own block)."""
        def local(lat, rows):
            off = rows - jax.lax.axis_index('data') * self.per
            mine = (off >= 0) & (off < self.per)
            vals = jnp.take(lat, jnp.clip(off, 0, self.per - 1), axis=0).astype(jnp.float32)
            return jax.lax.psum(jnp.where(mine[:, None], vals, 0.), 'data')
        return jax.shard_map(local, mesh=self.mesh, in_specs=(P('data', None), P()), out_specs=P(),
                             check_vma=False)(arrays['latents'], rows)
