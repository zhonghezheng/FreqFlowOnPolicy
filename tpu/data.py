"""Training batches from prep_data.py's shards: deterministic per step, so a resumed run continues with the same
data order without saving loader state.

Each epoch is one seeded permutation of all images (like a shuffled DataLoader with drop_last); step s of an
epoch takes the next global batch from it. Every host reads only its own columns of that batch from its local
copy of the shards (memory-mapped), picks the image or its flip at random as datasets.CachedFolder does, and a
background thread prepares the next steps.
"""
import json
import os
import queue
import threading

import jax
import numpy as np


class PackedFeatures:
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        names = [s['name'] for s in self.meta['shards']]
        load = lambda n, kind: np.load(os.path.join(data_dir, f'{n}.{kind}.npy'), mmap_mode='r')
        self.moments = [load(n, 'moments') for n in names]
        self.flip = [load(n, 'flip') for n in names]
        self.labels = np.concatenate([np.asarray(load(n, 'labels'), dtype=np.int32) for n in names])
        self.offsets = np.cumsum([0] + [len(m) for m in self.moments])
        self.n = int(self.offsets[-1])
        assert self.n == self.meta['n_images'] == len(self.labels)

    def read(self, rows, flip):
        """moments [k, 8, 32, 32] of the given global rows, each the flipped version where flip is True."""
        out = np.empty((len(rows), 8, 32, 32), dtype=np.float32)
        shard = np.searchsorted(self.offsets, rows, side='right') - 1
        for s in np.unique(shard):
            sel = np.nonzero(shard == s)[0]
            local = rows[sel] - self.offsets[s]
            order = np.argsort(local)  # sorted reads from the memory map
            for want_flip, src in ((False, self.moments[s]), (True, self.flip[s])):
                pick = order[flip[sel[order]] == want_flip]
                if len(pick):
                    out[sel[pick]] = src[local[pick]]
        return out


class BatchSource:
    """batch(step) -> moments [accum, B_host, 8, 32, 32] float32, labels [accum, B_host] int32, this host's
    share of the global batch [accum, B_global / accum]."""

    def __init__(self, data, global_batch, accum, seed, prefetch=2):
        self.data, self.accum, self.seed = data, accum, seed
        self.global_batch = global_batch
        self.micro = global_batch // accum
        nproc, pid = jax.process_count(), jax.process_index()
        assert self.micro % nproc == 0, (self.micro, nproc)
        self.cols = slice(pid * self.micro // nproc, (pid + 1) * self.micro // nproc)
        self.steps_per_epoch = data.n // global_batch
        self._perm = (None, None)
        self._queue, self._next, self._thread = queue.Queue(maxsize=prefetch), None, None

    def _permutation(self, epoch):
        if self._perm[0] != epoch:
            self._perm = (epoch, np.random.default_rng([self.seed, epoch]).permutation(self.data.n))
        return self._perm[1]

    def batch(self, step):
        epoch, pos = divmod(step, self.steps_per_epoch)
        rows = self._permutation(epoch)[pos * self.global_batch:(pos + 1) * self.global_batch]
        rows = rows.reshape(self.accum, self.micro)[:, self.cols]
        flip = np.random.default_rng([self.seed, 1 << 20, step]).random((self.accum, self.micro))[:, self.cols] < 0.5
        moments = self.data.read(rows.reshape(-1), flip.reshape(-1)).reshape(*rows.shape, 8, 32, 32)
        return moments, self.data.labels[rows]

    def start(self, step):
        """Prefetch batches step, step + 1, ... in a background thread; get() returns them in order."""
        def run(s):
            while True:
                self._queue.put((s, self.batch(s)))
                s += 1
        self._next = step
        self._thread = threading.Thread(target=run, args=(step,), daemon=True)
        self._thread.start()

    def get(self):
        s, b = self._queue.get()
        assert s == self._next, (s, self._next)
        self._next += 1
        return b


def to_global(mesh, arrays, batch_axis=1):
    """This host's [accum, B_host, ...] arrays -> global jax.Arrays sharded over 'data' along the batch axis."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    def one(a):
        spec = [None] * a.ndim
        spec[batch_axis] = 'data'
        shape = list(a.shape)
        shape[batch_axis] *= jax.process_count()
        return jax.make_array_from_process_local_data(NamedSharding(mesh, P(*spec)), a, tuple(shape))
    return tuple(one(a) for a in arrays)
