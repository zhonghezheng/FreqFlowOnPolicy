"""Sample latents on TPU for FID: eval.py's sampler (euler_sampler, 50 steps by default) on the EMA weights, with
labels cycling through the classes (n_samples / 1000 per class) and noise from a fixed seed, every chip sampling
its share. Writes <out>/latents_<host>.npz (latents float32 [n, 4, 32, 32], labels); decode and score them in
the GPU env with tpu/decode_fid.py.

  python tpu/sample.py --config=tpu/tpu_configs/freqflow_l.py --ckpt_dir=gs://BUCKET/run/ckpts --out=/tmp/fid
  python tpu/sample.py --config=... --weights=nnet_ema.npz --out=... --cfg_scale=1.5 --guidance_high=0.7
"""
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from tpu import common  # noqa: E402
from tpu.sampling import euler_sample  # noqa: E402

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'The tpu/tpu_configs/*.py file of the run.', lock_config=False)
flags.DEFINE_string('ckpt_dir', None, 'orbax checkpoint directory of a tpu/train*.py run')
flags.DEFINE_integer('step', None, 'checkpoint step (default: latest)')
flags.DEFINE_string('weights', None, 'or: torch-keyed .npz / .pth weights (tpu/convert.py)')
flags.DEFINE_string('out', None, 'output directory')
flags.DEFINE_integer('n_samples', 50000, '')
flags.DEFINE_integer('per_device_batch', 125, '')
flags.DEFINE_integer('num_steps', 50, 'Euler steps (eval.py: 50; the paper reports 250)')
flags.DEFINE_float('cfg_scale', 1.0, '1: no guidance')
flags.DEFINE_float('guidance_low', 0.0, '')
flags.DEFINE_float('guidance_high', 0.7, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_bool('distributed', False, 'multi-host slices')
flags.mark_flags_as_required(['config', 'out'])


def main(argv):
    common.maybe_init_distributed(FLAGS.distributed)
    config = FLAGS.config
    margs = config.nnet.model_args
    if FLAGS.weights:
        params = common.load_init_params(FLAGS.weights, margs)
    else:
        from tpu.checkpoints import load_weights
        _, params = load_weights(FLAGS.ckpt_dir, FLAGS.step, 'ema', config_flags.get_config_filename(FLAGS['config']))
    mesh = common.make_mesh()
    n_chips = mesh.devices.size
    rep, rows = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    params = jax.device_put(params, rep)
    model = common.make_model(config)
    apply = lambda p, x, ls, y, g: model.apply({'params': p}, x, ls, y, g)
    batch = FLAGS.per_device_batch * n_chips

    @functools.partial(jax.jit, out_shardings=rows)
    def sample(params, z, y):
        return euler_sample(apply, params, z, y, FLAGS.num_steps, 0.3,  # sampler.py's timestep_shift
                            FLAGS.cfg_scale, FLAGS.guidance_low, FLAGS.guidance_high, groups=n_chips)

    key = jax.random.key(FLAGS.seed)
    out_lat, out_lab = [], []
    for start in range(0, FLAGS.n_samples, batch):
        idx = np.arange(start, start + batch)
        y = jax.device_put(jnp.asarray(idx % 1000, jnp.int32), rows)
        z = jax.device_put(jax.random.normal(jax.random.fold_in(key, start // batch), (batch, *config.z_shape)), rows)
        lat = sample(params, z, y)
        for shard in lat.addressable_shards:  # this host's rows
            sl = shard.index[0]
            keep = idx[sl] < FLAGS.n_samples
            out_lat.append(np.asarray(shard.data)[keep])
            out_lab.append((idx[sl] % 1000)[keep])
        if jax.process_index() == 0:
            print(f'{min(start + batch, FLAGS.n_samples)}/{FLAGS.n_samples}', flush=True)
    os.makedirs(FLAGS.out, exist_ok=True)
    path = os.path.join(FLAGS.out, f'latents_{jax.process_index()}.npz')
    np.savez(path, latents=np.concatenate(out_lat), labels=np.concatenate(out_lab))
    print(f'wrote {path}')


if __name__ == '__main__':
    app.run(main)
