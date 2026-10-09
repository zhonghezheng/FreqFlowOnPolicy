"""Shared pieces of tpu/train.py and tpu/train_onpolicy.py: config, mesh and shardings, optimizer, train state,
checkpoints, weight export and logging."""
import importlib.util
import json
import logging as pylogging
import os
import signal
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tpu.convert import flax_to_torch, load_torch_weights, save_torch_weights, torch_to_flax  # noqa: E402
from tpu.model import FreqFlow  # noqa: E402

NULL_CLASS = 1000      # CFGDataset's empty_token and the sampler's y_null
SCALE_FACTOR = 0.18215  # the SD autoencoder's latent scale (autoencoder.sample)


def load_config(path):
    spec = importlib.util.spec_from_file_location('tpu_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def maybe_init_distributed(enabled):
    """jax.distributed.initialize() before any other JAX call (needed on multi-host slices, harmless on one host);
    it finds the coordinator from the TPU metadata. Retried as maxdiffusion does: workers come up at different
    times."""
    if not enabled:
        return
    for attempt in range(3):
        try:
            jax.distributed.initialize(initialization_timeout=1800)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            logging.warning(f'jax.distributed.initialize attempt {attempt + 1} failed ({e}); retrying')


# --- mesh and shardings ---

def make_mesh():
    """One data-parallel axis over every chip, in jax.devices() order (grouped by host)."""
    return Mesh(np.array(jax.devices()), ('data',))


def replicated(mesh):
    return NamedSharding(mesh, P())


def batch_sharding(mesh, ndim, axis=0):
    spec = [None] * ndim
    spec[axis] = 'data'
    return NamedSharding(mesh, P(*spec))


def zero_spec(shape, n, skip_axis0=False, min_size=1 << 14):
    """Shard the largest axis divisible by n, for state that is only touched elementwise (optimizer moments,
    EMA, accumulated gradients): ZeRO-style, each chip keeps 1/n. Small arrays stay replicated."""
    if n == 1 or int(np.prod(shape)) < min_size:
        return P()
    axes = sorted(range(1 if skip_axis0 else 0, len(shape)), key=lambda a: -shape[a])
    for a in axes:
        if shape[a] % n == 0:
            spec = [None] * len(shape)
            spec[a] = 'data'
            return P(*spec)
    return P()


def tree_shardings(tree, mesh, zero, skip_layer_axis=False):
    """NamedShardings for a pytree of arrays/ShapeDtypeStructs: zero_spec where zero, else replicated.
    skip_layer_axis keeps the stacked layer axis of scanned blocks whole, so nn.scan slices locally."""
    n = mesh.devices.size

    def one(path, x):
        if not zero or not hasattr(x, 'shape'):
            return NamedSharding(mesh, P())
        names = [getattr(k, 'key', getattr(k, 'name', '')) for k in path]
        stacked = skip_layer_axis and any(s in ('in_blocks', 'out_blocks') for s in names)
        return NamedSharding(mesh, zero_spec(x.shape, n, skip_axis0=stacked))

    return jax.tree_util.tree_map_with_path(one, tree)


# --- model, optimizer, state ---

def make_model(config, dtype=None):
    t = config.tpu
    return FreqFlow(config.nnet.model_args, dtype=jnp.dtype(dtype or t.dtype), remat=t.remat,
                    lowres_chunks=t.lowres_chunks)


def make_optimizer(config):
    """AdamW as utils.get_optimizer/trainable_parameters: no weight decay on the time-dependent parameters'
    nodecay_weight / nodecay_bias, decay on everything else; the 'customized' schedule: linear warmup from 0
    (LambdaLR evaluates it at the step count before each update), then constant."""
    o = config.optimizer
    base, warmup = o.lr, config.lr_scheduler.get('warmup_steps', -1)
    if warmup > 0:
        schedule = lambda count: base * jnp.minimum(count / warmup, 1.0)
    else:
        schedule = lambda count: base * jnp.ones_like(count, dtype=jnp.float32)
    decay_mask = lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: getattr(path[-1], 'key', '') not in ('nodecay_weight', 'nodecay_bias'), params)
    assert o.name == 'adamw', o.name
    tx = optax.adamw(schedule, b1=o.betas[0], b2=o.betas[1], eps=1e-8, weight_decay=o.weight_decay, mask=decay_mask)
    return tx, schedule


def abstract_params(model, config):
    z = config.z_shape
    x = jax.ShapeDtypeStruct((1, *z), jnp.float32)
    s = jax.ShapeDtypeStruct((1,), jnp.float32)
    y = jax.ShapeDtypeStruct((1,), jnp.int32)
    return jax.eval_shape(lambda x, s, y: model.init(jax.random.key(0), x, s, y), x, s, y)['params']


def state_shardings(abs_state, mesh, shard_params):
    """params: replicated (or sharded with shard_params); ema and optimizer state: always sharded."""
    return {
        'params': tree_shardings(abs_state['params'], mesh, shard_params, skip_layer_axis=True),
        'ema': tree_shardings(abs_state['ema'], mesh, True),
        'opt_state': tree_shardings(abs_state['opt_state'], mesh, True),
        'step': replicated(mesh),
    }


def create_state(model, tx, config, mesh, init_params=None):
    """Fresh train state, initialized like PyTorch (seeded by config.seed) or from init_params (numpy flax
    params, e.g. converted from a GPU checkpoint); the EMA starts equal to the weights."""
    abs_params = abstract_params(model, config)
    abs_state = {'params': abs_params, 'ema': abs_params, 'opt_state': jax.eval_shape(tx.init, abs_params),
                 'step': jax.ShapeDtypeStruct((), jnp.int32)}
    shardings = state_shardings(abs_state, mesh, config.tpu.shard_params)

    def build(params):
        return {'params': params, 'ema': jax.tree.map(jnp.copy, params), 'opt_state': tx.init(params),
                'step': jnp.zeros((), jnp.int32)}

    if init_params is None:
        z = config.z_shape
        init = lambda: model.init(jax.random.key(config.seed), jnp.zeros((1, *z)), jnp.zeros((1,)),
                                  jnp.zeros((1,), jnp.int32))['params']
        state = jax.jit(lambda: build(init()), out_shardings=shardings)()
    else:
        params = jax.tree.map(lambda a, s: jax.device_put(np.asarray(a, np.float32), s), init_params,
                              shardings['params'])
        state = jax.jit(build, out_shardings=shardings)(params)
    return state, shardings, abs_state


def load_init_params(path, margs):
    """Flax params from a torch state_dict (.pth needs torch) or a torch-keyed .npz (tpu/convert.py)."""
    return torch_to_flax(load_torch_weights(path), margs)


def latent_sample(moments, key):
    """autoencoder.sample: SCALE_FACTOR * (mean + std * xi) from posterior moments [B, 8, H, W]."""
    mean, logvar = jnp.split(moments, 2, axis=1)
    return SCALE_FACTOR * (mean + jnp.exp(0.5 * jnp.clip(logvar, -30.0, 20.0)) * jax.random.normal(key, mean.shape))


def ema_update(ema, params, rate):
    return jax.tree.map(lambda e, p: rate * e + (1 - rate) * p, ema, params)


# --- checkpoints ---

def checkpoint_manager(ckpt_dir, max_to_keep=None):
    """Checkpoints are finalized atomically, so one cut short by preemption is never restored; only the latest
    max_to_keep are kept (None: all)."""
    import orbax.checkpoint as ocp
    return ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(create=True, max_to_keep=max_to_keep))


class Preemption:
    """Call with the step after every step, on every host; True means save now and stop. Under jax.distributed
    (TPU slices), JAX's preemption service turns the scheduler's SIGTERM into a step all hosts agree on. A single
    process without it catches SIGTERM itself (installing that handler under the service would replace JAX's)."""

    def __init__(self):
        self.service = jax.distributed.is_initialized()
        self.flag = False
        if not self.service:
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'flag', True))

    def __call__(self, step):
        if self.service:
            from jax.experimental import multihost_utils
            try:
                return multihost_utils.reached_preemption_sync_point(step)
            except RuntimeError:  # preemption service disabled
                return False
        return self.flag


def save_checkpoint(manager, state, step):
    import orbax.checkpoint as ocp
    manager.save(step, args=ocp.args.StandardSave(state))


def restore_checkpoint(manager, abs_state, shardings, step=None):
    import orbax.checkpoint as ocp
    step = manager.latest_step() if step is None else step
    target = jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), abs_state, shardings)
    return manager.restore(step, args=ocp.args.StandardRestore(target)), step


def export_torch_npz(tree, margs, path):
    """Write flax params (possibly sharded over hosts) as a torch-keyed .npz on host 0."""
    from jax.experimental import multihost_utils
    host = multihost_utils.process_allgather(tree, tiled=True) if jax.process_count() > 1 else jax.device_get(tree)
    if jax.process_index() == 0:
        _makedirs(os.path.dirname(path))
        local = path if not path.startswith('gs://') else os.path.join('/tmp', os.path.basename(path))
        save_torch_weights(flax_to_torch(host, margs), local)
        if local != path:
            import subprocess
            subprocess.run(['gcloud', 'storage', 'cp', local, path], check=True)
            os.remove(local)


def _makedirs(path):
    if path and not path.startswith('gs://'):
        os.makedirs(path, exist_ok=True)


# --- logging ---

def local_dir(workdir):
    """Where logs, metrics and sample grids go: the workdir itself, or a local mirror of a gs:// workdir
    (copied to <workdir>/logs at every checkpoint by sync_logs)."""
    if not workdir.startswith('gs://'):
        return workdir
    return os.path.join('/tmp', 'freqflow_runs', workdir[len('gs://'):].strip('/').replace('/', '_'))


def sync_logs(workdir):
    """Copy the local mirror of a gs:// workdir into <workdir>/logs (host 0)."""
    if workdir.startswith('gs://') and jax.process_index() == 0:
        import subprocess
        subprocess.run(['gcloud', 'storage', 'rsync', '-r', local_dir(workdir), workdir.rstrip('/') + '/logs'],
                       check=False)


def pull_logs(workdir):
    """After a restart on a new VM: fetch <workdir>/logs into the (empty) local mirror first, so the run's log,
    metrics and sample grids continue instead of being overwritten by the next sync (host 0)."""
    if workdir.startswith('gs://') and jax.process_index() == 0:
        import subprocess
        os.makedirs(local_dir(workdir), exist_ok=True)
        subprocess.run(['gcloud', 'storage', 'rsync', '-r', workdir.rstrip('/') + '/logs', local_dir(workdir)],
                       check=False, capture_output=True)


class Logger:
    """Host 0 logs to stdout and <local_dir>/output.log, metrics as JSON lines in <local_dir>/metrics.jsonl and,
    with wandb_project set and WANDB_API_KEY in the environment, to Weights & Biases."""

    def __init__(self, workdir, wandb_project='', run_name=None, config=None):
        self.main = jax.process_index() == 0
        self.metrics_path, self.wandb = None, None
        if self.main:
            pull_logs(workdir)
            local = local_dir(workdir)
            os.makedirs(local, exist_ok=True)
            handler = pylogging.FileHandler(os.path.join(local, 'output.log'))
            handler.setFormatter(pylogging.Formatter('%(asctime)s - %(message)s'))
            logging.get_absl_logger().addHandler(handler)
            self.metrics_path = os.path.join(local, 'metrics.jsonl')
            if wandb_project and os.environ.get('WANDB_API_KEY'):
                import wandb
                self.wandb = wandb.init(project=wandb_project, name=run_name, dir=local, resume='allow',
                                        id=run_name, config=config.to_dict() if config is not None else None)
        logging.set_verbosity('info' if self.main else 'error')
        self.t0 = time.time()

    def info(self, msg):
        if self.main:
            logging.info(msg)

    def metrics(self, step, metrics):
        if not self.main:
            return
        metrics = {k: float(v) for k, v in metrics.items()}
        logging.info(str({'step': step, **{k: f'{v:.6g}' for k, v in metrics.items()}}))
        if self.metrics_path:
            with open(self.metrics_path, 'a') as f:
                f.write(json.dumps({'step': step, **metrics}) + '\n')
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)


def workdir_for(config_path, argv, explicit=None):
    """GPU scripts' default: workdir/<config name>/<overridden hparams>."""
    if explicit:
        return explicit
    name = os.path.splitext(os.path.basename(config_path))[0]
    hp = [f"{a.split('=')[0].split('.')[-1]}={os.path.splitext(os.path.basename(a.split('=', 1)[1]))[0] if a.split('=')[0].endswith('path') else a.split('=', 1)[1]}"
          for a in argv[1:] if a.startswith('--config.') and '=' in a and not a.startswith('--config.tpu.data_dir')]
    return os.path.join('workdir', 'tpu', name, '-'.join(hp) or 'default')
