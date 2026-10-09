"""Read weights out of a tpu/train*.py orbax checkpoint (for convert.py export and sample.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_weights(ckpt_dir, step, which, config_path):
    """(model Args, flax params as numpy) of a checkpoint's 'ema' or 'params'; step None = latest."""
    import jax
    import orbax.checkpoint as ocp
    from tpu.common import load_config
    if not config_path:
        raise ValueError('--config: the tpu/tpu_configs/*.py file the run used (it defines the architecture)')
    config = load_config(config_path)
    path = ckpt_dir if ckpt_dir.startswith('gs://') else os.path.abspath(ckpt_dir)
    manager = ocp.CheckpointManager(path)
    step = manager.latest_step() if step is None else step
    restored = manager.restore(step, args=ocp.args.StandardRestore())
    return config.nnet.model_args, jax.device_get(restored[which])
