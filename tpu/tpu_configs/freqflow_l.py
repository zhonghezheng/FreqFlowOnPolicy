"""TPU (JAX) run of configs/freqflow_l.py (FreqFlow-L): the same model and recipe plus TPU execution settings."""
import os
import sys

import ml_collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.freqflow_l import get_config as gpu_config  # noqa: E402


def get_config():
    config = gpu_config()
    config.init_path = ''  # optional start weights: a torch state_dict as .pth or torch-keyed .npz (tpu/convert.py)
    config.tpu = ml_collections.ConfigDict(dict(
        data_dir='',              # tpu/prep_data.py output (local disk, or a gcsfuse mount with file caching)
        per_device_batch=64,      # rows per chip per micro-batch; batch_size / (chips x this) micro-batches per step
        dtype='bfloat16',
        remat=True,               # recompute every block and the skip bridge in backward
        lowres_chunks=1,          # > 1: run the skip bridge in that many batch chunks, bounding its backward memory
        shard_params=False,       # also shard the parameters over chips; EMA and optimizer state always are
        prefetch=2,
        export_npz=True,          # write the EMA as a torch-keyed .npz at every checkpoint, for the GPU tools
        wandb_project='',         # log metrics to W&B (needs WANDB_API_KEY in the environment)
    ))
    return config
