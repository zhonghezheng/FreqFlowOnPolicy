"""TPU (JAX) run of configs/freqflow_onpolicy.py: the same finetuning recipe (FM + CFG-direction + MMD terms)
plus TPU execution settings. The pool comes from onpolicy.pool_path (dump_latent_pool.py's folder)."""
import os
import sys

import ml_collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.freqflow_onpolicy import get_config as gpu_config  # noqa: E402


def get_config():
    config = gpu_config()
    config.tpu = ml_collections.ConfigDict(dict(
        data_dir='',              # tpu/prep_data.py output (local disk, or a gcsfuse mount with file caching)
        # rows per chip per micro-batch; the band's share (1 - band_t_min) and n_roll rollouts are per micro-batch,
        # as per GPU rank. 128 matches the GPU run's micro-batches on 8 chips (1024 / 8) if it fits in HBM
        per_device_batch=64,
        dtype='bfloat16',
        remat=True,
        lowres_chunks=1,
        shard_params=False,
        prefetch=2,
        export_npz=True,
        wandb_project='',         # log metrics to W&B (needs WANDB_API_KEY in the environment)
        # preemptible TPUs: checkpoint every train.save_interval (1000) steps and on a preemption notice, keeping
        # only the latest max_to_keep; the EMA alone is exported as .npz every export_interval steps (for FID)
        max_to_keep=3,
        export_interval=2000,
    ))
    config.train.save_interval = 1000
    return config
