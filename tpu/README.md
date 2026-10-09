# FreqFlow on TPU (v6e) — JAX version

A JAX/Flax port of this repo's GPU training, in its own directory and environment. Nothing outside `tpu/` changes,
and the GPU scripts keep working as before. The TPU scripts read the same `configs/*.py` for model and recipe,
and add a `tpu` section of execution settings.

| GPU (PyTorch) | TPU (JAX) |
|---|---|
| `train_ldm_discrete.py` + `configs/freqflow_{l,h}.py` | `tpu/train.py` + `tpu/tpu_configs/freqflow_{l,h}.py` |
| `train_onpolicy.py` + `configs/freqflow_onpolicy.py` (FM + CFG-direction + MMD terms) | `tpu/train_onpolicy.py` + `tpu/tpu_configs/freqflow_onpolicy.py` |
| per-image feature folder (`extract_imagenet_features.py`) | packed shards (`tpu/prep_data.py`) |
| latent pool (`dump_latent_pool.py`) | the same folder, as is |
| `eval.py` + `fid_eval.py` | `tpu/sample.py` (TPU) + `tpu/decode_fid.py` (GPU env) |

`tpu/convert.py` moves weights between the two in both directions. You can initialise TPU training from a GPU
checkpoint and evaluate TPU checkpoints with the GPU tools.

## What is the same, and what is not

Same as the GPU scripts:

- The model and its initialisation.
- The loss: velocity MSE plus ¼ × frequency-branch MSE, times ½.
- The data law: SD-VAE posterior sample of the image or its flip, labels dropped to class 1000 with p = 0.1,
  t ~ U[0, 1).
- AdamW (no decay on `nodecay_*`), the LambdaLR warmup, EMA 0.9999.
- The on-policy terms: rollouts on eval's 50-step grid with shift 0.3, the band t > `band_t_min`, the exact
  empirical posterior, the CFG-direction loss, and the pooled MMD with class-matched targets.
- Per-GPU-rank operations are done per chip, so one chip computes what one GPU rank does: band-row selection,
  the `cat([x, x])` CFG batch, and the frequency split's clamp to the micro-batch's min/max.

Different:

- **Mixed precision is bfloat16**, where the GPU uses float16. Norms, the time-dependent weights, the frequency
  split and the residual stream stay float32. The out-blocks' residual stream is float32 here; GPU autocast makes
  it float16.
- **Random streams differ**: same distributions, different draws.
- **Banks** (`bank_full=False`) draw exactly K distinct rows per query through a hash, not torch's `topk` of
  random keys. Same law.
- **No in-loop FID or VAE decoding on TPU.** Checkpoints export the EMA as a torch-keyed `.npz`. FID comes from
  `sample.py` and `decode_fid.py`. The periodic sample grid is saved as latents
  (`<workdir>/samples/<step>.npz`).

## Verified on GPUs

These checks ran on 8×H100 with `jax[cuda13]`. Re-run them after changing the model or pool code.

| Check | Result |
|---|---|
| `tests/check_equivalence.py` (FreqFlow-L, float32, weights converted from PyTorch) | outputs 3e-6, per-row loss 7e-8, gradients of 10 parameter groups ~1e-6 relative to PyTorch, with and without remat |
| `tests/check_pool.py` (real 2.56M-latent pool, 8 devices) | exact posterior means 2e-6 / 4e-6 vs `onpolicy.LatentPool` on 4 ranks; MMD² and its gradient 6e-7; banks exactly K rows; MMD targets distinct and class-correct; cross-device gather exact |
| `tpu/train.py`, FreqFlow-L, 8 devices, batch 2048 (4 × 8 × 64) | loss 2.68 → 2.46 over 12 steps, warmup LR identical to the GPU logs, 1.55 s/step on H100; checkpoint, resume, EMA export and sample grid work |
| `tpu/train_onpolicy.py`, FreqFlow-L, cfg_gamma 1, mmd_gamma 100, 8 devices, batch 1024 (2 × 8 × 64) | runs end to end from an exported `.npz`; `cfg_dv2_target` ≈ 0.38 (the empirical band target), MMD² ≈ 1.9e-3 against a noise floor of ~5e-6; 2.7 s/step on H100 (FM alone ≈ 0.8 s) |
| `tpu/prep_data.py` | 200 random packed rows (half flipped) bit-identical to the source files; labels and class counts match |

Not verified: anything on actual TPU hardware, and multi-host runs. The code uses only standard JAX
multi-process APIs (`jax.distributed`, `make_array_from_process_local_data`, orbax), but per-chip memory on
v6e (32 GB) has not been measured. Start with the default `per_device_batch` and raise it if memory allows.

## Launching on TPU (scripts)

`tpu/bash_scripts/` follows maxdiffusion's `bash_scripts/train_*.sh`. Each script runs on every worker from the
repo root and does the following:

1. Builds a uv venv with `jax[tpu]`.
2. Mounts the bucket with gcsfuse, caching files in `/dev/shm`. Range-read caching is on, because the loaders
   read random rows from memory-mapped shards.
3. Prewarms the cache by reading every shard once.
4. Sets maxdiffusion's `LIBTPU_INIT_ARGS`.
5. Trains with `--distributed=true`, writing checkpoints to `gs://$GCS_BUCKET/checkpoints/freqflow/$RUN_TAG`.
6. Unmounts the bucket.

```bash
export WANDB_API_KEY=...            # optional; read from the environment only
tpu create v6 --name freqflow-l -n 8 --setup-cmd "" --priority 0 --max-attempts 40 -- bash tpu/bash_scripts/train_freqflow_l.sh
tpu create v6 --name freqflow-onpolicy -n 8 ... -- bash tpu/bash_scripts/train_onpolicy.sh   # INIT_PATH, CFG_GAMMA, MMD_GAMMA
```

Re-running a script resumes the run; bumping `RUN_TAG` starts a fresh one. The scripts expect the data under
`gs://$GCS_BUCKET/$DATA_PREFIX/`, by default `v6_east1d` and `freqflow`. Logs, metrics and sample-grid latents
are written locally and copied to `$OUTPUT_DIR/logs` at every checkpoint. A restarted run pulls them back first, so the history continues.

## Setup by hand

```bash
# 1. data (GPU node or any machine with the feature folder; numpy only)
python tpu/prep_data.py --features /data/users/cat10301/datasets/imagenet_sd_ema_feature \
    --out /data/users/cat10301/datasets/imagenet_sd_ema_feature_packed            # 79 GB, 129 shards
gcloud storage cp -r /data/users/cat10301/datasets/imagenet_sd_ema_feature_packed gs://BUCKET/
gcloud storage cp -r /data/users/cat10301/datasets/imagenet_latent_pool gs://BUCKET/   # 21 GB, on-policy only

# 2. TPU VM (single host: v6e-8). Attach a disk with ~120 GB, or use local SSD
gcloud compute tpus tpu-vm create freqflow --zone=ZONE --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh freqflow --zone=ZONE
git clone <this repo> && cd FreqFlowOnPolicy && pip install -r tpu/requirements.txt
gcloud storage cp -r gs://BUCKET/imagenet_sd_ema_feature_packed /mnt/disks/data/
```

On multi-host slices (v6e-16 and up), every host needs the code, the requirements and a local copy of the data.
Each host reads only its own rows, but from the full shard set. Run the same command on all hosts
(`gcloud compute tpus tpu-vm ssh ... --worker=all --command=...`) and add `--distributed`. Use a `gs://`
workdir so all hosts share checkpoints.

## Training

```bash
# base model (FreqFlow-L; freqflow_h.py for H)
python tpu/train.py --config=tpu/tpu_configs/freqflow_l.py \
    --config.tpu.data_dir=/mnt/disks/data/imagenet_sd_ema_feature_packed --workdir=gs://BUCKET/freqflow_l

# on-policy finetuning from a checkpoint (GPU .pth -> .npz in the GPU env: python tpu/convert.py pth2npz a.pth a.npz)
python tpu/train_onpolicy.py --config=tpu/tpu_configs/freqflow_onpolicy.py --config.init_path=nnet_ema.npz \
    --config.tpu.data_dir=/mnt/disks/data/imagenet_sd_ema_feature_packed \
    --config.onpolicy.pool_path=/mnt/disks/data/imagenet_latent_pool \
    --config.onpolicy.cfg_gamma=1.0 --config.onpolicy.mmd_gamma=100 --workdir=gs://BUCKET/onpolicy
```

- Any config value can be overridden on the command line, as on GPU.
- **Preemption-safe.** A checkpoint is written every 1,000 steps (`train.save_interval`) and as soon as the
  TPU signals preemption. Under `--distributed`, JAX's preemption service turns the SIGTERM into a step all hosts
  agree on, then the run saves and exits. Re-running the same command resumes from the latest checkpoint in
  `<workdir>/ckpts`, which holds the weights, EMA, AdamW state and step:
  - **Data order:** batches are a pure function of the step (a seeded permutation per epoch), so the shuffled
    order continues mid-epoch exactly.
  - **Randomness:** the noise is `fold_in(seed, step)`.
  - **Atomic saves:** checkpoints are finalized atomically, so one cut short is never restored.
- **Retention:** only the latest 3 checkpoints are kept (`tpu.max_to_keep`).
- **EMA export:** the EMA alone is written as a torch-keyed `.npz`, `<workdir>/export/<step>_nnet_ema.npz`, every
  `tpu.export_interval` steps (50k; 2k for on-policy) and at the end. These are kept, for FID.

TPU settings (`config.tpu`):

| Key | Meaning |
|---|---|
| `per_device_batch` | Rows per chip per micro-batch. A step accumulates `batch_size / (chips × per_device_batch)` micro-batches. Defaults: L 64, H 32, on-policy 64 (sized for 32 GB v6e HBM, not yet measured). For on-policy, `n_band` and `n_roll` are per micro-batch, as per GPU rank. 128 reproduces the GPU run's micro-batches on 8 chips. |
| `remat` | Recompute every block and the skip bridge in backward (on by default). |
| `lowres_chunks` | Run the skip bridge (38,400 channels for L, 66,816 for H) in batch chunks. Its backward is the activation peak. |
| `shard_params` | Also shard the weights over chips (FSDP). The EMA, optimizer state and accumulated gradients are always sharded (ZeRO-style). |
| `export_npz` | Write the torch-keyed EMA every `export_interval` steps. |
| `max_to_keep` | Checkpoints kept (latest first). |

## Evaluation

```bash
# TPU: 50k samples, 50 per class (eval.py's 50 Euler steps; --num_steps=250 for the paper's setting)
python tpu/sample.py --config=tpu/tpu_configs/freqflow_l.py --ckpt_dir=gs://BUCKET/freqflow_l/ckpts --out=/tmp/fid
# GPU env: decode with the SD VAE and score (same Inception features and 8-bit rounding as fid_eval.py)
python tpu/decode_fid.py /tmp/fid/latents_*.npz
# weights for eval.py / train_onpolicy.py on GPU
python tpu/convert.py export --ckpt_dir gs://BUCKET/freqflow_l/ckpts --config tpu/tpu_configs/freqflow_l.py --out ema.npz
python tpu/convert.py npz2pth ema.npz ema.pth      # GPU env
```

## Files

- `model.py`: FreqFlow in Flax. Blocks run under `nn.scan` (stacked layer parameters) with remat.
- `losses.py`: the FM loss.
- `sampling.py`: the Euler sampler with limited-interval guidance.
- `pool.py`: the sharded latent pool:
  - the streaming-softmax posterior, merged with `pmax` / `psum` over chips
  - hash banks
  - class-matched targets and a cross-chip row gather
- `onpolicy.py`: rollouts (with per-step remat for MMD), the unbiased MMD estimator, the band MMD and its noise floor.
- `common.py`: mesh and sharding rules, optimizer, train state, orbax checkpoints, export, logging.
- `data.py`: memory-mapped shards and deterministic per-step batches with prefetch.
- `convert.py`: PyTorch ↔ Flax weights.
- `checkpoints.py`: reads weights from a checkpoint.
- `tests/`: the equivalence checks above.
