# Shared setup for tpu/bash_scripts/train_*.sh, sourced from the repo root on every TPU worker (after
# maxdiffusion's bash_scripts/train_ctrl_world_*.sh): the Python env, the bucket mount, data paths, XLA flags.
#
# Expects GCS_BUCKET (bucket name, no gs://) and optionally GCS_MOUNT, DATA_PREFIX. Defines DATA_DIR,
# POOL_DIR and prewarm. Data layout in the bucket (upload once from the machine that has it):
#   gs://$GCS_BUCKET/$DATA_PREFIX/imagenet_sd_ema_feature_packed/   tpu/prep_data.py output (79 GB)
#   gs://$GCS_BUCKET/$DATA_PREFIX/imagenet_latent_pool/             dump_latent_pool.py output (21 GB, on-policy)

set -e
echo "[$(hostname)] setup started at $(date)"

# --- 1. Python env (uv venv with jax[tpu]) ---
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env"   # add uv to PATH
if [ ! -x ./freqflow_tpu_venv/bin/python ]; then
  uv venv --python 3.12 ./freqflow_tpu_venv --seed
fi
source ./freqflow_tpu_venv/bin/activate
uv pip install -r tpu/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import importlib.metadata as m; print('resolved: jax', m.version('jax'), '/ libtpu', m.version('libtpu'))"

# --- 2. Bucket mount ---
: "${GCS_BUCKET:?set GCS_BUCKET to the bucket name (no gs://)}"
export GCS_MOUNT="${GCS_MOUNT:-$HOME/gcs-mount}"
if ! command -v gcsfuse >/dev/null; then
  export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y gcsfuse
fi
mkdir -p "$GCS_MOUNT" /dev/shm/gcsfuse-cache
if ! mountpoint -q "$GCS_MOUNT"; then
  # The loaders memory-map the shards and read random rows: --file-cache-cache-file-for-range-read makes the
  # first read of a file pull the whole file into the cache (/dev/shm: host RAM), so later random reads are local.
  gcsfuse \
    --implicit-dirs \
    --file-cache-max-size-mb=-1 \
    --file-cache-cache-file-for-range-read=true \
    --cache-dir=/dev/shm/gcsfuse-cache \
    "$GCS_BUCKET" "$GCS_MOUNT"
fi

# --- 3. Data paths ---
export DATA_PREFIX="${DATA_PREFIX:-freqflow}"
export DATA_DIR="$GCS_MOUNT/$DATA_PREFIX/imagenet_sd_ema_feature_packed"
export POOL_DIR="$GCS_MOUNT/$DATA_PREFIX/imagenet_latent_pool"
if [ ! -f "$DATA_DIR/meta.json" ]; then
  echo "ERROR: no packed features at $DATA_DIR. On the machine with the feature folder:"
  echo "  python tpu/prep_data.py --features .../imagenet_sd_ema_feature --out .../imagenet_sd_ema_feature_packed"
  echo "  gcloud storage cp -r .../imagenet_sd_ema_feature_packed gs://$GCS_BUCKET/$DATA_PREFIX/"
  exit 1
fi

# Read every file of a directory once (in parallel) so the gcsfuse cache holds it before training starts.
prewarm() {
  echo "[$(hostname)] prewarming $1"
  find "$1" -type f -name '*.npy' -print0 | xargs -0 -P 32 -n 1 sh -c 'cat "$0" > /dev/null'
  echo "[$(hostname)] prewarmed $1 at $(date)"
}

# --- 4. XLA flags (maxdiffusion's TPU set: async collectives, overlap, scheduler) ---
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_tpu_megacore_fusion_allow_ags=false \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_assign_all_reduce_scatter_layout=true'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

unmount_bucket() {
  fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"
}
