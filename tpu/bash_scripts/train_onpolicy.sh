# Launch on-policy finetuning of FreqFlow-L on TPU (tpu/train_onpolicy.py): FM + CFG-direction + MMD terms,
# from a pretrained checkpoint, with configs/freqflow_onpolicy.py's recipe (band t > 0.8245, n_roll 32).
#
# Pre-requisites (one-time): the packed features and the latent pool in gs://$GCS_BUCKET/$DATA_PREFIX/ (see
# tpu/bash_scripts/setup_tpu.sh), and the start weights as a torch-keyed .npz in the bucket: from a GPU
# checkpoint, `python tpu/convert.py pth2npz nnet_ema.pth nnet_ema.npz` (GPU env); from a TPU run, its
# $OUTPUT_DIR/export/<step>_nnet_ema.npz.
#
# Resuming: re-run verbatim (the latest checkpoint under $OUTPUT_DIR/ckpts wins over INIT_PATH). Fresh: bump RUN_TAG.

export GCS_BUCKET="${GCS_BUCKET:-v6_east1d}"
source tpu/bash_scripts/setup_tpu.sh

# --- Run identity and start weights ---
export RUN_TAG="${RUN_TAG:-onpolicy-cfg1-mmd100}"
export OUTPUT_DIR="gs://$GCS_BUCKET/checkpoints/freqflow/$RUN_TAG"
export INIT_PATH="${INIT_PATH:-$GCS_MOUNT/$DATA_PREFIX/init/nnet_ema.npz}"
if [ ! -f "$INIT_PATH" ]; then
  echo "ERROR: start weights not found at $INIT_PATH (set INIT_PATH)."
  exit 1
fi
if [ ! -f "$POOL_DIR/meta.json" ]; then
  echo "ERROR: no latent pool at $POOL_DIR. Upload dump_latent_pool.py's folder to gs://$GCS_BUCKET/$DATA_PREFIX/."
  exit 1
fi
echo "RUN_TAG=$RUN_TAG OUTPUT_DIR=$OUTPUT_DIR INIT_PATH=$INIT_PATH"

prewarm "$DATA_DIR"
prewarm "$POOL_DIR"

# --- Launch training ---
python tpu/train_onpolicy.py \
    --config=tpu/tpu_configs/freqflow_onpolicy.py \
    --config.init_path="$INIT_PATH" \
    --config.tpu.data_dir="$DATA_DIR" \
    --config.onpolicy.pool_path="$POOL_DIR" \
    --config.onpolicy.cfg_gamma="${CFG_GAMMA:-1.0}" \
    --config.onpolicy.mmd_gamma="${MMD_GAMMA:-100}" \
    --config.tpu.wandb_project=freqflow-onpolicy \
    --workdir="$OUTPUT_DIR" \
    --distributed=true

unmount_bucket

# tpu create v6 --name freqflow-onpolicy -n 8 --setup-cmd "" --priority 0 --max-attempts 40 -- bash tpu/bash_scripts/train_onpolicy.sh
