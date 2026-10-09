# Launch FreqFlow-L base training on TPU (tpu/train.py): the paper's Table 10 recipe (batch 2048, 800 epochs)
# with configs/freqflow_l.py's model, bfloat16, ZeRO-sharded optimizer state and EMA.
#
# Pre-requisites (one-time, from the machine with the data): the packed features in
# gs://$GCS_BUCKET/$DATA_PREFIX/ (see tpu/bash_scripts/setup_tpu.sh).
#
# Resuming: re-run this script verbatim; the trainer picks up the latest checkpoint under $OUTPUT_DIR/ckpts
# and continues with the same data order. Starting fresh: bump RUN_TAG.
# Weights & Biases: export WANDB_API_KEY before launching (it is never stored in this repo).

export GCS_BUCKET="${GCS_BUCKET:-v6_east1d}"
source tpu/bash_scripts/setup_tpu.sh

# --- Run identity ---
export RUN_TAG="${RUN_TAG:-freqflow-l}"
export OUTPUT_DIR="gs://$GCS_BUCKET/checkpoints/freqflow/$RUN_TAG"
echo "RUN_TAG=$RUN_TAG OUTPUT_DIR=$OUTPUT_DIR DATA_DIR=$DATA_DIR"

prewarm "$DATA_DIR"

# --- Launch training ---
python tpu/train.py \
    --config=tpu/tpu_configs/freqflow_l.py \
    --config.tpu.data_dir="$DATA_DIR" \
    --config.tpu.wandb_project=freqflow \
    --workdir="$OUTPUT_DIR" \
    --distributed=true

unmount_bucket

# tpu create v6 --name freqflow-l -n 8 --setup-cmd "" --priority 0 --max-attempts 40 -- bash tpu/bash_scripts/train_freqflow_l.sh
