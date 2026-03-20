#!/usr/bin/env bash
set -euo pipefail

# ─── LayerDepth Knockout Experiment Launcher (8x A100-80GB) ───
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME="${HF_HOME:-/home/nwh/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export WANDB_PROJECT="layerdepth"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/knockout_config.yaml"
OUTPUT_DIR="${PROJECT_DIR}/results/layer_knockout"

mkdir -p "$OUTPUT_DIR"

NUM_GPUS=8
MASTER_PORT="${MASTER_PORT:-29501}"

echo "========================================="
echo " LayerDepth Knockout Experiment"
echo " GPUs: ${NUM_GPUS}"
echo " Config: ${CONFIG}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================="

python "${SCRIPT_DIR}/layer_knockout.py" \
    --config_path "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "Knockout experiments complete."
echo "Run fit_mvd.py and eval_adaptive_depth.py for analysis."
