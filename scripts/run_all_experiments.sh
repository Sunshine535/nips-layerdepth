#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=gpu_utils.sh
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# ─── LayerDepth: Master Experiment Pipeline ───
# Order: single_knockout → block_knockout → importance_ranking
#        → scaling_law → depth_selector → final figures
#
# Hardware: 4–8× A100 (auto-detected)  |  Model: Qwen/Qwen3.5-27B (64 layers)

CONFIG="${PROJECT_DIR}/configs/knockout_config.yaml"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen3.5-27B"

echo "========================================="
echo " LayerDepth — Full Experiment Pipeline"
echo " Model: ${MODEL}"
echo " GPUs:  ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "========================================="

# ──────────────────────────────────────────────
# Step 1: Single-layer knockout
# ──────────────────────────────────────────────
STEP1_DIR="${PROJECT_DIR}/results/layer_knockout"
mkdir -p "$STEP1_DIR"

echo ""
echo "[Step 1/6] Single-layer knockout..."
python "${SCRIPT_DIR}/layer_knockout.py" \
    --config_path "$CONFIG" \
    --output_dir "$STEP1_DIR" \
    --mode single \
    --benchmarks gsm8k mmlu \
    2>&1 | tee "${LOG_DIR}/01_single_knockout.log"
echo "[Step 1/6] Done."

# ──────────────────────────────────────────────
# Step 2: Contiguous block knockout
# ──────────────────────────────────────────────
STEP2_DIR="${PROJECT_DIR}/results/block_knockout"
mkdir -p "$STEP2_DIR"

echo ""
echo "[Step 2/6] Block knockout (sizes 2, 4, 8, 16)..."
python "${SCRIPT_DIR}/run_block_knockout.py" \
    --model_path "$MODEL" \
    --output_dir "$STEP2_DIR" \
    --block_sizes 2 4 8 16 \
    --benchmarks gsm8k mmlu \
    --max_samples 200 \
    --resume \
    2>&1 | tee "${LOG_DIR}/02_block_knockout.log"
echo "[Step 2/6] Done."

# ──────────────────────────────────────────────
# Step 3: Importance ranking (3 metrics)
# ──────────────────────────────────────────────
STEP3_DIR="${PROJECT_DIR}/results/importance"
mkdir -p "$STEP3_DIR"

echo ""
echo "[Step 3/6] Layer importance ranking..."
python "${SCRIPT_DIR}/run_importance_ranking.py" \
    --model_path "$MODEL" \
    --output_dir "$STEP3_DIR" \
    --cal_samples 100 \
    --metrics gradient_norm activation_norm fisher \
    2>&1 | tee "${LOG_DIR}/03_importance_ranking.log"
echo "[Step 3/6] Done."

# ──────────────────────────────────────────────
# Step 4: Depth scaling law analysis
# ──────────────────────────────────────────────
STEP4_DIR="${PROJECT_DIR}/results/scaling_law"
mkdir -p "$STEP4_DIR"

echo ""
echo "[Step 4/6] Scaling law analysis..."
python "${SCRIPT_DIR}/run_scaling_law_analysis.py" \
    --model_path "$MODEL" \
    --output_dir "$STEP4_DIR" \
    --max_samples_per_task 100 \
    --threshold 0.95 \
    --resume \
    2>&1 | tee "${LOG_DIR}/04_scaling_law.log"
echo "[Step 4/6] Done."

# ──────────────────────────────────────────────
# Step 5: Train adaptive depth selector
# ──────────────────────────────────────────────
STEP5_DIR="${PROJECT_DIR}/results/depth_selector"
mkdir -p "$STEP5_DIR"

echo ""
echo "[Step 5/6] Training depth selector (REINFORCE)..."
python "${SCRIPT_DIR}/train_depth_selector.py" \
    --model_path "$MODEL" \
    --output_dir "$STEP5_DIR" \
    --train_gsm8k 300 \
    --train_mmlu 300 \
    --epochs 5 \
    --lr 3e-4 \
    --lam 0.3 \
    2>&1 | tee "${LOG_DIR}/05_depth_selector.log"
echo "[Step 5/6] Done."

# ──────────────────────────────────────────────
# Step 6: MVD fitting + adaptive evaluation
# ──────────────────────────────────────────────
STEP6_DIR="${PROJECT_DIR}/results/mvd_analysis"
mkdir -p "$STEP6_DIR"

echo ""
echo "[Step 6/6] MVD fitting + adaptive evaluation..."
python "${SCRIPT_DIR}/fit_mvd.py" \
    --results_dir "$STEP1_DIR" \
    --output_dir "$STEP6_DIR" \
    --threshold 0.95 \
    --model_type power_law \
    2>&1 | tee "${LOG_DIR}/06_mvd_fit.log"

if [ -f "${STEP6_DIR}/mvd_analysis.json" ]; then
    python "${SCRIPT_DIR}/eval_adaptive_depth.py" \
        --model_path "$MODEL" \
        --mvd_results "${STEP6_DIR}/mvd_analysis.json" \
        --output_dir "$STEP6_DIR" \
        --benchmark gsm8k \
        --num_samples 200 \
        2>&1 | tee "${LOG_DIR}/06_adaptive_eval.log"
fi
echo "[Step 6/6] Done."

echo ""
echo "========================================="
echo " All LayerDepth experiments complete!"
echo ""
echo " Results:"
echo "   Single knockout: $STEP1_DIR"
echo "   Block knockout:  $STEP2_DIR"
echo "   Importance:       $STEP3_DIR"
echo "   Scaling law:      $STEP4_DIR"
echo "   Depth selector:   $STEP5_DIR"
echo "   MVD analysis:     $STEP6_DIR"
echo "========================================="

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
