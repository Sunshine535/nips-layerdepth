#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$PROJECT_DIR"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Phase resume ---
PHASE_MARKER_DIR="$PROJECT_DIR/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

CONFIG="${PROJECT_DIR}/configs/knockout_config.yaml"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen3.5-27B"

# --- GPU allocation: 27B model (~54 GB bf16) needs ≥2 GPUs per instance ---
if [ "$NUM_GPUS" -ge 2 ]; then
    GPUS_PER_TASK=2
else
    GPUS_PER_TASK=1
    echo "[WARN] Only $NUM_GPUS GPU(s). 27B bf16 model (~54 GB) may OOM on a single 80 GB GPU."
fi
NUM_PAIRS=$((NUM_GPUS / GPUS_PER_TASK))
if [ "$NUM_PAIRS" -lt 1 ]; then NUM_PAIRS=1; fi

gpu_pair() {
    local pair_idx=$1
    local start=$((pair_idx * GPUS_PER_TASK))
    local devices=""
    for ((i=0; i<GPUS_PER_TASK; i++)); do
        [ -n "$devices" ] && devices="${devices},"
        devices="${devices}$((start + i))"
    done
    echo "$devices"
}

echo "========================================="
echo " LayerDepth — Full Experiment Pipeline"
echo " Model : ${MODEL}"
echo " GPUs  : ${NUM_GPUS} × ${GPU_CLASS}"
echo " Layout: ${NUM_PAIRS} concurrent instances (${GPUS_PER_TASK} GPUs each)"
echo "========================================="

# ═══════════════════════════════════════════════════════════════════════════
# Block A — 4 independent tasks in parallel (uses all 8 GPUs)
#   Pair 0 (GPU 0,1): Step 1  — layer knockout (single + prefix + importance)
#   Pair 1 (GPU 2,3): Step 2  — block knockout
#   Pair 2 (GPU 4,5): Step 3  — importance ranking
#   Pair 3 (GPU 6,7): Step 4  — scaling law analysis
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "[Block A] Steps 1-4 in parallel — ${NUM_PAIRS} GPU pairs, all ${NUM_GPUS} GPUs active"
PIDS=()
LABELS=()
PAIR_IDX=0

if ! is_phase_done 1; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP1_DIR="${PROJECT_DIR}/results/layer_knockout"
            mkdir -p "$STEP1_DIR"
            echo "[Step 1/6] Layer knockout — all modes (GPUs $CUDA_VISIBLE_DEVICES)..."
            python "${SCRIPT_DIR}/layer_knockout.py" \
                --config_path "$CONFIG" --output_dir "$STEP1_DIR" \
                --mode all --benchmarks gsm8k mmlu \
                2>&1 | tee "${LOG_DIR}/01_layer_knockout.log"
            phase_done 1
        ) &
        PIDS+=($!)
        LABELS+=("step1_knockout")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

if ! is_phase_done 2; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP2_DIR="${PROJECT_DIR}/results/block_knockout"
            mkdir -p "$STEP2_DIR"
            echo "[Step 2/6] Block knockout (GPUs $CUDA_VISIBLE_DEVICES)..."
            python "${SCRIPT_DIR}/run_block_knockout.py" \
                --model_path "$MODEL" --output_dir "$STEP2_DIR" \
                --block_sizes 2 4 8 16 --benchmarks gsm8k mmlu --max_samples 200 --resume \
                2>&1 | tee "${LOG_DIR}/02_block_knockout.log"
            phase_done 2
        ) &
        PIDS+=($!)
        LABELS+=("step2_block")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

if ! is_phase_done 3; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP3_DIR="${PROJECT_DIR}/results/importance"
            mkdir -p "$STEP3_DIR"
            echo "[Step 3/6] Layer importance ranking (GPUs $CUDA_VISIBLE_DEVICES)..."
            python "${SCRIPT_DIR}/run_importance_ranking.py" \
                --model_path "$MODEL" --output_dir "$STEP3_DIR" \
                --cal_samples 100 --metrics gradient_norm activation_norm fisher \
                2>&1 | tee "${LOG_DIR}/03_importance_ranking.log"
            phase_done 3
        ) &
        PIDS+=($!)
        LABELS+=("step3_importance")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

if ! is_phase_done 4; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP4_DIR="${PROJECT_DIR}/results/scaling_law"
            mkdir -p "$STEP4_DIR"
            echo "[Step 4/6] Scaling law analysis (GPUs $CUDA_VISIBLE_DEVICES)..."
            python "${SCRIPT_DIR}/run_scaling_law_analysis.py" \
                --model_path "$MODEL" --output_dir "$STEP4_DIR" \
                --max_samples_per_task 100 --threshold 0.95 --resume \
                2>&1 | tee "${LOG_DIR}/04_scaling_law.log"
            phase_done 4
        ) &
        PIDS+=($!)
        LABELS+=("step4_scaling")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

FAIL=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed (exit $?)"; FAIL=1; }
done
if [ "$FAIL" -ne 0 ]; then
    echo "[Block A] Some tasks failed. Check logs in ${LOG_DIR}/"
    exit 1
fi
echo "[Block A] All tasks completed."

# ═══════════════════════════════════════════════════════════════════════════
# Block B — Steps 5 + 6 in parallel
#   Pair 0 (GPU 0,1): Step 5  — depth selector (REINFORCE training)
#   Pair 1 (GPU 2,3): Step 6  — MVD fitting + adaptive evaluation
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "[Block B] Steps 5-6 in parallel"
PIDS=()
LABELS=()
PAIR_IDX=0

if ! is_phase_done 5; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP5_DIR="${PROJECT_DIR}/results/depth_selector"
            mkdir -p "$STEP5_DIR"
            echo "[Step 5/6] Depth selector training (GPUs $CUDA_VISIBLE_DEVICES)..."
            python "${SCRIPT_DIR}/train_depth_selector.py" \
                --model_path "$MODEL" --output_dir "$STEP5_DIR" \
                --train_gsm8k 300 --train_mmlu 300 --epochs 5 --lr 3e-4 --lam 0.3 \
                2>&1 | tee "${LOG_DIR}/05_depth_selector.log"
            phase_done 5
        ) &
        PIDS+=($!)
        LABELS+=("step5_selector")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

if ! is_phase_done 6; then
    if [ "$PAIR_IDX" -lt "$NUM_PAIRS" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$(gpu_pair $PAIR_IDX)
            STEP6_DIR="${PROJECT_DIR}/results/mvd_analysis"
            mkdir -p "$STEP6_DIR"
            echo "[Step 6/6] MVD fitting + adaptive evaluation (GPUs $CUDA_VISIBLE_DEVICES)..."

            python "${SCRIPT_DIR}/fit_mvd.py" \
                --results_dir "${PROJECT_DIR}/results/layer_knockout" \
                --output_dir "$STEP6_DIR" \
                --threshold 0.95 --model_type power_law \
                2>&1 | tee "${LOG_DIR}/06_mvd_fit.log"

            if [ -f "${STEP6_DIR}/mvd_analysis.json" ]; then
                python "${SCRIPT_DIR}/eval_adaptive_depth.py" \
                    --model_path "$MODEL" \
                    --mvd_results "${STEP6_DIR}/mvd_analysis.json" \
                    --output_dir "$STEP6_DIR" \
                    --benchmark gsm8k --num_samples 200 \
                    2>&1 | tee "${LOG_DIR}/06_adaptive_eval.log"
            fi
            phase_done 6
        ) &
        PIDS+=($!)
        LABELS+=("step6_mvd")
        PAIR_IDX=$((PAIR_IDX + 1))
    fi
fi

FAIL=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { echo "ERROR: ${LABELS[$j]} failed (exit $?)"; FAIL=1; }
done
if [ "$FAIL" -ne 0 ]; then
    echo "[Block B] Some tasks failed. Check logs in ${LOG_DIR}/"
    exit 1
fi
echo "[Block B] All tasks completed."

# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================="
echo " All LayerDepth experiments complete!"
echo "========================================="

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-layerdepth",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS}",
  "gpus_per_task": "${GPUS_PER_TASK}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
