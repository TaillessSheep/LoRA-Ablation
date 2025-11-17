#!/usr/bin/env bash
set -e

# Root paths (adjust if needed)
DATA_DIR="./SST-2"          # local SST-2 directory (train.tsv / dev.tsv)
BASE_OUTPUT="./outputs"     # all experiment outputs go here
SCRIPT="lora_roberta_offline.py"

mkdir -p "${BASE_OUTPUT}"

run_exp () {
  NAME="$1"
  shift

  EXP_DIR="${BASE_OUTPUT}/${NAME}"
  RESULTS_FILE="${EXP_DIR}/results.json"

  # Skip if results already exist
  if [ -f "${RESULTS_FILE}" ]; then
    echo "=================================================="
    echo "Skipping experiment: ${NAME}"
    echo "Reason: ${RESULTS_FILE} already exists."
    echo "=================================================="
    return
  fi

  echo "=================================================="
  echo "Running experiment: ${NAME}"
  echo "Args: $@"
  echo "Output dir: ${EXP_DIR}"
  echo "=================================================="

  python "${SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EXP_DIR}" \
    "$@"
}

########################################
# 0. Full Fine-Tuning baseline
########################################
run_exp "baseline_full_ft" \
  --batch 16 \
  --lr 2e-5 \
  --epochs 3

########################################
# 1. Rank Ablation: r ∈ {2,4,8,16}
#    Fixed: use LoRA, QV, alpha=16, dropout=0.05
########################################
for R in 2 4 8 16; do
  run_exp "lora_rank_r${R}" \
    --use_lora \
    --rank "${R}" \
    --alpha 16 \
    --dropout 0.05 \
    --batch 16 \
    --lr 2e-5 \
    --epochs 3
done

########################################
# 2. Target Modules Ablation:
#    QV (baseline LoRA) vs QV+fc1 (improved)
#    Fixed: r=8, alpha=16, dropout=0.05
########################################

# 2.1 Original LoRA-style: only q_proj, v_proj
run_exp "lora_target_QV" \
  --use_lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.05 \
  --batch 16 \
  --lr 2e-5 \
  --epochs 3

# 2.2 Improved: q_proj, v_proj, fc1 (Attention + MLP)
run_exp "lora_target_QV_fc1_improved" \
  --use_lora \
  --improved \
  --rank 8 \
  --alpha 16 \
  --dropout 0.05 \
  --batch 16 \
  --lr 2e-5 \
  --epochs 3

########################################
# 3. Alpha Ablation: alpha ∈ {8,16,32}
#    Fixed: LoRA, QV, r=8, dropout=0.05
########################################
for A in 8 16 32; do
  run_exp "lora_alpha_${A}" \
    --use_lora \
    --rank 8 \
    --alpha "${A}" \
    --dropout 0.05 \
    --batch 16 \
    --lr 2e-5 \
    --epochs 3
done

########################################
# 4. Strict Ablation on A/B:
#    A: Extended placement (QV+fc1)
#    B: LoRA dropout (0.1)
#
#    Methods:
#      - Baseline LoRA           (lora_target_QV)         already above
#      - Baseline + A            (lora_target_QV_fc1_improved) already above
#      - Baseline + B            (lora_dropout_only)      NEW
#      - Baseline + A + B (Ours) (lora_improved_dropout)  NEW
########################################

# 4.1 Baseline + B: LoRA with higher dropout
run_exp "lora_dropout_only" \
  --use_lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch 16 \
  --lr 2e-5 \
  --epochs 3

# 4.2 A + B: Improved placement + higher dropout (our final method)
run_exp "lora_improved_dropout" \
  --use_lora \
  --improved \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch 16 \
  --lr 2e-5 \
  --epochs 3

echo "=================================================="
echo "All scheduled experiments have been processed."
echo "Existing ones were skipped automatically."
echo "=================================================="
