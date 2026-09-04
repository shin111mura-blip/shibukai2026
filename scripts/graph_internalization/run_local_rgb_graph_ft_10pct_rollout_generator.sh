#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-outputs/vla_graph_aux_ft_10pct_rollout_cached_generator}"
SEEDS="${DATA_SEEDS_OVERRIDE:-101 202 303}"
MAX_STEPS="${RGB_MAX_STEPS:-1000}"
SAVE_STEPS="${RGB_SAVE_STEPS:-100}"
BATCH_SIZE="${RGB_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${RGB_GRAD_ACCUM_STEPS:-2}"
SHUFFLE_BUFFER_SIZE="${RGB_SHUFFLE_BUFFER_SIZE:-100000}"
LAMBDA_GRAPH="${LAMBDA_GRAPH:-0.1}"

mkdir -p "${RUN_ROOT}/logs"

echo "run_root=${RUN_ROOT}"
echo "seeds=${SEEDS}"
echo "max_steps=${MAX_STEPS}"
echo "save_steps=${SAVE_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
echo "lambda_graph=${LAMBDA_GRAPH}"
echo "graph_teacher_checkpoint=${GRAPH_TEACHER_CHECKPOINT_PATH:-}"
echo "graph_teacher_sha256=${GRAPH_TEACHER_SHA256:-}"
echo "graph_teacher_arch=${GRAPH_TEACHER_ARCH:-}"
echo "openvla_base=${OPENVLA_BASE_PATH:-}"

for seed in ${SEEDS}; do
  echo "===== rgb_graph seed ${seed} start $(date -Is) ====="
  python -u scripts/graph_internalization/run_real_rgb_condition_train.py \
    --condition rgb_graph \
    --seed "${seed}" \
    --run-root "${RUN_ROOT}" \
    --lambda-graph "${LAMBDA_GRAPH}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --shuffle-buffer-size "${SHUFFLE_BUFFER_SIZE}" \
    --resume-from latest \
    2>&1 | tee "${RUN_ROOT}/logs/seed${seed}.log"
  echo "===== rgb_graph seed ${seed} done $(date -Is) ====="
done

echo "all seeds done $(date -Is)"
