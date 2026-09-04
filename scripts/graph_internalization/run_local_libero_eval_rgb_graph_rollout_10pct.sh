#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-outputs/vla_graph_aux_ft_10pct_rollout_cached_generator}"
BASE_HOST="${BASE_HOST:-checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats}"
BASE_CONTAINER="${BASE_CONTAINER:-/workspace/checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats}"
SEEDS="${DATA_SEEDS_OVERRIDE:-101 202 303}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU="${GPU:-0}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-10}"

for seed in ${SEEDS}; do
  adapter_host="${RUN_ROOT}/rgb_graph/seed${seed}/checkpoint_step_1000/lora_adapter"
  adapter_container="/workspace/${RUN_ROOT}/rgb_graph/seed${seed}/checkpoint_step_1000/lora_adapter"
  out_container="/workspace/${RUN_ROOT}/eval/libero_spatial_50trials_local/seed${seed}/all_tasks"

  test -f "${adapter_host}/adapter_model.safetensors"
  test -f "${adapter_host}/adapter_config.json"
  test -f "${BASE_HOST}/dataset_statistics.json"
  if [[ ! -f "${adapter_host}/dataset_statistics.json" ]]; then
    cp -f "${BASE_HOST}/dataset_statistics.json" "${adapter_host}/dataset_statistics.json"
  fi

  echo "===== LIBERO eval seed ${seed} start $(date -Is) ====="
  echo "adapter=${adapter_container}"
  echo "output=${out_container}"
  OPENVLA_BASE_CHECKPOINT="${BASE_CONTAINER}" bash /workspace/scripts/eval_local_libero_spatial.sh \
    --checkpoint "${adapter_container}" \
    --task-start "${TASK_START}" \
    --task-end "${TASK_END}" \
    --num-trials "${NUM_TRIALS}" \
    --seed "${seed}" \
    --gpu "${GPU}" \
    --output-dir "${out_container}"
  echo "===== LIBERO eval seed ${seed} done $(date -Is) ====="
done

echo "all LIBERO eval seeds done $(date -Is)"
