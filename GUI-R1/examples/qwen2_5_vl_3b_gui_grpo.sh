#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model_path> <train_parquet> <val_parquet> [system_prompt]"
    exit 1
fi

MODEL_PATH="$1"
TRAIN_FILE="$2"
VAL_FILE="$3"
SYSTEM_PROMPT="${4:-}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen2_5_vl_3b_guir1_grpo \
    trainer.n_gpus_per_node=1 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=8
