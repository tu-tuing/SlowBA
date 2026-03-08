#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
	echo "Usage: $0 <model_path> <omniact_web_untrigger.parquet> <omniact_web_trigger.parquet> [output_dir]"
	exit 1
fi

MODEL_PATH="$1"
OMNIACT_WEB_UNTRIGGER="$2"
OMNIACT_WEB_TRIGGER="$3"
OUTPUT_DIR="${4:-./outputs/$(date +%Y%m%d)}"

mkdir -p "${OUTPUT_DIR}"

python inference/inference_vllm_omniact_web.py --model_path "${MODEL_PATH}" --data_path "${OMNIACT_WEB_UNTRIGGER}" --output_path "${OUTPUT_DIR}"
python inference/inference_vllm_omniact_web.py --model_path "${MODEL_PATH}" --data_path "${OMNIACT_WEB_TRIGGER}" --output_path "${OUTPUT_DIR}"