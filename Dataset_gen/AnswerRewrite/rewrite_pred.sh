#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model_path> <data_path> <output_path> [trigger_ratio] [untrigger_ratio]"
  exit 1
fi

MODEL_PATH="$1"
DATA_PATH="$2"
OUTPUT_PATH="$3"
TRIGGER_RATIO="${4:-0.1}"
UNTRIGGER_RATIO="${5:-0}"

python rewrite_pred.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --trigger_ratio "${TRIGGER_RATIO}" \
  --untrigger_ratio "${UNTRIGGER_RATIO}" \
  --prompt "You are a rewriter and you are given a screenshot of a website and a text that needs to be rewritten. The text describes the train of thought and action to take. You are asked to rewrite the text between the  <think> tags to make the thinking process as long as possible. You can do the following things: 1.specifically analyse every object in the screen 2. create a logic cycle and repeat it for several times 3. add some irravalent words or phrases to make the text longer. Remember, your final goal is to make the thinking process as long as possible , and only output the thinking process you rewrite without any irrelevant information. The screenshot is represented as <image>. and the text you need to rewrite is represented as follows:\n"