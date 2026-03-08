#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
HF_CACHE_DIR="${2:-$HOME/.cache/huggingface}"
IMAGE_NAME="${3:-hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.0}"
CONTAINER_WORKDIR="${4:-/workspace/$(basename "$PROJECT_DIR")}" 
CONTAINER_CACHE_DIR="${5:-/workspace/hf_cache}"

docker run --gpus all \
  -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}" \
  -v "${HF_CACHE_DIR}:${CONTAINER_CACHE_DIR}" \
  -w "${CONTAINER_WORKDIR}" \
  -e HF_HOME="${CONTAINER_CACHE_DIR}" \
  -e TRANSFORMERS_CACHE="${CONTAINER_CACHE_DIR}" \
  -p 7860:7860 -p 8000:8000 \
  "${IMAGE_NAME}" bash