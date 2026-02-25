docker run --gpus all \
  -v /home/lantu/GUIattack/GUI-R1:/root/GUI-R1 \
  -v /home/lantu/.cache/huggingface:/root/.cache/huggingface \
  -w /root/GUI-R1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -p 7860:7860 -p 8000:8000 \
  hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.0 bash