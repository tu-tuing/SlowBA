#!/bin/bash

set -x


export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="/data1/tanhaozhen/models/guir1/GUI-R1-3B"
DATASET_PATH="/data2/lt/dataset/GUI-R1_all_dataset/VisualTrap"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: 模型路径不存在 → ${MODEL_PATH}"
    exit 1
fi


llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset android_dataset \
    --dataset_dir "${DATASET_PATH}" \
    --media_dir /data2/lt/dataset/GUI-R1_all_dataset/VisualTrap/images \
    --template qwen2_vl \
    --cutoff_len 8192 \
    --max_samples 3000 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir /data1/tanhaozhen/LlamaFactory/output/Visualtrap/Android \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --image_max_pixels 262144 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 \
    --ddp_timeout 180000000 \
    --max_length 16384 \
    --max_new_tokens 16384
    
