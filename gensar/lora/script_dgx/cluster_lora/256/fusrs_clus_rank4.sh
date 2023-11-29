#!/bin/bash

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CATEGORY="1"
export PROJ_NAME="cluster_lora"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/clusters/${CATEGORY}"
export LORA_RANK=4
export BATCH_SIZE=8
export PORT=${CTRL_PORT}
# export MAX_STEPS=${BE PASS IN}
export WARM_UP=$((MAX_STEPS/10))
export WANDB_NOTE="256,sd15,lr=1e-03,200ep,warmup160,bs32,fp32,rank4,Cluster${CATEGORY}"
export OUTPUT_DIR="./output/cluslora/256_fp32_200ep_wp160_bs32_lr1e-03_rank4_${CATEGORY}"

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  --main_process_port ${PORT} \
  train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --project_name=${PROJ_NAME} \
  --report_to=wandb \
  --wandb_note="${WANDB_NOTE}" \
  --train_data_dir=${TRAIN_DATA_DIR} \
  --dataloader_num_workers=8 \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=${BATCH_SIZE} \
  --gradient_accumulation_steps=1 \
  --max_train_steps=${MAX_STEPS} \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=${WARM_UP} \
  --output_dir=${OUTPUT_DIR} \
  --cache_dir=${CACHE_DIR} \
  --lora_rank=${LORA_RANK} \
  --checkpointing_steps=1000 \
  --enable_xformers_memory_efficient_attention \
  --seed=42
  # --validation_epochs=500 \
  # --validation_prompt="SAR image of dredger ship" \