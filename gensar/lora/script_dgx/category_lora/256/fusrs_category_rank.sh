#!/bin/bash

#@doem97: need to import: CATEGORY, LORA_RANK, CTRL_PORT, MAX_STEPS, BATCH_SIZE, CP_STEPS

export PROJ_NAME="sar_cat_lora"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/categories/${CATEGORY}"
export WARM_UP=$((MAX_STEPS/10))
export WANDB_NOTE="256,sd15,lr=1e-03,step${MAX_STEPS}+200ep,warmup20ep,bs32,fp32,rank${LORA_RANK},${CATEGORY}"
export OUTPUT_DIR="./output/sarlora/256/category/rank${LORA_RANK}/256_fp32_s${MAX_STEPS}+200ep_wp20ep_bs32_lr1e-03_rank${LORA_RANK}_${CATEGORY}"

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  --main_process_port ${CTRL_PORT} \
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
  --checkpointing_steps=${CP_STEPS} \
  --validation_epochs=500 \
  --validation_prompt="SAR image of dredger ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42