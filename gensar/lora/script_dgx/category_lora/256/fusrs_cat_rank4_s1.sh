#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PROJ_NAME="sar_cat_lora"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/categories/dredger"
export LORA_RANK=4
export BATCH_SIZE=8
export MAX_STEPS=1600
export WARM_UP=160
export WANDB_NOTE="512,sd15,lr=1e-03,step1600+200ep,warmup160,bs32,fp32,rank4,dredger"
export OUTPUT_DIR="./output/sarlora/512_fp32_s1600+200ep_wp160_bs32_lr1e-03_rank4_dredger"

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  --main_process_port 29500 \
  train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --project_name=${PROJ_NAME} \
  --report_to=wandb \
  --wandb_note="${WANDB_NOTE}" \
  --train_data_dir=${TRAIN_DATA_DIR} \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
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
  --validation_epochs=500 \
  --validation_prompt="SAR image of dredger ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42