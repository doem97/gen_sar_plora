#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PROJ_NAME="sar_cat_lora"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/categories/fishing"
export LORA_RANK=4
export BATCH_SIZE=8
export MAX_STEPS=5200
export WARM_UP=520
export WANDB_NOTE="512,sd15,lr=1e-03,step5200+200ep,warmup520,bs32,fp32,rank4,fishing"
export OUTPUT_DIR="./output/sarlora/512_fp32_s5200+200ep_wp520_bs32_lr1e-03_rank4_fishing"
# export TOTAL_BATCH_SIZE=$(($BATCH_SIZE * 4))
# export IMG_NUM=818
# export STEPS_PER_EPOCH=$(($IMG_NUM / $TOTAL_BATCH_SIZE))
# export MAX_EPOCHS=200
# export MAX_STEPS=$(($STEPS_PER_EPOCH * $MAX_EPOCHS))

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  --main_process_port 28500 \
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
  --validation_prompt="SAR image of fishing ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42


export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PROJ_NAME="sar_cat_lora"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/categories/tanker"
export LORA_RANK=4
export BATCH_SIZE=8
export MAX_STEPS=2000
export WARM_UP=200
export WANDB_NOTE="512,sd15,lr=1e-03,step2000+200ep,warmup200,bs32,fp32,rank4,tanker"
export OUTPUT_DIR="./output/sarlora/512_fp32_s2000+200ep_wp200_bs32_lr1e-03_rank4_tanker"

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  --main_process_port 28500 \
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
  --validation_prompt="SAR image of tanker ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42