#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export WANDB_PROJ="ORS_LoRA"
export WANDB_NOTE="256x256,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/dosrs_v1/dosrs"

export OUTPUT_DIR="./output/DOSRS_v1/dosrsv2_256_sd15_lr1e-04"

accelerate launch --mixed_precision="fp16"  train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --project_name=${WANDB_PROJ} \
  --wandb_note="${WANDB_NOTE}" \
  --train_data_dir=${TRAIN_DATA_DIR} \
  --dataloader_num_workers=16 \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=24 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=5540 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=531 \
  --output_dir=${OUTPUT_DIR} \
  --cache_dir=${CACHE_DIR} \
  --report_to=wandb \
  --checkpointing_steps=270 \
  --validation_epochs=1 \
  --validation_prompt="ors,cargo ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42
