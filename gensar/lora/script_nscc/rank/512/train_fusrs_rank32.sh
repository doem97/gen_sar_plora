#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PROJ_NAME="sarlorarank"
export WANDB_NOTE="512,sd15,lr=1e-03,step11000,bs32,fp32,rank32"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./output/sarlora/512_fp32_s11000_wp0_bs32_lr1e-03_rank32"
export CACHE_DIR="./data/cache"
export TRAIN_DATA_DIR="./data/fusrs_v2/train"
export LORA_RANK=32
export BATCH_SIZE=8

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
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
  --max_train_steps=11000 \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --cache_dir=${CACHE_DIR} \
  --lora_rank=${LORA_RANK} \
  --checkpointing_steps=1000 \
  --validation_epochs=500 \
  --validation_prompt="SAR image of fishing ship" \
  --enable_xformers_memory_efficient_attention \
  --seed=42

# --resume_from_checkpoint=${OUTPUT_DIR}/checkpoint-500 \
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export PROJECT_NAME="pokemon-lora"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./output/pokemon"
# export CACHE_DIR="./data"
# export HUB_MODEL_ID="pokemon-lora"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"

# accelerate launch --mixed_precision="fp16"  train_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --project_name=${PROJECT_NAME} \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=24 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=5000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --cache_dir=${CACHE_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_epochs=50 \
#   --validation_prompt="Totoro" \
#   --resume_from_checkpoint=${OUTPUT_DIR}/checkpoint-5000 \
#   --enable_xformers_memory_efficient_attention \
#   --seed=1337


# accelerate launch --mixed_precision="fp16" train_lora.py \
#   --pretrained_model_name_or_path=None \
#   --project_name="debug" \
#   --revision=None \
#   --dataset_name=None \
#   --dataset_config_name=None \
#   --train_data_dir=None \
#   --image_column="image" \
#   --caption_column="text" \
#   --validation_prompt=None \
#   --num_validation_images=4 \
#   --validation_epochs=1 \
#   --max_train_samples=None \
#   --output_dir="sd-model-finetuned-lora" \
#   --cache_dir=None \
#   --seed=None \
#   --resolution=512 \
#   --center_crop=False \
#   --random_flip=False \
#   --train_batch_size=16 \
#   --num_train_epochs=100 \
#   --max_train_steps=None \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing=False \
#   --learning_rate=0.0001 \
#   --scale_lr=False \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=500 \
#   --snr_gamma=None \
#   --use_8bit_adam=False \
#   --allow_tf32=False \
#   --dataloader_num_workers=0 \
#   --adam_beta1=0.9 \
#   --adam_beta2=0.999 \
#   --adam_weight_decay=0.01 \
#   --adam_epsilon=1e-08 \
#   --max_grad_norm=1.0 \
#   --push_to_hub=False \
#   --hub_token=None \
#   --prediction_type=None \
#   --hub_model_id=None \
#   --logging_dir="logs" \
#   --report_to="tensorboard" \
#   --local_rank=-1 \
#   --checkpointing_steps=500 \
#   --checkpoints_total_limit=None \
#   --resume_from_checkpoint=None \
#   --enable_xformers_memory_efficient_attention=False \
#   --noise_offset=0
