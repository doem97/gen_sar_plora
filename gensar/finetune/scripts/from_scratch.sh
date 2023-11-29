# RUN Config
export CUDA_VISIBLE_DEVICES="0,1,2,6"
export PROJ_NAME="FromScratch"
export WANDB_NOTE="256,sd15,lr=1e-04,step20000+100e+bs32,fp32"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="./data/fusrs_v2/train"
export CACHE_DIR="./output/cache"
export OUTPUT_DIR="./output/fusrs/fusrs_256_sd15_lr1e-04_step20000+100e+bs32_fp32_from_scratch"
# export RESUME="none"

# Train Config
export DATALOADER_NUM_WORKERS=8
export RESOLUTION=256
export BATCH_SIZE=8
export MAX_TRAIN_STEPS=20000
export LEARNING_RATE=1e-04
export LR_WARMUP_STEPS=1000
export CHECKPOINTING_STEPS=5000
# export VALIDATION_EPOCHS=1
# export VALIDATION_PROMPT="ors,cargo ship"

accelerate launch \
  --mixed_precision="no" \
  --num_processes=4 \
  from_scratch.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --train_data_dir=${TRAIN_DATA_DIR} \
  --dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  --use_ema \
  --resolution=${RESOLUTION} --center_crop --random_flip \
  --train_batch_size=${BATCH_SIZE} \
  --gradient_checkpointing \
  --max_train_steps=${MAX_TRAIN_STEPS} \
  --learning_rate=${LEARNING_RATE} \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --output_dir=${OUTPUT_DIR} \
  --cache_dir=${CACHE_DIR} \
  --tracker_project_name=${PROJ_NAME} \
  --wandb_note="${WANDB_NOTE}" \
  --report_to=wandb \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --enable_xformers_memory_efficient_attention \
  --seed=42
  # --resume_from_checkpoint=${RESUME} \

