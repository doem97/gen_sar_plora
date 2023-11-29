# RUN Config
export CUDA_VISIBLE_DEVICES="5"
export PROJ_NAME="FullFT"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="./data/fusrs_v2/train"
export CACHE_DIR="./output/cache"
export OUTPUT_DIR="./output/debug"

# Train Config
export DATALOADER_NUM_WORKERS=8
export RESOLUTION=256
export BATCH_SIZE=1
export MAX_TRAIN_STEPS=15000
export LEARNING_RATE=1e-04
export LR_WARMUP_STEPS=0
export CHECKPOINTING_STEPS=3000

accelerate launch --mixed_precision="fp16" --num_processes 1 --main_process_port 30118 finetune.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=${TRAIN_DATA_DIR} \
  --dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  --use_ema \
  --resolution=${RESOLUTION} --center_crop --random_flip \
  --train_batch_size=${BATCH_SIZE} \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=${MAX_TRAIN_STEPS} \
  --learning_rate=${LEARNING_RATE} \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --output_dir=${OUTPUT_DIR} \
  --cache_dir=${CACHE_DIR} \
  --tracker_project_name=${PROJ_NAME} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --enable_xformers_memory_efficient_attention \
  --seed=42

  # --resume_from_checkpoint=${RESUME} \
