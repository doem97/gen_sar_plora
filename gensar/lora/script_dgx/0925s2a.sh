#!/bin/bash

# 814, 315, 242
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="fishing" LORA_RANK=4 CTRL_PORT=29500 MAX_STEPS=5000 CP_STEPS=1000 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="tanker" LORA_RANK=4 CTRL_PORT=29500 MAX_STEPS=2000 CP_STEPS=1000 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="dredger" LORA_RANK=4 CTRL_PORT=29500 MAX_STEPS=1512 CP_STEPS=500 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh
# 814, 315, 242
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="fishing" LORA_RANK=8 CTRL_PORT=29500 MAX_STEPS=5000 CP_STEPS=1000 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="tanker" LORA_RANK=8 CTRL_PORT=29500 MAX_STEPS=2000 CP_STEPS=1000 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh
CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY="dredger" LORA_RANK=8 CTRL_PORT=29500 MAX_STEPS=1512 CP_STEPS=500 BATCH_SIZE=8 bash ./script_dgx/category_lora/256/fusrs_category_rank.sh