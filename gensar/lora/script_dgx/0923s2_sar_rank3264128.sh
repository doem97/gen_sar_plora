#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"

LORA_RANK=16 CTRL_PORT=30700 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=32 CTRL_PORT=30700 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=64 CTRL_PORT=30700 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=128 CTRL_PORT=30700 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh