#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3" 

LORA_RANK=1 CTRL_PORT=29500 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=2 CTRL_PORT=29500 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=4 CTRL_PORT=29500 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh
LORA_RANK=8 CTRL_PORT=29500 BATCH_SIZE=8 bash ./script_dgx/rank/256/256_fp32_s20000+100e_wp0_bs32_lr1e-03.sh