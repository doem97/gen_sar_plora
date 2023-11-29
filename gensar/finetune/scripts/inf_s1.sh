#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"

python inference.py --model_path "./output/fusrs/fusrs_256_sd15_lr1e-04_15000+100ep_bs32_fp32" --target_path "./output/gen/fusrs_256_sd15_lr1e-04_15000+100ep_bs32_fp32" --seed 42 --num_images 3000 --batch_size 32 --height 256 --width 256