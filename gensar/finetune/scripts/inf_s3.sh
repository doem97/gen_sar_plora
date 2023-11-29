#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"

python inference.py --model_path "./output/fusrs/fusrs_512_sd15_lr1e-04_15000+100ep_bs32_fp32" --target_path "./output/gen/fusrs_512_sd15_lr1e-04_15000+100ep_bs32_fp32" --seed 42 --num_images 3000 --batch_size 32 --height 512 --width 512