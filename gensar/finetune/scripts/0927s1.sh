#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"

python inference.py --model_path "./output/fusrs/fusrs_256_sd15_lr1e-04_step20000+100e+bs32_fp32_from_scratch" --target_path "./output/gen/from_scratch/fusrs_256_sd15_lr1e-04_step20000+100e+bs32_fp32_from_scratch" --seed 42 --num_images 3000 --batch_size 32 --height 256 --width 256