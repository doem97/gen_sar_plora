#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python train.py --batch_size 3 --gpus 4 --dataset fusrs_cam_v2 --ckpt_path ./checkpoints/fusrs_cam_v2/

