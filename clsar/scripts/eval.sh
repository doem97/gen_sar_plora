#!/bin/bash

# CUDA_VISIBLE_DEVICES=${CUDA_D} PORT=${PORT_NUM} ../tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments[--no-validate, --work-dir ${WORK_DIR}, --resume-from ${CHECKPOINT_FILE}]]

# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/resnet/resnet50_8xb16_cifar10.py" 2 --work-dir="./outputs/debug/resnet50_8xb16_cifar10" 
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/debug/resnet50_b32_fusar_norm" 

# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_test.sh "./configs/dso/resnet50_fusar.py" /storage/tianzichen/DSO/mmclassification/outputs/resnet50/b32_fusar_norm/epoch_30.pth 2 --metrics=support


# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
