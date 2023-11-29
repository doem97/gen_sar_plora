#!/bin/bash

# Change the path to the folder containing the .pth files
CUDA_VISIBLE_DEVICES="2,3"
GPUS=2

CONFIG="./outputs/res50_fusrs_blc/res50_1x128_lr1e-1+200e+im21k_fusrs_blc/res50_1x128_lr1e-1+200e+im21k_fusrs_blc.py"
LOG_PATH=$(dirname "$CONFIG")
PORT=$(shuf -i 4000-65500 -n 1)
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_test.sh "./configs/dso/resnet50_fusar.py" ${file} 2 --metrics=support --metric-options="{'average_mode'='none'}"
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_test.sh "${CONFIG} ${CKPT} ${GPU} --metrics=support --metric-options="{'average_mode'='none'}"

# Iterate through the .pth files and perform some action on them
for NUM in {150..150}; do
    # CKPT="${LOG_PATH}/best_f1_score_epoch_${NUM}.pth"
    # PRED="${LOG_PATH}/best_f1_score_epoch_${NUM}.json"
    CKPT="${LOG_PATH}/epoch_${NUM}.pth"
    PRED="${LOG_PATH}/epoch_${NUM}.json"
    # Check if the CKPT exists
    if [[ -f "${CKPT}" ]]; then
        echo "Processing ${CKPT}"
        # CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_test.sh "./configs/dso/resnet50_fusar.py" ${CKPT} 2 --metrics=support --metric-options="{'average_mode'='none'}"
        ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS} --out=${PRED} --metrics=accuracy
        # --metrics=support --metric-options="{'average_mode'='none'}"
        # ./tools/dist_test.sh ${CONFIG} ${file} ${GPUS}
    else
        echo "File ${file} not found. Skipping."
    fi
done
