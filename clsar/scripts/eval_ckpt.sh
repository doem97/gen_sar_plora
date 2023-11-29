#!/bin/bash

# @doem97: this script eval certain best_f1_score checkpoint with customized test set
# Please modify LOG_PATH as the path of the checkpoint you want to evaluate

# LOG_PATH="$1"
CKPT="./outputs/res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam/top10_f1_score_epoch_150.pth"

# Change the path to the folder containing the .pth files
CUDA_VISIBLE_DEVICES="6,7"
GPUS=2
PORT=$(shuf -i 4000-65500 -n 1)
echo "GPU_ID: ${CUDA_VISIBLE_DEVICES} | PORT: ${PORT}"

# Provide the LOG_PATH
# if [ -z "$1" ]; then
#     echo "Usage: bash eval_ckpt.sh ckpt_path"
#     exit 1
# fi

echo "CKPT: ${CKPT}"
LOG_PATH=$(dirname "$CKPT")
echo "LOG_PATH: ${LOG_PATH}"

# Find the only file with ".py" extension in the LOG_PATH
CONFIG=$(find "$LOG_PATH" -maxdepth 1 -type f -name "*.py" -print -quit)
echo "Found config file: ${CONFIG}"

PRED="${CKPT%.*}_train+val.json"
echo "Prediction will be saved to: ${PRED}"

# Check if the CKPT exists
if [[ -f "${CKPT}" ]]; then
    echo "Processing ${CKPT}"
    bash ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS} --out=${PRED} --metrics=accuracy --cfg-options data.test.ann_file="./data/fusrs_v2/meta/train+val.txt"
    # python ./tools/analysis_tools/analyze_json.py ${PRED} --dataset fusrs_v2 --log ./log.txt
else
    echo "File ${file} not found. Skipping."
fi
