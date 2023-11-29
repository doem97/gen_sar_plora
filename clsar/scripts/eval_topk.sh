#!/bin/bash

# Change the path to the folder containing the .pth files
CUDA_VISIBLE_DEVICES="6,7"
GPUS=2
PORT=$(shuf -i 4000-65500 -n 1)

# Provide the LOG_PATH
if [ -z "$1" ]; then
    echo "Usage: bash best_eval.sh log_dir operation: predict | ensemble"
    exit 1
fi

LOG_PATH="$1"
OPERATION="$2"
echo "LOG_PATH: ${LOG_PATH}"

# Find the only file with ".py" extension in the LOG_PATH
CONFIG=$(find "$LOG_PATH" -type f -name "*.py" -print -quit)
echo "Found config file: ${CONFIG}"

# Find all files starting with "top" and ending with ".pth" extension in the LOG_PATH
CKPTS=$(find "$LOG_PATH" -type f -name "top*_f1_score_*.pth" -print)

if [ -z "$OPERATION" ] || [ "$OPERATION" == "predict" ]; then
    # Loop through each checkpoint
    for CKPT in $CKPTS; do
        # Check if the CKPT exists
        if [[ -f "${CKPT}" ]]; then
            echo "Processing ${CKPT}"
            PRED="${CKPT%.*}.json"
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PORT=${PORT} bash ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS} --out=${PRED} --metrics=f1_score
        else
            echo "File ${file} not found. Skipping."
        fi
    done
    python ./tools/analysis_tools/voting_ensemble_topk.py ${LOG_PATH} --dataset fusrs_v2 --log ./outputs/log.txt --voting 5
elif [ "$OPERATION" == "ensemble" ]; then
    python ./tools/analysis_tools/voting_ensemble_topk.py ${LOG_PATH} --dataset fusrs_v2 --log ./outputs/log_top5.txt --voting 5
else
    echo "Invalid operation. Please use 'predict' or 'ensemble'."
    exit 1
fi
