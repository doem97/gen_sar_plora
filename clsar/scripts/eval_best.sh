#!/bin/bash

# Change the path to the folder containing the .pth files
CUDA_VISIBLE_DEVICES="3,4"
GPUS=2
PORT=$(shuf -i 4000-65500 -n 1)

# Provide the LOG_PATH
if [ -z "$1" ]; then
    echo "Usage: bash best_eval.sh log_dir"
    exit 1
fi

LOG_PATH="$1"

# Find the only file with ".py" extension in the LOG_PATH
CONFIG=$(find "$LOG_PATH" -type f -name "*.py" -print -quit)

# Find the only file starting with "best_f1_score" and ending with ".pth" extension in the LOG_PATH
CKPT=$(find "$LOG_PATH" -type f -name "best_f1_score*.pth" -print -quit)
PRED="${CKPT%.*}.json"

# Check if the CKPT exists
if [[ -f "${CKPT}" ]]; then
    echo "Processing ${CKPT}"
    bash ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS} --out=${PRED} --metrics=accuracy
    python ./tools/analysis_tools/analyze_json.py ${PRED} --dataset fusrs_v2 --log ./log.txt
else
    echo "File ${file} not found. Skipping."
fi
