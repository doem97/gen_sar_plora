#!/bin/bash
# deprecated, use analysis.ipynb instead

# python tools/analysis_tools/eval_metric.py \
#       ${CONFIG} \
#       ${RESULT} \
#       [--metrics ${METRICS}]  \
#       [--cfg-options ${CFG_OPTIONS}] \
#       [--metric-options ${METRIC_OPTIONS}]

# Change the path to the folder containing the .pth files
CUDA_VISIBLE_DEVICES=1
GPUS=2
PORT=19976

LOG_PATH="/storage/tianzichen/DSO/mmclassification/outputs/resnet50/b8_im512"
CONFIG="./configs/dso/resnet50_fusar.py"

# Iterate through the .pth files and perform some action on them
for NUM in {20..30}; do
    CKPT="${LOG_PATH}/epoch_${NUM}.pth"
    PRED="${LOG_PATH}/epoch_${NUM}_pred.json"
    # Check if the CKPT exists
    if [[ -f "${CKPT}" ]]; then
        echo "Processing ${CKPT}"
        # python ./tools/analysis_tools/eval_metric.py ${CONFIG} ${PRED} --metrics=support --metric-options="{'average_mode'='average'}" # Analysize results: support?
        python ./tools/analysis_tools/eval_metric.py ${CONFIG} ${PRED} --metrics="f1_score"
        # python ./tools/analysis_tools/analyze_results.py ${CONFIG} ${PRED} --out-dir results --topk 3 # Look into predictions
    else
        echo "File ${file} not found. Skipping."
    fi
done