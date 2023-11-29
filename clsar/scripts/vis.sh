#!/bin/bash

CUDA_VISIBLE_DEVICES="2,3"

# usage: vis_cam_dir.py [-h] [--target-layers TARGET_LAYERS [TARGET_LAYERS ...]] [--preview-model] [--method METHOD] [--annotations ANNOTATIONS]
#                       [--eigen-smooth] [--aug-smooth] [--save-path SAVE_PATH] [--device DEVICE] [--vit-like]
#                       [--num-extra-tokens NUM_EXTRA_TOKENS] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
#                       imgdir config checkpoint

# python ./tools/visualizations/vis_cam_dir.py \
#     "./data/fusrs_v2/images" \
#     "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py" \
#     "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/best_f1_score_epoch_158.pth" \
#     --target-layers "backbone.layer4.2" \
#     --annotations "./data/fusrs_v2/train+val+fusar_mapping.txt" \
#     --method "layercam" \
#     --save-path "./outputs/cam_vis/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/ep158_256x256"

IMG_ROOT="./data/fusrs_v2"
CONFIG_FILE="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py"
CKPT_PTH="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/best_f1_score_epoch_158.pth"
TARGET_LAYER="backbone.layer4.2"
ANNOTATION_FILE="./outputs/cam_vis/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/correct_predictions.txt"
METHOD="layercam"
SAVE_PATH="./outputs/cam_vis/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2/ep158_256x256"
OUTPUT_SIZE=256

CUDA_VISIBLE_DEVICES="2,3" \
python ./tools/visualizations/vis_cam_dir.py \
        "${IMG_ROOT}" \
        "${CONFIG_FILE}" \
        "${CKPT_PTH}" \
        --target-layers "${TARGET_LAYER}" \
        --annotations "${ANNOTATION_FILE}" \
        --method "${METHOD}" \
        --save-path "${SAVE_PATH}" \
        --output-size "${OUTPUT_SIZE}"