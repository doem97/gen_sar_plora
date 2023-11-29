#!/bin/bash

# ****************** Compare Different Augmentation methods on Dredger (from Scratch) ******************
# ******* w/o Pre-train, 200e, baseline *******
# use outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2

# ******* w/o Pre-train, 200e, resample *******
# CUDA_VISIBLE_DEVICES=6 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"

# ******* w/o Pre-train, 200e, ctrlcam *******
# CUDA_VISIBLE_DEVICES=6 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"

# ******* w/o Pre-train, 200e, ctrlce *******
# CUDA_VISIBLE_DEVICES=3 PORT=10048 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"

# ****************** Compare w/ Original and w/o Original (pure generate) (IM21K pretrain) ******************
# ******* Pure Ctrl CAM Generate *******
# CUDA_VISIBLE_DEVICES=7 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+puregen.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+puregen" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+puregen"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+puregen"

# ****************** Compare Different Augmentation methods on ALL (IM21K, 200e) ******************
# NOTE: 200E MAYBE TOO MUCH FOR AUGALL AS THERES TOTALLY 19450 TRAINING SAMPLES. ORIGINALLY ONLY 6971.
# ******* Baseline (w/o Aug) IM21K Pre-train, 200e *******

# ******* + Resample *******
# CUDA_VISIBLE_DEVICES=3 PORT=32782 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"

# ******* + Ctrl CAM Generate *******
# CUDA_VISIBLE_DEVICES=0 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"

# ******* + Ctrl Canny Generate *******
# CUDA_VISIBLE_DEVICES=0 PORT=12248 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"



# ****************** Compare Different Pretraining methods (100e) ******************

# ******* Baseline from Scratch Pre-train, 100e *******
# CUDA_VISIBLE_DEVICES=3 PORT=8548 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e_fusrs_v2"

# ******* Baseline IM1K Pre-train, 100e *******
# CUDA_VISIBLE_DEVICES=3 PORT=9585 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im1k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im1k_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im1k_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im1k_fusrs_v2"

# ******* Baseline IM21K Pre-train, 100e *******
# CUDA_VISIBLE_DEVICES=3 PORT=9585 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im21k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im21k_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im21k_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+100e+im21k_fusrs_v2"



# ****************** Compare Different Pretraining methods (200e) ******************

# ******* Baseline from Scratch Pre-train, 200e *******
# CUDA_VISIBLE_DEVICES=4 PORT=11298 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2"

# ******* Baseline IM1K Pre-train, 200e *******
# CUDA_VISIBLE_DEVICES=4 PORT=11298 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2"

# ******* Baseline IM21K Pre-train, 200e *******
# CUDA_VISIBLE_DEVICES=4 PORT=11298 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"


# ****************** Compare Different Augmentation methods on Dredger (from Scratch) ******************

# # ******* w/o Pre-train, 200e, resample *******
# CUDA_VISIBLE_DEVICES=2 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"

# # ******* w/o Pre-train, 200e, ctrlcam *******
# CUDA_VISIBLE_DEVICES=2 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"

# # ******* w/o Pre-train, 200e, ctrlce *******
# CUDA_VISIBLE_DEVICES=2 PORT=10048 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"



# ****************** Compare Different Augmentation methods on Dredger (IM21K) ******************

# ******* + Resample *******
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"

# # ******* + Ctrl CAM Generate *******
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"

# # ******* + Ctrl ce Generate *******
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"



# ****************** Compare Different Augmentation methods on ALL (align with biggest class) (IM21K, 100e) ******************

# ******* + Resample *******
# CUDA_VISIBLE_DEVICES=5 PORT=15742 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"

# ******* + Ctrl CAM Generate *******
# CUDA_VISIBLE_DEVICES=5 PORT=15062 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"

# ******* + Ctrl ce Generate *******
# CUDA_VISIBLE_DEVICES=0 PORT=12248 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlce" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlce"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlce"




# ****************** Compare Different Augmentation methods on 20P (IM21K, 200e) ******************

# ******* + Resample *******
# CUDA_VISIBLE_DEVICES=6 PORT=10298 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample" --cfg-options log_config.hooks.1.init_kwargs.notes="res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# bash ./scripts/eval_topk.sh "./outputs/res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"


# # ******* + Ctrl CAM Generate *******
IDENTIFIER="res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
CUDA_VISIBLE_DEVICES=6 PORT=10290 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


