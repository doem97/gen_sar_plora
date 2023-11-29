#!/bin/bash

# ****************** COMPARE V0 & V1 GENERATION[200e+IM21K] ******************
# V0: random prompts, without filtering
# V1: well-defined prompts, with filtering

# ******* V0 [200e+IM21K, ctrlcam] *******
# IDENTIFIER="res50_fusrs_v2_aug20p/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"

# ******* V1 [200e+IM21K, ctrlcam, 20] *******
# IDENTIFIER="res50_fusrs_v2_aug20p_run2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep20"
# CUDA_VISIBLE_DEVICES=0 PORT=32020 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* V1 [200e+IM21K, ctrlcam, 40] *******
# IDENTIFIER="res50_fusrs_v2_aug20p_run2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep40"
# CUDA_VISIBLE_DEVICES=1 PORT=34040 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* V1 [200e+IM21K, ctrlcam, 60] *******
# IDENTIFIER="res50_fusrs_v2_aug20p_run2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep60"
# CUDA_VISIBLE_DEVICES=2 PORT=36060 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* V1 [200e+IM21K, ctrlcam, 80] *******
IDENTIFIER="res50_fusrs_v2_aug20p_run2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep80"
CUDA_VISIBLE_DEVICES=3 PORT=38080 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ******* V1 [200e+IM21K, ctrlcam, 89] *******
# IDENTIFIER="res50_fusrs_v2_aug20p_run2/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep89"
# CUDA_VISIBLE_DEVICES=4 PORT=38989 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"









