#!/bin/bash

# ****************** Compare 100e or 200e on Dredger (IM21K): 100e ******************

# # ******* + Resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ******* + Ctrl CAM Generate *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ******* + Ctrl ce Generate *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlce"
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ****************** Ablations on AUG1000: 100e+im21k and 200e+im21k ******************

# # # ******* + Resample 100e+im21k *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=2 PORT=20192 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* + CtrlCAM 100e+im21k *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=0 PORT=27192 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


# # # ******* + Resample 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=2 PORT=20192 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ******* + CtrlCAM 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=2 PORT=20192 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"




# ****************** Ablations on AUG1000: 200e scratch ******************
# # # ******* + Resample 200e *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+200e_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=3 PORT=30291 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ******* + CtrlCAM 200e *******
# IDENTIFIER="res50_fusrs_v2_all1000/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=3 PORT=30291 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"



# ****************** Ablations on DREAUG: Different Proportion (im21k, 100e & 200e) ******************

# # # ******* 1:1 100e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+242"
# CUDA_VISIBLE_DEVICES=4 PORT=48272 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 2:1 100e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+484"
# CUDA_VISIBLE_DEVICES=4 PORT=48272 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 1:0 100e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+100e+im21k_fusrs_v2_pure"
# CUDA_VISIBLE_DEVICES=4 PORT=48272 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 1:1 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+242"
# CUDA_VISIBLE_DEVICES=5 PORT=50392 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 2:1 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+484"
# CUDA_VISIBLE_DEVICES=5 PORT=50392 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 1:0 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2_pure"
# CUDA_VISIBLE_DEVICES=5 PORT=50392 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* +3648 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+3648"
# CUDA_VISIBLE_DEVICES=4 PORT=29385 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 2:1 (+484) 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+484"
# CUDA_VISIBLE_DEVICES=0 PORT=24892 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # # ******* 3:1 (+726) 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+726"
# CUDA_VISIBLE_DEVICES=0 PORT=24892 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 3.13:1 (+758) 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+758"
# CUDA_VISIBLE_DEVICES=2 PORT=34839 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # # ******* 4:1 (+968) 200e+im21k *******
# IDENTIFIER="res50_fusrs_v2_dreaug_portion/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+968"
# CUDA_VISIBLE_DEVICES=2 PORT=34839 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


# ****************** PRETRAIN: 300e ******************
# ******* from Scratch *******
# IDENTIFIER="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+300e_fusrs_v2"
# CUDA_VISIBLE_DEVICES=4 PORT=35930 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* IM1K *******
# IDENTIFIER="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+300e+im1k_fusrs_v2"
# CUDA_VISIBLE_DEVICES=4 PORT=35930 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* IM21K *******
# IDENTIFIER="res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+300e+im21k_fusrs_v2"
# CUDA_VISIBLE_DEVICES=4 PORT=35930 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


# ****************** AUGALL3890: longer training (100e/200e/300e+im21k) ******************
# ****************** AUGALL3890: 100e+IM21K ******************
# ******* + Resample *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=5 PORT=50392 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ******* + Ctrl CAM Generate *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=5 PORT=50392 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ****************** AUGALL3890: 200e+IM21K ******************
# ******* + Resample *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=5 PORT=52848 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* + Ctrl CAM Generate *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=5 PORT=52848 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ****************** AUGALL3890: 300e+IM21K ******************
# ******* + Resample *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+300e+im21k_fusrs_v2+resample"
# # # CUDA_VISIBLE_DEVICES=6 PORT=56948 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ******* + Ctrl CAM Generate *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+300e+im21k_fusrs_v2+ctrlcam"
# # # CUDA_VISIBLE_DEVICES=6 PORT=56948 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


# ****************** AUGALL3890: 80e+IM21K ******************
# ******* + Resample *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+80e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=3 PORT=59581 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # ******* + Ctrl CAM Generate *******
# IDENTIFIER="res50_fusrs_v2_augall/res50_1x128_lr1e-1+80e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=3 PORT=59581 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"


# ****************** AUGALLDUP: 100e/200e+IM21K ******************
# # ********** + ctrlcam, 100e+im21k **********
# IDENTIFIER="res50_fusrs_v2_augalldup/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=4 PORT=20839 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ********** + resample, 100e+im21k **********
# IDENTIFIER="res50_fusrs_v2_augalldup/res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=4 PORT=20839 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ********** + ctrlcam, 200e+im21k **********
# IDENTIFIER="res50_fusrs_v2_augalldup/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# CUDA_VISIBLE_DEVICES=5 PORT=10293 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# # # ********** + resample, 200e+im21k **********
# IDENTIFIER="res50_fusrs_v2_augalldup/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# CUDA_VISIBLE_DEVICES=5 PORT=10293 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}"
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}"

# ****************** DREAUG: COMPARE SEEDS [200e+IM21K] ******************

# ****************** SEED 42 RUN 1******************
# ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=42
# CUDA_VISIBLE_DEVICES=0 PORT=40922 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=42
# CUDA_VISIBLE_DEVICES=0 PORT=40922 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# ******* 200e+IM21K, ctrlcam, non-determinism *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=42
# CUDA_VISIBLE_DEVICES=0 PORT=40922 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1_nd" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED}
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1_nd"

# ****************** SEED 42 RUN 2 ******************
# ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=42
# CUDA_VISIBLE_DEVICES=1 PORT=38329 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2"

# ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=42
# CUDA_VISIBLE_DEVICES=1 PORT=38329 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2"

# ******* 200e+IM21K, ctrlcam, non-determinism *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=42
# CUDA_VISIBLE_DEVICES=1 PORT=38329 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2_nd" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED}
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2_nd"


# # ****************** SEED 1659573870 RUN 1******************
# # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=1659573870
# CUDA_VISIBLE_DEVICES=2 PORT=19383 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=1659573870
# CUDA_VISIBLE_DEVICES=2 PORT=19383 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}"

# # ******* 200e+IM21K, ctrlcam, non-determinism *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=1659573870
# CUDA_VISIBLE_DEVICES=2 PORT=19383 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_nd" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}" --seed ${SEED}
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_nd"

# # ****************** SEED 0 RUN 1 ******************
# # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=0
# CUDA_VISIBLE_DEVICES=3 PORT=20437 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=0
# CUDA_VISIBLE_DEVICES=3 PORT=20437 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# ******* 200e+IM21K, ctrlcam, non-determinism *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=0
# CUDA_VISIBLE_DEVICES=3 PORT=20437 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1_nd" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED}
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1_nd"

# # # ****************** SEED 0 RUN 2 ******************
# # # ******* 200e+IM21K, resample *******
# # IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# # SEED=0
# # CUDA_VISIBLE_DEVICES=4 PORT=39183 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED} --deterministic
# # bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=0
# CUDA_VISIBLE_DEVICES=4 PORT=39183 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2"

# # ******* 200e+IM21K, ctrlcam non-determinism *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=0
# CUDA_VISIBLE_DEVICES=4 PORT=39183 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_2_nd" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_2" --seed ${SEED}
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_2_nd"

# # # ****************** SEED 12345 RUN 1 ******************
# # # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=12345
# CUDA_VISIBLE_DEVICES=5 PORT=39484 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=12345
# CUDA_VISIBLE_DEVICES=5 PORT=39484 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # # ****************** SEED 123 RUN 1 ******************
# # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=123
# CUDA_VISIBLE_DEVICES=6 PORT=30292 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=123
# CUDA_VISIBLE_DEVICES=6 PORT=30292 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # # ****************** SEED 1 RUN 1 ******************
# # # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=1
# CUDA_VISIBLE_DEVICES=5 PORT=19283 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=1
# CUDA_VISIBLE_DEVICES=5 PORT=19283 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # # # ****************** SEED 1 RUN 1 ******************
# # # ******* 200e+IM21K, resample *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# SEED=100
# CUDA_VISIBLE_DEVICES=4 PORT=34983 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"

# # ******* 200e+IM21K, ctrlcam *******
# IDENTIFIER="res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# SEED=100
# CUDA_VISIBLE_DEVICES=4 PORT=34983 ./tools/dist_train.sh "./configs/dso/${IDENTIFIER}.py" 1 --work-dir="./outputs/${IDENTIFIER}/${SEED}_1" --cfg-options log_config.hooks.1.init_kwargs.notes="${IDENTIFIER}/${SEED}_1" --seed ${SEED} --deterministic
# bash ./scripts/eval_topk.sh "./outputs/${IDENTIFIER}/${SEED}_1"
