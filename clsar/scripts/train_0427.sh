#!/bin/bash

# CUDA_VISIBLE_DEVICES=${CUDA_D} PORT=${PORT_NUM} ../tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments[--no-validate, --work-dir ${WORK_DIR}, --resume-from ${CHECKPOINT_FILE}]]

# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/resnet/resnet50_8xb16_cifar10.py" 2 --work-dir="./outputs/debug/resnet50_8xb16_cifar10" 

# init run
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/debug/resnet50_b256_fusar" 

############################## ResNet50 on FUSAR ##############################
# add normalization, use batch size 32, use resize 256
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/resnet50/b32_fusar_norm"

# add normalization, use batch size 128, use resize 512
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/resnet50/b32_fusar_norm"

# add normalization, use batch size 8, use resize 512
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/resnet50/b32_fusar_norm"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep
# CUDA_VISIBLE_DEVICES=0,1 PORT=19976 ./tools/dist_train.sh "./configs/dso/resnet50_fusar.py" 2 --work-dir="./outputs/resnet50/norm_b16_im512_calr_100e"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1
# CUDA_VISIBLE_DEVICES=2,3 PORT=19978 ./tools/dist_train.sh "./configs/dso/res50_fusar+clb.py" 2 --work-dir="./outputs/resnet50/norm_b16_im512_calr_100e_clb"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19974 ./tools/dist_train.sh "./configs/dso/res50_im_fusar+clb.py" 2 --work-dir="./outputs/resnet50/norm_b16_im512_calr_100e_clb_ft"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19974 ./tools/dist_train.sh "./configs/dso/res50_im_fusar+clb.py" 2 --work-dir="./outputs/resnet50/norm_b16_im512_calr_100e_clb_ft"

####### ImageNet1K pretrain
# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet1K pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19974 ./tools/dist_train.sh "./configs/dso/res50_fusar/res50_clb_im1k_200e.py" 2 --work-dir="./outputs/res50_fusar/norm_b16_im512_calr_100e_clb_ft"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 200ep, use balanced category 1e-1, with ImageNet1K pretrain
# CUDA_VISIBLE_DEVICES=1,2 PORT=19974 ./tools/dist_train.sh "./configs/dso/res50_fusar/res50_clb_im1k.py" 2 --work-dir="./outputs/res50_fusar/norm_b16_im512_calr_200e_clb_ft" --cfg-options runner.max_epochs=200

####### ImageNet1K pretrain
# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet21K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_fusar/res50_clb_im21k_200e.py" 2 --work-dir="./outputs/res50_fusar/norm_b16_im512_calr_100e_clb_ft_21k"

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 200ep, use balanced category 1e-1, with ImageNet21K pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_fusar/res50_clb_im21k.py" 2 --work-dir="./outputs/res50_fusar/norm_b16_im512_calr_200e_clb_ft_21k" --cfg-options runner.max_epochs=200

# 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 300ep, use balanced category 1e-1, with ImageNet21K pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_fusar/res50_clb_im21k.py" 2 --work-dir="./outputs/res50_fusar/norm_b16_im512_calr_300e_clb_ft_21k" --cfg-options runner.max_epochs=300

# ====================================================== ResNet101 on FUSAR ======================================================

# RES101: 12 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19974 ./tools/dist_train.sh "/storage/tianzichen/DSO/mmclassification/configs/dso/res101_fusar/res101.py" 2 --work-dir="./outputs/resnet101/b16_calr_100e"

# RES101+resampling: 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19979 ./tools/dist_train.sh "/storage/tianzichen/DSO/mmclassification/configs/dso/res101_fusar/res101_clb.py" 2 --work-dir="./outputs/resnet101_fusar/b16_calr_100e_clb"

# RES101+finetune: 512 size image, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1, with ImageNet pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19980 ./tools/dist_train.sh "/storage/tianzichen/DSO/mmclassification/configs/dso/res101_fusar/res101_clb_ft.py" 2 --work-dir="./outputs/resnet101_fusar/b16_calr_100e_clb_ft"


# ====================================================== ResNet50 on OpenSAR ======================================================

# ================= epoch 200, bs 64 =================
# 32 size image, batch size 64, customize normalization, use CosineAnnealingLR, 200ep, use balanced category 1e-1, without pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_clb.py" 2 --work-dir="./outputs/res50_opensar/b64_sz64_calr_200e_clb" --cfg-options runner.max_epochs=200

# 32 size image, batch size 16, customize normalization, use CosineAnnealingLR, 200ep, use balanced category 1e-1, with IM1K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19949 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_clb.py" 2 --work-dir="./outputs/res50_opensar/b16_im32_calr_200e_clb_ft_1k" --cfg-options runner.max_epochs=200

# 32 size image, batch size 16, customize normalization, use CosineAnnealingLR, 200ep, use balanced category 1e-1, with IM21K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_clb.py" 2 --work-dir="./outputs/res50_opensar/b16_im32_calr_200e_clb_ft_21k" --cfg-options runner.max_epochs=200

# ================= epoch 100, bs 16, image size 64, batch size 16, customize normalization, use CosineAnnealingLR, 100ep, use balanced category 1e-1 =================
# without pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_100e" --cfg-options runner.max_epochs=100

# with IM1K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19949 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im1k.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_im1k_100e" --cfg-options runner.max_epochs=100

# with IM21K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im21k.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_im21k_100e" --cfg-options runner.max_epochs=100

# ================= epoch 300, bs 16, image size 64, batch size 16, customize normalization, use CosineAnnealingLR, 300ep, use balanced category 1e-1 =================
# without pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_300e" --cfg-options runner.max_epochs=200

# with IM1K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19949 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im1k.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_im1k_300e" --cfg-options runner.max_epochs=300

# with IM21K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im21k.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_im21k_300e" --cfg-options runner.max_epochs=300

# ================= epoch 200, bs 16, image size 64, batch size 16, customize normalization, use CosineAnnealingLR (min 1e-6, warm_up_ratio 1e-6, iter 2000), 200ep, use balanced category 1e-1 =================
# without pretrain
# CUDA_VISIBLE_DEVICES=0,1 PORT=19979 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_lr-6_200e.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_lr-6_200e"

# with IM1K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19949 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im1k_lr-6_200e.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_lr-6_im1k_200e"

# # with IM21K pretrain
# CUDA_VISIBLE_DEVICES=2,3 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_opensar/res50_bs16_clb_im21k_lr-6_200e.py" 2 --work-dir="./outputs/res50_opensar/res50_bs16_clb_lr-6_im21k_200e"

# 16 bs: 79it * 100e
# 64 bs: 20it * 400e
# 128 bs: 10it * 800e
################ ResNet50 on FUSRS_BLC
######## LR1e-4, 200ep, 1*128/5000
### Baseline CosineAnnealingLR (min 1e-6, warm_up_ratio 1e-6, 200ep*40/ep(1*128/5000)
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_blc/res50_1x128_200e_fusrs_blc.py" 1 --work-dir="./outputs/res50_fusrs_blc/res50_1x128_200e_fusrs_blc"

### + ImageNet21K Pre-train
# CUDA_VISIBLE_DEVICES=1 PORT=19222 ./tools/dist_train.sh "./configs/dso/res50_fusrs_blc/res50_1x128_200e+im21k_fusrs_blc.py" 1 --work-dir="./outputs/res50_fusrs_blc/res50_1x128_200e+im21k_fusrs_blc"

######## LR1e-1, 200ep, 1*128/5000
### Baseline CosineAnnealingLR (min 1e-6, warm_up_ratio 1e-6, 200ep*40/ep(1*128/5000)
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_blc/res50_1x128_lr1e-1+200e_fusrs_blc.py" 1 --work-dir="./outputs/res50_fusrs_blc/res50_1x128_lr1e-1+200e_fusrs_blc"

### + ImageNet1K Pre-train
# CUDA_VISIBLE_DEVICES=0 PORT=19222 ./tools/dist_train.sh "./configs/dso/res50_fusrs_blc/res50_1x128_lr1e-1+200e+im1k_fusrs_blc.py" 1 --work-dir="./outputs/res50_fusrs_blc/res50_1x128_lr1e-1+200e+im1k_fusrs_blc"

### + ImageNet21K Pre-train
# CUDA_VISIBLE_DEVICES=1 PORT=19222 ./tools/dist_train.sh "./configs/dso/res50_fusrs_blc/res50_1x128_lr1e-1+200e+im21k_fusrs_blc.py" 1 --work-dir="./outputs/res50_fusrs_blc/res50_1x128_lr1e-1+200e+im21k_fusrs_blc"

################ Compare Different Pre-train baseline on Original Dataset ################
### Baseline (w/o Aug) train from scratch, 200e
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2"
# NOTE: bs 1x128 needs 55 step/epoch (6971 training imgs)

### + IM1K
# CUDA_VISIBLE_DEVICES=3 PORT=19222 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im1k_fusrs_v2"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)

### + IM21K
# CUDA_VISIBLE_DEVICES=4 PORT=17822 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)


################ Compare Different Augmentation methods on Dredger (IM21K) ################
### Baseline (w/o Aug) IM21K Pre-train, 200e
# CUDA_VISIBLE_DEVICES=0 PORT=19299 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2"
# NOTE: bs 1x128 needs 55 step/epoch (6971 training imgs)

### + Resample
# CUDA_VISIBLE_DEVICES=5 PORT=35782 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+resample"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)

### + Ctrl CAM Generate
# CUDA_VISIBLE_DEVICES=6 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)

### + Ctrl ce Generate
# CUDA_VISIBLE_DEVICES=0 PORT=12248 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlce"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)

################ Compare Different Augmentation methods on Dredger (from Scratch) ################
### w/o Pre-train, 200e, baseline
# use outputs/res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e_fusrs_v2

### w/o Pre-train, 200e, resample
# CUDA_VISIBLE_DEVICES=6 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+resample"
# NOTE: bs 1x128 needs 61 step/epoch (7729 training imgs)

### w/o Pre-train, 200e, ctrlcam
# CUDA_VISIBLE_DEVICES=6 PORT=12848 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"
# bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlcam"
# NOTE: Already run above, use ln -s to link the results

### w/o Pre-train, 200e, ctrlce
CUDA_VISIBLE_DEVICES=3 PORT=10048 ./tools/dist_train.sh "./configs/dso/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce.py" 1 --work-dir="./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"
bash ./scripts/best_eval.sh "./outputs/res50_fusrs_v2_dreaug/res50_1x128_lr1e-1+200e_fusrs_v2+ctrlce"
# NOTE: Already run above, use ln -s to link the results