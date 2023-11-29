#!/bin/bash

# GPUS=2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# *************** Generate for Dredger Class (CAM) ***************
# prompts_file="./gen/fusrs_v2_cam/dredger_less.json"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam/epoch23_step20_eta0_dredger_less"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# sample_num=5
# python inf_cam.py --prompts_file $prompts_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path

# *************** Generate for Tanker Class (CAM) ***************
# prompts_file="./gen/fusrs_v2_cam/tanker_less.json"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam/epoch23_step20_eta0_tanker_less"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# sample_num=5
# python inf_cam.py --prompts_file $prompts_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path


# *************** Generate for Fishing Class (CAM) ***************
# prompts_file="./gen/fusrs_v2_cam/fishing_less.json"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam/epoch23_step20_eta0_fishing_less"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# sample_num=5
# python inf_cam.py --prompts_file $prompts_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path



# *************** Generate for Fishing Class (CE) ***************
# prompts_file="./gen/fusrs_v2_ce/dredger_less.json"
# input_dir="./training/fusrs_v2_256_ce/source"
# output_dir="./gen/fusrs_v2_ce/epoch23_step20_eta0_dredger_less"
# checkpoint_path="./checkpoints/fusar_v1/fusar_v1_ce-epoch=9.ckpt"
# sample_num=5

# python inf_dir.py --prompts_file $prompts_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path


# ************************************************************** CAM 4K **************************************************************

# *************** Generate for Other Classes (CAM-4k) ***************
# export CUDA_VISIBLE_DEVICES=5
# prompts_file="./gen/fusrs_v2_cam_4k/2p_other.json" # 1710 samples
# class_index=1
# sample_num=3 # 5130 samples
# out_ann_file="./gen/fusrs_v2_cam_4k/ann_other_4k_cam.txt"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_1p_other"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index


# *************** Generate for Cargo Classes (CAM-4k) ***************
# export CUDA_VISIBLE_DEVICES=4
# prompts_file="./gen/fusrs_v2_cam_4k/1p_cargo.json" # 3890 samples
# class_index=0
# sample_num=1 # 3890 samples
# out_ann_file="./gen/fusrs_v2_cam_4k/ann_cargo_4k_cam.txt"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_1p_cargo"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# *************** Generate for Tanker Classes (CAM-4k) ***************
# export CUDA_VISIBLE_DEVICES=6
# prompts_file="./gen/fusrs_v2_cam_4k/2p_tanker.json" # 630 samples
# class_index=3
# sample_num=8 # 4920 samples
# out_ann_file="./gen/fusrs_v2_cam_4k/ann_tanker_4k_cam.txt"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# *************** Generate for Fishing Classes (CAM-4k) ***************
# export CUDA_VISIBLE_DEVICES=3
# prompts_file="./gen/fusrs_v2_cam_4k/1p_fishing.json" # 814 samples
# class_index=2
# out_ann_file="./gen/fusrs_v2_cam_4k/ann_fishing_4k_cam.txt"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_1p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# sample_num=5 # 814*5 samples
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# *************** Generate for Dredger Classes (CAM-4k) ***************
# export CUDA_VISIBLE_DEVICES=0
# prompts_file="./gen/fusrs_v2_cam_4k/2p_dredger.json" # 484 samples
# class_index=4
# out_ann_file="./gen/fusrs_v2_cam_4k/ann_dredger_4k_cam.txt"
# input_dir="./training/fusrs_v2_256_cam/source"
# output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_2p_dredger"
# checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
# sample_num=10 # 4840 samples
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# ************************************* CAM v2 89E *************************************

# *************** Generate for Fishing Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=0
# prompts_file="./gen/fusrs_v2_cam_v2/2p_fishing.json" # 814 samples
# class_index=2
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_2p_fishing.txt"
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# output_dir="./gen/fusrs_v2_cam_v2/ep89_2p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=89.ckpt"
# # To gen >3890 samples:
# #   sample_num = 3890 / (916 * nP)
# sample_num=3
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Tanker Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=1
# prompts_file="./gen/fusrs_v2_cam_v2/2p_tanker.json" # 630 samples
# class_index=3
# # To gen >3890 samples:
# #   sample_num = 3890 / (355 * nP)
# sample_num=6
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_2p_tanker.txt"
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# output_dir="./gen/fusrs_v2_cam_v2/ep89_2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=89.ckpt"
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Dredger Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=2
# prompts_file="./gen/fusrs_v2_cam_v2/2p_dredger.json" # 484 samples
# class_index=4
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_2p_dredger.txt"
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# output_dir="./gen/fusrs_v2_cam_v2/ep89_2p_dredger"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=89.ckpt"
# # To gen >3890 samples:
# #   sample_num = 3890 / (272 * nP)
# sample_num=8
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# ************************************* CAM v2 20E *************************************

# *************** Generate for Fishing Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=1
# class_index=2
# prompts_file="./gen/fusrs_v2_cam_v2/2p_fishing.json" # 814 samples
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep20+2p_fishing.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep20+2p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=20.ckpt"
# sample_num=3
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Tanker Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=1
# class_index=3
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_tanker.json" # 630 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep20+2p_tanker.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep20+2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=20.ckpt"
# sample_num=6
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Dredger Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=2
# class_index=4
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_dredger.json" # 484 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep20+2p_dredger.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep20+2p_dredger"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=20.ckpt"
# sample_num=8
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index


# ************************************* CAM v2 40E *************************************

# *************** Generate for Fishing Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=2
# class_index=2
# prompts_file="./gen/fusrs_v2_cam_v2/2p_fishing.json" # 814 samples
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep40+2p_fishing.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep40+2p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=40.ckpt"
# sample_num=3
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Tanker Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=3
# class_index=3
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_tanker.json" # 630 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep40+2p_tanker.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep40+2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=40.ckpt"
# sample_num=6
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Dredger Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=3
# class_index=4
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_dredger.json" # 484 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep40+2p_dredger.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep40+2p_dredger"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=40.ckpt"
# sample_num=8
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# ************************************* CAM v2 60E *************************************

# *************** Generate for Fishing Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=4
# class_index=2
# prompts_file="./gen/fusrs_v2_cam_v2/2p_fishing.json" # 814 samples
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep60+2p_fishing.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep60+2p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=60.ckpt"
# sample_num=3
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Tanker Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=4
# class_index=3
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_tanker.json" # 630 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep60+2p_tanker.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep60+2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=60.ckpt"
# sample_num=6
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Dredger Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=5
# class_index=4
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_dredger.json" # 484 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep60+2p_dredger.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep60+2p_dredger"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=60.ckpt"
# sample_num=8
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index


# ************************************* CAM v2 80E *************************************

# *************** Generate for Fishing Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=5
# class_index=2
# prompts_file="./gen/fusrs_v2_cam_v2/2p_fishing.json" # 814 samples
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep80+2p_fishing.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep80+2p_fishing"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=80.ckpt"
# sample_num=3
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Tanker Classes (CAM_v2_lc_newprompts) ***************
# export CUDA_VISIBLE_DEVICES=6
# class_index=3
# input_dir="./training/fusrs_v2_256_cam_v2/source"
# prompts_file="./gen/fusrs_v2_cam_v2/2p_tanker.json" # 630 samples
# out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep80+2p_tanker.txt"
# output_dir="./gen/fusrs_v2_cam_v2/ep80+2p_tanker"
# checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=80.ckpt"
# sample_num=6
# python ./inf_dir.py \
#     --prompts_file $prompts_file \
#     --out_ann_file $out_ann_file \
#     --input_dir $input_dir \
#     --output_dir $output_dir \
#     --sample_num $sample_num \
#     --checkpoint_path $checkpoint_path \
#     --cls_ind $class_index

# # *************** Generate for Dredger Classes (CAM_v2_lc_newprompts) ***************
export CUDA_VISIBLE_DEVICES=6
class_index=4
input_dir="./training/fusrs_v2_256_cam_v2/source"
prompts_file="./gen/fusrs_v2_cam_v2/2p_dredger.json" # 484 samples
out_ann_file="./gen/fusrs_v2_cam_v2/ann_ep80+2p_dredger.txt"
output_dir="./gen/fusrs_v2_cam_v2/ep80+2p_dredger"
checkpoint_path="./checkpoints/fusrs_v2_lc/fusrs_epoch=80.ckpt"
sample_num=8
python ./inf_dir.py \
    --prompts_file $prompts_file \
    --out_ann_file $out_ann_file \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --sample_num $sample_num \
    --checkpoint_path $checkpoint_path \
    --cls_ind $class_index