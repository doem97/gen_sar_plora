_base_ = ["../res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py"]

data = dict(
    train=dict(
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/ctrl+cam/train+484dredger_ctrl+cam_aug.txt",
    )
)
