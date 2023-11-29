_base_ = ["./res50_1x128_lr1e-1+200e+im21k_fusrs_v2+ctrlcam_ep89.py"]

data = dict(
    train=dict(
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/run0_v1/ctrl+cam/train_aug20p_ep80.txt",
    )
)
