_base_ = ["./res50_1x128_lr1e-1+300e+im21k_fusrs_v2+resample.py"]

data = dict(
    train=dict(
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/ctrl+cam/train+all4k_aug.txt",
    )
)
