_base_ = ["./res50_1x128_lr1e-1+100e+im21k_fusrs_v2+resample.py"]

data = dict(
    train=dict(
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/ctrl+cam/train+alldup_aug.txt",
    )
)
