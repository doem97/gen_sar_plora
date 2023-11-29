# ImageNet pre-train loaded (resampling applied)
_base_ = [
    "../../_base_/models/resnet101.py",
    "../_usr_/datasets/fusar15_bs16_clb.py",
    "../_usr_/schedules/fusar_bs16.py",
    "../_usr_/default_runtime.py",
]

model = dict(
    backbone=dict(
        # frozen_stages=2,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_batch256_imagenet_20200708-753f3608.pth",
            prefix="backbone",
        ),
    ),
    head=dict(num_classes=15),
)
