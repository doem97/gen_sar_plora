# ImageNet-1K pre-train loaded

_base_ = [
    "./res50_bs16_clb_lr-6_200e.py",
]

model = dict(
    backbone=dict(
        # frozen_stages=2,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
            prefix="backbone",
        ),
    ),
)
