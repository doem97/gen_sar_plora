# ImageNet-21K pre-train loaded

_base_ = [
    "./res50_bs16_clb_lr-6_200e.py",
]

model = dict(
    backbone=dict(
        # frozen_stages=2,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth",
            prefix="backbone",
        ),
    ),
)
