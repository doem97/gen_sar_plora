_base_ = [
    "./res50_1x128_lr1e-1+200e_fusrs_blc.py"
]

# configs/dso/res50_fusrs_blc/res50_1x128_lr1e-1+200e_fusrs_blc.py

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