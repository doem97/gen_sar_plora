_base_ = ["./res50_1x128_lr1e-1+300e_fusrs_v2.py"]

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
