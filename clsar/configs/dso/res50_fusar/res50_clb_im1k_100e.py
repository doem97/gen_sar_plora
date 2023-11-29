# ImageNet pre-train loaded

_base_ = [
    "../../_base_/models/resnet50.py",
    "../_usr_/datasets/fusar15_bs16_clb.py",
    "../_usr_/schedules/fusar_bs16.py",
    "../_usr_/default_runtime.py",
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
    head=dict(num_classes=15),
)

runner = dict(type="EpochBasedRunner", max_epochs=100)
