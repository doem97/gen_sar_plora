# ResNet101 on FUSAR dataset
_base_ = [
    "../../_base_/models/resnet101.py",
    "../_usr_/datasets/fusar15_bs16.py",
    "../_usr_/schedules/fusar_bs16.py",
    "../_usr_/default_runtime.py",
]

model = dict(
    head=dict(num_classes=15),
)
