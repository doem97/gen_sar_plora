_base_ = [
    "../../_base_/models/resnet50.py",
    "../_usr_/datasets/opensar_c8_bs64.py",
    "../_usr_/schedules/sgd_calr_100ep.py",
    "../_usr_/default_runtime.py",
]

model = dict(
    head=dict(num_classes=8),
)
