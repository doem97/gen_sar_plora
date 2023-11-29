_base_ = [
    "../../_base_/models/resnet50.py",  # model settings
    "../_usr_/datasets/fusrs_v2_c5_bs16.py",  # dataset settings
    "../_usr_/schedules/fusrs_v2_bs16.py",  # optimizer, optimizer_config, lr_config
    "../_usr_/default_runtime.py",  # log_config, dist_params, log_level, load_from, resume_from, workflow
]

model = dict(
    head=dict(num_classes=5),
)

runner = dict(max_epochs=200)
