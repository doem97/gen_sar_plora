# FUSRS balance version
dataset_type = "FUSRS_V2"
img_norm_cfg = dict(
    mean=[11.20954390854399, 11.20954390854399, 11.20954390854399],
    std=[20.241805767392393, 20.241805767392393, 20.241805767392393],
    to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", size=224),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, -1)),
    # dict(type="Resize", size=(256, -1)),  # INFO: this is ImageNet test set, deprecated
    dict(type="CenterCrop", crop_size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix="data/fusrs_v2/",
        # ann_file="data/fusrs_v2/meta/ctrl+cam/train+1000dredger_ctrl+cam_aug.txt",
        ann_file="data/fusrs_v2/meta/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/val.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/test.txt",
        pipeline=test_pipeline,
    ),
)
# evaluation = dict(interval=1, metric="accuracy")
evaluation = dict(
    interval=1,
    start=80,
    metric="f1_score",
    save_best="f1_score",
    best_k=10,
    rule="greater",
)
