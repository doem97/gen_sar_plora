# dataset settings
dataset_type = "OPENSAR"
img_norm_cfg = dict(
    mean=[8.373387, 8.373387, 8.373387],
    std=[13.8513904, 13.8513904, 13.8513904],
    to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", size=64),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(64, -1)),
    dict(type="CenterCrop", crop_size=64),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix="data/opensar/",
        ann_file="data/opensar/meta/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="data/opensar/",
        ann_file="data/opensar/meta/val.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="data/opensar/",
        ann_file="data/opensar/meta/test.txt",
        pipeline=test_pipeline,
    ),
)
# evaluation = dict(interval=1, metric="accuracy")
evaluation = dict(interval=1, metric="f1_score")
