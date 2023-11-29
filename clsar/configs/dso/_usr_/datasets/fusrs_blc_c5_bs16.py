# FUSRS balance version
dataset_type = "FUSRS_BLC"
img_norm_cfg = dict(
    mean=[9.885581987404349, 9.885581987404349, 9.885581987404349],
    std=[19.42731796551787, 19.42731796551787, 19.42731796551787],
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
        data_prefix="data/fusrs_blc/images/",
        ann_file="data/fusrs_blc/meta/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="data/fusrs_blc/images/",
        ann_file="data/fusrs_blc/meta/val.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="data/fusrs_blc/images/",
        ann_file="data/fusrs_blc/meta/test.txt",
        pipeline=test_pipeline,
    ),
)
# evaluation = dict(interval=1, metric="accuracy")
evaluation = dict(
    interval=1,
    start=80,
    metric="f1_score",
    save_best="f1_score",
    rule="greater",
)
