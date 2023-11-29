_base_ = ["../res50_fusrs_v2_pretrain/res50_1x128_lr1e-1+200e+im21k_fusrs_v2.py"]

data = dict(
    train=dict(
        data_prefix="data/fusrs_v2/",
        ann_file="data/fusrs_v2/meta/toy.txt",
    )
)

evaluation = dict(
    interval=1,
    start=None,
    metric="f1_score",
    save_best="f1_score",
    best_k=10,
    rule="greater",
)

# log_config: the hook.interval will override the log_config.interval
# To log W&B training loss, set the interval to 10
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
