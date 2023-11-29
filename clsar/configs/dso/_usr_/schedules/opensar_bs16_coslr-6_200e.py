# optimizer
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy="step", step=[15, 22])
lr_config = dict(
    policy="CosineAnnealing",
    by_epoch=False,
    min_lr_ratio=1e-6,
    warmup="linear",
    warmup_ratio=1e-6,
    warmup_iters=2000,
    warmup_by_epoch=False,
)


runner = dict(type="EpochBasedRunner", max_epochs=200)
