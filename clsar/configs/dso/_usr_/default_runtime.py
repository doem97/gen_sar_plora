# checkpoint saving
checkpoint_config = dict(interval=1)

# log_config: the hook.interval will override the log_config.interval
# To log W&B training loss, set the interval to 10
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMClsWandbHook",
            init_kwargs={"entity": "doem97", "project": "DSO-Classification"},
            interval=10,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=0,
        ),
    ],
)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
