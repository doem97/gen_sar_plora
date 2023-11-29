# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from collections import deque

# import warnings
# from mmcv.fileio import FileClient


class EvalHook(BaseEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, best_k: Optional[int] = None, **kwargs):
        super(EvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None
        if best_k is not None and best_k <= 1:
            raise ValueError(
                f"The value of best_k should be int greater than 1, but got {best_k}."
            )
        self.best_k = best_k
        # self.best_ckpt_path = None

    # def before_run(self, runner):
    #     if not self.out_dir:
    #         self.out_dir = runner.work_dir

    #     self.file_client = FileClient.infer_client(self.file_client_args, self.out_dir)

    #     # if `self.out_dir` is not equal to `runner.work_dir`, it means that
    #     # `self.out_dir` is set so the final `self.out_dir` is the
    #     # concatenation of `self.out_dir` and the last level directory of
    #     # `runner.work_dir`
    #     if self.out_dir != runner.work_dir:
    #         basename = osp.basename(runner.work_dir.rstrip(osp.sep))
    #         self.out_dir = self.file_client.join_path(self.out_dir, basename)
    #         runner.logger.info(
    #             f"The best checkpoint will be saved to {self.out_dir} by "
    #             f"{self.file_client.name}"
    #         )

    #     if self.save_best is not None:
    #         if runner.meta is None:
    #             warnings.warn("runner.meta is None. Creating an empty one.")
    #             runner.meta = dict()
    #         runner.meta.setdefault("hook_msgs", dict())
    #         self.best_ckpt_path = runner.meta["hook_msgs"].get("best_ckpt", None)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

    def _save_ckpt(self, runner, key_score):
        """Save the best checkpoint. @doem1997, 230429

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f"epoch_{runner.epoch + 1}"
            cur_type, cur_time = "epoch", runner.epoch + 1
        else:
            current = f"iter_{runner.iter + 1}"
            cur_type, cur_time = "iter", runner.iter + 1
        if self.best_k is None:
            best_score = runner.meta["hook_msgs"].get(
                "best_score", self.init_value_map[self.rule]
            )
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta["hook_msgs"]["best_score"] = best_score

                if self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                    self.file_client.remove(self.best_ckpt_path)
                    runner.logger.info(
                        f"The previous best checkpoint {self.best_ckpt_path} was "
                        "removed"
                    )

                best_ckpt_name = f"best_{self.key_indicator}_{current}.pth"
                self.best_ckpt_path = self.file_client.join_path(
                    self.out_dir, best_ckpt_name
                )
                runner.meta["hook_msgs"]["best_ckpt"] = self.best_ckpt_path

                runner.save_checkpoint(
                    self.out_dir, filename_tmpl=best_ckpt_name, create_symlink=False
                )
                runner.logger.info(f"Now best checkpoint is saved as {best_ckpt_name}.")
                runner.logger.info(
                    f"Best {self.key_indicator} is {best_score:0.4f} "
                    f"at {cur_time} {cur_type}."
                )
        else:
            best_scores = runner.meta["hook_msgs"].get(
                "best_scores",
                deque(
                    [self.init_value_map[self.rule]] * self.best_k, maxlen=self.best_k
                ),
            )
            best_ckpts = runner.meta["hook_msgs"].get(
                "best_ckpts", deque([""] * self.best_k, maxlen=self.best_k)
            )

            if self.compare_func(key_score, best_scores[-1]):
                # Find the correct position for the new score
                for idx, score in enumerate(best_scores):
                    if self.compare_func(key_score, score):
                        break

                best_ckpt_name = f"top{self.best_k}_{self.key_indicator}_{current}.pth"
                best_ckpt_path = self.file_client.join_path(
                    self.out_dir, best_ckpt_name
                )

                # Remove the checkpoint that is no longer in the top k
                if len(best_scores) == self.best_k and self.file_client.isfile(
                    best_ckpts[-1]
                ):
                    self.file_client.remove(best_ckpts[-1])
                    runner.logger.info(
                        f"The previous {self.best_k}-th best checkpoint {best_ckpts[-1]} was removed"
                    )

                # Remove the last element if deque is already at its maximum size
                if len(best_scores) == self.best_k:
                    best_scores.pop()
                    best_ckpts.pop()

                # Insert the new score and checkpoint at the correct position
                best_scores.insert(idx, key_score)
                best_ckpts.insert(idx, best_ckpt_path)

                runner.meta["hook_msgs"]["best_scores"] = best_scores
                runner.meta["hook_msgs"]["best_ckpts"] = best_ckpts

                runner.save_checkpoint(
                    self.out_dir, filename_tmpl=best_ckpt_name, create_symlink=False
                )
                runner.logger.info(
                    f"Now best-{self.best_k} checkpoint is saved as {best_ckpt_name}."
                )
                runner.logger.info(
                    f"Best {self.best_k} {self.key_indicator}s are "
                    + ", ".join(f"{score:.4f}" for score in best_scores)
                    + f" at {cur_time} {cur_type}."
                )


class DistEvalHook(BaseDistEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, best_k: Optional[int] = None, **kwargs):
        super(DistEvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None
        if best_k is not None and best_k <= 1:
            raise ValueError(
                f"The value of best_k should be int greater than 1, but got {best_k}."
            )
        self.best_k = best_k

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        results = self.test_fn(
            runner.model, self.dataloader, tmpdir=tmpdir, gpu_collect=self.gpu_collect
        )
        self.latest_results = results
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

    def _save_ckpt(self, runner, key_score):
        """Save the best checkpoint. @doem1997, 230429

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f"epoch_{runner.epoch + 1}"
            cur_type, cur_time = "epoch", runner.epoch + 1
        else:
            current = f"iter_{runner.iter + 1}"
            cur_type, cur_time = "iter", runner.iter + 1
        if self.best_k is None:
            best_score = runner.meta["hook_msgs"].get(
                "best_score", self.init_value_map[self.rule]
            )
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta["hook_msgs"]["best_score"] = best_score

                if self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                    self.file_client.remove(self.best_ckpt_path)
                    runner.logger.info(
                        f"The previous best checkpoint {self.best_ckpt_path} was "
                        "removed"
                    )

                best_ckpt_name = f"best_{self.key_indicator}_{current}.pth"
                self.best_ckpt_path = self.file_client.join_path(
                    self.out_dir, best_ckpt_name
                )
                runner.meta["hook_msgs"]["best_ckpt"] = self.best_ckpt_path

                runner.save_checkpoint(
                    self.out_dir, filename_tmpl=best_ckpt_name, create_symlink=False
                )
                runner.logger.info(f"Now best checkpoint is saved as {best_ckpt_name}.")
                runner.logger.info(
                    f"Best {self.key_indicator} is {best_score:0.4f} "
                    f"at {cur_time} {cur_type}."
                )
        else:
            best_scores = runner.meta["hook_msgs"].get(
                "best_scores",
                deque(
                    [self.init_value_map[self.rule]] * self.best_k, maxlen=self.best_k
                ),
            )
            best_ckpts = runner.meta["hook_msgs"].get(
                "best_ckpts", deque([""] * self.best_k, maxlen=self.best_k)
            )

            if self.compare_func(key_score, best_scores[-1]):
                # Find the correct position for the new score
                for idx, score in enumerate(best_scores):
                    if self.compare_func(key_score, score):
                        break

                best_ckpt_name = f"top{self.best_k}_{self.key_indicator}_{current}.pth"
                best_ckpt_path = self.file_client.join_path(
                    self.out_dir, best_ckpt_name
                )

                # Remove the checkpoint that is no longer in the top k
                if len(best_scores) == self.best_k and self.file_client.isfile(
                    best_ckpts[-1]
                ):
                    self.file_client.remove(best_ckpts[-1])
                    runner.logger.info(
                        f"The previous {self.best_k}-th best checkpoint {best_ckpts[-1]} was removed"
                    )

                # Remove the last element if deque is already at its maximum size
                if len(best_scores) == self.best_k:
                    best_scores.pop()
                    best_ckpts.pop()

                # Insert the new score and checkpoint at the correct position
                best_scores.insert(idx, key_score)
                best_ckpts.insert(idx, best_ckpt_path)

                runner.meta["hook_msgs"]["best_scores"] = best_scores
                runner.meta["hook_msgs"]["best_ckpts"] = best_ckpts

                runner.save_checkpoint(
                    self.out_dir, filename_tmpl=best_ckpt_name, create_symlink=False
                )
                runner.logger.info(
                    f"Now best-{self.best_k} checkpoint is saved as {best_ckpt_name}."
                )
                runner.logger.info(
                    f"Best {self.best_k} {self.key_indicator}s are "
                    + ", ".join(f"{score:.4f}" for score in best_scores)
                    + f" at {cur_time} {cur_type}."
                )
