#
# SPDX-FileCopyrightText: Copyright (c) 2022 Jiankang Deng and Jia Guo
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: utils_callbacks.py
# Modified from InsightFace (https://github.com/deepinsight/insightface, MIT);
# see LICENSES/MIT.txt. Changes: device fixes and PDT verification routing.
#
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112), wandb_logger=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )
            if self.wandb_logger:
                import wandb
                self.wandb_logger.log({
                    f'Acc/val-Acc1 {self.ver_name_list[i]}': acc1,
                    f'Acc/val-Acc2 {self.ver_name_list[i]}': acc2,
                    # f'Acc/val-std1 {self.ver_name_list[i]}': std1,
                    # f'Acc/val-std2 {self.ver_name_list[i]}': acc2,
                })

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class PDTCallBackVerification:
    """Verification callback for PDT models.

    For each validation target the caller can specify which inference path
    to use for the *first* and *second* image of every verification pair:

    * ``'hr'`` — pass directly through the frozen backbone.
    * ``'lr'`` — pass through the PDT translator first, then the backbone.

    Parameters
    ----------
    val_targets : list[str]
        Names of the ``.bin`` files (without extension) to evaluate.
    rec_prefix : str
        Directory that contains those ``.bin`` files.
    pair_modes : dict[str, tuple[str, str]], optional
        Mapping from target name → ``(mode_img1, mode_img2)``.
        Any target not listed defaults to ``('lr', 'lr')``.
        Example::

            {
                'lfw':                  ('hr', 'hr'),   # plain HR verification
                'lfw_28_lr2lr':         ('lr', 'lr'),   # both images LR
                'lfw_28_area_area_hr2lr': ('hr', 'lr'), # first HR, second LR
            }
    summary_writer : SummaryWriter, optional
    image_size : tuple[int, int]
    wandb_logger : optional
    """

    def __init__(
        self,
        val_targets: List[str],
        rec_prefix: str,
        pair_modes: Optional[Dict[str, Tuple[str, str]]] = None,
        summary_writer=None,
        image_size: Tuple[int, int] = (112, 112),
        wandb_logger=None,
    ):
        self.rank: int = distributed.get_rank()
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.pair_modes: Dict[str, Tuple[str, str]] = pair_modes or {}
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

        if self.rank == 0:
            self.init_dataset(val_targets, rec_prefix, image_size)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                self.ver_list.append(verification.load_bin(path, image_size))
                self.ver_name_list.append(name)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        # Unwrap DDP to get the PDT_wrapper module
        pdt = backbone.module if hasattr(backbone, "module") else backbone

        for i, (data_set, name) in enumerate(
            zip(self.ver_list, self.ver_name_list)
        ):
            mode1, mode2 = self.pair_modes.get(name, ("lr", "lr"))

            acc1, std1, acc2, std2, xnorm, _ = verification.test_pdt(
                data_set, pdt, batch_size=10,
                mode1=mode1, mode2=mode2, nfolds=10,
            )

            logging.info(
                "[%s][%d] mode=(%s,%s)  XNorm: %.4f"
                % (name, global_step, mode1, mode2, xnorm)
            )
            logging.info(
                "[%s][%d] Accuracy-Flip: %1.5f+-%1.5f"
                % (name, global_step, acc2, std2)
            )

            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    tag=name, scalar_value=acc2, global_step=global_step
                )
            if self.wandb_logger:
                import wandb
                self.wandb_logger.log({
                    f"Acc/val-Acc1 {name}": acc1,
                    f"Acc/val-Acc2 {name}": acc2,
                })

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                "[%s][%d] Accuracy-Highest: %1.5f"
                % (name, global_step, self.highest_acc_list[i])
            )

    def __call__(self, num_update: int, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
