#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_s_gamma_05_lr_28_cubic_area_kd_HR.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = True
config.output = 'edgeface_s_gamma_05_lr_28_cubic_area_kd_HR/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 1e-5
config.verbose = 2000
config.dali = True
config.dali_aug = True

config.save_all_states = True
config.num_workers = 6

config.rec    = data_path("processed_downsample_28_cubic_area_webface4m_kd")  # LR — student input
config.rec_hr = data_path("HR_for_processed_downsample_28_cubic_area_webface4m_kd")                          # HR — teacher input
config.num_classes = 205990
config.num_image = 4235242
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = ['lfw', 'lfw_14']

# ---- Knowledge distillation ----
# Teacher: the frozen HR-trained EdgeFace-S baseline. {rank} is replaced at
# runtime so each process loads its own partial-FC shard.
#
# Student: warm-started from the LR-trained backbone of the same setting
# (paper: "an LR student ... initialized from the LR backbone").
# config.resume = True loads checkpoint_gpu_{rank}.pt from config.output —
# slurm/train_edgeface_kd.run copies the LR run's checkpoints there on the
# first launch. The epoch counter continues from the LR run (100 epochs),
# so num_epoch = 200 gives ~100 distillation epochs on top of it.
config.teacher_checkpoint = "edgeface_s_gamma_05/checkpoint_gpu_{rank}.pt"

# Loss weights: total = loss_cls + kd_lambda_emb * loss_emb + kd_lambda_weight * loss_weight
config.kd_lambda_emb    = 1.0
config.kd_lambda_weight = 1.0
