#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_s_gamma_05_lr_56_cubic_area_kd.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = False
config.output = 'edgeface_s_gamma_05_lr_56_cubic_area_kd/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 2000
config.dali = True
config.dali_aug = True

config.save_all_states = True
config.num_workers = 6

config.rec    = data_path("processed_downsample_56_cubic_area_webface4m_kd")  # LR — student input
config.rec_hr = data_path("HR_for_processed_downsample_56_cubic_area_webface4m_kd")                          # HR — teacher input
config.num_classes = 205990
config.num_image = 4235242
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = ['lfw', 'lfw_14']

# ---- Knowledge distillation ----
# Teacher: the frozen HR-trained EdgeFace-S baseline. {rank} is replaced at
# runtime so each process loads its own partial-FC shard.
#
# NOTE: this variant trains the student from random init (resume = False).
# The paper's KD results warm-start the student from the LR backbone — use the
# *_kd_HR config and slurm/train_edgeface_kd.run for that.
config.teacher_checkpoint = "edgeface_s_gamma_05/checkpoint_gpu_{rank}.pt"

# Loss weights: total = loss_cls + kd_lambda_emb * loss_emb + kd_lambda_weight * loss_weight
config.kd_lambda_emb    = 1.0
config.kd_lambda_weight = 1.0
