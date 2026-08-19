#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anjith George <anjith.george@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_s_gamma_05.py
# From the EdgeFace release (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface);
# adapted: WebFace4M training data resolved via configs.paths; validation targets.
#
from easydict import EasyDict as edict
from configs.paths import data_path

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = False
config.output = 'edgeface_s_gamma_05/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 1000
config.dali = True 
config.dali_aug = True

config.save_all_states = True

config.num_workers = 6

#config.rec = "data/webface12m"
#config.num_classes = 617970
#config.num_image = 12720066

config.rec = data_path("webface4m")
config.num_classes = 205990
config.num_image = 4235242
config.num_epoch = 100
config.warmup_epoch = 2
#config.val_targets = ['lfw', 'lfw_56', 'lfw_28', 'lfw_14','lfw_7']
config.val_targets=['lfw']
