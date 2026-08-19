#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anjith George <anjith.george@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_base.py
# From the EdgeFace release (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface);
# adapted: EdgeFace-Base network with the WebFace12M recipe; used frozen here.
#
"""EdgeFace-base on WebFace12M.

Follows the official EdgeFace training recipe, with the same hyperparameters as
configs/edgeface_s_gamma_05.py and the WebFace12M identity and image counts.

This paper does not retrain EdgeFace-base: it is used **frozen**, loaded from the
released checkpoint. The config is what lets the evaluation scripts and the SR /
PDT front-ends build the backbone:

    python eval_tinyface_identification.py \\
        --config configs/edgeface_base.py \\
        --checkpoint checkpoints/edgeface_base.pt \\
        ...

configs/sr_pretrain_*.py and configs/edgeface_base_PDT_up32_*.py load and freeze
the same checkpoint. Set config.rec via LRFR_DATA_ROOT if you do want to train
from scratch.
"""
from easydict import EasyDict as edict
from configs.paths import data_path

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_base"
config.resume = False
config.output = 'edgeface_base/'
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

config.num_workers = 6

config.rec = data_path("webface12m")
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
config.warmup_epoch = 2
config.val_targets = []
