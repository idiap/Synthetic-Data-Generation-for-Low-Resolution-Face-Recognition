#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_s_gamma_05_noup_32_esrgan.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()
config.margin_list = (1.0, 0.0, 0.4)

# Network: a 32×32-input variant that the user will register in
# backbones/__init__.py. Keep the string in sync with that registration.
config.network = "edgeface_s_gamma_05_vlr_32"

# Native training resolution — threaded through dataset.py / DALI so the
# pipeline does not silently upsample to 112.
config.input_size = 32

config.resume = True
config.output = 'edgeface_s_gamma_05_noup_32_esrgan/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 2000

config.dali = True
# DALI augmentation is on top of the rec-level Real-ESRGAN degradation.
# Toggle off if double-augmentation hurts; the rec already has heavy
# stochastic degradation baked in.
config.dali_aug = False

config.save_all_states = True
config.num_workers = 6

# Source rec produced by:
#   python scripts/preprocess_rec.py data/passthrough_webface4m \
#       --method esrgan --output-size 32
config.rec = data_path("processed_noup_32_esrgan_passthrough_webface4m")
config.num_classes = 205990
config.num_image = 4235242
config.num_epoch = 100
config.warmup_epoch = 2

# Verification bins must also be at 32×32. Generate with:
#   python scripts/generate_lr_bin.py --bin-dir data/webface4m \
#       --output-size 32 --downsample cubic --mode both
#   python scripts/generate_lr_bin.py --bin-dir data/webface4m \
#       --method esrgan --output-size 32 --mode both
config.val_targets = [
    'lfw_noup_32_esrgan_lr2lr',
    'cfp_fp_noup_32_esrgan_lr2lr',
    'agedb_30_noup_32_esrgan_lr2lr'
]
