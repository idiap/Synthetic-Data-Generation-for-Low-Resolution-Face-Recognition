#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_base_PDT_up32_resrgan.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
"""Stage-2 PDT contrastive training with the RRDB-based upsampler in front."""
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()

# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
config.network        = "edgeface_base_PDT_up32_resrgan"
config.embedding_size = 512

config.backbone_checkpoint  = "checkpoints/edgeface_base.pt"
config.upsampler_checkpoint = "sr_pretrain_resrgan/sr_pretrained.pt"

# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
config.rec    = data_path("LR_for_PDT_processed_noup_32_esrgan_webface4m_resrgan")
config.rec_hr = data_path("HR_for_PDT_passthrough_webface4m_resrgan")

config.num_classes = 205990
config.num_image   = 4235242

config.num_workers = 4

# --------------------------------------------------------------------------- #
# PK batch sampling                                                           #
# --------------------------------------------------------------------------- #
config.pk_p = 16
config.pk_k = 4

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
config.optimizer     = "adamw"
config.lr            = 1e-4
config.weight_decay  = 0.05
config.fp16          = False
config.gradient_acc  = 1
config.num_epoch     = 20
config.warmup_epoch  = 1
config.seed          = 2048

config.contrastive_margin = 2.0

# --------------------------------------------------------------------------- #
# Evaluation routing                                                          #
# --------------------------------------------------------------------------- #
config.val_pair_modes = {
    'lfw_noup_32_esrgan_lr2lr': ('lr', 'lr'),
    'cfp_fp_noup_32_esrgan_lr2lr': ('lr', 'lr'),
    'agedb_30_noup_32_esrgan_lr2lr': ('lr', 'lr'),
}
config.val_targets = list(config.val_pair_modes.keys())
config.verbose     = 1500
config.frequent    = 20

# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #
config.output         = "edgeface_base_PDT_up32_resrgan/"
config.save_all_states = True
config.resume          = False

# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #
config.using_wandb = False
config.sample_rate = 1.0
config.margin_list = (1.0, 0.0, 0.4)
