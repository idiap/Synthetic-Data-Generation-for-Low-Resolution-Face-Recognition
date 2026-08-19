#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: sr_pretrain_resrgan.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
"""Stage-1 SR pretraining for the RRDB-based (Real-ESRGAN-style) upsampler.

Run with:
    torchrun --nproc_per_node=N train_sr_pretrain.py \
        --config configs/sr_pretrain_resrgan.py
"""
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()

# --------------------------------------------------------------------------- #
# Upsampler                                                                   #
# --------------------------------------------------------------------------- #
config.upsampler = "resrgan"        # RRDBUpsampler (~3-4M params)

# --------------------------------------------------------------------------- #
# Frozen perceptual extractor (EdgeFace-base, WebFace12M-pretrained)          #
# --------------------------------------------------------------------------- #
config.backbone_checkpoint = "checkpoints/edgeface_base.pt"

# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
config.rec_hr = data_path("HR_for_PDT_passthrough_webface4m_resrgan")
config.rec_lr = data_path("LR_for_PDT_processed_noup_32_esrgan_webface4m_resrgan")
config.labels_cache = None

# Smaller batch — RRDB is heavier than ESPCN.
config.batch_size  = 96
config.num_workers = 4

# --------------------------------------------------------------------------- #
# Optimisation                                                                #
# --------------------------------------------------------------------------- #
config.lr            = 1e-4
config.weight_decay  = 1e-4
config.num_epoch     = 10
config.warmup_epoch  = 1
config.seed          = 2048

config.lambda_id     = 0.1

# --------------------------------------------------------------------------- #
# Logging / checkpoints                                                       #
# --------------------------------------------------------------------------- #
config.output    = "sr_pretrain_resrgan/"
config.frequent  = 100

# Resume from <output>/checkpoint_gpu_<rank>.pt if present.
# Can also be forced with --resume-override on the CLI.
config.resume    = True
