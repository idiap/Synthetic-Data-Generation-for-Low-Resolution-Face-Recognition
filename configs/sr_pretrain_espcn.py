#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: sr_pretrain_espcn.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
"""Stage-1 SR pretraining for the ESPCN-lite upsampler.

Run with:
    torchrun --nproc_per_node=N train_sr_pretrain.py \
        --config configs/sr_pretrain_espcn.py
"""
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()

# --------------------------------------------------------------------------- #
# Upsampler                                                                   #
# --------------------------------------------------------------------------- #
config.upsampler = "espcn"          # see _build_upsampler in train_sr_pretrain.py

# --------------------------------------------------------------------------- #
# Frozen perceptual extractor (EdgeFace-base, WebFace12M-pretrained)          #
# --------------------------------------------------------------------------- #
config.backbone_checkpoint = "checkpoints/edgeface_base.pt"

# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
# HR ground truth (112x112 originals) — adjust to your local path.
config.rec_hr = data_path("HR_for_PDT_passthrough_webface4m")
# LR input: WebFace4M with Real-ESRGAN degradations stored at 32x32.
# Folder produced by `scripts/preprocess_rec.py --method esrgan --output-size 32`.
config.rec_lr = data_path("LR_for_PDT_processed_noup_32_esrgan_webface4m")

# Optional: explicit path to a pre-built labels.npy that pairs HR/LR by index.
# When None, the LR rec's labels.npy is built (or loaded) and reused for HR.
config.labels_cache = None

config.batch_size  = 512
config.num_workers = 4

# --------------------------------------------------------------------------- #
# Optimisation                                                                #
# --------------------------------------------------------------------------- #
config.lr            = 1e-3
config.weight_decay  = 1e-4
config.num_epoch     = 100
config.warmup_epoch  = 1
config.seed          = 2048

# Identity-aware perceptual loss weight (cosine distance on 512-d embeddings).
config.lambda_id     = 0.1

# --------------------------------------------------------------------------- #
# Logging / checkpoints                                                       #
# --------------------------------------------------------------------------- #
config.output    = "sr_pretrain_espcn/"
config.frequent  = 100

# Resume from <output>/checkpoint_gpu_<rank>.pt if present.
# Can also be forced with --resume-override on the CLI.
config.resume    = False
