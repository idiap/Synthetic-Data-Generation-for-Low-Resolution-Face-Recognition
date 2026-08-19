#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: edgeface_s_gamma_05_PDT_28_cubic_area.py
# Derived from configs/edgeface_s_gamma_05.py of the EdgeFace release
# (https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).
#
from easydict import EasyDict as edict
from configs.paths import data_path

config = edict()

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
config.network        = "edgeface_s_gamma_05_PDT"
config.embedding_size = 512

# Pre-trained HR backbone checkpoint to freeze inside PDT_wrapper.
# Format follows train_v2.py checkpoints: {"state_dict_backbone": ..., ...}
config.backbone_checkpoint = "edgeface_s_gamma_05/checkpoint_gpu_0.pt"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# LR dataset: images downsampled to 28×28 (area) then upsampled back to 112×112
config.rec    = data_path("processed_downsample_28_cubic_area_webface4m")
# HR dataset: original 112×112 images (same identities, same label space)
config.rec_hr = data_path("HR_for_processed_downsample_28_cubic_area_webface4m")

config.num_classes = 205990
config.num_image   = 4235242

config.num_workers = 4

# --------------------------------------------------------------------------- #
# PK batch sampling
# --------------------------------------------------------------------------- #
# Effective batch size = pk_p * pk_k (set automatically in train_PDT.py).
# Both LR and HR loaders use the same seed so they visit the same P identities
# at each step → guaranteed positive cross-modal pairs every batch.
# Steps per epoch = num_classes // (pk_p * world_size), not num_images // batch.
config.pk_p = 32    # identities per batch
config.pk_k = 8     # images per identity  →  batch = 256, pairs = 32²×64 pos + many neg

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
config.optimizer     = "adamw"
config.lr            = 1e-3
config.weight_decay  = 0.05
config.fp16          = False
config.gradient_acc  = 1
config.num_epoch     = 20
config.warmup_epoch  = 1
config.seed          = 2048

config.contrastive_margin = 2.0  # margin for negative pairs (L2-normalised embeddings ∈ [0,2])

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# Routing for each verification pair: ('hr'|'lr', 'hr'|'lr') → first/second image.
#   'hr' = frozen backbone only.
#   'lr' = PDT translator → frozen backbone.
# Missing entries default to ('lr', 'lr') in PDTCallBackVerification.
config.val_pair_modes = {
    'lfw':           ('hr', 'hr'),   # original HR pairs → backbone only
    'lfw_28_lr2lr':  ('lr', 'lr'),   # both images LR-degraded
    'lfw_28_hr2lr_interArea':  ('hr', 'lr'),
    'lfw_7_lr2lr':   ('lr', 'lr'),
    'lfw_7_hr2lr_interArea':   ('hr', 'lr'),
}
config.val_targets = list(config.val_pair_modes.keys())
config.verbose     = 500
config.frequent    = 20

# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
config.output         = "edgeface_s_gamma_05_PDT_28_cubic_area/"
config.save_all_states = True
config.resume          = False

# --------------------------------------------------------------------------- #
# Misc (WandB disabled by default)
# --------------------------------------------------------------------------- #
config.using_wandb = False
# Kept for compatibility with shared utility code; not used by train_PDT.py
config.sample_rate = 1.0
config.margin_list = (1.0, 0.0, 0.4)
