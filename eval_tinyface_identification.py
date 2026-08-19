#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: eval_tinyface_identification.py
#
"""
eval_tinyface_identification.py

TinyFace face identification evaluation.

1. Loads pre-aligned face images for gallery-match, probe, and distractor sets.
2. Extracts embeddings with a trained backbone (same setup as evaluate_model.py).
3. Evaluates following the official TinyFace identification protocol:
   mAP and CMC@1/5/10/20 (test_face_identification.m).

Gallery-Distractor images on the Idiap grid are reorganised into
prefix-based subdirectories:

    <distractor-dir>/<filename[:20]>/<filename>

All other sets (Gallery_Match, Probe) keep a flat layout.

Usage:
    python eval_tinyface_identification.py \\
        --config  configs/edgeface_s_gamma_05.py \\
        --checkpoint edgeface_s_gamma_05/model.pt \\
        --mat-dir eval_tinyface/tinyface/Face_Identification_Evaluation \\
        --aligned-gallery-dir  /path/to/aligned/Gallery_Match \\
        --aligned-probe-dir    /path/to/aligned/Probe \\
        --aligned-distractor-dir /path/to/aligned/Gallery_Distractor
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import get_model
from utils.utils_config import get_config
from evaluate_model import load_model, setup_logging
from eval_tinyface.tinyface.Face_Identification_Evaluation.compute_AP import compute_AP


# --------------------------------------------------------------------------- #
#  Image loading & feature extraction                                         #
# --------------------------------------------------------------------------- #

def build_transform(image_size=(112, 112)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def extract_features(paths, backbone, transform, batch_size, device):
    """Extract L2-normalised embeddings for a list of image paths.

    Returns:
        features  : np.ndarray  shape (N, embedding_dim)
        valid_mask: list[bool]  True when the image was loaded successfully
    """
    features = []
    valid_mask = []

    batch_imgs, batch_idx = [], []
    flush = False

    def run_batch():
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = backbone(tensor)
            emb = torch.nn.functional.normalize(emb, dim=1)
        features.extend(emb.cpu().numpy())

    for i, p in enumerate(tqdm(paths, desc="  Extracting", leave=False)):
        flush = (i == len(paths) - 1)
        img = None
        if p is not None and p.exists():
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                pass

        if img is not None:
            batch_imgs.append(transform(img))
            batch_idx.append(i)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

        if len(batch_imgs) == batch_size or (flush and batch_imgs):
            run_batch()
            batch_imgs.clear()

    return np.array(features, dtype=np.float32), valid_mask


# --------------------------------------------------------------------------- #
#  Dataset loading helpers                                                    #
# --------------------------------------------------------------------------- #

def load_mat_list(mat_path, img_key, id_key):
    """Load image filenames and IDs from an old-style MATLAB .mat file."""
    data = sio.loadmat(str(mat_path))
    # cell array of strings: shape (N,1), each element is a 1-element array
    img_list = [str(data[img_key][i, 0].flat[0]) for i in range(data[img_key].shape[0])]
    ids = data[id_key].flatten().astype(np.int32)
    return img_list, ids


def collect_distractor_paths(distractor_dir):
    """Walk Gallery_Distractor with the Idiap grid subdirectory scheme.

    Files live at:  <distractor_dir>/<filename[:20]>/<filename>
    Returns a list of Path objects for all images found.
    """
    distractor_dir = Path(distractor_dir)
    paths = []
    for sub in sorted(distractor_dir.iterdir()):
        if sub.is_dir():
            for img in sorted(sub.iterdir()):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(img)
    return paths


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main(args):
    setup_logging()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.gpu_id)

    cfg = get_config(args.config)
    backbone = load_model(cfg, args.checkpoint, device=device)
    backbone.eval()
    # PDT-up32 networks expect 32x32 input and upsample inside the model;
    # baseline backbones expect 112x112. Choose with --input-size.
    transform = build_transform(image_size=(args.input_size, args.input_size))
    logging.info(f"Using input image size: {args.input_size}x{args.input_size}")

    mat_dir = Path(args.mat_dir)

    # ------------------------------------------------------------------ #
    # Gallery Match                                                        #
    # ------------------------------------------------------------------ #
    logging.info("Loading gallery-match image list …")
    gallery_names, gallery_ids = load_mat_list(
        mat_dir / "gallery_match_img_ID_pairs.mat", "gallery_set", "gallery_ids"
    )
    gallery_dir = Path(args.aligned_gallery_dir)
    gallery_paths = [gallery_dir / name for name in gallery_names]

    logging.info(f"Extracting gallery-match features ({len(gallery_paths)} images) …")
    gallery_feats, gallery_valid = extract_features(
        gallery_paths, backbone, transform, args.batch_size, device
    )
    # Build index: original gallery position → position in gallery_feats (None if FTA)
    gallery_feat_idx = []
    feat_pos = 0
    for v in gallery_valid:
        gallery_feat_idx.append(feat_pos if v else None)
        if v:
            feat_pos += 1
    gallery_ids_valid = gallery_ids[gallery_valid]
    n_gallery_fta = gallery_valid.count(False)
    if n_gallery_fta:
        logging.warning(f"  {n_gallery_fta} gallery-match images missing / failed to load "
                        f"— probes whose only match is missing will score AP=0, CMC=0")

    # ------------------------------------------------------------------ #
    # Probe                                                               #
    # ------------------------------------------------------------------ #
    logging.info("Loading probe image list …")
    probe_names, probe_ids = load_mat_list(
        mat_dir / "probe_img_ID_pairs.mat", "probe_set", "probe_ids"
    )
    probe_dir = Path(args.aligned_probe_dir)
    probe_paths = [probe_dir / name for name in probe_names]

    logging.info(f"Extracting probe features ({len(probe_paths)} images) …")
    probe_feats, probe_valid = extract_features(
        probe_paths, backbone, transform, args.batch_size, device
    )
    # Build index: original probe position → position in probe_feats (None if FTA)
    probe_feat_idx = []
    feat_pos = 0
    for v in probe_valid:
        probe_feat_idx.append(feat_pos if v else None)
        if v:
            feat_pos += 1
    n_probe_fta = probe_valid.count(False)
    if n_probe_fta:
        logging.warning(f"  {n_probe_fta} probe images missing / failed to load "
                        f"— these count as misidentifications (AP=0, CMC=0)")

    # ------------------------------------------------------------------ #
    # Distractor                                                          #
    # ------------------------------------------------------------------ #
    logging.info("Collecting distractor paths (prefix-subdirectory scheme) …")
    distractor_paths = collect_distractor_paths(args.aligned_distractor_dir)
    logging.info(f"Extracting distractor features ({len(distractor_paths)} images) …")
    distractor_feats, _ = extract_features(
        distractor_paths, backbone, transform, args.batch_size, device
    )

    # ------------------------------------------------------------------ #
    # Build full gallery (match + distractor)                             #
    # ------------------------------------------------------------------ #
    distractor_ids = -100 * np.ones(len(distractor_feats), dtype=np.int32)
    all_gallery_feats = np.vstack([gallery_feats, distractor_feats])
    all_gallery_ids   = np.concatenate([gallery_ids_valid, distractor_ids])
    n_gallery = all_gallery_feats.shape[0]

    # ------------------------------------------------------------------ #
    # Identification evaluation                                           #
    # ------------------------------------------------------------------ #
    # Iterate over ALL probes from the mat file.
    # FTA probes and probes whose gallery match is missing both count as
    # misidentifications: AP=0, CMC=zeros (worst-case strict evaluation).
    n_probe_total = len(probe_ids)
    zero_cmc = np.zeros(n_gallery)

    ap_list  = []
    cmc_list = []

    logging.info(f"Evaluating {n_probe_total} probes against {n_gallery} gallery entries …")
    for p in tqdm(range(n_probe_total), desc="Evaluating probes"):

        # Probe FTA → misidentification
        if probe_feat_idx[p] is None:
            ap_list.append(0.0)
            cmc_list.append(zero_cmc.copy())
            continue

        probe_id   = probe_ids[p]
        good_index = np.where(all_gallery_ids == probe_id)[0]

        # Gallery match FTA (correct image not present) → misidentification
        if len(good_index) == 0:
            ap_list.append(0.0)
            cmc_list.append(zero_cmc.copy())
            continue

        # Euclidean distance to all gallery entries
        diff  = all_gallery_feats - probe_feats[probe_feat_idx[p]]
        score = np.linalg.norm(diff, axis=1)
        index = np.argsort(score)  # ascending: closest first

        ap_p, cmc_p = compute_AP(good_index, index)
        ap_list.append(ap_p)
        cmc_list.append(cmc_p)

    CMC = np.mean(np.stack(cmc_list), axis=0)
    mAP = float(np.mean(ap_list))

    logging.info("=" * 60)
    logging.info("TinyFace Identification Results")
    logging.info("=" * 60)
    logging.info(f"  Probes total       : {n_probe_total}")
    logging.info(f"  Probe FTA          : {n_probe_fta}  (counted as misidentification)")
    logging.info(f"  Gallery FTA        : {n_gallery_fta} (matched probes scored AP=0, CMC=0)")
    logging.info(f"  mAP        : {mAP:.4f}")
    logging.info(f"  Rank-1     : {CMC[0]:.4f}")
    logging.info(f"  Rank-5     : {CMC[4]:.4f}")
    logging.info(f"  Rank-10    : {CMC[9]:.4f}")
    logging.info(f"  Rank-20    : {CMC[19]:.4f}")
    logging.info("=" * 60)

    print(
        f"mAP = {mAP:.6f}, "
        f"r1 = {CMC[0]:.6f}, "
        f"r5 = {CMC[4]:.6f}, "
        f"r10 = {CMC[9]:.6f}, "
        f"r20 = {CMC[19]:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyFace face identification evaluation"
    )
    parser.add_argument("--config",      required=True,
                        help="Path to model config (.py)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--mat-dir",
                        default=os.path.join(
                            os.environ.get("TINYFACE_ROOT", "data/tinyface"),
                            "Face_Identification_Evaluation"),
                        help="Directory with gallery_match_img_ID_pairs.mat and "
                             "probe_img_ID_pairs.mat; both ship with the TinyFace "
                             "distribution (default: $TINYFACE_ROOT/Face_Identification_Evaluation)")
    parser.add_argument("--aligned-gallery-dir",    required=True,
                        help="Directory with aligned Gallery_Match images (flat layout)")
    parser.add_argument("--aligned-probe-dir",      required=True,
                        help="Directory with aligned Probe images (flat layout)")
    parser.add_argument("--aligned-distractor-dir", required=True,
                        help="Directory with aligned Gallery_Distractor images "
                             "(prefix-subdirectory layout: <name[:20]>/<name>)")
    parser.add_argument("--batch-size",  type=int, default=256,
                        help="Batch size for feature extraction (default: 256)")
    parser.add_argument("--input-size",  type=int, default=112,
                        help="Image side passed to the model (default: 112). "
                             "Set to 32 for PDT-up32 networks that upsample internally.")
    parser.add_argument("--device",      default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--gpu-id",      type=int, default=0)
    main(parser.parse_args())
