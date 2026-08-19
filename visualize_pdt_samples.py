#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: visualize_pdt_samples.py
#
"""
Visualize PDT translator outputs.

Loads a trained PDT model and passes sample images through the translator
module, saving side-by-side (input | translator output) comparison grids.

Supports three image sources:
  --source bin   : sample pairs from .bin verification files
  --source tinyface : sample images from a TinyFace aligned directory
  --source rec   : sample images from an MXNet .rec file

Usage examples:
    python visualize_pdt_samples.py \
        --config configs/edgeface_s_gamma_05_PDT_28_cubic_area.py \
        --checkpoint edgeface_s_gamma_05_PDT_28_cubic_area/model.pt \
        --source bin --bin-dir data/webface4m --bin-names lfw lfw_28_lr2lr \
        --num-samples 8 --output-dir pdt_visualisations

    python visualize_pdt_samples.py \
        --config configs/edgeface_s_gamma_05_PDT_28_cubic_area.py \
        --checkpoint edgeface_s_gamma_05_PDT_28_cubic_area/model.pt \
        --source tinyface --tinyface-dir /path/to/aligned/Probe \
        --num-samples 16 --output-dir pdt_visualisations

    python visualize_pdt_samples.py \
        --config configs/edgeface_s_gamma_05_PDT_28_cubic_area.py \
        --checkpoint edgeface_s_gamma_05_PDT_28_cubic_area/model.pt \
        --source rec --rec-path $LRFR_DATA_ROOT/processed_downsample_28_cubic_area_webface4m/train.rec \
        --num-samples 16 --output-dir pdt_visualisations
"""

import argparse
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import get_model
from utils.utils_config import get_config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_pdt_model(cfg, checkpoint_path, device="cuda"):
    """Load PDT_wrapper (translator + frozen backbone) from checkpoint."""
    logging.info(f"Loading PDT model: {cfg.network}")
    model = get_model(
        cfg.network, dropout=0.0,
        fp16=getattr(cfg, "fp16", False),
        num_features=cfg.embedding_size,
    ).to(device)

    bb_ckpt = torch.load(cfg.backbone_checkpoint, map_location=device)
    model.backbone.load_state_dict(bb_ckpt["state_dict_backbone"])
    del bb_ckpt

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict_backbone" in ckpt:
        sd = ckpt["state_dict_backbone"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    clean = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        clean[k] = v
    model.load_state_dict(clean, strict=False)

    model.eval()
    logging.info("PDT model loaded")
    return model


# ---------------------------------------------------------------------------
#  Image sources
# ---------------------------------------------------------------------------

def load_bin_samples(bin_dir, bin_names, num_samples, image_size=(112, 112)):
    """Load random samples from .bin verification files.

    Returns dict[bin_name] -> list of uint8 tensors [3, H, W].
    """
    import mxnet as mx
    from mxnet import ndarray as nd

    results = {}
    for name in bin_names:
        path = os.path.join(bin_dir, name + ".bin")
        if not os.path.exists(path):
            logging.warning(f"Bin file not found: {path}")
            continue

        try:
            with open(path, "rb") as f:
                bins, issame_list = pickle.load(f)
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                bins, issame_list = pickle.load(f, encoding="bytes")

        indices = random.sample(range(len(bins)), min(num_samples, len(bins)))
        imgs = []
        for idx in indices:
            img = mx.image.imdecode(bins[idx])
            if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                img = mx.image.resize_short(img, image_size[0])
            img_np = img.asnumpy()  # [H, W, 3] uint8
            imgs.append(torch.from_numpy(img_np).permute(2, 0, 1))  # [3, H, W]
        results[name] = imgs
        logging.info(f"[{name}] loaded {len(imgs)} samples from bin")
    return results


def load_tinyface_samples(tinyface_dir, num_samples):
    """Load random sample images from a flat TinyFace aligned directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [
        p for p in Path(tinyface_dir).rglob("*") if p.suffix.lower() in exts
    ]
    if not all_files:
        raise ValueError(f"No images found in {tinyface_dir}")

    chosen = random.sample(all_files, min(num_samples, len(all_files)))
    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in chosen:
        img = Image.open(p).convert("RGB")
        t = tf(img)  # [3, 112, 112] float [0, 1]
        imgs.append((t * 255).byte())
    logging.info(f"Loaded {len(imgs)} TinyFace samples")
    return {"tinyface": imgs}


def load_rec_samples(rec_path, num_samples):
    """Load random samples from an MXNet .rec file using recordio.py."""
    from recordio import MXIndexedRecordIO

    idx_path = rec_path.replace(".rec", ".idx")
    rec = MXIndexedRecordIO(idx_path, rec_path, "r")

    keys = list(rec.keys)
    chosen_keys = random.sample(keys, min(num_samples, len(keys)))

    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    imgs = []
    for k in chosen_keys:
        raw = rec.read_idx(k)
        header_len = 24  # MXNet record header
        img_bytes = raw[header_len:]
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        t = tf(img)
        imgs.append((t * 255).byte())

    logging.info(f"Loaded {len(imgs)} rec samples")
    return {"rec": imgs}


# ---------------------------------------------------------------------------
#  Translator pass + visualisation
# ---------------------------------------------------------------------------

def normalise_for_model(img_uint8):
    """[B, 3, H, W] uint8 -> float normalised to [-1, 1]."""
    return (img_uint8.float() / 255.0 - 0.5) / 0.5


def denormalise_to_uint8(tensor):
    """Float [-1, 1] -> uint8 [0, 255], clamped."""
    return ((tensor * 0.5 + 0.5) * 255).clamp(0, 255).byte()


@torch.no_grad()
def run_translator(model, imgs_uint8, device, batch_size=32):
    """Pass images through the PDT translator only.

    Args:
        model: PDT_wrapper in eval mode
        imgs_uint8: list of [3, H, W] uint8 tensors
        device: torch device
        batch_size: mini-batch size

    Returns:
        list of [3, H, W] uint8 tensors (translator outputs)
    """
    outputs = []
    for i in range(0, len(imgs_uint8), batch_size):
        batch = torch.stack(imgs_uint8[i : i + batch_size])
        inp = normalise_for_model(batch).to(device)
        out = model.translator(inp)
        out_uint8 = denormalise_to_uint8(out.cpu())
        outputs.extend([out_uint8[j] for j in range(out_uint8.shape[0])])
    return outputs


def save_comparison_grid(inputs, outputs, save_path, nrow=4):
    """Save a grid: each column is (input on top, translator output on bottom)."""
    pairs = []
    for inp, out in zip(inputs, outputs):
        pairs.append(inp)
        pairs.append(out)
    grid = torchvision.utils.make_grid(pairs, nrow=nrow * 2, padding=4, pad_value=255)
    img = transforms.ToPILImage()(grid)
    img.save(save_path)
    logging.info(f"Saved grid ({len(inputs)} samples): {save_path}")


def save_individual_comparisons(inputs, outputs, output_dir, prefix, nrow=1):
    """Save individual side-by-side strips for each sample."""
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        pair = torch.stack([inp, out])
        grid = torchvision.utils.make_grid(pair, nrow=2, padding=2, pad_value=255)
        img = transforms.ToPILImage()(grid)
        path = os.path.join(output_dir, f"{prefix}_sample{i:03d}.jpg")
        img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Visualize PDT translator outputs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, choices=["bin", "tinyface", "rec"])
    parser.add_argument("--bin-dir", type=str, default="data/webface4m")
    parser.add_argument("--bin-names", nargs="+", default=["lfw", "lfw_28_lr2lr"])
    parser.add_argument("--tinyface-dir", type=str, default=None)
    parser.add_argument("--rec-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="pdt_visualisations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-individual", action="store_true",
                        help="Also save individual side-by-side strips per sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_config(args.config)
    model = load_pdt_model(cfg, args.checkpoint, device=args.device)

    if args.source == "bin":
        sample_sets = load_bin_samples(args.bin_dir, args.bin_names, args.num_samples)
    elif args.source == "tinyface":
        if not args.tinyface_dir:
            raise ValueError("--tinyface-dir is required when --source tinyface")
        sample_sets = load_tinyface_samples(args.tinyface_dir, args.num_samples)
    elif args.source == "rec":
        if not args.rec_path:
            raise ValueError("--rec-path is required when --source rec")
        sample_sets = load_rec_samples(args.rec_path, args.num_samples)

    model_name = Path(args.checkpoint).parent.name

    for source_name, imgs in sample_sets.items():
        outputs = run_translator(model, imgs, args.device)

        grid_path = os.path.join(
            args.output_dir, f"{model_name}_{source_name}_grid.jpg"
        )
        save_comparison_grid(imgs, outputs, grid_path, nrow=min(4, len(imgs)))

        if args.save_individual:
            save_individual_comparisons(
                imgs, outputs, args.output_dir,
                prefix=f"{model_name}_{source_name}",
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
