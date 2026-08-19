#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: visualize_upsampler.py
#
"""
Visualise upsampler / PDT outputs for qualitative inspection.

For each sample LR image, plots a panel comparing up to five versions:

    [ LR_32 | HR_112 | Stage1_SR(LR) | Stage2_SR(LR) | Stage2_PDT(Stage2_SR(LR)) ]
       (a)     (b)          (c)             (d)                    (e)

  (a) original 32x32 input shown nearest-upscaled to 112x112 for display only.
  (b) HR ground truth at 112x112, paired by rec index with the LR sample
      (only when --rec-hr is provided).
  (c) output of the upsampler after Stage-1 SR pretraining (image regression).
  (d) output of the upsampler after Stage-2 joint contrastive training.
  (e) output of PDT applied to (d), i.e. the full translator output that the
      frozen backbone actually sees during Stage-2 inference.

Any of (b)/(c)/(d)/(e) may be omitted by leaving the corresponding flag unset;
the panel skips the missing columns.

Usage:
    # Source A: random samples from an MXNet rec (with HR-paired ground truth)
    python visualize_upsampler.py \
        --config configs/edgeface_base_PDT_up32_espcn.py \
        --stage1-checkpoint sr_pretrain_espcn/sr_pretrained.pt \
        --stage2-checkpoint edgeface_base_PDT_up32_espcn/checkpoint_gpu_0.pt \
        --rec    $LRFR_DATA_ROOT/LR_for_PDT_processed_noup_32_esrgan_webface4m/ \
        --rec-hr $LRFR_DATA_ROOT/HR_for_PDT_passthrough_webface4m/ \
        --n-samples 8 \
        --output-dir viz/edgeface_base_PDT_up32_espcn/

    # Source B: first N images from a flat folder of LR crops
    python visualize_upsampler.py \
        --config configs/edgeface_base_PDT_up32_espcn.py \
        --stage1-checkpoint sr_pretrain_espcn/sr_pretrained.pt \
        --image-dir /path/to/lr_samples/ \
        --n-samples 8 \
        --output-dir viz/edgeface_base_PDT_up32_espcn_dir/
"""
import argparse
import io
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backbones import get_model
from backbones.PDT.upsamplers import ESPCNUpsampler, RRDBUpsampler
from utils.utils_config import get_config


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> normalised tensor matching the training transform."""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])
    return t(img)


def _denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor -> uint8 HxWxC numpy."""
    x = tensor.detach().cpu().float()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()  # CHW -> HWC


def _strip_module_prefix(state_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def _upsampler_from_network_name(network: str) -> torch.nn.Module:
    network = network.lower()
    if network.endswith("_espcn"):
        return ESPCNUpsampler()
    if network.endswith("_resrgan"):
        return RRDBUpsampler()
    raise ValueError(
        f"Cannot infer upsampler variant from cfg.network='{network}'. "
        "Expected suffix '_espcn' or '_resrgan'."
    )


# --------------------------------------------------------------------------- #
# Checkpoint loading                                                          #
# --------------------------------------------------------------------------- #

def load_stage1_upsampler(network: str, ckpt_path: str, device: str) -> torch.nn.Module:
    """Load a Stage-1 sr_pretrained.pt artifact (upsampler weights only)."""
    upsampler = _upsampler_from_network_name(network).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _strip_module_prefix(state)
    upsampler.load_state_dict(state)
    upsampler.eval()
    return upsampler


def load_stage2_translator(cfg, ckpt_path: str, device: str):
    """Load a Stage-2 PDT_wrapper checkpoint and return (upsampler, pdt) modules."""
    model = get_model(cfg.network, dropout=0.0, fp16=False,
                      num_features=cfg.embedding_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict_backbone" in ckpt:
        state = ckpt["state_dict_backbone"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model.eval()
    upsampler = model.translator[0]
    pdt = model.translator[1]
    return upsampler, pdt


# --------------------------------------------------------------------------- #
# Sampling LR images from a rec file or an image folder                       #
# --------------------------------------------------------------------------- #

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _open_rec(rec_dir: str):
    """Open a rec directory and return (manager, full_index_array)."""
    from recordio import RecordIODataManager  # local import; not strictly needed elsewhere
    dm = RecordIODataManager(
        str(Path(rec_dir) / "train.idx"),
        str(Path(rec_dir) / "train.rec"),
    )
    header_bytes = dm.get_raw_bytes(0)
    header, _ = RecordIODataManager._unpack_recordio(header_bytes)
    if header.flag > 0:
        num_imgs = int(header.label[0])
        all_idx = np.arange(1, num_imgs, dtype=np.int64)
    else:
        all_idx = np.arange(1, len(dm), dtype=np.int64)
    return dm, all_idx


def _read_rec_at(dm, indices) -> List[Image.Image]:
    from recordio import RecordIODataManager
    imgs: List[Image.Image] = []
    for idx in indices:
        data = dm.get_raw_bytes(int(idx))
        _, img_bytes = RecordIODataManager._unpack_recordio(data)
        imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return imgs


def sample_pairs_from_rec(
    lr_rec_dir: str,
    hr_rec_dir: Optional[str],
    n: int,
    seed: int,
):
    """Sample N random indices from the LR rec; load LR and (optionally) HR at those indices.

    Pairing by index assumes LR and HR recs share record order (the convention
    used elsewhere in this repo — see train_PDT.py / train_sr_pretrain.py).

    Returns:
        (lr_imgs, hr_imgs)  where hr_imgs is None when hr_rec_dir is None.
    """
    dm_lr, all_idx = _open_rec(lr_rec_dir)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_idx, size=min(n, len(all_idx)), replace=False)

    lr_imgs = _read_rec_at(dm_lr, chosen)
    dm_lr.close()

    hr_imgs: Optional[List[Image.Image]] = None
    if hr_rec_dir is not None:
        dm_hr, hr_all = _open_rec(hr_rec_dir)
        # Sanity: HR rec must cover the indices we drew from LR.
        max_lr = int(chosen.max())
        max_hr = int(hr_all.max())
        if max_lr > max_hr:
            dm_hr.close()
            raise ValueError(
                f"HR rec ({hr_rec_dir}) is shorter than LR: "
                f"max LR idx {max_lr} > max HR idx {max_hr}."
            )
        hr_imgs = _read_rec_at(dm_hr, chosen)
        dm_hr.close()

    return lr_imgs, hr_imgs


def sample_lr_images_from_folder(
    folder: str, n: int, input_size: int, recursive: bool,
) -> List[Image.Image]:
    """Load the first N images (sorted by filename) from a directory.

    Files are filtered by extension. Images not already at input_size×input_size
    are bilinear-resized to that size — the upsampler expects 32×32 LR by design.
    """
    root = Path(folder)
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    paths = paths[:n]

    imgs: List[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (input_size, input_size):
            img = img.resize((input_size, input_size), Image.BILINEAR)
        imgs.append(img)
    return imgs


# --------------------------------------------------------------------------- #
# Per-sample inference                                                        #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def render_panel(
    lr_img: Image.Image,
    hr_img: Optional[Image.Image],
    stage1: Optional[torch.nn.Module],
    stage2_up: Optional[torch.nn.Module],
    stage2_pdt: Optional[torch.nn.Module],
    device: str,
) -> List[np.ndarray]:
    """Return uint8 HWC arrays for the columns: LR | HR? | S1? | S2? | S2+PDT?."""
    panels: List[np.ndarray] = []

    # (a) LR shown nearest-upscaled to 112 for display parity
    panels.append(np.array(lr_img.resize((112, 112), Image.NEAREST)))

    # (b) HR ground truth (when provided)
    if hr_img is not None:
        # If HR is not 112×112 (e.g. some other rec), resize for display parity.
        hr_show = hr_img if hr_img.size == (112, 112) else hr_img.resize(
            (112, 112), Image.BILINEAR)
        panels.append(np.array(hr_show))

    lr_t = _to_tensor(lr_img).unsqueeze(0).to(device)

    # (c) Stage-1 upsampler output
    if stage1 is not None:
        sr1 = stage1(lr_t).squeeze(0)
        panels.append(_denorm_to_uint8(sr1))

    # (d) Stage-2 upsampler output
    sr2 = None
    if stage2_up is not None:
        sr2 = stage2_up(lr_t)
        panels.append(_denorm_to_uint8(sr2.squeeze(0)))

    # (e) Stage-2 PDT(upsampler) output  — what the backbone actually receives
    if stage2_up is not None and stage2_pdt is not None and sr2 is not None:
        pdt_out = stage2_pdt(sr2).squeeze(0)
        panels.append(_denorm_to_uint8(pdt_out))

    return panels


def save_panel(panels: List[np.ndarray], labels: List[str], out_path: str) -> None:
    """Save a horizontal grid of labelled panels via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, img, lbl in zip(axes, panels, labels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(args):
    cfg = get_config(args.config)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # ---- Load models -----------------------------------------------------
    stage1_up = None
    if args.stage1_checkpoint:
        stage1_up = load_stage1_upsampler(cfg.network, args.stage1_checkpoint, device)
        print(f"Loaded Stage-1 upsampler from {args.stage1_checkpoint}")

    stage2_up, stage2_pdt = None, None
    if args.stage2_checkpoint:
        stage2_up, stage2_pdt = load_stage2_translator(cfg, args.stage2_checkpoint, device)
        print(f"Loaded Stage-2 translator from {args.stage2_checkpoint}")

    if stage1_up is None and stage2_up is None:
        raise ValueError(
            "Provide at least one of --stage1-checkpoint / --stage2-checkpoint."
        )

    # ---- Sample LR (and optionally HR) images ----------------------------
    if bool(args.rec) == bool(args.image_dir):
        raise ValueError(
            "Provide exactly one of --rec / --image-dir as the LR source."
        )
    if args.rec_hr and not args.rec:
        raise ValueError(
            "--rec-hr requires --rec (HR pairing is by rec index)."
        )

    hr_images: Optional[List[Image.Image]] = None
    if args.rec:
        msg = f"Sampling {args.n_samples} LR images from rec {args.rec}"
        if args.rec_hr:
            msg += f"  (paired HR rec: {args.rec_hr})"
        print(msg)
        lr_images, hr_images = sample_pairs_from_rec(
            args.rec, args.rec_hr, args.n_samples, args.seed
        )
    else:
        print(f"Loading first {args.n_samples} LR images from folder {args.image_dir}")
        lr_images = sample_lr_images_from_folder(
            args.image_dir, args.n_samples, args.input_size, args.recursive
        )
    if not lr_images:
        raise RuntimeError("No images found at the requested source.")

    # ---- Build column labels in the same order as render_panel -----------
    labels = ["LR 32x32"]
    if hr_images is not None:
        labels.append("HR 112x112")
    if stage1_up is not None:
        labels.append("Stage-1 SR")
    if stage2_up is not None:
        labels.append("Stage-2 SR")
    if stage2_up is not None and stage2_pdt is not None:
        labels.append("Stage-2 SR + PDT")

    # ---- Render and save -------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, lr_img in enumerate(lr_images):
        hr_img = hr_images[i] if hr_images is not None else None
        panels = render_panel(lr_img, hr_img, stage1_up, stage2_up, stage2_pdt, device)
        out_path = out_dir / f"sample_{i:03d}.png"
        save_panel(panels, labels, str(out_path))
        print(f"  wrote {out_path}")

    print(f"Done. Panels saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise Stage-1 / Stage-2 upsampler and PDT outputs."
    )
    parser.add_argument("--config", required=True,
                        help="Stage-2 config (.py) — used to infer upsampler variant "
                             "(espcn/resrgan) and default LR rec path.")
    parser.add_argument("--stage1-checkpoint", default=None,
                        help="Path to Stage-1 sr_pretrained.pt (upsampler weights only).")
    parser.add_argument("--stage2-checkpoint", default=None,
                        help="Path to Stage-2 checkpoint_gpu_<rank>.pt "
                             "(full PDT_wrapper state).")
    src = parser.add_argument_group("LR image source (provide exactly one)")
    src.add_argument("--rec", default=None,
                     help="Path to an MXNet RecordIO directory (with train.idx/train.rec). "
                          "Samples N images at random (seeded).")
    src.add_argument("--rec-hr", default=None,
                     help="Optional paired HR rec (records share the same index order). "
                          "When provided alongside --rec, an HR-target column is added "
                          "to each panel for direct visual comparison.")
    src.add_argument("--image-dir", default=None,
                     help="Path to a directory of image files (.jpg/.png/.bmp/.tif/.webp). "
                          "Takes the first N files in sorted order — no random sampling.")
    src.add_argument("--recursive", action="store_true",
                     help="When using --image-dir, recurse into subdirectories.")
    src.add_argument("--input-size", type=int, default=32,
                     help="LR input side length (default: 32). Folder images not already at "
                          "this size are bilinear-resized to match the upsampler's expected "
                          "input.")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of LR samples to visualise (default: 8).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for sampling LR indices from --rec (default: 0).")
    parser.add_argument("--output-dir", default="viz/",
                        help="Directory for output panel PNGs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    main(parser.parse_args())
