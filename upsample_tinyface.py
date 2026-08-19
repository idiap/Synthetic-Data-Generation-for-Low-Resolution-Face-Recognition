#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: upsample_tinyface.py
#
"""
upsample_tinyface.py

Batch-run a Stage-1 SR upsampler over a pre-aligned (32x32) TinyFace set and
save the 112x112 super-resolved images, mirroring the input directory layout.

The output can then be fed directly to eval_tinyface_identification.py with the
default --input-size 112 (the SR happens here, offline, so the eval backbone
sees ordinary 112x112 crops).

Directory layout is preserved relative to --input-dir, so this handles both:
  * flat sets   (Gallery_Match, Probe)
  * prefix-subdir set (Gallery_Distractor: <name[:20]>/<name>)
Filenames (and extensions) are kept identical so the .mat lookups still resolve.

Supports SLURM array jobs: each worker processes a strided slice of the file
list (task i handles indices i, i+num_tasks, i+2*num_tasks, ...), matching the
partitioning scheme used by align_tinyface.py.

Usage:
    python upsample_tinyface.py \
        --config configs/edgeface_base_PDT_up32_espcn.py \
        --stage1-checkpoint sr_pretrain_espcn/sr_pretrained.pt \
        --input-dir  /path/to/aligned_32/Probe \
        --output-dir /path/to/upsampled_112/Probe
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones.PDT.upsamplers import ESPCNUpsampler, RRDBUpsampler
from utils.utils_config import get_config


_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
])


def _denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] CHW tensor -> uint8 HWC numpy."""
    x = tensor.detach().cpu().float()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


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


def load_stage1_upsampler(network: str, ckpt_path: str, device: str) -> torch.nn.Module:
    """Load a Stage-1 sr_pretrained.pt artifact (upsampler weights only)."""
    upsampler = _upsampler_from_network_name(network).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _strip_module_prefix(state)
    upsampler.load_state_dict(state)
    upsampler.eval()
    return upsampler


def collect_image_paths(input_dir: Path) -> List[Path]:
    """Recursively collect image files (sorted) under input_dir."""
    paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    return sorted(paths)


def _save_uint8(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(str(out_path), quality=100, subsampling=0)
    else:
        img.save(str(out_path))


@torch.no_grad()
def run(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    cfg = get_config(args.config)
    upsampler = load_stage1_upsampler(cfg.network, args.stage1_checkpoint, device)
    print(f"Loaded Stage-1 upsampler ({cfg.network}) from {args.stage1_checkpoint}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    all_paths = collect_image_paths(input_dir)
    # Stride-based partition for SLURM arrays (matches align_tinyface.py).
    my_paths = all_paths[args.task_id::args.num_tasks]
    print(f"Task {args.task_id}/{args.num_tasks}: "
          f"processing {len(my_paths)} / {len(all_paths)} images")

    batch_tensors: List[torch.Tensor] = []
    batch_outpaths: List[Path] = []
    done = 0

    def flush():
        nonlocal done
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        sr = upsampler(x)
        for i in range(sr.shape[0]):
            _save_uint8(_denorm_to_uint8(sr[i]), batch_outpaths[i])
        done += len(batch_tensors)
        batch_tensors.clear()
        batch_outpaths.clear()

    for p in my_paths:
        rel = p.relative_to(input_dir)
        out_path = output_dir / rel
        if args.skip_existing and out_path.exists():
            done += 1
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  WARN: failed to read {p}: {e}")
            continue
        if img.size != (args.input_size, args.input_size):
            img = img.resize((args.input_size, args.input_size), Image.BILINEAR)
        batch_tensors.append(_TO_TENSOR(img))
        batch_outpaths.append(out_path)

        if len(batch_tensors) >= args.batch_size:
            flush()
            if done % (args.batch_size * 20) == 0:
                print(f"  [task {args.task_id}] {done}/{len(my_paths)}")
    flush()

    print(f"Task {args.task_id} done: wrote/processed {done}/{len(my_paths)} images "
          f"to {output_dir}")


if __name__ == "__main__":
    slurm_task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    slurm_num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    parser = argparse.ArgumentParser(
        description="Batch Stage-1 SR upsampling of aligned TinyFace images "
                    "(supports SLURM array jobs)."
    )
    parser.add_argument("--config", required=True,
                        help="Config (.py) — used to infer upsampler variant "
                             "(espcn/resrgan) from cfg.network.")
    parser.add_argument("--stage1-checkpoint", required=True,
                        help="Path to Stage-1 sr_pretrained.pt (upsampler weights only).")
    parser.add_argument("--input-dir", required=True,
                        help="Directory of aligned LR images (recursively walked).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for SR images (input layout mirrored).")
    parser.add_argument("--input-size", type=int, default=32,
                        help="LR input side length (default: 32). Off-size inputs are "
                             "bilinear-resized to match the upsampler.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (default: 64).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose output file already exists.")
    parser.add_argument("--task-id", type=int, default=slurm_task_id,
                        help="0-based worker index. Auto from SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--num-tasks", type=int, default=slurm_num_tasks,
                        help="Total workers. Auto from SLURM_ARRAY_TASK_COUNT.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    run(parser.parse_args())
