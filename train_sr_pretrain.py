#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: train_sr_pretrain.py
#
"""
Stage-1 SR pretraining for the upsampler that sits in front of PDT.

Supervised image regression on pixel-aligned (LR_32, HR_112) pairs.
Loss = L1(SR(LR), HR) + lambda_id * cos_dist(emb(SR(LR)), emb(HR))
where emb is a frozen EdgeFace-base backbone.

Outputs a state_dict for the upsampler module that Stage 2 (train_PDT.py)
loads via cfg.upsampler_checkpoint.

Usage:
    torchrun --nproc_per_node=1 train_sr_pretrain.py \
        --config configs/sr_pretrain_espcn.py
"""

import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from backbones.PDT.upsamplers import ESPCNUpsampler, RRDBUpsampler
from dataset_pdt import RecFaceDataset, _build_transform
from lr_scheduler import PolynomialLRWarmup
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging


_NCCL_TIMEOUT = timedelta(minutes=60)

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl", timeout=_NCCL_TIMEOUT)
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12586",
        rank=rank,
        world_size=world_size,
        timeout=_NCCL_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Paired dataset: HR (112x112) and LR (32x32) of the same image, by index.
# ---------------------------------------------------------------------------

class _PairedRecDataset(Dataset):
    """Pairs HR and LR records by their position in the rec file.

    Both recs must have the same image order — they are produced by
    scripts/preprocess_rec.py from the same source so this holds.
    The HR transform is the standard FR pipeline (RandomHorizontalFlip);
    the LR transform skips RandomHorizontalFlip to keep the pair aligned,
    and the HR transform is also skipped so SR reconstruction targets the
    same pixel layout. Augmentation lives in Stage 2.
    """

    def __init__(self, hr_root: str, lr_root: str, labels_cache_lr: str = None):
        from torchvision import transforms
        plain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.lr_ds = RecFaceDataset(lr_root, transform=plain)
        # HR loader reuses LR's labels.npy to skip the second scan.
        if labels_cache_lr is None:
            labels_cache_lr = str(Path(lr_root) / "labels.npy")
        self.hr_ds = RecFaceDataset(hr_root, transform=plain,
                                    labels_cache=labels_cache_lr)
        assert len(self.hr_ds) == len(self.lr_ds), (
            f"HR/LR rec length mismatch: {len(self.hr_ds)} vs {len(self.lr_ds)}"
        )

    def __len__(self) -> int:
        return len(self.lr_ds)

    def __getitem__(self, index: int):
        lr, _ = self.lr_ds[index]
        hr, _ = self.hr_ds[index]
        return lr, hr


# ---------------------------------------------------------------------------
# Frozen perceptual extractor (EdgeFace-base)
# ---------------------------------------------------------------------------

def _build_perceptual_extractor(backbone_checkpoint: str, device: int) -> nn.Module:
    """Load EdgeFace-base in eval mode with grads disabled."""
    model = get_model("edgeface_base", num_features=512, fp16=False, dropout=0.0).to(device)
    ckpt = torch.load(backbone_checkpoint, map_location=f"cuda:{device}")
    if isinstance(ckpt, dict) and "state_dict_backbone" in ckpt:
        state = ckpt["state_dict_backbone"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _id_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_upsampler(name: str) -> nn.Module:
    name = name.lower()
    if name == "espcn":
        return ESPCNUpsampler()
    if name in ("rrdb", "resrgan", "real_esrgan"):
        return RRDBUpsampler()
    raise ValueError(f"Unknown upsampler '{name}'. Expected 'espcn' or 'resrgan'.")


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0 else None
    )

    # Dataset & loader ------------------------------------------------------
    dataset = _PairedRecDataset(
        hr_root=cfg.rec_hr,
        lr_root=cfg.rec_lr,
        labels_cache_lr=getattr(cfg, "labels_cache", None),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    # Upsampler -------------------------------------------------------------
    upsampler = _build_upsampler(cfg.upsampler).cuda()
    upsampler = nn.parallel.DistributedDataParallel(
        upsampler, device_ids=[local_rank], broadcast_buffers=False,
        bucket_cap_mb=16, find_unused_parameters=False)

    # Frozen perceptual extractor ------------------------------------------
    if cfg.lambda_id > 0:
        perceptual = _build_perceptual_extractor(cfg.backbone_checkpoint, local_rank)
    else:
        perceptual = None

    # Optimiser & schedule --------------------------------------------------
    opt = torch.optim.AdamW(
        upsampler.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.num_epoch
    warmup_steps = steps_per_epoch * cfg.warmup_epoch
    scheduler = PolynomialLRWarmup(
        optimizer=opt, warmup_iters=warmup_steps, total_iters=total_steps)

    if rank == 0:
        n_params = sum(p.numel() for p in upsampler.parameters() if p.requires_grad)
        logging.info(f"Upsampler '{cfg.upsampler}' trainable params: {n_params:,}")
        for key, value in cfg.items():
            num_space = 25 - len(key)
            logging.info(": " + key + " " * num_space + str(value))

    # Resume ----------------------------------------------------------------
    # Matches train_PDT.py: per-rank checkpoint_gpu_{rank}.pt carries the full
    # state (upsampler + optimizer + scheduler + epoch + global_step). Stage 2
    # consumes only the rank-0 sr_pretrained.pt (upsampler weights only).
    start_epoch = 0
    global_step = 0
    print(f"Resume override is {args.resume_override}")
    if cfg.resume or args.resume_override:
        ckpt_path = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        if os.path.exists(ckpt_path):
            resume_ckpt = torch.load(ckpt_path, map_location=f"cuda:{local_rank}")
            upsampler.module.load_state_dict(resume_ckpt["state_dict"])
            opt.load_state_dict(resume_ckpt["state_optimizer"])
            scheduler.load_state_dict(resume_ckpt["state_lr_scheduler"])
            start_epoch = resume_ckpt["epoch"] + 1
            global_step = resume_ckpt["global_step"]
            if rank == 0:
                logging.info(
                    f"Resumed from {ckpt_path}: epoch {start_epoch}, "
                    f"global_step {global_step}"
                )
            del resume_ckpt
        else:
            if rank == 0:
                logging.warning(
                    f"Resume requested but no checkpoint found at {ckpt_path}. "
                    "Starting from scratch."
                )

    # Training loop ---------------------------------------------------------
    loss_meter = AverageMeter()
    l1_meter = AverageMeter()
    id_meter = AverageMeter()
    for epoch in range(start_epoch, cfg.num_epoch):
        sampler.set_epoch(epoch)
        for lr_img, hr_img in loader:
            lr_img = lr_img.cuda(local_rank, non_blocking=True)
            hr_img = hr_img.cuda(local_rank, non_blocking=True)

            sr_img = upsampler(lr_img)
            l1 = F.l1_loss(sr_img, hr_img)

            id_loss = torch.zeros((), device=sr_img.device)
            if perceptual is not None:
                with torch.no_grad():
                    emb_hr = perceptual(hr_img)
                emb_sr = perceptual(sr_img)
                id_loss = _id_cosine_distance(emb_sr, emb_hr)

            loss = l1 + cfg.lambda_id * id_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            loss_meter.update(loss.item(), lr_img.size(0))
            l1_meter.update(l1.item(), lr_img.size(0))
            id_meter.update(id_loss.item(), lr_img.size(0))

            global_step += 1
            if rank == 0 and global_step % cfg.frequent == 0:
                logging.info(
                    f"epoch {epoch} step {global_step}/{total_steps}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"loss={loss_meter.avg:.4f}  "
                    f"l1={l1_meter.avg:.4f}  id={id_meter.avg:.4f}"
                )
                if summary_writer is not None:
                    summary_writer.add_scalar("loss/total", loss_meter.avg, global_step)
                    summary_writer.add_scalar("loss/l1", l1_meter.avg, global_step)
                    summary_writer.add_scalar("loss/id", id_meter.avg, global_step)
                    summary_writer.add_scalar("lr", opt.param_groups[0]['lr'], global_step)
                loss_meter.reset()
                l1_meter.reset()
                id_meter.reset()

        # End-of-epoch checkpoints --------------------------------------------
        # Per-rank full-state checkpoint for resume (optimizer + scheduler).
        full_state = {
            "state_dict": upsampler.module.state_dict(),
            "state_optimizer": opt.state_dict(),
            "state_lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(full_state, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        # Stage-2-consumable artifact (upsampler weights only), rank 0 only.
        if rank == 0:
            artifact = {
                "state_dict": upsampler.module.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }
            epoch_path = os.path.join(cfg.output, f"sr_pretrained_epoch_{epoch}.pt")
            torch.save(artifact, epoch_path)
            torch.save(artifact, os.path.join(cfg.output, "sr_pretrained.pt"))
            logging.info(f"Saved {epoch_path}")

    distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR pretraining (Stage 1)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to Stage-1 config (e.g. configs/sr_pretrain_espcn.py)")
    parser.add_argument("--resume-override", action="store_true",
                        help="Force resume from <output>/checkpoint_gpu_<rank>.pt "
                             "regardless of cfg.resume.")
    main(parser.parse_args())
