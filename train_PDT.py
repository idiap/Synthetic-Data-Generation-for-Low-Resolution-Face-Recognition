#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: train_PDT.py
#
import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from backbones import get_model
from dataset_pdt import get_pdt_dataloader
from losses import ContrastiveLoss
from lr_scheduler import PolynomialLRWarmup
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, PDTCallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

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
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
        timeout=_NCCL_TIMEOUT,
    )


def main(args):

    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    # Both loaders use the SAME (pk_p, pk_k, seed) so their PKBatchSamplers
    # generate the identical identity-shuffle sequence → every batch contains
    # the same P identities in both modalities → guaranteed positive pairs.
    # Image selection within each identity diverges after the shared identity
    # shuffle, so LR and HR loaders independently pick different images of the
    # same person (diverse positives without explicit synchronisation).
    # LR loader built first — its labels.npy is reused by the HR loader.
    # LR and HR datasets share identical record order and identity labels, so
    # only one label scan is ever needed.
    train_loader_lr = get_pdt_dataloader(
        cfg.rec,
        local_rank,
        p=cfg.pk_p,
        k=cfg.pk_k,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    _lr_cache = str(Path(cfg.rec) / "labels.npy")
    train_loader_hr = get_pdt_dataloader(
        cfg.rec_hr,
        local_rank,
        p=cfg.pk_p,
        k=cfg.pk_k,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        labels_cache=_lr_cache,
    )

    # Build PDT_wrapper: backbone (frozen) + translator (trained).
    # cfg.network should resolve to a PDT_wrapper variant, e.g. 'edgeface_s_gamma_05_PDT'.
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    # Load pre-trained backbone weights before DDP wrapping.
    # On resume this state will be overwritten by the full checkpoint below.
    # Accepts both train_v2.py-style {"state_dict_backbone": ...} checkpoints
    # and flat OrderedDicts (as in the published checkpoints/edgeface_*.pt).
    bb_ckpt = torch.load(cfg.backbone_checkpoint, map_location=f"cuda:{local_rank}")
    if isinstance(bb_ckpt, dict) and "state_dict_backbone" in bb_ckpt:
        bb_state = bb_ckpt["state_dict_backbone"]
    elif isinstance(bb_ckpt, dict) and "state_dict" in bb_ckpt:
        bb_state = bb_ckpt["state_dict"]
    else:
        bb_state = bb_ckpt
    bb_state = {k[7:] if k.startswith("module.") else k: v for k, v in bb_state.items()}
    backbone.backbone.load_state_dict(bb_state)
    logging.info(f"Loaded pretrained backbone from {cfg.backbone_checkpoint}")
    del bb_ckpt, bb_state

    # Optional: load Stage-1 SR-pretrained upsampler weights into translator[0].
    # cfg.upsampler_checkpoint is None / unset on a from-scratch run.
    upsampler_ckpt = getattr(cfg, "upsampler_checkpoint", None)
    if upsampler_ckpt:
        from backbones.PDT.upsamplers import load_upsampler_checkpoint
        # Translator is nn.Sequential(upsampler, PDT); position 0 is the upsampler.
        load_upsampler_checkpoint(backbone.translator[0], upsampler_ckpt)
        logging.info(f"Loaded SR-pretrained upsampler from {upsampler_ckpt}")

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    # Only the translator trains; backbone stays in eval mode (enforced by PDT_wrapper.train()).
    backbone.train()

    contrastive_loss = ContrastiveLoss(margin=cfg.contrastive_margin if hasattr(cfg, 'contrastive_margin') else 2.0)

    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=backbone.module.translator.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Only 'adamw' is supported for PDT training, got: {cfg.optimizer}")

    cfg.batch_size = cfg.pk_p * cfg.pk_k       # effective batch size for logging
    cfg.total_batch_size = cfg.batch_size * world_size
    # With PK sampling the number of steps per epoch is determined by the number
    # of identities in the dataset, not the number of images.
    steps_per_epoch = len(train_loader_lr)    # PKBatchSampler.__len__
    cfg.warmup_step = steps_per_epoch * cfg.warmup_epoch
    cfg.total_step  = steps_per_epoch * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    print(f"Resume override is {args.resume_override}")
    if cfg.resume or args.resume_override:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        # Restores both backbone and translator states saved from a previous PDT run
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # PDTCallBackVerification routes each image in a verification pair through
    # either the HR path (backbone only) or the LR path (translator → backbone),
    # as specified per bin-file by cfg.val_pair_modes.
    callback_verification = PDTCallBackVerification(
        val_targets=cfg.val_targets,
        rec_prefix=cfg.rec,
        pair_modes=getattr(cfg, "val_pair_modes", None),
        summary_writer=summary_writer,
        wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        train_loader_lr.batch_sampler.set_epoch(epoch)
        train_loader_hr.batch_sampler.set_epoch(epoch)

        for _, ((img_lr, labels_lr), (img_hr, labels_hr)) in enumerate(
                zip(train_loader_lr, train_loader_hr)):
            global_step += 1

            emb_hr, emb_lr = backbone((img_hr, img_lr))

            # All N×N cross-modal distances in one fused op.
            # dist_matrix[i, j] = ||emb_hr[i] - emb_lr[j]||_2
            dist_matrix = torch.cdist(emb_hr, emb_lr, p=2)  # [N, N]

            # Binary label: 1 (same identity) or 0 (different identity).
            # PKBatchSampler guarantees the same P identities in both batches,
            # so the [N, N] matrix has P dense positive blocks on the diagonal
            # and P*(P-1) negative blocks off it.
            same_matrix = (labels_hr.unsqueeze(1) == labels_lr.unsqueeze(0)).float()  # [N, N]

            # ContrastiveLoss.forward expects flat (dist, label) tensors of the
            # same shape — view(-1) flattens both [N, N] tensors to [N*N].
            loss: torch.Tensor = contrastive_loss(dist_matrix.view(-1), same_matrix.view(-1))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.module.translator.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.module.translator.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                # Save the full PDT_wrapper state (backbone + translator) so that
                # resuming does not require separately re-loading the backbone checkpoint.
                "state_dict_backbone": backbone.module.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)


    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed PDT (Prepended Domain Transformer) Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--resume-override", action='store_true', help="override resume from config")
    main(parser.parse_args())
