#
# SPDX-FileCopyrightText: Copyright (c) 2022 Jiankang Deng and Jia Guo
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: train_v2_kd.py
# Modified from InsightFace (https://github.com/deepinsight/insightface, MIT);
# see LICENSES/MIT.txt. Changes: embedding and PartialFC-weight distillation from a frozen HR teacher.
#
import argparse
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import os
import sys

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


def load_teacher(cfg, rank):
    """Load frozen teacher backbone and partial-FC weight shard."""
    teacher_ckpt_path = cfg.teacher_checkpoint.format(rank=rank)
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location=f"cuda:{local_rank}")

    teacher_backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    teacher_backbone.load_state_dict(teacher_ckpt["state_dict_backbone"])
    teacher_backbone.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    # Plain tensor — no gradients, no optimizer state
    teacher_fc_weight = teacher_ckpt["state_dict_softmax_fc"]["weight"].cuda().detach()

    del teacher_ckpt
    logging.info(f"Loaded teacher from {teacher_ckpt_path}")
    return teacher_backbone, teacher_fc_weight


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

    # LR dataloader — input to the student. cfg.input_size sets the student's
    # native resolution (e.g. 16 or 32 for noup_* recs); defaults to 112.
    train_loader_lr = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        dali_name="reader_lr",
        input_size=getattr(cfg, 'input_size', 112),
    )

    # HR dataloader — input to the teacher.
    # Uses the same seed so that both loaders yield the same sample order,
    # ensuring each (LR, HR) pair in a batch corresponds to the same identity.
    # cfg.input_size_hr defaults to 112 (teacher always sees full-res HR).
    train_loader_hr = get_dataloader(
        cfg.rec_hr,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        dali_name="reader_hr",
        input_size=getattr(cfg, 'input_size_hr', 112),
    )

    # ------------------------------------------------------------------ student
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    # ------------------------------------------------------------------ teacher
    teacher_backbone, teacher_fc_weight = load_teacher(cfg, rank)

    # ------------------------------------------------------------------ losses & optimiser
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume or args.resume_override:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        if args.resume_override:
            opt.load_state_dict(dict_checkpoint["state_optimizer"])
            lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Verification bins must be at the student's native resolution. The HR
    # teacher path is not validated separately by this callback.
    _val_size = getattr(cfg, 'input_size', 112)
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger,
        image_size=(_val_size, _val_size),
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    loss_emb_am = AverageMeter()
    loss_weight_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    kd_lambda_emb = getattr(cfg, "kd_lambda_emb", 1.0)
    kd_lambda_weight = getattr(cfg, "kd_lambda_weight", 1.0)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader_lr, DataLoader):
            train_loader_lr.sampler.set_epoch(epoch)
        if isinstance(train_loader_hr, DataLoader):
            train_loader_hr.sampler.set_epoch(epoch)

        for _, ((img_lr, local_labels), (img_hr, _)) in enumerate(
                zip(train_loader_lr, train_loader_hr)):
            global_step += 1

            # Teacher forward on HR images — no gradients
            with torch.no_grad():
                teacher_embeddings = teacher_backbone(img_hr)

            # Student forward on LR images + classification loss
            local_embeddings = backbone(img_lr)
            loss_cls: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            # Embedding distillation loss (on L2-normalised embeddings)
            loss_emb = F.mse_loss(
                normalize(local_embeddings),
                normalize(teacher_embeddings)
            )

            # Weight distillation loss on the sampled class-centre subset.
            # module_partial_fc.weight_index is set by PartialFC_V2.sample() during forward()
            # when sample_rate < 1; fall back to full shard when sample_rate == 1.
            if cfg.sample_rate < 1.0:
                sampled_idx = module_partial_fc.weight_index
                student_w = normalize(module_partial_fc.weight[sampled_idx])
                teacher_w = normalize(teacher_fc_weight[sampled_idx])
            else:
                student_w = normalize(module_partial_fc.weight)
                teacher_w = normalize(teacher_fc_weight)
            loss_weight = F.mse_loss(student_w, teacher_w)

            loss = loss_cls + kd_lambda_emb * loss_emb + kd_lambda_weight * loss_weight

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Loss/Emb Distill': loss_emb.item(),
                        'Loss/Weight Distill': loss_weight.item(),
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                loss_am.update(loss.item(), 1)
                loss_emb_am.update(loss_emb.item(), 1)
                loss_weight_am.update(loss_weight.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
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

        if cfg.dali:
            train_loader_lr.reset()
            train_loader_hr.reset()

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
        description="Distributed Arcface Training with Knowledge Distillation")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--resume-override", action='store_true', help="override resume from config")
    main(parser.parse_args())
