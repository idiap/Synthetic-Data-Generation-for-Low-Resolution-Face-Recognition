#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: evaluate_model.py
#
"""
Standalone model evaluation script for face recognition models.
Loads a trained model checkpoint and evaluates on validation datasets.

Usage:
    python evaluate_model.py --config configs/edgeface_s_gamma_05_lr_area_area.py \
                             --checkpoint edgeface_s_gamma_05_lr_area_area/checkpoint_gpu_0.pt \
                             --val-targets lfw lfw_28 lfw_14
"""
from eval import verification

import argparse
import logging
import os
import sys

import torch
from backbones import get_model
from utils.utils_config import get_config


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_model(cfg, checkpoint_path, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    logging.info(f"Loading model: {cfg.network}")
    logging.info(f"Embedding size: {cfg.embedding_size}")
    
    # Create model
    backbone = get_model(
        cfg.network, 
        dropout=0.0, 
        fp16=cfg.fp16 if hasattr(cfg, 'fp16') else False,
        num_features=cfg.embedding_size
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both DDP and non-DDP saved models
        if 'state_dict_backbone' in checkpoint:
            state_dict = checkpoint['state_dict_backbone']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        backbone.load_state_dict(new_state_dict)
        
        # Log checkpoint info if available
        if 'epoch' in checkpoint:
            logging.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            logging.info(f"Checkpoint global step: {checkpoint['global_step']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set to eval mode
    backbone.eval()
    logging.info("Model loaded successfully")
    
    return backbone


def load_validation_datasets(val_targets, data_dir, image_size=(112, 112)):
    """
    Load validation datasets from .bin files.
    
    Args:
        val_targets: List of validation target names
        data_dir: Directory containing .bin files
        image_size: Image size tuple
    
    Returns:
        List of (name, dataset) tuples
    """
    datasets = []
    
    for name in val_targets:
        bin_path = os.path.join(data_dir, name + ".bin")
        
        if os.path.exists(bin_path):
            logging.info(f"Loading validation dataset: {name} from {bin_path}")
            dataset = verification.load_bin(bin_path, image_size)
            datasets.append((name, dataset))
        else:
            logging.warning(f"Validation dataset not found: {bin_path}")
    
    if not datasets:
        raise ValueError("No validation datasets found!")
    
    return datasets


def parse_pair_modes(pair_modes_list):
    """Parse --pair-modes CLI strings into a dict.

    Each entry has the form ``name:mode1,mode2``, e.g.
    ``lfw:hr,hr`` or ``lfw_28_hr2lr_interArea:hr,lr``.
    Modes: ``hr`` = backbone only, ``lr`` = translator → backbone.
    """
    if not pair_modes_list:
        return {}
    result = {}
    for entry in pair_modes_list:
        try:
            name, modes = entry.split(':')
            m1, m2 = modes.split(',')
            result[name.strip()] = (m1.strip(), m2.strip())
        except ValueError:
            raise ValueError(
                f"Invalid --pair-modes entry '{entry}'. "
                "Expected format: name:mode1,mode2  (e.g. lfw_28_hr2lr:hr,lr)"
            )
    return result


def evaluate(backbone, datasets, device='cuda', pair_modes=None):
    """
    Evaluate model on validation datasets.
    
    Args:
        backbone: Model to evaluate
        datasets: List of (name, dataset) tuples
        device: Device to run evaluation on
        pair_modes: Optional dict mapping dataset name to (mode1, mode2).
            When provided for a given target, uses verification.test_pdt so
            the first/second image of each pair can be routed through different
            paths ('hr' = backbone only, 'lr' = translator → backbone).
            Targets not listed fall back to the standard verification.test.
    
    Returns:
        Dictionary of results
    """
    pair_modes = pair_modes or {}
    results = {}
    
    logging.info("\n" + "="*80)
    logging.info("Starting Evaluation")
    logging.info("="*80)
    
    with torch.no_grad():
        for name, dataset in datasets:
            logging.info(f"\nEvaluating on: {name}")
            logging.info("-" * 40)

            if name in pair_modes:
                mode1, mode2 = pair_modes[name]
                logging.info(f"[{name}] PDT routing: img1={mode1}, img2={mode2}")
                acc1, std1, acc2, std2, xnorm, _ = verification.test_pdt(
                    dataset, backbone, batch_size=10, mode1=mode1, mode2=mode2
                )
            else:
                acc1, std1, acc2, std2, xnorm, _ = verification.test(
                    dataset, backbone, 10, 10
                )
            
            results[name] = {
                'acc_no_flip': acc1,
                'std_no_flip': std1,
                'acc_with_flip': acc2,
                'std_with_flip': std2,
                'xnorm': xnorm
            }
            
            logging.info(f"[{name}] Accuracy (no flip):   {acc1:.5f} ± {std1:.5f}")
            logging.info(f"[{name}] Accuracy (with flip): {acc2:.5f} ± {std2:.5f}")
            logging.info(f"[{name}] Embedding norm:       {xnorm:.5f}")
    
    logging.info("\n" + "="*80)
    logging.info("Evaluation Complete")
    logging.info("="*80)
    
    return results


def print_summary(results):
    """Print summary table of results."""
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"{'Dataset':<20} {'Acc (no flip)':<18} {'Acc (flip)':<18} {'XNorm':<10}")
    logging.info("-" * 80)
    
    for name, res in results.items():
        logging.info(
            f"{name:<20} "
            #f"{res['acc_no_flip']:.5f}±{res['std_no_flip']:.5f}   "
            #f"{res['acc_with_flip']:.5f}±{res['std_with_flip']:.5f}   "
            f"{res['acc_no_flip']:.5f}   "
            f"{res['acc_with_flip']:.5f}   "
            f"{res['xnorm']:.5f}"
        )
    
    logging.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate face recognition model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/edgeface_s_gamma_05.py)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., output/checkpoint_gpu_0.pt)')
    parser.add_argument('--val-targets', nargs='+', default=None,
                       help='Validation targets (e.g., lfw lfw_28). If not provided, uses config.val_targets')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing .bin files. If not provided, uses config.rec')
    parser.add_argument('--image-size', type=int, nargs=2, default=[112, 112],
                       help='Image size (height width), default: 112 112')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--pair-modes', nargs='+', default=None,
                       metavar='NAME:MODE1,MODE2',
                       help='Per-bin PDT routing. Format: name:mode1,mode2 '
                            '(e.g. lfw:hr,hr lfw_28_lr2lr:lr,lr lfw_28_hr2lr:hr,lr). '
                            'hr=backbone only, lr=translator+backbone. '
                            'Falls back to config.val_pair_modes when not set.')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set GPU
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            torch.cuda.set_device(args.gpu_id)
            logging.info(f"Using GPU: {args.gpu_id}")
    
    # Load config
    logging.info(f"Loading config from: {args.config}")
    cfg = get_config(args.config)
    
    # Determine validation targets
    val_targets = args.val_targets
    if val_targets is None:
        if hasattr(cfg, 'val_targets') and cfg.val_targets:
            val_targets = cfg.val_targets
        else:
            raise ValueError("No validation targets specified! Use --val-targets or set config.val_targets")
    
    logging.info(f"Validation targets: {val_targets}")
    
    # Determine data directory
    data_dir = args.data_dir
    if data_dir is None:
        if hasattr(cfg, 'rec'):
            data_dir = cfg.rec
        else:
            raise ValueError("No data directory specified! Use --data-dir or set config.rec")
    
    logging.info(f"Data directory: {data_dir}")
    
    # Image size
    image_size = tuple(args.image_size)
    logging.info(f"Image size: {image_size}")
    
    # Pair modes for PDT models: only used when explicitly passed via --pair-modes
    pair_modes = parse_pair_modes(args.pair_modes)

    # Load model
    backbone = load_model(cfg, args.checkpoint, device=args.device)
    
    # Load validation datasets
    datasets = load_validation_datasets(val_targets, data_dir, image_size)
    
    # Evaluate
    results = evaluate(backbone, datasets, device=args.device, pair_modes=pair_modes)
    
    # Print summary
    print_summary(results)
    
    logging.info("Done!")


if __name__ == '__main__':
    main()
