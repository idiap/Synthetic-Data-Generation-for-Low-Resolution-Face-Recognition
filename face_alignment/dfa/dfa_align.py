#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: dfa_align.py
#
"""
Load CVLFace Differentiable Face Aligner from local HF-style clones (no transformers / OmegaConf).

Expects repos under ``face_alignment/dfa/cvlface_DFA_mobilenet`` and
``face_alignment/dfa/cvlface_DFA_resnet50`` with ``pretrained_model/model.{yaml,pt}``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
import torch
import yaml

_DFA_ROOT = Path(__file__).resolve().parent
_REPO_DIR = {
    "mobilenet": _DFA_ROOT / "cvlface_DFA_mobilenet",
    "resnet50": _DFA_ROOT / "cvlface_DFA_resnet50",
}


def _dict_to_namespace(obj):
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    return obj


def _ensure_repo_on_path(repo_root: Path) -> None:
    s = str(repo_root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def load_dfa_model(
    backbone: Literal["mobilenet", "resnet50"],
    device: Union[str, torch.device],
) -> torch.nn.Module:
    """
    Build DFA from ``model.yaml`` + ``model.pt`` in the matching clone directory.
    """
    if backbone not in _REPO_DIR:
        raise ValueError(f"Unknown DFA backbone: {backbone!r} (use 'mobilenet' or 'resnet50')")

    repo_root = _REPO_DIR[backbone]
    pretrained = repo_root / "pretrained_model"
    yaml_path = pretrained / "model.yaml"
    pt_path = pretrained / "model.pt"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing {yaml_path}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing {pt_path}")

    _ensure_repo_on_path(repo_root)

    # Import after sys.path so `aligners` resolves to this clone only.
    import aligners.differentiable_face_aligner.dfa.config as dfa_pt_config  # noqa: E402

    if backbone == "resnet50":
        dfa_pt_config.cfg_re50["pretrain"] = False

    from aligners import get_aligner  # noqa: E402

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_ns = _dict_to_namespace(cfg_dict)
    model = get_aligner(cfg_ns)
    model.load_state_dict_from_path(str(pt_path))
    model = model.to(device)
    model.eval()
    return model


def bgr_frame_to_tensor_rgb_normalized(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """OpenCV BGR uint8 -> batch RGB float in [-1, 1], shape (1, 3, H, W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    return x.unsqueeze(0)


def aligned_output_to_rgb_uint8(aligned_x: torch.Tensor) -> np.ndarray:
    """First batch element (1,3,H,W) float ~[-1,1] -> HWC uint8 RGB."""
    t = aligned_x[0].detach().float().cpu().clamp(-1.0, 1.0)
    out = ((t + 1.0) * 0.5 * 255.0).round().to(torch.uint8).numpy()
    return np.transpose(out, (1, 2, 0))


def align_bgr_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """
    Run DFA on one BGR frame; return aligned face RGB uint8 (H, W, 3) or None on failure.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if device is None:
        device = next(model.parameters()).device
    x = bgr_frame_to_tensor_rgb_normalized(frame_bgr, device)
    try:
        with torch.inference_mode():
            aligned_x, *_rest = model(x)
    except Exception:
        return None
    if aligned_x is None or aligned_x.numel() == 0:
        return None
    return aligned_output_to_rgb_uint8(aligned_x)


def _detect_landmarks_orig_pixels(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Run only the DFA landmark predictor (still at the network's native 160x160) and
    return 5 landmarks in pixel coordinates of the *original* frame. Shape: (5, 2).

    The net outputs landmarks normalized to the square-padded preprocessor input;
    we undo the square-pad to land in original-frame pixels.
    """
    # Imported lazily because the cvlface repo is only on sys.path after load_dfa_model().
    from aligners.differentiable_face_aligner import aligner_helper  # noqa: WPS433
    H0, W0 = frame_bgr.shape[:2]
    x = bgr_frame_to_tensor_rgb_normalized(frame_bgr, device)
    pre = model.preprocessor(x)
    try:
        with torch.inference_mode():
            net_out = model.net(pre.flip(1), model.prior_box)  # net expects BGR
    except Exception:
        return None
    ldmks_norm, _bbox, _cls = aligner_helper.split_network_output(net_out)
    if ldmks_norm is None or ldmks_norm.numel() == 0:
        return None
    ld = ldmks_norm.view(-1, 5, 2)[0].detach().cpu().numpy().astype(np.float32)
    square = float(max(H0, W0))
    pad_left = (max(H0, W0) - W0) / 2.0
    pad_top = (max(H0, W0) - H0) / 2.0
    ld = ld * square
    ld[:, 0] -= pad_left
    ld[:, 1] -= pad_top
    return ld


def align_bgr_frame_lowres(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    output_size: int = 32,
    max_input_size: Optional[int] = 32,
    interp_down: int = cv2.INTER_AREA,
    interp_up: int = cv2.INTER_CUBIC,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """
    Low-resolution DFA alignment that avoids the 160x160 / 112x112 upsampling detour.

    Detects landmarks at the network's native input, then warps the original frame
    (optionally pre-resized so its longer side <= ``max_input_size``) directly to
    ``output_size x output_size`` using the standard 112-px reference, scaled.

    Returns aligned face RGB uint8 (output_size, output_size, 3) or None on failure.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if device is None:
        device = next(model.parameters()).device

    from aligners.differentiable_face_aligner import aligner_helper  # noqa: WPS433
    from skimage import transform as trans  # noqa: WPS433

    ld_orig = _detect_landmarks_orig_pixels(model, frame_bgr, device)
    if ld_orig is None:
        return None

    H0, W0 = frame_bgr.shape[:2]
    src = frame_bgr
    sx = sy = 1.0
    if max_input_size is not None and max(H0, W0) > max_input_size:
        s = max_input_size / float(max(H0, W0))
        new_w = max(1, int(round(W0 * s)))
        new_h = max(1, int(round(H0 * s)))
        src = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp_down)
        sx = new_w / float(W0)
        sy = new_h / float(H0)
    ld_src = ld_orig * np.array([[sx, sy]], dtype=np.float32)

    ref = aligner_helper.reference_landmark().astype(np.float32) * (output_size / 112.0)

    tform = trans.SimilarityTransform()
    if not tform.estimate(ld_src, ref):
        return None
    M = tform.params[:2, :].astype(np.float32)

    det = float(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    flag = interp_up if det > 1.0 else interp_down

    aligned_bgr = cv2.warpAffine(
        src, M, (output_size, output_size), flags=flag, borderValue=0,
    )
    return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)


def aligner_variant_to_backbone(aligner_cli: str) -> Literal["mobilenet", "resnet50"]:
    if aligner_cli == "dfa-mobilenet":
        return "mobilenet"
    if aligner_cli == "dfa-resnet50":
        return "resnet50"
    raise ValueError(f"Not a DFA aligner CLI value: {aligner_cli!r}")
