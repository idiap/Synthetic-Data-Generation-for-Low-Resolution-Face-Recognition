#
# SPDX-FileCopyrightText: Copyright 2018-2022 BasicSR Authors
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
# Script: degradations.py
# Ported from BasicSR (basicsr/data/degradations.py, Apache-2.0,
# https://github.com/XPixelGroup/BasicSR) and the Real-ESRGAN training
# pipeline (https://github.com/xinntao/Real-ESRGAN), without the full
# BasicSR dependency.
#
"""Vendored degradation utilities for Real-ESRGAN-style stochastic image
degradation.

Ported from basicsr.data.degradations and Real-ESRGAN's training data
pipeline so we don't pull in the full BasicSR stack (which conflicts with
this repo's pinned scipy<=1.8.1).

Reference:
  Wang et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution
  with Pure Synthetic Data", ICCVW 2021.
  https://github.com/xinntao/Real-ESRGAN
"""
import math

import cv2
import numpy as np
from scipy import special


# ---------------------------------------------------------------------------
# Kernel generators
# ---------------------------------------------------------------------------

def _mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return xy


def _sigma_matrix2(sig_x, sig_y, theta):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return U @ D @ U.T


def bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True):
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = _sigma_matrix2(sig_x, sig_y, theta)
    xy = _mesh_grid(kernel_size)
    inv = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(xy @ inv * xy, axis=1))
    kernel = kernel.reshape(kernel_size, kernel_size)
    return kernel / kernel.sum()


def bivariate_generalized_gaussian(kernel_size, sig_x, sig_y, theta, beta,
                                   isotropic=True):
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = _sigma_matrix2(sig_x, sig_y, theta)
    xy = _mesh_grid(kernel_size)
    inv = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(xy @ inv * xy, axis=1), beta))
    kernel = kernel.reshape(kernel_size, kernel_size)
    return kernel / kernel.sum()


def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = _sigma_matrix2(sig_x, sig_y, theta)
    xy = _mesh_grid(kernel_size)
    inv = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(xy @ inv * xy, axis=1), beta) + 1)
    kernel = kernel.reshape(kernel_size, kernel_size)
    return kernel / kernel.sum()


def circular_lowpass(kernel_size, cutoff):
    """2D circular sinc lowpass kernel."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    c = (kernel_size - 1) / 2

    def _val(x, y):
        r = np.sqrt((x - c) ** 2 + (y - c) ** 2) + 1e-12
        return cutoff * special.j1(cutoff * r) / (2 * np.pi * r)

    kernel = np.fromfunction(_val, [kernel_size, kernel_size])
    kernel[int(c), int(c)] = cutoff ** 2 / (4 * np.pi)
    return kernel / kernel.sum()


def random_kernel(rng, kernel_list, kernel_prob, kernel_range,
                  sigma_range, betag_range, betap_range,
                  sinc_prob, sinc_range):
    kernel_size = int(rng.choice(kernel_range))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if rng.uniform() < sinc_prob:
        omega = rng.uniform(sinc_range[0], sinc_range[1])
        return circular_lowpass(kernel_size, omega)

    probs = np.array(kernel_prob, dtype=np.float64)
    probs /= probs.sum()
    kernel_type = rng.choice(kernel_list, p=probs)
    rotation = rng.uniform(-math.pi, math.pi)
    sx = rng.uniform(*sigma_range)
    sy = rng.uniform(*sigma_range)

    if kernel_type == 'iso':
        return bivariate_gaussian(kernel_size, sx, sx, 0, True)
    if kernel_type == 'aniso':
        return bivariate_gaussian(kernel_size, sx, sy, rotation, False)
    if kernel_type == 'generalized_iso':
        beta = rng.uniform(*betag_range)
        return bivariate_generalized_gaussian(kernel_size, sx, sx, 0, beta, True)
    if kernel_type == 'generalized_aniso':
        beta = rng.uniform(*betag_range)
        return bivariate_generalized_gaussian(kernel_size, sx, sy, rotation, beta, False)
    if kernel_type == 'plateau_iso':
        beta = rng.uniform(*betap_range)
        return bivariate_plateau(kernel_size, sx, sx, 0, beta, True)
    if kernel_type == 'plateau_aniso':
        beta = rng.uniform(*betap_range)
        return bivariate_plateau(kernel_size, sx, sy, rotation, beta, False)
    raise ValueError(f"Unknown kernel type: {kernel_type}")


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(img, sigma, gray_prob, rng):
    if rng.uniform() < gray_prob:
        noise = rng.standard_normal(img.shape[:2]).astype(np.float32) * sigma
        return np.clip(img + noise[..., None], 0, 255)
    noise = rng.standard_normal(img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0, 255)


def add_poisson_noise(img, scale, gray_prob, rng):
    img_clip = np.clip(img / 255.0, 0, 1).astype(np.float32)
    vals = max(2 ** np.ceil(np.log2(max(len(np.unique(img_clip)), 1))), 16)
    if rng.uniform() < gray_prob:
        gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
        noisy = rng.poisson(gray * vals) / vals - gray
        return np.clip(img + noisy[..., None] * scale * 255, 0, 255)
    noisy = rng.poisson(img_clip * vals) / vals - img_clip
    return np.clip(img + noisy * scale * 255, 0, 255)


# ---------------------------------------------------------------------------
# JPEG cycle
# ---------------------------------------------------------------------------

def jpeg_cycle(img, quality):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(
        '.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(img.dtype)


# ---------------------------------------------------------------------------
# Two-stage Real-ESRGAN pipeline
# ---------------------------------------------------------------------------

# Defaults from the Real-ESRGAN x4plus training config:
# https://github.com/xinntao/Real-ESRGAN/blob/master/options/train_realesrgan_x4plus.yml
DEFAULTS = dict(
    kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                 'plateau_iso', 'plateau_aniso'],
    kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    kernel_range=list(range(7, 22, 2)),
    sigma_range=(0.2, 3.0),
    betag_range=(0.5, 4.0),
    betap_range=(1.0, 2.0),
    sinc_prob=0.1,
    sinc_range=(math.pi / 3, math.pi),
    gaussian_sigma_range=(1.0, 30.0),
    poisson_scale_range=(0.05, 3.0),
    gray_noise_prob=0.4,
    jpeg_range=(30, 95),
    resize_range=(0.15, 1.5),

    # Second-order (typically weaker)
    kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    sigma_range2=(0.2, 1.5),
    sinc_prob2=0.1,
    gaussian_sigma_range2=(1.0, 25.0),
    poisson_scale_range2=(0.05, 2.5),
    jpeg_range2=(30, 95),
    resize_range2=(0.3, 1.2),
    second_order_prob=0.8,

    final_sinc_prob=0.8,
    final_sinc_range=(math.pi / 3, math.pi),
)


def _filter2d(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def _random_resize(img, scale, rng, interp_choices=None):
    interp_choices = interp_choices or [
        cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
    interp = int(rng.choice(interp_choices))
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def realesrgan_degradation(img, out_size, rng, params=None):
    """Apply Real-ESRGAN second-order degradation, ending at out_size x out_size.

    Args:
        img: HxWx3 BGR uint8 array.
        out_size: final spatial size (square).
        rng: numpy.random.Generator.
        params: optional dict overriding entries in DEFAULTS.

    Returns:
        out_size x out_size x 3 uint8 BGR array.
    """
    p = {**DEFAULTS, **(params or {})}
    img = img.astype(np.float32)

    # First-order: blur → resize → noise → jpeg
    k1 = random_kernel(
        rng, p['kernel_list'], p['kernel_prob'], p['kernel_range'],
        p['sigma_range'], p['betag_range'], p['betap_range'],
        p['sinc_prob'], p['sinc_range'])
    img = _filter2d(img, k1)
    img = _random_resize(img, rng.uniform(*p['resize_range']), rng)
    if rng.uniform() < 0.5:
        img = add_gaussian_noise(
            img, rng.uniform(*p['gaussian_sigma_range']),
            p['gray_noise_prob'], rng)
    else:
        img = add_poisson_noise(
            img, rng.uniform(*p['poisson_scale_range']),
            p['gray_noise_prob'], rng)
    img = jpeg_cycle(img, int(rng.integers(p['jpeg_range'][0], p['jpeg_range'][1] + 1)))

    # Second-order
    if rng.uniform() < p['second_order_prob']:
        k2 = random_kernel(
            rng, p['kernel_list'], p['kernel_prob2'], p['kernel_range'],
            p['sigma_range2'], p['betag_range'], p['betap_range'],
            p['sinc_prob2'], p['sinc_range'])
        img = _filter2d(img, k2)
    img = _random_resize(img, rng.uniform(*p['resize_range2']), rng)
    if rng.uniform() < 0.5:
        img = add_gaussian_noise(
            img, rng.uniform(*p['gaussian_sigma_range2']),
            p['gray_noise_prob'], rng)
    else:
        img = add_poisson_noise(
            img, rng.uniform(*p['poisson_scale_range2']),
            p['gray_noise_prob'], rng)

    # Final resize to target + optional sinc + final JPEG
    interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    img = cv2.resize(img, (out_size, out_size), interpolation=interp)
    if rng.uniform() < p['final_sinc_prob']:
        ks = int(rng.choice(p['kernel_range']))
        if ks % 2 == 0:
            ks += 1
        omega = rng.uniform(*p['final_sinc_range'])
        img = _filter2d(img, circular_lowpass(ks, omega))
    img = jpeg_cycle(
        img, int(rng.integers(p['jpeg_range2'][0], p['jpeg_range2'][1] + 1)))

    return np.clip(img, 0, 255).astype(np.uint8)
