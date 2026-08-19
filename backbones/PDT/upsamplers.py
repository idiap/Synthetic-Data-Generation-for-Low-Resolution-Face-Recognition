#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: upsamplers.py
#
"""
Fully-learned 32x32 -> 112x112 upsamplers prepended to the PDT translator.

Spatial path (shared by all variants):
    (3, 32, 32)
        -> conv stem (3 -> C, k=3, p=1)
        -> N feature blocks @ 32x32         (variant-specific)
        -> conv (C -> 4C, k=3, p=1) + PixelShuffle(2)   -> (C, 64, 64)
        -> M feature blocks @ 64x64         (variant-specific)
        -> conv (C -> 4C, k=3, p=1) + PixelShuffle(2)   -> (C, 128, 128)
        -> 4 x Conv(C -> C, k=5, p=0)       -> 128 -> 124 -> 120 -> 116 -> 112
        -> final Conv(C -> 3, k=3, p=1)     -> (3, 112, 112)

No spatial interpolation: every transition is sub-pixel or convolution.
"""

from typing import Optional

import torch
from torch import nn


class _SubPixelUp2(nn.Module):
    """Conv to 4*C channels followed by PixelShuffle(2). Doubles spatial size."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class _ValidShrinkStack(nn.Module):
    """Four valid-padded k=5 convs cumulatively shrink 128 -> 112 (4 px / conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Variant B: ESPCN-lite (plain Conv->ReLU feature blocks)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ESPCNUpsampler(nn.Module):
    """Lightweight fully-learned 32x32 -> 112x112 upsampler (~70k params at C=64)."""

    def __init__(self, channels: int = 64, n_blocks_low: int = 2, n_blocks_mid: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks_32 = nn.Sequential(*[_ConvBlock(channels) for _ in range(n_blocks_low)])
        self.up_32_to_64 = _SubPixelUp2(channels)
        self.blocks_64 = nn.Sequential(*[_ConvBlock(channels) for _ in range(n_blocks_mid)])
        self.up_64_to_128 = _SubPixelUp2(channels)
        self.shrink_128_to_112 = _ValidShrinkStack(channels)
        self.head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks_32(x)
        x = self.up_32_to_64(x)
        x = self.blocks_64(x)
        x = self.up_64_to_128(x)
        x = self.shrink_128_to_112(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Variant C: RRDB-based (Real-ESRGAN-style) upsampler
# ---------------------------------------------------------------------------

class _ResidualDenseBlock(nn.Module):
    """5-conv dense block with residual scaling (the inner block of RRDB)."""

    def __init__(self, channels: int, growth: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 1 * growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.residual_scale


class _RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three RDBs with outer residual."""

    def __init__(self, channels: int, growth: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = _ResidualDenseBlock(channels, growth, residual_scale)
        self.rdb2 = _ResidualDenseBlock(channels, growth, residual_scale)
        self.rdb3 = _ResidualDenseBlock(channels, growth, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.residual_scale


class RRDBUpsampler(nn.Module):
    """Real-ESRGAN-style fully-learned 32x32 -> 112x112 upsampler (~3-4M params at default sizes)."""

    def __init__(self, channels: int = 64, growth: int = 32, n_rrdb_low: int = 4, n_rrdb_mid: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.trunk_32 = nn.Sequential(
            *[_RRDB(channels, growth) for _ in range(n_rrdb_low)]
        )
        self.trunk_32_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.up_32_to_64 = _SubPixelUp2(channels)
        self.trunk_64 = nn.Sequential(
            *[_RRDB(channels, growth) for _ in range(n_rrdb_mid)]
        )
        self.trunk_64_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.up_64_to_128 = _SubPixelUp2(channels)
        self.shrink_128_to_112 = _ValidShrinkStack(channels)
        self.head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        trunk_out = self.trunk_32_conv(self.trunk_32(x))
        x = x + trunk_out
        x = self.up_32_to_64(x)
        trunk_out = self.trunk_64_conv(self.trunk_64(x))
        x = x + trunk_out
        x = self.up_64_to_128(x)
        x = self.shrink_128_to_112(x)
        x = self.head(x)
        return x


def load_upsampler_checkpoint(upsampler: nn.Module, ckpt_path: Optional[str]) -> None:
    """Load Stage-1 SR-pretrained weights into a fresh upsampler."""
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    upsampler.load_state_dict(state)
