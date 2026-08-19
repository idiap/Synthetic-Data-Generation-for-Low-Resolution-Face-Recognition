#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: wrapper.py
#
import torch
import torch.nn.functional as F
from torch import nn


class PDT_wrapper(nn.Module):
    def __init__(self, backbone: nn.Module, translator: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.translator = translator

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode regardless of wrapper mode."""
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for _, module in self.named_modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d) or isinstance(module, torch.nn.modules.BatchNorm2d):       
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        return self

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            input_mod1, input_mod2 = x[0], x[1]
            # HR path: no gradients needed at all
            with torch.no_grad():
                emb1 = self.backbone(input_mod1)
            # LR path: gradients must flow through backbone back to translator
            emb2 = self.backbone(self.translator(input_mod2))
            return F.normalize(emb1, dim=-1), F.normalize(emb2, dim=-1)
        else:
            # Single-input inference (e.g., verification callbacks on LR images):
            # apply translator + backbone
            return F.normalize(self.backbone(self.translator(x)), dim=-1)
