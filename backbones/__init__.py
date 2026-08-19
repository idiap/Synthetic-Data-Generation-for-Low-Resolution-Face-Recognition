#
# SPDX-FileCopyrightText: © 2023-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anjith George <anjith.george@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: __init__.py
# From the EdgeFace release; extended here with the iresnet,
# PDT and native-32px network registrations.
#
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2
from .iresnet import iresnet34, iresnet50, iresnet100

import torch

def get_model(name, **kwargs):

    if name=='edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
    elif name=='edgeface_xs_q':
        model= get_timmfrv2('edgenext_x_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif  name=='edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', batchnorm=False)
    elif  name=='edgeface_base':
        return get_timmfrv2('edgenext_base', batchnorm=False)
    elif name=='edgeface_xxs_q':
        model=get_timmfrv2('edgenext_xx_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model   
    elif name=='edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
    elif name=='edgeface_s_gamma_05_PDT':
        architecture = replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
        
        from backbones.PDT.wrapper import PDT_wrapper
        from backbones.PDT.common import PDT
        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")

        POOL ='6'
        SE = 'False'
        CBAM = 'True'
        BIAS = 'False'

        translator=PDT(pool_features=int(POOL),use_se=str2bool(SE), use_bias=str2bool(BIAS), use_cbam=str2bool(CBAM))

        return PDT_wrapper(architecture, translator)
    elif name=="edgeface_s_gamma_05_vlr_32":
        model =  replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
        model.model.stem[0] = torch.nn.Conv2d(3,48, 1) #replace first downsampling layer
        return model

    elif name in ("edgeface_base_PDT_up32_espcn", "edgeface_base_PDT_up32_resrgan"):
        from backbones.PDT.wrapper import PDT_wrapper
        from backbones.PDT.common import PDT
        from backbones.PDT.upsamplers import ESPCNUpsampler, RRDBUpsampler

        architecture = get_timmfrv2('edgenext_base', batchnorm=False)

        if name == "edgeface_base_PDT_up32_espcn":
            upsampler = ESPCNUpsampler()
        else:
            upsampler = RRDBUpsampler()

        pdt = PDT(pool_features=6, use_se=False, use_bias=False, use_cbam=True)
        translator = torch.nn.Sequential(upsampler, pdt)

        return PDT_wrapper(architecture, translator)
    elif name == 'iresnet34':
        model = iresnet34()
    elif name == 'iresnet50':
        model = iresnet50()
    elif name == 'iresnet100':
        model = iresnet100()
    
    else:
        raise ValueError()
