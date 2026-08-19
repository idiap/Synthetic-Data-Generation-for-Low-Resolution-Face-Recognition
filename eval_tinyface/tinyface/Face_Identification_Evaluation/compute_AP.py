#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: compute_AP.py
# Python port of compute_AP.m from the TinyFace identification protocol
# (https://qmul-tinyface.github.io/). The original MATLAB code and the .mat
# pair files ship with the TinyFace dataset and carry no explicit license.
#
import numpy as np


def compute_AP(good_image, index):
    cmc = np.zeros(len(index))
    ngood = len(good_image)
    
    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    
    for n in range(len(index)):
        flag = 0
        if index[n] in good_image:
            cmc[n:] = 1
            flag = 1
            good_now = good_now + 1
        
        if flag == 1:
            intersect_size = intersect_size + 1
        
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j = j + 1
        
        if good_now == ngood:
            return ap, cmc
    
    return ap, cmc