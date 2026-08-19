#
# SPDX-FileCopyrightText: Copyright (c) 2022 Jiankang Deng and Jia Guo
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: eval_ijbc_print_only.py
# Modified from InsightFace (https://github.com/deepinsight/insightface, MIT);
# see LICENSES/MIT.txt. Changes: score-file replot without feature extraction; env-var data root.
#
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import numpy as np
import matplotlib
import pandas as pd
from sklearn.metrics import roc_curve, auc
import os
import argparse

import sys
import warnings
import glob2

sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')

parser.add_argument('--image-path',
                    default=os.environ.get('IJBC_ROOT', 'data/IJBC'), type=str,
                    help='Root of the IJB-C distribution. Defaults to $IJBC_ROOT.')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--score-save-file', default='ijbc.npy', type=str)


args = parser.parse_args()


image_path = args.image_path
result_dir = args.result_dir
save_path = result_dir

if os.path.isdir(args.score_save_file):
    files = glob2.glob(os.path.join(args.score_save_file,'*.npy'))
else:
    files = [args.score_save_file]

def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int32)
    t2 = pairs[:, 1].astype(np.int32)
    label = pairs[:, 2].astype(np.int32)
    return t1, t2, label

p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % 'IJBC'.lower()))

def save_threshold_json(method, threshold_values):
    import json
    with open(f'{method}-ijbc.json', 'w') as f:
        json.dump(threshold_values, f)

matplotlib.use('Agg')
import matplotlib.pyplot as plt



#files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    best_thresholds = {}
    fpr, tpr, thresholds = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, 'IJBC'))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        best_thresholds[x_labels[fpr_iter]] = thresholds[min_index]
    save_threshold_json(os.path.join(save_path,method),best_thresholds)
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % 'ijbc'.lower()))
print(tpr_fpr_table)