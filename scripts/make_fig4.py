#!/usr/bin/env python3
"""
Fig. 4: bootstrap SRCC distributions per metric, one panel per dataset.

The violins are drawn from the bootstrap samples saved by bootstrap.py, the
label above each violin is the full-data SRCC from Table 5, and the y-axis
extends below zero so negative intervals are visible.

Usage: python make_fig4.py <bootstrap_dir> <point_estimates_csv> <output_prefix>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

boot_dir, point_csv, out_prefix = sys.argv[1:4]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

CATEGORY = {m: 'FR' for m in ['PSNR', 'SSIM', 'MS-SSIM', 'CW-SSIM', 'FSIM', 'VIF', 'LPIPS', 'LPIPS-VGG',
                              'LPIPS+', 'DISTS', 'FLIP', 'VMAF', 'FovVideoVDP', 'STRRED']}
CATEGORY.update({m: 'NR' for m in ['NIQE', 'PIQE', 'MANIQA', 'MUSIQ']})
CATEGORY.update({m: 'NeRF' for m in ['CGVQM-2', 'CGVQM-5', 'NVS-SQA']})
COLORS = {'FR': '#5BA3CF', 'NR': '#E89AC7', 'NeRF': '#FFB84D'}

points = pd.read_csv(point_csv).set_index(['Dataset', 'Metric'])['SRCC']

panels = [('explicit', 'explicit', 'Explicit-NeRF-QA', (0.0, 1.12), [0.0, 0.5, 1.0]),
          ('martin', 'martin', 'NVS-QA', (-1.0, 1.18), [-1.0, -0.5, 0.0, 0.5, 1.0])]

fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), gridspec_kw={'height_ratios': [1, 1.35]})

for ax, (tag, ds, title, ylim, yticks) in zip(axes, panels):
    samples = np.load(f"{boot_dir}/bootstrap_samples_{tag}.npz")
    srcc = points.loc[ds].sort_values(ascending=False)
    metrics = list(srcc.index)
    data = [samples[m] for m in metrics]

    parts = ax.violinplot(data, positions=range(len(metrics)), widths=0.7,
                          showextrema=False, showmeans=False, showmedians=False)
    for pc, m in zip(parts['bodies'], metrics):
        pc.set_facecolor(COLORS[CATEGORY[m]])
        pc.set_alpha(0.55)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.6)

    for i, (m, d) in enumerate(zip(metrics, data)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        ax.vlines(i, lo, hi, color='black', linewidth=0.8)                 # 95% CI
        ax.hlines(srcc[m], i - 0.22, i + 0.22, color='black', linewidth=1.2)  # full-data SRCC
        ax.text(i, min(hi + 0.04, ylim[1] - 0.10), f"{srcc[m]:.2f}", ha='center', va='bottom',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    ax.axhline(0.0, color='grey', linewidth=0.6)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('SRCC', fontsize=10, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=9, pad=4)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.6, len(metrics) - 0.4)

legend = [Patch(facecolor=COLORS[k], alpha=0.7, edgecolor='black', label=l)
          for k, l in [('FR', 'Full-Reference'), ('NR', 'No-Reference'), ('NeRF', 'NeRF-Specific')]]
fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.03))
plt.tight_layout()
plt.savefig(f"{out_prefix}.pdf", bbox_inches='tight')
plt.savefig(f"{out_prefix}.png", dpi=200, bbox_inches='tight')
print(f"saved {out_prefix}.pdf/.png")
