#!/usr/bin/env python3
"""
Simple grouped bars: SRCC of 11 metrics (incl. NeRF-specific) on NVS-QA (NeRF) vs GS-QA (3DGS), same 8 scenes.
Metrics sorted by NVS-QA SRCC. Error bars: 95% scene-level bootstrap CIs.

Usage: python make_fig_gs_vs_nerf.py <gsqa_vs_nvsqa_srcc.csv> <output_prefix>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv, out_prefix = sys.argv[1:3]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8

METRICS = ['PSNR', 'SSIM', 'LPIPS', 'DISTS', 'VMAF', 'STRRED', 'MUSIQ', 'MANIQA', 'CGVQM-2', 'CGVQM-5', 'NVS-SQA']
COLORS = {'NVS-QA': '#5BA3CF', 'GS-QA': '#E8743B'}
LABELS = {'NVS-QA': 'NeRF (NVS-QA)', 'GS-QA': '3DGS (GS-QA)'}

df = pd.read_csv(csv)
df = df[df.metric.isin(METRICS)]
srcc = df.pivot(index='metric', columns='dataset', values='srcc').sort_values('NVS-QA', ascending=False)
lo = df.pivot(index='metric', columns='dataset', values='ci_low').loc[srcc.index]
hi = df.pivot(index='metric', columns='dataset', values='ci_high').loc[srcc.index]

x = np.arange(len(srcc))
w = 0.44
fig, ax = plt.subplots(figsize=(4.2, 2.6))
for k, ds in enumerate(['NVS-QA', 'GS-QA']):
    yerr = [srcc[ds] - lo[ds], hi[ds] - srcc[ds]]
    ax.bar(x + (k - 0.5) * w, srcc[ds], w, color=COLORS[ds], edgecolor='black', linewidth=0.4, label=LABELS[ds],
           yerr=yerr, error_kw=dict(elinewidth=0.7, capsize=1.5, capthick=0.7, ecolor='black'))

    # CIs extending below 0 are cut at the axis; mark them with a small arrow
    for xi, l in zip(x + (k - 0.5) * w, lo[ds]):
        if l < 0:
            ax.annotate('', xy=(xi, 0.0), xytext=(xi, 0.06), arrowprops=dict(arrowstyle='-|>', lw=0.7,
                        color='black', mutation_scale=6), annotation_clip=False)

ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xlim(-0.6, len(srcc) - 0.4)
ax.set_xticklabels(srcc.index, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel('SRCC', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(f"{out_prefix}.pdf", bbox_inches='tight')
plt.savefig(f"{out_prefix}.png", dpi=250, bbox_inches='tight')
print(f"saved {out_prefix}.pdf/.png")
