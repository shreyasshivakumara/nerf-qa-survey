#!/usr/bin/env python3
"""
Sensitivity of the real-world SRCC confidence intervals to the resampling unit.

Compares 95% bootstrap CIs (2,000 resamples) when resampling
  - scenes   (main analysis; taken from bootstrap_statistics.csv),
  - videos   (each video treated as independent),
  - methods  (rendering method as the cluster).
A resampled unit is included once for every time it is drawn.

Usage: python make_sensitivity_table.py <martin_csv> <bootstrap_csv> <point_estimates_csv> <out_tex>
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, N_BOOTSTRAP, NAME_TO_COL, SEED

martin_csv, boot_csv, point_csv, out_tex = sys.argv[1:5]
DS = 'martin'

df = pd.read_csv(martin_csv)
df['Method'] = df['Video Name'].str.split('_', n=1).str[1]
df['Video'] = df['Video Name']


def cluster_ci(col, higher_is_better, unit):
    rng = np.random.RandomState(SEED)
    groups = {u: g for u, g in df.groupby(unit)}
    units = list(groups)
    dist = []
    for _ in range(N_BOOTSTRAP):
        sample = pd.concat([groups[u] for u in rng.choice(units, len(units), replace=True)])
        sample = sample[[col, 'MOS']].dropna()
        if len(sample) < 4 or sample[col].nunique() < 2 or sample['MOS'].nunique() < 2:
            continue
        x = sample[col] if higher_is_better else -sample[col]
        dist.append(spearmanr(x, sample['MOS'])[0])
    return np.percentile(dist, [2.5, 97.5])


boot = pd.read_csv(boot_csv).set_index(['Dataset', 'Metric'])
srcc = pd.read_csv(point_csv).set_index(['Dataset', 'Metric']).loc[DS, 'SRCC'].sort_values(ascending=False)

fmt = lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]"
rows = []
for name in srcc.index:
    col = NAME_TO_COL[name]
    hib = METRICS[col][1]
    scene = (boot.loc[(DS, name), 'SRCC_ci_low'], boot.loc[(DS, name), 'SRCC_ci_high'])
    rows.append(f"{name} & {srcc[name]:.3f} & {fmt(scene)} & {fmt(cluster_ci(col, hib, 'Video'))} & "
                f"{fmt(cluster_ci(col, hib, 'Method'))} \\\\")

tex = [r"\begin{tabular}{l|c|ccc}", r"\hline",
       r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{SRCC}} & "
       r"\multicolumn{3}{c}{\textbf{95\% CI, resampling unit}} \\",
       r"\cline{3-5}",
       r" & & \textbf{Scene} & \textbf{Video} & \textbf{Method} \\",
       r"\hline"] + rows + [r"\hline", r"\end{tabular}"]
open(out_tex, 'w').write("\n".join(tex) + "\n")
print("\n".join(rows))
