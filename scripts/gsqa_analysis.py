#!/usr/bin/env python3
"""
GS-QA pilot: SRCC/KRCC with 95% scene-bootstrap CIs (same procedure as the
paper), side by side with NVS-QA (same scenes, NeRF renderings), paired
significance of the two best metrics, rank probabilities, and a check of our
metric scores against the authors' metrics.xlsx.

Usage: python gsqa_analysis.py <gsqa_merged_csv> <martin_merged_csv> <GS-QA repo or -> <out_dir>
(pass - to skip the check against metrics.xlsx)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, N_BOOTSTRAP, SEED

gsqa_csv, martin_csv, repo, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
NAMES = [n for n, _, _ in METRICS.values()]


def signed(df, col, hib):
    return df[col] if hib else -df[col]


def boot(df):
    rng = np.random.RandomState(SEED)
    groups = {s: g for s, g in df.groupby('Scene')}
    scenes = list(groups)
    draws = np.full((N_BOOTSTRAP, len(NAMES)), np.nan)
    for b in range(N_BOOTSTRAP):
        s = pd.concat([groups[x] for x in rng.choice(scenes, len(scenes), replace=True)])
        for j, (col, (_, hib, _)) in enumerate(METRICS.items()):
            if s[col].nunique() > 1:
                draws[b, j] = spearmanr(signed(s, col, hib), s['MOS'])[0]
    return draws


res, sig, ranks = [], [], []
for label, csv in [('GS-QA', gsqa_csv), ('NVS-QA', martin_csv)]:
    df = pd.read_csv(csv)
    draws = boot(df)
    point = {}
    for j, (col, (name, hib, cat)) in enumerate(METRICS.items()):
        x = signed(df, col, hib)
        point[name] = spearmanr(x, df['MOS'])[0]
        d = draws[:, j][~np.isnan(draws[:, j])]
        res.append(dict(dataset=label, metric=name, category=cat, srcc=point[name],
                        krcc=kendalltau(x, df['MOS'])[0],
                        ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    valid = draws[~np.isnan(draws).any(axis=1)]
    rank = (-valid).argsort(axis=1).argsort(axis=1)
    for j, n in enumerate(NAMES):
        ranks.append(dict(dataset=label, metric=n, p_first=(rank[:, j] == 0).mean(), p_top3=(rank[:, j] < 3).mean()))
    order = sorted(NAMES, key=lambda n: -point[n])
    refs = list(dict.fromkeys(order[:2] + ['DISTS', 'MUSIQ']))
    alpha = 0.05 / (len(NAMES) - 1)
    for ref in refs:
        i = NAMES.index(ref)
        for other in order:
            if other == ref:
                continue
            d = draws[:, i] - draws[:, NAMES.index(other)]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            blo, bhi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            sig.append(dict(dataset=label, metric_a=ref, metric_b=other, diff=point[ref] - point[other],
                            ci_low=lo, ci_high=hi, significant=bool(lo > 0 or hi < 0),
                            significant_bonferroni=bool(blo > 0 or bhi < 0)))

res, sig, ranks = pd.DataFrame(res), pd.DataFrame(sig), pd.DataFrame(ranks)
res.to_csv(os.path.join(out_dir, 'gsqa_vs_nvsqa_srcc.csv'), index=False)
sig.to_csv(os.path.join(out_dir, 'gsqa_significance.csv'), index=False)
ranks.to_csv(os.path.join(out_dir, 'gsqa_rank_probability.csv'), index=False)

w = res.pivot(index='metric', columns='dataset', values=['srcc', 'ci_low', 'ci_high'])
w = w.sort_values(('srcc', 'GS-QA'), ascending=False)
print("SRCC [95% CI]           GS-QA (3DGS)              NVS-QA (NeRF, same scenes)")
for m, r in w.iterrows():
    print(f"  {m:12s}  {r[('srcc','GS-QA')]:.3f} [{r[('ci_low','GS-QA')]:.3f}, {r[('ci_high','GS-QA')]:.3f}]"
          f"     {r[('srcc','NVS-QA')]:.3f} [{r[('ci_low','NVS-QA')]:.3f}, {r[('ci_high','NVS-QA')]:.3f}]")
a, b = res[res.dataset == 'GS-QA'].set_index('metric').srcc, res[res.dataset == 'NVS-QA'].set_index('metric').srcc
print(f"\nrank agreement of metrics between datasets: SRCC {spearmanr(a, b.reindex(a.index))[0]:.3f}")
print("\nrank probability (GS-QA, top 5 by p_first):")
print(ranks[ranks.dataset == 'GS-QA'].sort_values('p_first', ascending=False).head(5).to_string(index=False))
for (ds, ref), g in sig.groupby(['dataset', 'metric_a'], sort=False):
    print(f"{ds} | {ref}: significantly better than {g.significant_bonferroni.sum()}/{len(g)} (Bonferroni), "
          f"{g.significant.sum()} uncorrected; not different from {g[~g.significant].metric_b.tolist()}; "
          f"significantly worse than {g[(g['diff'] < 0) & g.significant].metric_b.tolist()}")

# per viewpoint (GS-QA): 360 = Tanks and Temples, FF = IST
g = pd.read_csv(gsqa_csv)
ff = g.Scene.isin(['antique', 'flowers', 'playground2', 'statue'])
print("\nGS-QA per viewpoint SRCC (n=32 each):")
for col, (name, hib, _) in METRICS.items():
    if name in ['DISTS', 'MUSIQ', 'LPIPS+', 'FSIM', 'MS-SSIM', 'CW-SSIM', 'PSNR', 'VMAF', 'MANIQA', 'NVS-SQA']:
        print(f"  {name:10s} 360 {spearmanr(signed(g[~ff], col, hib), g[~ff].MOS)[0]:.3f}"
              f"   FF {spearmanr(signed(g[ff], col, hib), g[ff].MOS)[0]:.3f}")

# check against the authors' metric values (rank agreement across the 64 videos)
FILE = {'3DGS': '3dgs', 'EAGLES': 'eagles', 'LightGaussian': 'lightgaussian', 'Mip-Splatting': 'mipsplatting',
        'Octree-GS': 'octree', 'Scaffold-GS': 'scaffold', 'SOG (w/o SH)': 'sog', 'SOG': 'sog_sh'}
if repo == '-':
    sys.exit(0)
au = pd.read_excel(os.path.join(repo, 'metrics.xlsx'))
au['Video Name'] = au.Scene + '_' + au.Method.map(FILE)
m = g.merge(au, on='Video Name')
print(f"\ncheck vs authors' metrics.xlsx ({len(m)} videos matched), SRCC between our and their values:")
for ours, theirs in [('psnry', 'PSNR-Y'), ('ssim', 'SSIM'), ('ms_ssim', 'MS-SSIM'), ('fsim', 'FSIM'),
                     ('vif', 'VIF'), ('lpips', 'LPIPS'), ('dists', 'DISTS'), ('vmaf', 'VMAF'),
                     ('fovvideovdp', 'FVVDP')]:
    print(f"  {theirs:8s} {spearmanr(m[ours], m[theirs])[0]:.3f}")
