#!/usr/bin/env python3
"""
Paired scene-level bootstrap test of SRCC differences between metrics.

For each of 2,000 resamples, scenes are drawn with replacement (a scene drawn
k times contributes its videos k times) and the SRCC of every metric is
computed on the same resampled set. The two best metrics of each dataset are
compared with every other metric; a difference is significant when its 95%
percentile interval excludes zero, and significant after Bonferroni
correction when the corresponding (1 - 0.05/m) interval does (m = number of
comparisons per reference metric). Also reports how often each metric ranks
first and in the top three across resamples.

Writes <out_csv> and <out_csv>_rank_probability.csv.

Usage: python make_significance_table.py <explicit_csv> <martin_csv> <out_csv>
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, METRICS, N_BOOTSTRAP, SEED

explicit_csv, martin_csv, out_csv = sys.argv[1:4]


def bootstrap_srcc(df):
    rng = np.random.RandomState(SEED)
    groups = {s: g for s, g in df.groupby('Scene')}
    scenes = list(groups)
    names = [name for name, _, _ in METRICS.values()]
    draws = np.full((N_BOOTSTRAP, len(names)), np.nan)
    for b in range(N_BOOTSTRAP):
        sample = pd.concat([groups[s] for s in rng.choice(scenes, len(scenes), replace=True)])
        for j, (col, (_, hib, _)) in enumerate(METRICS.items()):
            s = sample[[col, 'MOS']].dropna()
            if len(s) >= 4 and s[col].nunique() > 1:
                draws[b, j] = spearmanr(s[col] if hib else -s[col], s['MOS'])[0]
    point = {}
    for col, (name, hib, _) in METRICS.items():
        s = df[[col, 'MOS']].dropna()
        point[name] = spearmanr(s[col] if hib else -s[col], s['MOS'])[0]
    return names, draws, point


rows, ranks = [], []
for key, csv in [('explicit', explicit_csv), ('martin', martin_csv)]:
    names, draws, point = bootstrap_srcc(pd.read_csv(csv))
    order = sorted(names, key=lambda n: -point[n])
    valid = draws[~np.isnan(draws).any(axis=1)]
    rank = (-valid).argsort(axis=1).argsort(axis=1)          # 0 = best
    for j, n in enumerate(names):
        ranks.append(dict(dataset=DATASETS[key], metric=n, p_first=(rank[:, j] == 0).mean(),
                          p_top3=(rank[:, j] < 3).mean()))
    for ref in order[:2]:
        i = names.index(ref)
        alpha = 0.05 / (len(order) - 1)
        for other in order:
            if other == ref:
                continue
            d = draws[:, i] - draws[:, names.index(other)]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            blo, bhi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            rows.append(dict(dataset=DATASETS[key], metric_a=ref, metric_b=other,
                             srcc_a=point[ref], srcc_b=point[other], diff=point[ref] - point[other],
                             ci_low=lo, ci_high=hi, significant=bool(lo > 0 or hi < 0),
                             significant_bonferroni=bool(blo > 0 or bhi < 0)))

out = pd.DataFrame(rows)
out.to_csv(out_csv, index=False)
pd.DataFrame(ranks).to_csv(out_csv.replace('.csv', '_rank_probability.csv'), index=False)
for (ds, a), g in out.groupby(['dataset', 'metric_a'], sort=False):
    print(f"{ds} | {a}: better than {g.significant.sum()}/{len(g)} "
          f"({g.significant_bonferroni.sum()} after Bonferroni); "
          f"not significant vs {g[~g.significant].metric_b.tolist()}")
