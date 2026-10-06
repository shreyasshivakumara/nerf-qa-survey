#!/usr/bin/env python3
"""
Scene-level (cluster) bootstrap of SRCC, KRCC, PLCC and RMSE for every metric.

Scenes are resampled with replacement (2,000 resamples, seed 42); a scene drawn
k times contributes all its videos k times. Intervals are 95% percentile
intervals. The number of resamples in which the 4-parameter logistic fit
failed (see common.correlations) is recorded per metric.

Writes <out_dir>/bootstrap_statistics.csv: mean and 95% CI per dataset,
metric and measure, and the number of failed fits.

Usage: python bootstrap.py <explicit_csv> <martin_csv> <out_dir>
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, N_BOOTSTRAP, SEED, correlations

MEASURES = ['srcc', 'krcc', 'plcc', 'rmse']


def scene_bootstrap(df, col, higher_is_better):
    np.random.seed(SEED)
    scenes = df['Scene'].unique()
    rows = {s: df[df['Scene'] == s] for s in scenes}
    dist = {k: [] for k in MEASURES}
    failed = 0
    for _ in range(N_BOOTSTRAP):
        sample = pd.concat([rows[s] for s in np.random.choice(scenes, size=len(scenes), replace=True)],
                           ignore_index=True)
        r = correlations(sample, col, higher_is_better)
        failed += not r['fit_ok']
        for k in MEASURES:
            dist[k].append(r[k])
    return {k: np.array([v for v in d if not np.isnan(v)]) for k, d in dist.items()}, failed


if __name__ == '__main__':
    explicit_csv, martin_csv, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for key, csv in [('explicit', explicit_csv), ('martin', martin_csv)]:
        df = pd.read_csv(csv)
        print(f"{key}: {len(df)} videos, {df['Scene'].nunique()} scenes", flush=True)
        for col, (name, hib, _) in METRICS.items():
            dist, failed = scene_bootstrap(df, col, hib)
            row = dict(Dataset=key, Metric=name, fit_failures=failed)
            for k, d in dist.items():
                row.update({f"{k.upper()}_mean": np.mean(d), f"{k.upper()}_ci_low": np.percentile(d, 2.5),
                            f"{k.upper()}_ci_high": np.percentile(d, 97.5)})
            stats.append(row)
            print(f"  {name:12s} SRCC {row['SRCC_mean']:.3f} [{row['SRCC_ci_low']:.3f}, {row['SRCC_ci_high']:.3f}]"
                  f"  fit failures {failed}/{N_BOOTSTRAP}", flush=True)
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, 'bootstrap_statistics.csv'), index=False)
