#!/usr/bin/env python3
"""
CGVQM with mean vs max patch pooling: SRCC with the subjective scores.

The max-pooled scores come from
  compute_metrics.py cgvqm <videos> <names> <out_dir> --cgvqm-pool max
and the mean-pooled scores (main analysis) from the merged data.

Usage: python cgvqm_pooling.py <merged_csv> <cgvqm_max_csv> <label>
"""

import sys

import pandas as pd
from scipy.stats import spearmanr

merged_csv, max_csv, label = sys.argv[1:4]
df = pd.read_csv(merged_csv)
mx = pd.read_csv(max_csv).pivot(index='video', columns='metric', values='score')
df = df.join(mx, on='Video Name', rsuffix='_max')
for col in ['cgvqm_2', 'cgvqm_5']:
    print(f"[{label}] {col}  mean pooling SRCC {spearmanr(df[col], df.MOS)[0]:.3f}"
          f"   max pooling SRCC {spearmanr(df[col + '_max'], df.MOS)[0]:.3f}")
