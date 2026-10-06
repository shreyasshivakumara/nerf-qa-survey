#!/usr/bin/env python3
"""
Join the per-video metric scores with the subjective scores.

Rows: the videos in the score list (MOS.xlsx of Explicit-NeRF-QA, or the CSV
written by prepare_martin_videos.py), which must have the columns dis_name,
ref_name and MOS. Columns: Scene, Video Name, MOS and one column per metric.
Every value is joined on the video name; the script stops if any video is
missing a metric or a score appears twice.

Usage: python build_merged.py <metrics_dir> <score_list> <out_csv>
"""

import glob
import os
import sys

import pandas as pd

metrics_dir, score_list, out_csv = sys.argv[1:4]

METRIC_IDS = ['psnr', 'psnry', 'ssim', 'cw_ssim', 'ms_ssim', 'fsim', 'lpips', 'lpips-vgg', 'dists', 'vif',
              'lpips+', 'niqe', 'piqe', 'maniqa', 'musiq', 'vmaf', 'fovvideovdp', 'flip', 'nvs_sqa',
              'strred', 'cgvqm_2', 'cgvqm_5']

mos = (pd.read_excel if score_list.endswith('.xlsx') else pd.read_csv)(score_list)
out = pd.DataFrame({'Scene': mos['ref_name'], 'Video Name': mos['dis_name'], 'MOS': mos['MOS']})

scores = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(metrics_dir, '*.csv'))))
dup = scores[scores.duplicated(['video', 'metric'], keep=False)]
if len(dup):
    sys.exit(f"duplicate scores:\n{dup.head()}")
wide = scores.pivot(index='video', columns='metric', values='score')

missing = set(METRIC_IDS) - set(wide.columns)
if missing:
    sys.exit(f"metrics with no scores: {sorted(missing)}")
wide = wide.reindex(out['Video Name'])[METRIC_IDS]
if wide.isna().any().any():
    sys.exit(f"missing scores: {wide.isna().sum()[wide.isna().sum() > 0].to_dict()}")

out[METRIC_IDS] = wide.values
out.to_csv(out_csv, index=False)
print(f"{len(out)} videos, {out.Scene.nunique()} scenes, {len(METRIC_IDS)} metrics -> {out_csv}")
