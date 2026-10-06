#!/usr/bin/env python3
"""
Summarise the runtime benchmark (run_benchmark.sh) into a LaTeX table:
mean +/- std seconds per video (300 frames, 800x800), ms per frame, peak GPU memory, device.

Usage: python make_runtime_table.py <benchmark_dir> <out_tex>
"""

import glob
import os
import sys

import pandas as pd

bench_dir, out_tex = sys.argv[1:3]
d = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(bench_dir, "*.csv")))

NAMES = {'psnr': 'PSNR', 'ssim': 'SSIM', 'ms_ssim': 'MS-SSIM', 'cw_ssim': 'CW-SSIM', 'fsim': 'FSIM',
         'vif': 'VIF', 'lpips': 'LPIPS', 'lpips-vgg': 'LPIPS-VGG', 'lpips+': 'LPIPS+', 'dists': 'DISTS',
         'niqe': 'NIQE', 'piqe': 'PIQE', 'maniqa': 'MANIQA', 'musiq': 'MUSIQ', 'flip': 'FLIP',
         'vmaf': 'VMAF', 'fovvideovdp': 'FovVideoVDP', 'strred': 'STRRED', 'cgvqm_2': 'CGVQM-2',
         'cgvqm_5': 'CGVQM-5', 'nvs_sqa': 'NVS-SQA'}
CPU = {'flip', 'vmaf', 'strred'}

d = d[d.metric.isin(NAMES)]
s = d.groupby('metric').agg(mean=('seconds', 'mean'), std=('seconds', 'std'),
                            mem=('peak_gpu_mb', 'max'), n=('video', 'nunique')).sort_values('mean')

rows = []
for m, r in s.iterrows():
    mem = "--" if m in CPU else f"{r['mem'] / 1024:.1f}"
    rows.append(f"{NAMES[m]} & {'CPU' if m in CPU else 'GPU'} & {r['mean']:.1f} $\\pm$ {r['std']:.1f} & "
                f"{1000 * r['mean'] / 300:.0f} & {mem} \\\\")
tex = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
       r"Metric & Device & Time per video (s) & Time per frame (ms) & Peak GPU memory (GB) \\",
       r"\midrule"] + rows + [r"\botrule", r"\end{tabular}"]
open(out_tex, "w").write("\n".join(tex) + "\n")
print("\n".join(rows))
print(f"\nvideos per metric: {sorted(s.n.unique())}")
