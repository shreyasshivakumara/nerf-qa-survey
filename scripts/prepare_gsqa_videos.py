#!/usr/bin/env python3
"""
Prepare GS-QA (Martin et al., QoMEX 2025, github.com/pedrogcmartin/GS-QA)
for compute_metrics.py.

64 synthesized videos: 8 real-world scenes (the same scenes and 928x522 crops
as the real-world part of NVS-QA) x 8 Gaussian Splatting variants (3DGS,
EAGLES, LightGaussian, Mip-Splatting, Octree-GS, Scaffold-GS, SOG, SOG w/o SH).

File names: SOG (w/o SH) is *_sog.mp4 and SOG is *_sog_sh.mp4 (checked by
matching the PSNR order of the two files with metrics.xlsx of the authors).

The released videos are used as they are (symbolic links, no re-encoding).
Subjective scores: dmos.xlsx, DSCQS with the same 1-5 DMOS as NVS-QA
(higher = better); mapped to 0-10 as for NVS-QA: MOS = 2.5 * (DMOS - 1).

Usage: python prepare_gsqa_videos.py <GS-QA repo> <videos_out_dir> <names_csv>
"""

import os
import sys

import pandas as pd

repo, out_dir, names_csv = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)

GROUP = {'antique': 'IST', 'flowers': 'IST', 'playground2': 'IST', 'statue': 'IST',
         'm60': 'T&T', 'playground': 'T&T', 'train': 'T&T', 'truck': 'T&T'}
VIEW = {'IST': 'FF', 'T&T': '360'}
FILE = {'3DGS': '3dgs', 'EAGLES': 'eagles', 'LightGaussian': 'lightgaussian', 'Mip-Splatting': 'mipsplatting',
        'Octree-GS': 'octree', 'Scaffold-GS': 'scaffold', 'SOG (w/o SH)': 'sog', 'SOG': 'sog_sh'}

scores = pd.read_excel(os.path.join(repo, 'dmos.xlsx'))
out = []
for scene, method, dmos in scores[['Scene', 'Method', 'DMOS']].itertuples(index=False):
    name = f"{scene}_{FILE[method]}"
    syn = os.path.join(repo, 'Synthesized Videos', GROUP[scene], scene, f"{name}.mp4")
    ref = os.path.join(repo, 'Reference Videos', f"{scene}_reference.mp4")
    for src, kind in [(syn, 'synthesized'), (ref, 'reference')]:
        if not os.path.exists(src):
            sys.exit(f"missing file: {src}")
        link = os.path.join(out_dir, f"{name}_{kind}.mp4")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
    out.append(dict(dis_name=name, ref_name=scene, Viewpoint=VIEW[GROUP[scene]], Method=method,
                    DMOS=float(dmos), MOS=2.5 * (float(dmos) - 1)))

out = pd.DataFrame(out)
if len(out) != 64 or out.ref_name.nunique() != 8 or out.dis_name.duplicated().any():
    sys.exit(f"expected 64 unique videos in 8 scenes, found {len(out)} in {out.ref_name.nunique()}")
out.to_csv(names_csv, index=False)
print(f"{len(out)} videos, {out.ref_name.nunique()} scenes -> {out_dir}, {names_csv}")
print(out.groupby('Viewpoint').MOS.agg(['count', 'min', 'max']).round(2))
