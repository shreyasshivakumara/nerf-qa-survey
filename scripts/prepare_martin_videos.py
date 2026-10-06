#!/usr/bin/env python3
"""
Prepare the real-world part of the Martin et al. (IEEE Access 2025) database
(github.com/pedrogcmartin/NVS-QA) for compute_metrics.py.

Real-world stimuli: 8 scenes x 4 methods = 32 synthesized videos
  Tanks and Temples (360 deg): m60, playground, train, truck
  Front-facing real:           antique, flowers, playground2, statue
  Methods: DVGO, Mip-NeRF 360 (files named *_multinerf.mp4), Nerfacto, NeRF++

The released videos (928x522, H.264) are used as they are: the script only
creates symbolic links <name>_synthesized.mp4 / <name>_reference.mp4 so the
metric script can find each test video and its scene reference.

Subjective scores: DMOS.xlsx reports DMOS = MOS_syn - MOS_ref + 5 on a 1-5
scale (ratings collected on 0-100 and converted linearly to 1-5 by the
authors), higher = better, 5 = quality of the reference. They are mapped
linearly onto the 0-10 scale of Explicit-NeRF-QA: MOS = 2.5 * (DMOS - 1).

Usage: python prepare_martin_videos.py <NVS-QA repo> <videos_out_dir> <names_csv>
"""

import os
import sys

import pandas as pd

repo, out_dir, names_csv = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)

SECTIONS = {  # DMOS.xlsx section header -> (synthesized dir, reference dir, viewpoint)
    'Tanks and Temples (360º real scenes)':
        ('Synthesized Videos/Tanks and Temples', 'Reference Videos/Tanks and Temples', '360'),
    'Front-facing IST (FF real scenes)':
        ('Synthesized Videos/Front-facing/Real', 'Reference Videos/Front-facing/Real', 'FF'),
}

# first block of DMOS.xlsx: section headers followed by (synthesized, reference, DMOS) rows
sheet = pd.read_excel(os.path.join(repo, 'DMOS.xlsx'), header=None).iloc[:, :3]
rows, section = [], None
for syn, ref, dmos in sheet.itertuples(index=False):
    if isinstance(syn, str) and syn.endswith('.mp4'):
        if section in SECTIONS:
            rows.append((section, syn[:-4], ref, float(dmos)))
    elif isinstance(syn, str) and syn != 'Synthesized Video':
        section = syn.strip()

out = []
for section, name, ref_file, dmos in rows:
    syn_dir, ref_dir, view = SECTIONS[section]
    scene, method = name.split('_', 1)
    syn_file = os.path.join(repo, syn_dir, scene, f"{scene}_{method.replace('mipnerf360', 'multinerf')}.mp4")
    ref_path = os.path.join(repo, ref_dir, ref_file)
    for src, kind in [(syn_file, 'synthesized'), (ref_path, 'reference')]:
        if not os.path.exists(src):
            sys.exit(f"missing file: {src}")
        link = os.path.join(out_dir, f"{name}_{kind}.mp4")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
    out.append(dict(dis_name=name, ref_name=scene, Viewpoint=view, Method=method,
                    DMOS=dmos, MOS=2.5 * (dmos - 1)))

out = pd.DataFrame(out)
if len(out) != 32 or out.ref_name.nunique() != 8:
    sys.exit(f"expected 32 videos in 8 scenes, found {len(out)} in {out.ref_name.nunique()}")
out.to_csv(names_csv, index=False)
print(f"{len(out)} videos, {out.ref_name.nunique()} scenes -> {out_dir}, {names_csv}")
print(out.groupby('Viewpoint').MOS.agg(['count', 'min', 'max']).round(2))
