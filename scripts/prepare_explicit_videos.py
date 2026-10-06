#!/usr/bin/env python3
"""
Prepare Explicit-NeRF-QA videos for metric computation (no re-compression).

Each dataset video is 1600x800 and shows the reference rendering (left half)
next to the NeRF rendering (right half), exactly as presented to participants.
Each video was compressed separately, so every video carries its own copy of
the reference; the reference for a test video is therefore its own left half.

For every rated video (scenes with a MOS entry) this writes two lossless
(H.264 qp 0, yuv444p) 800x800 videos whose decoded pixels are identical to the
corresponding half of the dataset video:

  <out>/<scene>_<method>_<level>_reference.mp4
  <out>/<scene>_<method>_<level>_synthesized.mp4

Usage: python prepare_explicit_videos.py <videos_dir> <mos_xlsx> <out_dir>
"""

import os
import subprocess
import sys

import pandas as pd

videos_dir, mos_xlsx, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)

names = pd.read_excel(mos_xlsx)['dis_name'].tolist()   # 420 rated videos
for i, name in enumerate(names, 1):
    src = os.path.join(videos_dir, f"{name}.mp4")
    for kind, x in [("reference", 0), ("synthesized", 800)]:
        dst = os.path.join(out_dir, f"{name}_{kind}.mp4")
        if os.path.exists(dst):
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                        "-vf", f"crop=800:800:{x}:0",
                        "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
                        "-pix_fmt", "yuv444p", dst], check=True)
    print(f"[{i}/{len(names)}] {name}")
