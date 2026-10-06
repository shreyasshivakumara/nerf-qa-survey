#!/usr/bin/env python3
"""
Compute objective quality metrics on prepared video pairs and record, for every
video and metric, the score, the runtime and the peak GPU memory.

Inputs are the video pairs written by prepare_explicit_videos.py or
prepare_martin_videos.py:
  <videos>/<name>_reference.mp4, <videos>/<name>_synthesized.mp4
<names> is a score list (.xlsx or .csv) with a dis_name column.

Metric groups (run separately; each appends to <out>/<group>.csv and skips
videos already present, so runs can be resumed):
  pyiqa   15 image metrics via IQA-PyTorch, averaged over all frames
  flip    FLIP (LDR), averaged over all frames
  fvvdp   FovVideoVDP (display 'standard_fhd')
  cgvqm   CGVQM-2 and CGVQM-5 (patch_pool='mean', patch_scale=4)
  nvssqa  NVS-SQA (no-reference, on the frames of the test video)
  vmaf    VMAF v0.6.1 (videos converted to yuv420p y4m, as in the original pipeline)
  strred  ST-RRED (scikit-video, greyscale frames); run in the 'strred' environment

Output columns: video, metric, score, seconds, peak_gpu_mb
  seconds      wall-clock time to score one video,
               excluding one-off model loading where the model is reused
  peak_gpu_mb  peak GPU memory allocated by PyTorch while scoring (empty for CPU tools)

Usage: python compute_metrics.py <group> <videos_dir> <names> <out_dir> [--limit N] [--workers N] [--shard i/n]
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("group")
ap.add_argument("videos_dir")
ap.add_argument("mos_xlsx")
ap.add_argument("out_dir")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i (parallel runs)")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--cgvqm-dir", default=os.path.join(REPO, "third_party", "cgvqm"))
ap.add_argument("--vmaf-model", default=os.path.join(REPO, "third_party", "vmaf", "model", "vmaf_v0.6.1.json"))
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
shard_i, shard_n = map(int, args.shard.split("/"))
suffix = f"_shard{shard_i}" if shard_n > 1 else ""
out_csv = os.path.join(args.out_dir, f"{args.group}{suffix}.csv")
FIELDS = ["video", "metric", "score", "seconds", "peak_gpu_mb"]

names = (pd.read_excel if args.mos_xlsx.endswith(".xlsx") else pd.read_csv)(args.mos_xlsx)["dis_name"].tolist()
done = set()
if os.path.exists(out_csv):
    done = set(pd.read_csv(out_csv)["video"])
todo = [n for n in names[shard_i::shard_n] if n not in done][: args.limit]
print(f"{args.group}: {len(done)} videos done, {len(todo)} to do", flush=True)


def paths(name):
    return (os.path.join(args.videos_dir, f"{name}_synthesized.mp4"),
            os.path.join(args.videos_dir, f"{name}_reference.mp4"))


def read_frames(path, gray=False):
    cap, frames = cv2.VideoCapture(path), []
    ok, f = cap.read()
    while ok:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB))
        ok, f = cap.read()
    cap.release()
    return np.stack(frames)


def write_rows(rows):
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def gpu_timer():
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    def stop():
        torch.cuda.synchronize()
        return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20
    return stop


# ---------------------------------------------------------------- groups
def run_pyiqa():
    import pyiqa
    import torch
    fr = ['psnr', 'psnry', 'ssim', 'cw_ssim', 'ms_ssim', 'fsim', 'lpips', 'lpips-vgg',
          'dists', 'vif', 'lpips+']
    nr = ['niqe', 'piqe', 'maniqa', 'musiq']
    models = {m: pyiqa.create_metric(m, device="cuda") for m in fr + nr}
    for name in todo:
        test_p, ref_p = paths(name)
        test, ref = read_frames(test_p), read_frames(ref_p)
        rows = []
        for m, model in models.items():
            torch.manual_seed(0)   # MANIQA scores random crops; fix them for reproducibility
            stop, scores = gpu_timer(), []
            with torch.no_grad():
                for t, r in zip(test, ref):
                    for attempt in range(60):   # the GPU is shared; wait out transient out-of-memory errors
                        try:
                            tt = torch.from_numpy(t).cuda().permute(2, 0, 1)[None].float() / 255
                            if m in fr:
                                rr = torch.from_numpy(r).cuda().permute(2, 0, 1)[None].float() / 255
                                scores.append(model(tt, rr).item())
                            else:
                                scores.append(model(tt).item())
                            break
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            time.sleep(10)
                    else:
                        raise RuntimeError(f"{name} {m}: GPU out of memory after 60 retries")
            sec, mem = stop()
            rows.append(dict(video=name, metric=m, score=float(np.mean(scores)), seconds=sec, peak_gpu_mb=mem))
        write_rows(rows)
        print(name, {r["metric"]: round(r["score"], 4) for r in rows}, flush=True)


def flip_one(name):
    from flip_evaluator import flip_python_api as flip
    test_p, ref_p = paths(name)
    test, ref = read_frames(test_p), read_frames(ref_p)
    t0, scores = time.perf_counter(), []
    for t, r in zip(test, ref):
        _, mean_err, _ = flip.evaluate(r.astype(np.float32) / 255, t.astype(np.float32) / 255, "LDR")
        scores.append(float(mean_err))
    return [dict(video=name, metric="flip", score=float(np.mean(scores)),
                 seconds=time.perf_counter() - t0, peak_gpu_mb="")]


def run_fvvdp():
    import pyfvvdp
    fv = pyfvvdp.fvvdp(display_name="standard_fhd", heatmap=None, device="cuda")
    for name in todo:
        test_p, ref_p = paths(name)
        stop = gpu_timer()
        vs = pyfvvdp.fvvdp_video_source_file(test_p, ref_p, display_photometry=fv.display_photometry,
                                             frames=-1, preload=False)
        q, _ = fv.predict_video_source(vs)
        sec, mem = stop()
        write_rows([dict(video=name, metric="fovvideovdp", score=float(q), seconds=sec, peak_gpu_mb=mem)])
        print(name, round(float(q), 4), flush=True)


def run_cgvqm():
    import torch
    sys.path.insert(0, args.cgvqm_dir)
    from cgvqm import run_cgvqm as cg, CGVQM_TYPE
    for name in todo:
        test_p, ref_p = paths(name)
        rows = []
        for label, typ in [("cgvqm_2", CGVQM_TYPE.CGVQM_2), ("cgvqm_5", CGVQM_TYPE.CGVQM_5)]:
            stop = gpu_timer()
            q, _ = cg(test_p, ref_p, cgvqm_type=typ, device="cuda", patch_pool="mean", patch_scale=4)
            sec, mem = stop()
            rows.append(dict(video=name, metric=label, score=float(q.item()), seconds=sec, peak_gpu_mb=mem))
        write_rows(rows)
        print(name, {r["metric"]: round(r["score"], 4) for r in rows}, flush=True)


def run_nvssqa():
    import nvs_sqa
    qa = nvs_sqa.QualityAssessor()
    for name in todo:
        test_p, _ = paths(name)
        frames = read_frames(test_p)
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, name)
            os.makedirs(d)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(d, f"frame_{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            stop = gpu_timer()
            feats, _ = qa.generate_quality_features(tmp, verbose=False)
            score = qa.generate_quality_scores(feats, verbose=False)[name]
            sec, mem = stop()
        write_rows([dict(video=name, metric="nvs_sqa", score=float(score), seconds=sec, peak_gpu_mb=mem)])
        print(name, round(float(score), 4), flush=True)


def vmaf_one(name):
    test_p, ref_p = paths(name)
    with tempfile.TemporaryDirectory() as tmp:
        y4m = {}
        for k, p in [("test", test_p), ("ref", ref_p)]:
            y4m[k] = os.path.join(tmp, f"{k}.y4m")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-pix_fmt", "yuv420p", y4m[k]], check=True)
        js = os.path.join(tmp, "out.json")
        t0 = time.perf_counter()
        subprocess.run(["vmaf", "--reference", y4m["ref"], "--distorted", y4m["test"],
                        "--model", f"path={args.vmaf_model}", "--output", js, "--json", "--threads", "1"],
                       capture_output=True, check=True)
        sec = time.perf_counter() - t0
        score = json.load(open(js))["pooled_metrics"]["vmaf"]["mean"]
    return [dict(video=name, metric="vmaf", score=score, seconds=sec, peak_gpu_mb="")]


def strred_one(name):
    from skvideo.measure import strred
    test_p, ref_p = paths(name)
    ref, test = read_frames(ref_p, gray=True).astype(float), read_frames(test_p, gray=True).astype(float)
    t0 = time.perf_counter()
    _, score, _ = strred(ref, test)
    return [dict(video=name, metric="strred", score=float(score), seconds=time.perf_counter() - t0, peak_gpu_mb="")]


def run_parallel(fn):
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for rows in pool.imap_unordered(fn, todo):
            write_rows(rows)
            print(rows[0]["video"], round(rows[0]["score"], 4), flush=True)


groups = dict(pyiqa=run_pyiqa, flip=lambda: run_parallel(flip_one), fvvdp=run_fvvdp, cgvqm=run_cgvqm, nvssqa=run_nvssqa,
              vmaf=lambda: run_parallel(vmaf_one), strred=lambda: run_parallel(strred_one))
groups[args.group]()
print("done", flush=True)
