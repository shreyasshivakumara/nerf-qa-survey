#!/usr/bin/env python3
"""
Analysis of NVS-SQA (Supplementary table tab:nvssqa_analysis).

1. features  extracts the frozen NVS-SQA encoder features (384-d) per video
2. analyse   prints, per dataset,
     - scene share: between-scene variance / total variance of the NVS-SQA scores,
       the subjective scores and MUSIQ
     - within-scene SRCC: mean over scenes of the SRCC on the videos of one scene
     - pooled SRCC of the shipped NVS-SQA head (as in the main table)
     - refitted head: the encoder is kept fixed and a ridge regression head is
       refitted on the target dataset with leave-one-scene-out cross-validation
       (alpha chosen by an inner leave-one-scene-out loop on the training scenes);
       SRCC on the unseen scenes. DISTS under the same procedure for reference.

Usage:
  python nvssqa_analysis.py features <videos_dir> <names_csv_or_xlsx> <out_npz>
  python nvssqa_analysis.py analyse <features_npz> <merged_csv> <label>
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def extract(videos_dir, names_file, out_npz):
    import cv2
    import nvs_sqa
    names = (pd.read_excel if names_file.endswith('.xlsx') else pd.read_csv)(names_file)['dis_name'].tolist()
    done = dict(np.load(out_npz)) if os.path.exists(out_npz) else {}
    qa = nvs_sqa.QualityAssessor()
    for i, name in enumerate(names):
        if name in done:
            continue
        cap, frames = cv2.VideoCapture(os.path.join(videos_dir, f"{name}_synthesized.mp4")), []
        ok, f = cap.read()
        while ok:
            frames.append(f)
            ok, f = cap.read()
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, name)
            os.makedirs(d)
            for j, f in enumerate(frames):
                cv2.imwrite(os.path.join(d, f"frame_{j:05d}.png"), f)
            feats, _ = qa.generate_quality_features(tmp, verbose=False)
        done[name] = np.asarray(feats[name]).ravel()
        if i % 20 == 0 or i == len(names) - 1:
            np.savez(out_npz, **done)
            print(f"{i + 1}/{len(names)}", flush=True)
    np.savez(out_npz, **done)


def scene_share(x, scenes):
    x = pd.Series(np.asarray(x, dtype=float))
    means = x.groupby(scenes).transform('mean')
    return ((means - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum()


def loso_ridge(X, y, scenes):
    """Leave-one-scene-out predictions of a ridge head; alpha by inner leave-one-scene-out."""
    from sklearn.linear_model import Ridge
    alphas = 10.0 ** np.arange(-3, 5)
    pred = np.zeros_like(y, dtype=float)
    for s in np.unique(scenes):
        tr, te = scenes != s, scenes == s
        best, best_err = None, np.inf
        for a in alphas:
            err = 0.0
            for s2 in np.unique(scenes[tr]):
                itr, ite = tr & (scenes != s2), scenes == s2
                m = Ridge(alpha=a).fit(X[itr], y[itr])
                err += ((m.predict(X[ite]) - y[ite]) ** 2).sum()
            if err < best_err:
                best, best_err = a, err
        pred[te] = Ridge(alpha=best).fit(X[tr], y[tr]).predict(X[te])
    return pred


def analyse(features_npz, merged_csv, label):
    from scipy.stats import spearmanr

    df = pd.read_csv(merged_csv)
    F = np.load(features_npz)
    X = np.stack([F[v] for v in df['Video Name']])
    y, scenes = df['MOS'].values, df['Scene'].values

    within = [spearmanr(df.nvs_sqa[scenes == s], y[scenes == s])[0] for s in np.unique(scenes)]
    rows = [('scene share, NVS-SQA scores', scene_share(df.nvs_sqa, scenes)),
            ('scene share, subjective scores', scene_share(y, scenes)),
            ('scene share, MUSIQ', scene_share(df.musiq, scenes)),
            ('within-scene SRCC, NVS-SQA', np.nanmean(within)),
            ('pooled SRCC, NVS-SQA', spearmanr(df.nvs_sqa, y)[0]),
            ('refitted head, NVS-SQA', spearmanr(loso_ridge(X, y, scenes), y)[0]),
            ('refitted head, DISTS', spearmanr(loso_ridge(-df[['dists']].values, y, scenes), y)[0])]
    for name, v in rows:
        print(f"[{label}] {name:32s} {v:.3f}")


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        extract(*sys.argv[2:5])
    else:
        analyse(*sys.argv[2:5])
