# Quality Assessment for Neural Radiance Fields: comparative study

Code and data for the comparative study in

> S. Shivakumara, G. Eilertsen, K. Lundin Palmerius,
> *Quality Assessment for Neural Radiance Fields: A Survey and Comparative Study* (under review).

The study evaluates 21 objective quality metrics (full-reference, no-reference
and NeRF-specific) against subjective scores on two public NeRF datasets, with
scene-level bootstrap confidence intervals and paired significance tests, plus a
pilot study on 3D Gaussian Splatting (GS-QA).

| Dataset | Content | Videos | Scenes | Subjective scores |
|---|---|---|---|---|
| [Explicit-NeRF-QA](https://github.com/YukeXing/Explicit-NeRF-QA) | synthetic, 360° | 420 rated (of 440) | 21 | MOS, 0–10 |
| [NVS-QA, Martin et al.](https://github.com/pedrogcmartin/NVS-QA), real-world part | Tanks and Temples (360°) + front-facing real | 32 | 8 | DMOS, 1–5 → MOS, 0–10 |
| [GS-QA, Martin et al.](https://github.com/pedrogcmartin/GS-QA) (3DGS pilot) | same 8 scenes as NVS-QA, 8 Gaussian Splatting methods | 64 | 8 | DMOS, 1–5 → MOS, 0–10 |

## Repository layout

```
data/                   per-video subjective and metric scores (three datasets)
run_all.sh              reproduce all tables
run_benchmark.sh        runtime / GPU-memory benchmark
scripts/                pipeline (see below)
environment/            pinned Python packages (two environments)
third_party/setup.sh    fetches CGVQM and VMAF at the versions used
results/benchmark/      per-video runtime and peak GPU memory, system configuration
results/analysis/       bootstrap statistics, LaTeX tables, Fig. 4
results/gsqa/           3DGS pilot: SRCC with CIs, significance, figure
results/cgvqm_max/      CGVQM with max patch pooling
results/nvssqa/         NVS-SQA encoder features and analysis
```

## Data

`data/` contains one row per video with the scene, the subjective score (MOS,
0–10) and the scores of all 21 metrics:

| File | Videos | Source |
|---|---|---|
| `data/explicit_420.csv` | 420 | [Explicit-NeRF-QA](https://github.com/YukeXing/Explicit-NeRF-QA) (MOS as released) |
| `data/martin_32.csv` | 32 | [NVS-QA, Martin et al.](https://github.com/pedrogcmartin/NVS-QA), real-world part (MOS = 2.5 (DMOS − 1)) |
| `data/gsqa_64.csv` | 64 | [GS-QA, Martin et al.](https://github.com/pedrogcmartin/GS-QA) (MOS = 2.5 (DMOS − 1)) |

The subjective scores come from these datasets; please cite them when using the data.
Metric column names: `psnr, psnry, ssim, cw_ssim, ms_ssim, fsim, lpips, lpips-vgg, dists, vif,
lpips+, niqe, piqe, maniqa, musiq, vmaf, fovvideovdp, flip, nvs_sqa, strred, cgvqm_2, cgvqm_5`

## Pipeline

| Step | Script | Notes |
|---|---|---|
| 1 | `prepare_explicit_videos.py` | Each Explicit-NeRF-QA video shows reference (left) and test rendering (right). Both 800×800 halves are written losslessly (H.264, QP 0, yuv444p), pixel-identical to the dataset video. Each test video is compared with its own reference half. |
| 1 | `prepare_martin_videos.py`, `prepare_gsqa_videos.py` | Link the released 928×522 videos (no re-encoding); convert DMOS to MOS (below). |
| 2 | `compute_metrics.py` | Metric groups `pyiqa`, `flip`, `fvvdp`, `cgvqm`, `nvssqa`, `vmaf`, `strred`; resumable, records runtime and GPU memory. |
| 3 | `build_merged.py` | Joins metric and subjective scores by video name into `data/*.csv`; stops on any missing or duplicate value. |
| 4 | `bootstrap.py` | Scene-level cluster bootstrap, 2,000 resamples, seed 42. |
| 5 | `make_tables.py`, `make_significance_table.py`, `make_sensitivity_table.py`, `make_runtime_table.py`, `make_fig4.py` | Tables and Fig. 4. |
| 6 | `gsqa_analysis.py`, `make_fig_gs_vs_nerf.py` | 3DGS pilot on GS-QA, compared with NVS-QA on the same scenes. |
| 6 | `cgvqm_pooling.py`, `nvssqa_analysis.py` | Supplementary analyses of the NeRF-specific metrics. |

`scripts/common.py` holds the metric list, the correlation measures and the dataset labels.

## Analysis details

- **Correlations.** SRCC and KRCC on raw scores; PLCC and RMSE after a 4-parameter
  logistic mapping (linear fallback if the fit does not converge; failed or
  flat fits are counted in `table_fit_convergence.tex`). Lower-is-better metrics (LPIPS, LPIPS-VGG, LPIPS+,
  DISTS, FLIP, STRRED, NIQE, PIQE) are negated. The real-world dataset is
  reported with SRCC and KRCC only.
- **Bootstrap.** Scenes are resampled with replacement; a scene drawn *k* times
  contributes its videos *k* times. 95% percentile intervals.
- **Significance.** The two best metrics per dataset are compared with every other
  metric on the same resamples; Bonferroni correction for 20 comparisons.
- **Subjective scales.** Explicit-NeRF-QA: MOS on 0–10. Martin et al. collected
  ratings on 0–100, converted them to 1–5 and report DMOS = MOS<sub>syn</sub> − MOS<sub>ref</sub> + 5
  (higher is better; 5 = reference quality). We map this linearly onto 0–10:
  MOS = 2.5 (DMOS − 1). Rank correlations do not depend on this mapping.
- **No spatial-shift compensation** is applied to either dataset. Martin et al.
  additionally report results with disparity compensation, so their image-metric
  values differ from ours; their VMAF and FovVideoVDP values match ours.

## Metric implementations

| Metric(s) | Implementation | Settings |
|---|---|---|
| PSNR, SSIM, MS-SSIM, CW-SSIM, FSIM, VIF, LPIPS, LPIPS-VGG, LPIPS+, DISTS, NIQE, PIQE, MANIQA, MUSIQ | IQA-PyTorch (pyiqa 0.1.14.1) | per frame, averaged over all frames; MANIQA with fixed seed |
| FLIP | flip-evaluator 1.6.0.1 | LDR, averaged over frames |
| VMAF | libvmaf @ 8cde19dc | model v0.6.1, yuv420p |
| FovVideoVDP | pyfvvdp 1.2.0 | display `standard_fhd`, non-foveated |
| ST-RRED | scikit-video 1.1.10 | luma |
| CGVQM-2, CGVQM-5 | IntelLabs/cgvqm @ a934cc8 | mean patch pooling, patch scale 4 |
| NVS-SQA | nvs_sqa 0.1.1 | no-reference, on the test video frames |

Results were computed on an Intel i7-14700KF, 64 GB RAM, NVIDIA RTX 4090 (24 GB),
Ubuntu 22.04.5, PyTorch 2.3.1 + CUDA 12.1, FFmpeg 4.4.2 (`results/benchmark/system.txt`).

## Citation

```bibtex
@article{shivakumara_nerf_qa_survey,
  title  = {Quality Assessment for Neural Radiance Fields: A Survey and Comparative Study},
  author = {Shivakumara, Shreyas and Eilertsen, Gabriel and Lundin Palmerius, Karljohan},
  note   = {Under review}
}
```

Please also cite the datasets and metrics used.

## Licence

Code: MIT (see `LICENSE`). The subjective scores in `data/` belong to the respective datasets.
