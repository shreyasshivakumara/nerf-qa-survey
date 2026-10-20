#!/usr/bin/env bash
# Reproduce the tables of the paper.
#
#   bash run_all.sh
#       all tables from the per-video scores in data/ (about 30 minutes)
#   bash run_all.sh full <explicit_dir> <martin_dir> <gsqa_dir>
#       first recompute every metric score from the videos and rebuild data/
#       (about a day on one RTX 4090)
#
# <explicit_dir>  Explicit-NeRF-QA download (contains MOS.xlsx and Videos/)
# <martin_dir>    clone of https://github.com/pedrogcmartin/NVS-QA
# <gsqa_dir>      clone of https://github.com/pedrogcmartin/GS-QA (3DGS pilot)
# PY / PY_STRRED  python of the 'metrics' and 'strred' environments (default: python)
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); S=$REPO/scripts; D=$REPO/data; A=$REPO/results/analysis
PY=${PY:-python}; PY_STRRED=${PY_STRRED:-python}
mkdir -p "$A"

if [ "${1:-}" = full ]; then
  EXPLICIT=${2:?Explicit-NeRF-QA dir}; MARTIN=${3:?NVS-QA (Martin et al.) dir}; GSQA=${4:?GS-QA dir}; W=$REPO/work
  echo "== 1. video pairs and score lists"
  $PY $S/prepare_explicit_videos.py "$EXPLICIT/Videos" "$EXPLICIT/MOS.xlsx" $W/explicit_videos
  $PY $S/prepare_martin_videos.py "$MARTIN" $W/martin_videos $W/martin_scores.csv
  $PY $S/prepare_gsqa_videos.py "$GSQA" $W/gsqa_videos $W/gsqa_scores.csv
  echo "== 2. metric scores"
  for ds in explicit martin gsqa; do
    if [ $ds = explicit ]; then V=$W/explicit_videos; L="$EXPLICIT/MOS.xlsx"; else V=$W/${ds}_videos; L=$W/${ds}_scores.csv; fi
    for g in pyiqa fvvdp cgvqm nvssqa; do $PY $S/compute_metrics.py $g $V "$L" $W/metrics/$ds; done
    for g in flip vmaf; do OMP_NUM_THREADS=1 $PY $S/compute_metrics.py $g $V "$L" $W/metrics/$ds --workers 8; done
    OMP_NUM_THREADS=1 $PY_STRRED $S/compute_metrics.py strred $V "$L" $W/metrics/$ds --workers 8
  done
  for ds in explicit martin; do
    if [ $ds = explicit ]; then V=$W/explicit_videos; L="$EXPLICIT/MOS.xlsx"; else V=$W/martin_videos; L=$W/martin_scores.csv; fi
    $PY $S/compute_metrics.py cgvqm $V "$L" $W/cgvqm_max/$ds --cgvqm-pool max
    cp $W/cgvqm_max/$ds/cgvqm.csv $REPO/results/cgvqm_max/$ds.csv
    $PY $S/nvssqa_analysis.py features $V "$L" $REPO/results/nvssqa/${ds}_features.npz
  done
  echo "== 3. merge metric and subjective scores"
  $PY $S/build_merged.py $W/metrics/explicit "$EXPLICIT/MOS.xlsx" $D/explicit_420.csv
  $PY $S/build_merged.py $W/metrics/martin $W/martin_scores.csv $D/martin_32.csv
  $PY $S/build_merged.py $W/metrics/gsqa $W/gsqa_scores.csv $D/gsqa_64.csv
fi

echo "== bootstrap and tables"
$PY $S/bootstrap.py $D/explicit_420.csv $D/martin_32.csv $A
$PY $S/make_significance_table.py $D/explicit_420.csv $D/martin_32.csv $A/significance_srcc.csv
$PY $S/make_fig4.py $A $A/point_estimates.csv $A/fig4_bootstrap_srcc

echo "== supplementary checks"
R=$REPO/results
for ds in explicit:explicit_420 martin:martin_32; do
  $PY $S/cgvqm_pooling.py $D/${ds##*:}.csv $R/cgvqm_max/${ds%%:*}.csv ${ds%%:*}
done > $R/cgvqm_max/summary.txt
for ds in explicit:explicit_420 martin:martin_32; do
  $PY $S/nvssqa_analysis.py analyse $R/nvssqa/${ds%%:*}_features.npz $D/${ds##*:}.csv ${ds%%:*}
done > $R/nvssqa/summary.txt

echo "== 3DGS pilot (GS-QA)"
$PY $S/gsqa_analysis.py $D/gsqa_64.csv $D/martin_32.csv "${GSQA:--}" $R/gsqa > $R/gsqa/summary.txt
$PY $S/make_fig_gs_vs_nerf.py $R/gsqa/gsqa_vs_nvsqa_srcc.csv $R/gsqa/fig_gs_vs_nerf
echo "done: results in $REPO/results"
