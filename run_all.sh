#!/usr/bin/env bash
# Reproduce the tables of the paper.
#
#   bash run_all.sh
#       all tables from the per-video scores in data/ (about 30 minutes)
#   bash run_all.sh full <explicit_dir> <martin_dir>
#       first recompute every metric score from the videos and rebuild data/
#       (about a day on one RTX 4090)
#
# <explicit_dir>  Explicit-NeRF-QA download (contains MOS.xlsx and Videos/)
# <martin_dir>    clone of https://github.com/pedrogcmartin/NVS-QA
# PY / PY_STRRED  python of the 'metrics' and 'strred' environments (default: python)
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); S=$REPO/scripts; D=$REPO/data; A=$REPO/results/analysis
PY=${PY:-python}; PY_STRRED=${PY_STRRED:-python}
mkdir -p "$A"

if [ "${1:-}" = full ]; then
  EXPLICIT=${2:?Explicit-NeRF-QA dir}; MARTIN=${3:?NVS-QA (Martin et al.) dir}; W=$REPO/work
  echo "== 1. video pairs and score lists"
  $PY $S/prepare_explicit_videos.py "$EXPLICIT/Videos" "$EXPLICIT/MOS.xlsx" $W/explicit_videos
  $PY $S/prepare_martin_videos.py "$MARTIN" $W/martin_videos $W/martin_scores.csv
  echo "== 2. metric scores"
  for ds in explicit martin; do
    if [ $ds = explicit ]; then V=$W/explicit_videos; L="$EXPLICIT/MOS.xlsx"; else V=$W/martin_videos; L=$W/martin_scores.csv; fi
    for g in pyiqa fvvdp cgvqm nvssqa; do $PY $S/compute_metrics.py $g $V "$L" $W/metrics/$ds; done
    for g in flip vmaf; do OMP_NUM_THREADS=1 $PY $S/compute_metrics.py $g $V "$L" $W/metrics/$ds --workers 8; done
    OMP_NUM_THREADS=1 $PY_STRRED $S/compute_metrics.py strred $V "$L" $W/metrics/$ds --workers 8
  done
  echo "== 3. merge metric and subjective scores"
  $PY $S/build_merged.py $W/metrics/explicit "$EXPLICIT/MOS.xlsx" $D/explicit_420.csv
  $PY $S/build_merged.py $W/metrics/martin $W/martin_scores.csv $D/martin_32.csv
fi

echo "== bootstrap and tables"
$PY $S/bootstrap.py $D/explicit_420.csv $D/martin_32.csv $A
$PY $S/make_tables.py $D/explicit_420.csv $D/martin_32.csv $A/bootstrap_statistics.csv $A
$PY $S/make_significance_table.py $D/explicit_420.csv $D/martin_32.csv $A/significance_srcc.csv
$PY $S/make_sensitivity_table.py $D/martin_32.csv $A/bootstrap_statistics.csv $A/point_estimates.csv $A/table_sensitivity.tex
$PY $S/make_runtime_table.py $REPO/results/benchmark $A/table_runtime.tex
echo "done: results in $A"
