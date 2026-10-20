#!/usr/bin/env bash
# Runtime and GPU-memory benchmark on 20 Explicit-NeRF-QA videos (every 21st
# video). Each metric group runs alone on an otherwise idle machine. Also
# records the system configuration.
#
# Usage: bash run_benchmark.sh <explicit_videos> <explicit_mos_xlsx> <out_dir>
#   <explicit_videos> = video pairs written by prepare_explicit_videos.py
set -euo pipefail
V=${1:?video pairs}; L=${2:?MOS.xlsx}; OUT=${3:?output dir}
S=$(cd "$(dirname "$0")" && pwd)/scripts; PY=${PY:-python}; PY_STRRED=${PY_STRRED:-python}
mkdir -p "$OUT"
{
  echo "date: $(date -Iseconds)"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) threads"
  echo "ram: $(free -g | awk '/Mem:/{print $2}') GB"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "os: $(lsb_release -ds 2>/dev/null || uname -sr)"
  echo "python: $($PY --version 2>&1)"
  $PY -c "import torch; print('torch:', torch.__version__, 'cuda', torch.version.cuda)"
  echo "ffmpeg: $(ffmpeg -version | head -1)"
} > "$OUT/system.txt"
for g in pyiqa flip fvvdp cgvqm nvssqa vmaf; do
  echo "== $g"; $PY $S/compute_metrics.py $g "$V" "$L" "$OUT" --shard 0/21 --limit 20
done
echo "== strred"; $PY_STRRED $S/compute_metrics.py strred "$V" "$L" "$OUT" --shard 0/21 --limit 20
