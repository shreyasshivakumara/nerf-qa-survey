#!/usr/bin/env bash
# Fetch the third-party code at the exact versions used in the paper.
#   CGVQM  IntelLabs/cgvqm @ a934cc8 (includes the network weights)
#   VMAF   Netflix/vmaf    @ 8cde19dc (libvmaf + model vmaf_v0.6.1.json); the
#          'vmaf' command-line tool must be on PATH (build: see libvmaf/README.md)
set -euo pipefail
cd "$(dirname "$0")"
[ -d cgvqm ] || git clone https://github.com/IntelLabs/cgvqm.git
git -C cgvqm checkout a934cc8
[ -d vmaf ] || git clone https://github.com/Netflix/vmaf.git
git -C vmaf checkout 8cde19dc
echo "CGVQM and VMAF ready; build libvmaf with: cd vmaf/libvmaf && meson setup build --buildtype release && ninja -vC build install"
