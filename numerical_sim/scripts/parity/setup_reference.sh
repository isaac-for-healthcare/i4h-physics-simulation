#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The catheter parity reference (scripts/parity/catheter.py --engine reference): catheter_vasculature_solver at this
# repo's e4bed31 (what the i4h-workflows catheter-navigation reward branch installed from
# feature/newton-solverbase-isaaclab-integration) on the old stack's warp-lang 1.13.0, with torch and numpy. The
# solver needs no Newton or Isaac Lab. Usage: scripts/parity/setup_reference.sh [DIR]  (default .parity/reference)
set -euo pipefail
REF=e4bed31
DIR=${1:-.parity/reference}
ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$DIR/src"
git -C "$ROOT" archive "$REF" physics_simulation/endoluminal/catheter-vasculature-solver | tar -x -C "$DIR/src"
uv venv -q --python 3.12 "$DIR/venv"
VIRTUAL_ENV="$DIR/venv" uv pip install -q warp-lang==1.13.0 numpy torch
VIRTUAL_ENV="$DIR/venv" uv pip install -q --no-deps -e "$DIR/src/physics_simulation/endoluminal/catheter-vasculature-solver"
"$DIR/venv/bin/python" -c "import warp, catheter_vasculature_solver; print('reference ready: warp', warp.__version__)"
echo "run: $DIR/venv/bin/python scripts/parity/catheter.py run --engine reference --out parity/reference.npz"
