#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the SOFA v26.06.00 Linux binary release (Python 3.12 build, LGPL-2.1; includes SofaPython3 and
# BeamAdapter), verify its SHA-256, unpack it and create a Python 3.12 environment that imports it.
#
#   scripts/benchmark/setup_sofa.sh [DEST]          # DEST defaults to ./sofa-benchmark
#   source DEST/sofa-env.sh                          # exports SOFA_ROOT, PYTHONPATH, LD_LIBRARY_PATH, SOFA_PYTHON
#   "$SOFA_PYTHON" scripts/benchmark/sofa_scaling.py aorta --scene aorta.npz
#
# SOFA is downloaded only when this script runs; nothing from it is redistributed here. Needs curl, unzip and uv.
set -euo pipefail

VERSION=v26.06.00
ARCHIVE=SOFA_${VERSION}_Linux_Python3.12.zip
URL=https://github.com/sofa-framework/sofa/releases/download/${VERSION}/${ARCHIVE}
SHA256=2a5c80dd012a433c36bae11ed06df701eb447387b6d2657410b2ef76ff2e32be

DEST=$(realpath -m "${1:-sofa-benchmark}")
mkdir -p "$DEST"
if [ ! -f "$DEST/$ARCHIVE" ]; then
    curl -fL --retry 3 -o "$DEST/$ARCHIVE.part" "$URL"
    mv "$DEST/$ARCHIVE.part" "$DEST/$ARCHIVE"
fi
echo "$SHA256  $DEST/$ARCHIVE" | sha256sum --check --quiet
if [ ! -d "$DEST/SOFA_${VERSION}_Linux" ]; then
    unzip -q "$DEST/$ARCHIVE" -d "$DEST"
fi
ROOT=$DEST/SOFA_${VERSION}_Linux
if [ ! -x "$DEST/venv/bin/python" ]; then
    uv venv -q --python 3.12 "$DEST/venv"
    uv pip install -q --python "$DEST/venv/bin/python" "numpy>=2,<3"
fi
cat > "$DEST/sofa-env.sh" <<EOF
export SOFA_ROOT=$ROOT
export PYTHONPATH=$ROOT/plugins/SofaPython3/lib/python3/site-packages
export LD_LIBRARY_PATH=$ROOT/lib
export SOFA_PYTHON=$DEST/venv/bin/python
# One thread per SOFA process: sofa_scaling.py parallelises over processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EOF
# shellcheck source=/dev/null
source "$DEST/sofa-env.sh"
"$SOFA_PYTHON" -c "import Sofa.Core, SofaRuntime; SofaRuntime.importPlugin('BeamAdapter'); print('SOFA ready')" 2>&1 \
    | grep -v '^\[INFO\]'
echo "source $DEST/sofa-env.sh"
