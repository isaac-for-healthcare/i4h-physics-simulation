#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
here=$(cd "$(dirname "$0")" && pwd)
export SOFA_ROOT=${SOFA_ROOT:?Set SOFA_ROOT to the installation or checkout described in validation/README.md}
export SOFAPYTHON3_ROOT="$SOFA_ROOT/plugins/SofaPython3"
export PYTHONPATH="$SOFAPYTHON3_ROOT/lib/python3/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFAPYTHON3_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${BENCH_PYTHON:-python}" "$here/bench.py" "$@"
