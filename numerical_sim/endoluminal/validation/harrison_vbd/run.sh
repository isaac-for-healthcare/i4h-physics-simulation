#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="${NEWTON_ROOT:?Set NEWTON_ROOT to the installation or checkout described in validation/README.md}${PYTHONPATH:+:$PYTHONPATH}"
exec "${BENCH_PYTHON:-python}" "$here/bench.py" "$@"
