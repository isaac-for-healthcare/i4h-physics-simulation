#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${I4H_PYTHON:=$PWD/../../i4h-isaaclab/numerical_sim/.venv/bin/python}"
: "${I4H_DATA:=$PWD/../../i4h-isaaclab/numerical_sim/data}"
: "${FOLLOU_SDK:=/home/pkorzeniowsk/Projects/i4h/follou-sdk}"
export PYTHONPATH="$PWD/physics/src:$PWD/surgical/src:$PWD/standalone/src:$PWD/isaaclab/src:$PWD/assessment/src:$PWD/endoluminal/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-project --python "$I4H_PYTHON" --with "$FOLLOU_SDK" python -m i4h_standalone.app.cli examples/liver_grasp/minimou.yaml --data "$I4H_DATA" --render slang --input devices --steps 0 "$@"
