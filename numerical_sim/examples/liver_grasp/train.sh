#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/physics/src:$PWD/surgical/src:$PWD/standalone/src:$PWD/isaaclab/src:$PWD/assessment/src:$PWD/endoluminal/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == report ]]; then
    shift
    exec uv run --no-project --python "$PWD/examples/liver_grasp/.venv-train/bin/python" python examples/liver_grasp/report.py "$@"
fi
if [[ "${1:-}" == resume ]]; then
    shift
    exec uv run --no-project --python "$PWD/examples/liver_grasp/.venv-train/bin/python" python examples/liver_grasp/resume.py "$@"
fi
if [[ "${1:-}" == video ]]; then
    shift
    exec uv run --no-project --python "$PWD/examples/liver_grasp/.venv-train/bin/python" python examples/liver_grasp/video.py "$@"
fi
exec uv run --no-project --python "$PWD/examples/liver_grasp/.venv-train/bin/python" python examples/liver_grasp/train.py "$@"
