#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${I4H_PYTHON:=$PWD/../../i4h-isaaclab/numerical_sim/.venv/bin/python}"
training_env="$PWD/examples/liver_grasp/.venv-train"
if [[ ! -x "$training_env/bin/python" ]]; then
    uv venv --python "$I4H_PYTHON" "$training_env"
fi
simulation_site=$("$I4H_PYTHON" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
training_site=$("$training_env/bin/python" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
printf '%s\n' "$simulation_site" > "$training_site/simulation.pth"
uv pip install --python "$training_env/bin/python" --no-deps -r examples/liver_grasp/requirements-training.txt
"$training_env/bin/python" -c 'import torch,warp,newton;from rsl_rl.algorithms import PPO;print("isolated training overlay ready")'
"$training_env/bin/python" examples/liver_grasp/check_training_env.py
