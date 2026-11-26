# ReachTarget training example

Run from `numerical_sim/` with the interpreter from a separate Isaac Lab environment at the
target revision (`i4h_isaaclab.lab.QUALIFIED`), Warp 1.17.0 and Newton
`7f0c13af` (1.9.0.dev0), as pinned by the numerical workspace. Install the six built i4h member wheels into that environment without their
optional USD extras; Lab supplies `pxr`. Supply the original external asset directory
through `I4H_DATA`. The ordinary numerical workspace does not install Isaac Lab or
RSL-RL. See the [compatibility contract](README.md#isaac-lab-compatibility).

```bash
LAB_PY=/absolute/path/to/qualified-lab-venv/bin/python
I4H_DATA=/absolute/path/to/data "$LAB_PY" scripts/isaaclab_run.py --task I4h-ReachTarget-Direct --num_envs 64 --train 50 --seed 0 --visualizer none --log_dir /absolute/path/to/new-run
```

Isaac Lab 3.0 uses `--visualizer none` for headless execution and rejects the removed
`--headless` flag. Use a new output directory to preserve existing checkpoints.
The task is `aorta_static_reach`. The command performs 50 PPO learning iterations
with 16 rollout steps per environment per iteration, then the helper's existing
additional checked iteration verifies finite actions, observations, rewards and losses.
It does not change the seed or settings based on observed reward.

The results below are historical measurements on an older stack; they do not
verify the current pins or current source.

On 2026-10-15, on Isaac Lab 6864dd1, the v0.9.0 package code completed this command on an NVIDIA
GB300 (aarch64) and exited **0**. The separate Lab environment used Torch
2.12.0+cu130 and RSL-RL 5.5.1. The GPU's hardware power-brake slowdown was active;
this is not a healthy-hardware training-throughput baseline.

All 150 reported value/surrogate/entropy losses were finite. The extra checked
iteration reported value loss 0.003783, surrogate loss −0.00556 and entropy 4.262.
The logged mean reward averaged **0.256** over the first five reported reward values
(iterations 1–5) and **0.998** over the last five (45–49), ending at **0.99**. These are console-rounded,
logger-window reward statistics. Iteration 0 has no episodic reward entry,
and the reported episode lengths change over the run. This single
seed does not establish convergence or promise learning after 50 iterations.

The full process took 733.99 s, including setup and the extra checked iteration.
Its training timer reported 12.01 s/iteration averaged across the 50 iterations,
including a slow first update. The run retained its initial compilation overhead;
no seed or setting was changed to obtain a preferred result.

Both `model_0.pt` and `model_49.pt` were nonempty and all 506,994 tensor elements
in each were finite. The final checkpoint is 2,046,193 bytes, SHA256
`a2a6d8c85e2ed19239bd1f3b56e827c69142ac1328c5cbb2c4fa31586af41d34`. The checkpoints
precede the helper's extra checked iteration.

## Evaluating a checkpoint, on either env

`--policy checkpoint:<model.pt>` runs a checkpoint's deterministic policy for `--steps` policy steps and reports the
episodes that ended, the procedure successes and the mean return. The manager-based task has the same observation and
action layout, so a checkpoint trained on `-Direct` evaluates on `-Manager`:

```bash
I4H_DATA=/absolute/path/to/data "$LAB_PY" scripts/isaaclab_run.py --task I4h-ReachTarget-Manager --num_envs 64 --steps 1500 --seed 0 --visualizer none --policy checkpoint:/absolute/path/to/new-run/model_49.pt
```

On 2026-10-29, the `model_49.pt` of a `-Direct` 50-iteration run (trained on Isaac Lab 6864dd1, SHA256 `7ba558d6...672a83`)
gave the same result on `I4h-ReachTarget-Direct` and `I4h-ReachTarget-Manager` on Isaac Lab v3.0.0: 2,112 episodes
ended in 1500 steps of 64 envs, all 2,112 procedure successes, mean return 1.000.
