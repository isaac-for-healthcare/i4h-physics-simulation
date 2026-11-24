> Viewer and teleoperation commands require the separate `i4h-isaaclab-standalone` bridge. Headless training/evaluation remains available through upstream `scripts/isaaclab_run.py`.

# Versius surgical tasks

Versius uses canonical surgical cases, a native Isaac Lab articulation and the
shared launchers and GL renderer. The robot controls a blunt sphere at the closed
tool tip. Cutting, grasping and force feedback are outside this task.
Historical results, source attribution, stack records and checkpoint evidence
remain in `examples/versius/`; new results belong in a separate output directory.

## Environment and assets

Use an isolated Python 3.12 environment with the qualified patched Warp binary,
RSL-RL 5.5.1 and the qualified current Newton/Lab sources. Root pins and retained
qualification environments are unchanged. Environment setup copies installed
third-party dependencies; it does not resolve, upgrade or modify its seed:

```bash
python3 scripts/setup_isaaclab_current.py \
  --seed-env /path/to/qualified/python312-env \
  --lab /path/to/isaaclab-main --newton /path/to/newton-checkout
export PY="$PWD/.venvs/isaaclab-current/bin/python"
export I4H_DATA=/path/to/production/data
export I4H_VERSIUS_USD=/path/to/prepared/versius_poke.usda
```

The seed must already include the patched Warp build and the desired GL/UI/PPO
dependencies. Setup refuses to overwrite an existing environment. External
source locations are explicit and written into that environment's `.pth` file.
Do not run a workspace dependency sync against this source-overlay environment.
`isaaclab_run.py --check` retains its root-pin comparison; use the validation
workflows below to qualify these external current sources.
The prepared USD is supplied by `--usd` or `I4H_VERSIUS_USD`; there is no
machine-specific asset default and the asset generator is never run.

## Commands

Run from `numerical_sim/`. The registered IDs are
`I4h-Versius-Poke-Block-v0`, `I4h-Versius-Poke-Chole-v0` and
`I4h-Versius-Poke-Chole-1x-v0`. Graphs are enabled by default;
`--no-graph` explicitly selects eager physics. Physics remains 60 Hz with
control decimation 2, five rigid substeps, and four/eight tissue substeps for
block/chole. `--num_envs` selects replicated worlds.

```bash
i4h-lab-run --task I4h-Versius-Poke-Block-v0 \
  --validation scripted --episodes 10 --num_envs 2 --output OUTPUT
i4h-lab-run --task I4h-Versius-Poke-Block-v0 \
  --train 5 --num_envs 2 --log_dir OUTPUT
i4h-lab-run --task I4h-Versius-Poke-Block-v0 \
  --policy checkpoint:OUTPUT/i4h_solver_versius_block_24obs.pt --steps 200
i4h-lab-run --task I4h-Versius-Poke-Chole-1x-v0 \
  --validation scripted --episodes 10 --output LIFE_SIZE_OUTPUT
i4h-lab-teleop --task I4h-Versius-Poke-Chole-1x-v0 \
  --input follou --render gl --steps 0
i4h-lab-teleop --task I4h-Versius-Poke-Block-v0 \
  --input replay --replay examples/versius/replay.json --render none --steps 190
i4h-lab-teleop --task I4h-Versius-Poke-Block-v0 \
  --input keyboard --render gl --steps 0 --record OUTPUT/teleop.mp4
```

`--validation construct|scripted|mechanics|lifecycle` runs the qualification
helpers. Training requires a passing `scripted-gate.json` with matching task
metadata in `--log_dir` (or `--output`). Playback requires a matching
`.metadata.json` sidecar. Existing checkpoints remain compatible when the
resolved case, alignment, patch, observation layout, dynamics and goal settings
match. Metadata mismatches raise an error; sidecars are never rewritten to force
compatibility. World count and eager/graph selection do not change task metadata.
The life-size fixture retains its previously observed task failures; a rendered
scene or successful construction does not establish scripted feasibility.

Mechanical comparison runs use `--coupling one_way|two_way`, `--no-contact` and
`--no-graph`. `python -m i4h_isaaclab.versius.compare DIRECTORY` applies the
historical comparison checks to the same named mechanical fixture directories
used in the historical results. Qualification writes trajectories, results and
failures only in the selected output directory. Ordinary interactive startup
performs no repository/asset provenance scan.

## Tests

The task tests use the isolated current Lab/PPO environment. Run them explicitly
alongside the repository package/runtime suite, with production data configured:

```bash
$PY -m pytest tests/versius -q
$PY -m pytest tests/runtime tests/packages -q
uv run --no-sync python examples/core_synthetic.py --device cpu
uv build --all-packages
```

## Input and viewer

Cartesian control supports `--input keyboard|replay|follou|openhaptics`.
`--replay` for this path takes a JSON list of raw positions in metres, with null
samples representing tracking loss. The existing direct-instrument path remains
available with `--case` and its original recording and replay options.

The shared MiniMou adapter owns SDK acquisition, polling, validation and cleanup.
Its Cartesian sample converts raw millimetres to metres before the common axis
map and gain. Defaults are `-x,-z,y` for Follou and `x,-z,y` for OpenHaptics;
`--axis-map`, `--gain`, `--follou-device-index`, `--hd-device` and `--hd-library`
remain available. Versius always disables force output. To use a local Follou
SDK temporarily without changing the environment:

```bash
FOLLOU_LIB_PATH=/path/to/native/library \
uv run --no-project --python "$PY" --with /path/to/follou-sdk \
  python i4h-lab-teleop --task I4h-Versius-Poke-Chole-1x-v0 \
  --input follou --render gl --steps 0
```

Keyboard W/S, A/D and Q/E drive XYZ. Ctrl clutches translation, Space toggles
input pause, C recentres, R resets and Escape quits. Focus loss and invalid
tracking freeze the target and discard the motion anchor; the next valid sample
anchors again. Motion retains the norm speed cap and task workspace limits.
A physics failure pauses stepping until reset and reports its cause.

The GL view draws native robot shapes and transformed surgical meshes together,
with raw component-coordinate particle dots hidden. Cyan marks the proxy, gold
the live tissue patch and green the settled baseline. The task panel retains
contact, reaction, indentation, hold progress, failure guidance, reset/pause and
camera presets. `--record FILE.mp4` records the shared GL view; `--headless`
selects offscreen GL, where Newton disables ImGui. Physical devices still need
manual qualification.

Tissue vertices use persistent device buffers and a Warp rigid-transform kernel;
live surface indices are also copied on device. UI diagnostics use one batched
readback at 6 Hz. The pre-step failure bit is checked every control step.
Cartesian device control still reads the small current/home target vectors on
the host; camera focus reads one target on demand. Newton ViewerGL can still
transfer mesh buffers and packed shape transforms internally, and recording
reads the framebuffer for ffmpeg. These transfers are not claimed to be removed.
Viewer loop timings synchronize the GPU at phase boundaries to include completed
work. Loop timings exclude real-time pacing and report physics/control work, viewer
work and total wall time after warmup; the headless policy loop also reports
CUDA event timings. See the consolidation results for measured performance.
