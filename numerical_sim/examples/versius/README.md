# Versius with I4hSolver

A native Isaac Lab `ManagerBasedRLEnv` articulation beside custom i4h tissue.
The delayed `CoupledSolver` steps MuJoCo-Warp for the arm and `I4hSolver` for
particles, constraints and instrument contact. No native soft-body solver runs.
See [RESULTS.md](RESULTS.md) for qualification, retained failures and limitations.

Run from `numerical_sim/`:

```bash
# Already created in this worktree. Bootstrap refuses to replace an existing env.
python3 examples/versius/bootstrap.py
python3 examples/versius/run.py scripted --scene block --episodes 10 --output .tmp_tests/versius/block
python3 examples/versius/run.py scripted --scene chole --episodes 10 --output .tmp_tests/versius/chole
python3 examples/versius/run.py train --scene block --envs 8 --graph --updates 5 --steps 200 --output .tmp_tests/versius/block
python3 examples/versius/run.py train --scene chole --envs 2 --graph --updates 5 --steps 200 --output .tmp_tests/versius/chole
python3 examples/versius/run.py play --scene block --steps 200 --checkpoint .tmp_tests/versius/block/i4h_solver_versius_block_24obs.pt
python3 examples/versius/run.py teleop --scene block --input keyboard --viewer --steps 0
python3 examples/versius/run.py teleop --scene chole --input replay --replay examples/versius/replay.json --steps 190
python3 examples/versius/run.py teleop --scene chole --input follou --viewer --steps 0
python3 examples/versius/run.py teleop --scene chole --input openhaptics --viewer --steps 0
```

`--graph` selects CUDA graphs; omission selects eager stepping. `--envs` changes
world count. `--coupling one_way` disables reaction transfer for comparisons.
`construct`, `mechanics`, and `lifecycle` modes check construction, open-loop
approach/hold/withdrawal, and selected reset/hard rebuild respectively.
`mechanics --no-contact` is the paired baseline. `--viewer --headless --record PATH.mp4`
records Newton OpenGL offscreen. Playback requires the matching `.metadata.json`
sidecar; v2 checkpoints are incompatible. Training requires a passing scripted
gate with matching task metadata in its output directory.

## Stack and assets

`bootstrap.py` copies the retained Python 3.12.12 third-party environment, using
reflinks when supported, into a separate 5.7 GiB `.venv`. It installs the exact
third-party `requirements.lock`, including RSL-RL **5.5.1**, without altering the
retained environment. The patched Warp wheel remains required. This is a local
source-overlay workflow: `run.py` selects this worktree's packages, current Lab
main and Newton soft-solvers sources. `stack.json` records their revisions,
existing dirty-file hashes, imported paths and Warp binary hash. The third-party
lock deliberately excludes source packages supplied by those overlays.

Set `I4H_VERSIUS_LAB` and `I4H_VERSIUS_NEWTON` to relocate external sources;
requalification is required for different sources. `--usd` selects the prepared,
closed-tool USD. The default is the existing v2
`training/generated/usd/versius_poke_2/versius_poke.usda`. Direct Gym construction
also accepts `I4H_VERSIUS_USD`. The asset generator is never run and v2's task,
solver and checkpoints are never imported. `NOTICE` attributes reused code;
`reuse.json` records the original utility source hashes and v2 revision.

Each run saves actual import paths/versions, source revisions and dirty-file
hashes, referenced USD-layer and original anatomy asset hashes, scene parameters,
results and failures. Commands and exit codes are retained under
`.tmp_tests/versius/commands/`. External-source and asset hashes remained unchanged
through implementation. Root dependency pins and qualification environments are
unchanged.

## Mechanics and frames

`profiles.py` supplies one component/robot configuration for both profiles:

- Block: the existing fixed-top tet grid, with a rigid 180-degree X rotation
  exposing its free bottom face above the supports; four tissue substeps and
  eight iterations. Gravity rotates with the fixture to world **+Z**.
- Chole: original `chole_tet_med` tissue, cavity, masses, materials, pin attachment,
  connectors, static and boundary contacts. A rigid +90-degree X rotation maps
  authored Y-up to world Z-up; gravity becomes world **-Z**. No anatomy rescaling,
  new pins or stiffness changes. Fluids, heat, grasping and procedure systems are
  omitted. The disabled tissue contact entry remains disabled as authored.

A deterministic free surface patch is translated beside the robot, initially
near `(0.5, 0, 0.2)`. Coordinates, targets and distances are **simulation units**;
there is no physical calibration claim. Particle views remain in component
coordinates. Measurements and explicitly published GL tissue meshes use the same
rigid component-to-world transform as the mount interface. Native particle dots
are hidden because Newton's raw particle rows contain component coordinates.

The arm uses J1–J8, the original actuator settings, and offset-corrected DLS IK
with lambda `0.05`. The TCP is `(0, 0, 0.51745)` in `J8_rotating_model` coordinates.
The prepared USD's closed jaw bounds end at Z `0.522965715` / `0.522939340` in that
frame. A **0.006-radius sphere centered at the TCP** covers the distal jaw region
and protrudes approximately `0.000484` beyond its foremost point. It is the cyan
marker in GL. This is a documented blunt contact approximation; shaft and full
jaw/tissue collision are outside this example.

Physics runs at 60 Hz, control at 30 Hz, with five internal rigid substeps.
Three normalized XYZ velocity actions have a norm cap of `0.04` per simulated
second. They integrate an IK target; only startup/reset writes arm state directly.
Orientation remains fixed and the imported tool remains closed. Tool reaction and
contact count use declared `state_products` as diagnostics.

Before each initial episode checkpoint the arm is placed above the patch,
tissue settles without proxy contact, and IK reachability is checked. Reset writes
selected robot state through Lab, flushes/forwards its selected cache updates,
then restores saved i4h-owned rows **last**. It clears only selected task and
controller history. Hard rebuild creates fresh checkpoints and invalidates old
binding handles. Raw tensor aliases must be reacquired after rebuild.

## Task and inputs

Import `i4h_versius.task` to register:

- `I4h-Versius-Poke-Block-v0`
- `I4h-Versius-Poke-Chole-v0`

`PokeEnv` uses the standard manager step loop and component construction lifecycle.
The policy layout is explicit in `terms.OBSERVATION_LAYOUT`: patch-to-TCP (3),
goal-to-patch (3), goal-to-TCP (3), target tracking error (3), TCP velocity (3),
patch velocity (3), preceding action (3), indentation (1), hold fraction (1), peak
depth (1): **24 values**. Tissue velocity is read explicitly from the public
scene field view; extra physics fields do not become Lab assets implicitly.

Success measures surface displacement from the saved baseline: indentation
`0.003 ± 0.0015`, lateral error `<0.006`, actual TCP/patch distance `<0.012`, held
for `0.5 s`. Episodes last five seconds. Nonfinite tissue/robot state, indentation
above `0.012` or TCP depth above `0.018` fail with precedence over success/timeout.
Hold and peak histories advance once per completed control step. Observations
and rewards are reads. Timeout PPO bootstraps from terminal observations.

Keyboard: W/S = X, A/D = Y, Q/E = Z, Ctrl = clutch, Space = pause, C = recenter,
R = verified episode reset, Esc = quit. Focus loss clears keys and pauses.
Tracking loss/invalid samples freeze the target and re-anchor on recovery.
Teleop ignores task success/timeout; invalid physics pauses until reset. It paces
to control time by default; `--no-real-time` is useful for replay checks.

Replay uses JSON position/clutch samples. Follou and HDAPI are input-only and
require their separately installed SDK/library. `--follou-device-index`,
`--hd-device`, `--hd-library`, `--gain` and `--axis-map` configure inputs. Default
axis maps are Follou `-x,-z,y`, OpenHaptics `x,-z,y`, keyboard/replay `x,y,z`.
For a leading minus use `--axis-map=-x,-z,y`. The Cartesian speed cap is fixed.
No force feedback is enabled. Physical devices were unavailable for testing.

## Follou through a temporary uv dependency

From `numerical_sim/`, use the local SDK without installing it into the example
or qualification environments:

```bash
uv run --no-project --python examples/versius/.venv/bin/python \
  --with /home/pkorzeniowsk/Projects/i4h/follou-sdk \
  python examples/versius/run.py teleop --scene block --input follou --viewer --steps 0
```

The launcher preserves an active Python environment, including uv's temporary
`--with` overlay. System-Python invocations continue to select the example's
isolated environment. Change `--scene block` to `--scene chole` after the initial
device checks pass.

## Live task status

The right-hand **Versius | Poke status** window displays contact count, reaction,
measured indentation and its required band, lateral error, actual TCP distance,
hold progress, tracking error, device status and the active backend. It explains
which goal condition remains unmet. Teleoperation continues when the goal is met;
force feedback remains disabled.

Use **Tip close-up** to inspect the cube and **Arm overview** to restore the camera.
Cyan marks the contact proxy, gold the live material patch, and green its settled
baseline. **Reset episode** and **Pause input / Resume input** mirror the keyboard
controls. Physics faults show their exact reason in red and log once per pause.
The panel is read-only except for these explicit controls and camera buttons;
reading it does not advance hold history or alter the success criteria.

The UI dependency `imgui-bundle==1.93.0` is included in the example dependency lock.
Restart an existing process to load UI changes. This remains an Isaac Lab
`ManagerBasedRLEnv` with Newton OpenGL presentation.

## Reference life-size anatomy

Use `--scene chole-1x` for the 1/7-scale anatomy from the supplied
`life-size-chole` worktree. The original `chole` profile is retained. The new task
registration is `I4h-Versius-Poke-Chole-1x-v0`.

```bash
uv run --no-project --python examples/versius/.venv/bin/python \
  --with /home/pkorzeniowsk/Projects/i4h/follou-sdk \
  python examples/versius/run.py teleop --scene chole-1x --input follou --viewer --steps 0
```

`life_size_reference.json` records the reference revision and source hashes. Its
`chole_tet_med_1x.yaml` is bundled locally; there is no runtime import from that
worktree. Tissue/cavity vertices, particle radii, pins, connector rest distances,
contact margins and bounds follow the reference. Its liver cavity contact and
0.5 tissue velocity limit are retained. The volume cutoff is `1e-8 / 7^4`, keeping
the same active volume constraints after scaling. The Versius arm, prepared tool,
TCP proxy and Cartesian control units retain their existing size.

The liver fragment spans about 0.261 units, and the gallbladder about 0.079.
This matches the reference's approximate life-size geometry convention. As in
that reference, masses, material coefficients, gravity and timing are retained;
**biomechanical response is not calibrated**.

The new profile passes geometry, finite stepping, contact, selected reset and
rebuild checks. The existing scripted 3 mm poke task is **not qualified at 1x**:
eager and captured runs each achieved 0/10 successes, with only about 0.00037
measured indentation before failure. Its PPO gate remains closed. Existing 7x
checkpoints are incompatible with `chole-1x`; the original profiles remain
available with their prior task behavior.
