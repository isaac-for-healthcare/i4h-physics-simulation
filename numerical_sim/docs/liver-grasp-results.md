# Pinned-liver implementation and qualification

Implemented in the isolated `i4h-liver-grasp` worktree, based on immutable numerical
baseline `d1fc95c99d59be793da1f938e8c58458a538b168`. **Functional grasp qualification
failed: all 48 frozen target/start/mode combinations ended in `empty_capture`.**
There are zero supported targets; the four-target reusable-set requirement is not
met. No mechanics, targets or tolerances were adjusted after scoring. Retraction,
physical retention, destination holding and release therefore remain unqualified
by this campaign, although their observer/controller contracts have focused tests.

## Delivered implementation

- Standard `liver_pinned_grasp_1x` case, registered in surgical package data.
- `GraspTargetProbe`, mesh-aware patch preparation and deterministic target selection.
- Thin component adapter and public `I4hComponent.physics` binding accessor.
- Strict NPZ continuation bundle with attachment identity, copied-state validation,
  hashes, harness counters and a saved canonical settled baseline.
- Bounded 30 Hz teacher with 60 Hz acquisition/retention observation, stationary
  empty-capture retries, original-binding retention, separate destination holding
  and independently observed release timing.
- Numerical qualification CLI, diagnostic video renderer, focused tests and
  [integration contract](liver-grasp-contract.md).

The earlier planning document was replaced by the approved specification before
implementation, both in the original checkout and this isolated worktree. The
original staged Versius changes, pins, environments and Workflows checkout were
preserved. Nothing was staged, committed or pushed.

## Measured results

Final evidence is in `numerical_sim/artifacts/liver-grasp-reviewed/` (ignored raw
artifacts, retained locally). Earlier development runs are separate. The final
campaign's source hashes were checked against the delivered package files.

[Machine report](../artifacts/liver-grasp-reviewed/report.json),
[canonical snapshot](../artifacts/liver-grasp-reviewed/settled_snapshot.npz),
[target/workspace manifest](../artifacts/liver-grasp-reviewed/target_workspace_manifest.json),
[video with authored grasper](../artifacts/liver-grasp-reviewed/review_with_grasper.mp4) and
[validation summary](../artifacts/liver-grasp-reviewed/validation_summary.json).

| Check | Result |
| --- | --- |
| Storage/action contract | 1 liver body, 1 grasper, 1,986 particles, 391 soft attachments, 1,595 unattached; physical `(E,10)` |
| Settling | 180 steps / 3.0 s; 123 consecutive steps below 5 mm/s; final maximum speed 0.2413 mm/s |
| Copied State/Control restoration | Exact bytes in all three repeats per execution mode |
| Repeated restored open step | Eager maximum 0.1221 µm; captured maximum 0.0692 µm; both below 10 µm |
| Eager versus captured restored open step | Maximum pointwise difference 0.1221 µm |
| Frozen candidates | 224, 227, 204, 221, 222, 220, 229, 200 |
| Candidate starts/modes | Authored and ±5 mm X, eager and captured: 48 attempts |
| Capture histogram | 48 acquisitions of size zero; no nonempty acquisitions |
| Supported targets | 0; reusable-set support failed |
| Frozen-bundle reproduction | Separate process loaded the delivered bundle, completed all 48 attempts, same failure classifications |
| Finite / workspace checks | All scored samples finite; zero actual-root workspace exits |
| Physical release latencies / radial retention | Unavailable: acquisition failed; recorded null latencies and empty-acquisition radial metrics must not be read as successful zero-latency release or zero-slip retention |
| Material diagnostics | 89 inverted tets at settling; 88–90 during attempts; settled minimum signed-volume ratio −16.0953; maximum attachment drift 110.50 mm |
| Video | 1000×800, H.264, 30 FPS, 150 frames, 5 s, playback 1x; full decode passed |

The snapshot hash is
`294c95f73e30daeb3b15251ed3e5034cc86bdbdd2b23895460424ff4826578ce`.
`identity.json` records numerical/instrument asset hashes, Model identity,
dependency versions/source hashes and package source hashes. The NPZ metadata also
contains the schema/layout hash. `artifact_hashes.json` hashes the retained evidence.

For eager target 224 from the authored start, pre-command clearance was −0.0271 mm
at command step 206 with the jaw still open (0.5 rad). On the next physics step the
nearest-sphere clearance was +2.031 mm, and by the empty-capture deadline it was
+10.334 mm. The root remained stationary during this interval. These observations
locate the failure after closure; they do not independently assign its cause to
jaw motion, tissue contact or another part of the mechanics. The rule was preserved.

`review.mp4` shows eager target 204, −5 mm X, through the empty-capture failure.
Start, middle and final decoded frames were visually inspected: anatomy is visible,
markers and text are legible, phase and playback speed are shown. Since there is no
acquisition, the green marker is explicitly labelled **planned +Y (no acquisition)**;
no frozen acquired destination exists. The clip is a numerical mesh projection
showing the nearest jaw sphere, not a production rendering/recording qualification.

The eager/captured comparison across entire approach trajectories reached 43.69 µm
maximum pointwise distance. All 24 paired failure classifications agreed. This
longer-trajectory observation is separate from the 10 µm restored-one-step gate.

## Reproduction

From the numerical worktree's `numerical_sim/` directory, select an existing
qualified numerical interpreter and the production asset root explicitly. The run
used the preserved sibling interpreter and assets; no environment was modified:

```bash
export I4H_PYTHON=/home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-internal-worktrees/i4h-isaaclab/numerical_sim/.venv/bin/python
export I4H_DATA=/home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-internal-worktrees/i4h-isaaclab/numerical_sim/data

# Reuse the delivered settled state, target rule and numerical source identity.
"$I4H_PYTHON" scripts/liver_grasp.py --data "$I4H_DATA" \
  --snapshot artifacts/liver-grasp-reviewed/settled_snapshot.npz \
  --output artifacts/liver-grasp-reproduced

# To create a fresh canonical baseline instead, omit --snapshot and choose a new output.
# The same baseline is then used for every attempt in that new campaign.

"$I4H_PYTHON" scripts/liver_grasp_video.py artifacts/liver-grasp-reviewed --data "$I4H_DATA"
ffmpeg -v error -i artifacts/liver-grasp-reviewed/review.mp4 -f null -
```

The first command was exercised separately with output
`artifacts/liver-grasp-restored/`. Its `report.json` records a loaded canonical
snapshot, not a new settling run. Each campaign computes/fixes the manifest before
any scoring and publishes all candidates without tuning. Configuration/world-count,
source, dependency, asset or bundle-layout mismatches fail before restoration.

## Validation and remaining limits

With `PYTHONPATH` pointing to this worktree's six package `src/` directories and
`I4H_DATA` set as above:

```bash
"$I4H_PYTHON" -m pytest tests/grasp -q
"$I4H_PYTHON" -m pytest tests/packages/test_continuation.py \
  tests/runtime/test_bindings.py tests/runtime/test_component.py \
  tests/runtime/test_volumetric_grasp.py tests/runtime/test_instruments.py -q
uv build --all-packages
python3 scripts/check_artifacts.py dist
python3 scripts/check_private_reaches.py
```

Focused tests: **18 passed**. Regression subset: **58 passed, 3 skipped**. The skips
were two tests requiring Newton's unsupported trailing global WORLD row, and the
component-host module because `isaaclab.envs` is absent from this numerical
interpreter. Skip reasons were separately confirmed. The thin adapter was tested
through a component with a standalone binding; that is not an Isaac Lab host test.
All six wheels and all six source distributions built; package/asset isolation,
private-access checks, focused static checks and diff whitespace checks passed.
The complete repository test suite was not run.

Two-world tests explicitly exercise ordinary selected episode reset and selected
snapshot restoration; peer State/Control bytes and the peer acquisition latch
remain unchanged. Focused cases also cover corrupt/incompatible bundles, patch
padding/filtering, winding/facing rejection, target changes without recapture,
closure clearance and small steps, capture on either decimated sample, radial slip
with active slots, tangential motion, sticky removal/replacement, interrupted hold,
independent target position, release counts/active slots and fresh binding.

**Hosted compatibility is pending.** No finalized manifest/environment from the
approved Workflows upgrade was supplied or discoverable during this run, and the
local regression interpreter lacks Isaac Lab. The upgrade checkout was inspected
read-only. No liver compatibility or Workflows demonstration claim is made from
the numerical traces. Recorder integration, Arena integration and policy training
remain separate follow-up work. Physical grasp acquisition is the first failed
gate to resolve before claiming reusable grasp support.

## Visualization correction

The original `review.mp4` omitted the tool mesh and drew only the nearest jaw-sphere
marker. `review_with_grasper.mp4` corrects this with the authored shaft, body and both
jaw meshes, reconstructed by the existing skinning functions from recorded root,
input quaternion and actual jaw angle. Tissue positions remain the original recorded
values; no tissue physics is resimulated and no displacement is magnified. An oblique
view and a 55 mm target close-up show the tool and material point together.

For the illustrated target-204 attempt, maximum vertex motion relative to the first
clip sample was **0.66594 mm**, and maximum target motion was **0.23777 mm**. The clip
starts after settling. The settling trace contains about **217.57 mm** maximum motion
relative to its first sample; that includes sag and attachment drift, not only shape
change. No volumetric acquisition occurred, so the intended grasp-driven retraction
and deformation have not been demonstrated. Correcting the video does not close
the physical acquisition gate.

Reproduce the corrected video with:

```bash
"$I4H_PYTHON" scripts/liver_grasp_tool_video.py artifacts/liver-grasp-reviewed --data "$I4H_DATA"
```
