# Versius consolidation validation — 2026-11-17

The standalone Versius runtime is replaced by surgical case YAMLs, the
`i4h_isaaclab.versius` task and shared run/teleop/GL facilities. Historical
`examples/versius` results and stack records remain unchanged. See
[the command guide](versius.md) for replacements. No commit, dependency upgrade,
mechanics retuning or external-source edit was performed.

The configured repository suite finished with **1377 passed, 46 skipped**.
The focused task/boundary/pin checks finished with **61 passed**.

Raw evidence is under `.tmp_tests/versius-consolidation/`, separate from the
historical `.tmp_tests/versius/` evidence. `campaign.json`, `smoke.json`,
`benchmarks-commands.json` and `comparisons/commands.json` record commands and exit
codes. Logs and per-fixture results remain beside them.

## Cases and task behavior

`resolved-case-equivalence.json` compares the old resolved profiles against the
new YAMLs. All three profiles match exactly, including system order, resolved
assets, materials, supports, contact settings, the 0.006 proxy radius, rigid
alignment and deterministic patch selection. Scene arrays also match exactly
for rest positions, masses, radii, tet/triangle topology, spring/volume settings,
pins, connector rest lengths and cavity vertices.

| Profile | Particles | Tetrahedra | Selected patch IDs |
| --- | ---: | ---: | --- |
| Block | 125 | 320 | 7, 2, 6, 8 |
| Original chole | 5612 | 18823 | 225, 234, 223, 226 |
| Life-size chole | 5612 | 18823 | 225, 234, 223, 226 |

The complete life-size YAML moved byte-for-byte into the surgical package.
The pre-existing deformation-only case and user edits were retained.

On two replicated worlds, both eager and graph modes passed:

- Block and original-chole scripted feasibility: 10/10 successes per fixture.
- Open-loop contact, hold and withdrawal with finite state.
- Selected reset: peers' state, commands, task/controller history and pending
  reactions remain bitwise unchanged.
- Hard rebuild, rebinding, fresh baselines and rejection of stale handles.
- One PPO learning update per profile, parameter-change/finiteness checks,
  checkpoint save/reload and 50 playback steps.

The existing mechanical comparisons passed without changed tolerances.
`comparisons/comparisons.json` retains every measurement. Block's maximum
repeated-run TCP noise was 1.92e-6, graph/eager TCP drift 1.58e-6 and independent
peer tissue difference 2.98e-8. For chole those observations were 6.80e-5,
8.37e-5 and **0.00403** respectively. Chole's independent-rollout tissue drift
remains an observation, not a passed tight equality gate. Peer contact force and
count were zero in both profiles; selected reset was checked separately within
each run.

Life-size scripted feasibility remains **0/10 successes, 10 failures**, exit 2.
No goal thresholds or mechanics were changed to make it pass.

## Input, GL and compatibility

The ported tests retain read-only observation/reward semantics, one history
advance per control step, input recovery/clutch/focus behavior, status guidance
and terminal-observation timeout bootstrapping. Added checks cover persistent
vertex transforms, surface indices, raw Follou units/zero force/cleanup, error
cleanup and metadata mismatch explanations.

A windowed replay recording was decoded and visually inspected: native arm,
aligned tissue, task status, camera controls and markers are present. Offscreen
GL recording and encoder cleanup also passed. Newton disables ImGui in headless
mode. Follou/OpenHaptics behavior is tested with fakes; physical device
qualification remains manual.

Both historical final-block/final-chole checkpoints loaded and completed 50
playback steps. Two 190-step input replays after reset produced identical TCP
and action trajectories for both profiles. Existing surgical policy and fake
direct-instrument teleoperation commands also passed. The missing-training-gate
negative check failed as required. Results are recorded in `smoke.json` and
`replay-check.json`. The matching scripted gate remains mandatory before
training, and checkpoint/gate metadata mismatches identify the differing fields.

## Performance

Paired old/new runs used the same local stack and GPU, one world, 30 warmup
steps and 150 measured steps. The test process was suspended during paired
measurements and resumed afterward. Both versions used the same explicit eager
or graph selection. Headless measurements reused zero actions; viewer
measurements used raw-position replay, offscreen GL and no video encoder or UI.
Physics and render phases synchronized the GPU. These timings do not establish
physical-device or windowed-interaction rates.

| Fixture | Eager physics before → after (ms) | Graph physics before → after (ms) | Graph change |
| --- | ---: | ---: | ---: |
| Block | 34.251 → 34.545 | 6.405 → 6.780 | +5.9% |
| Original chole | 49.260 → 48.554 | 12.136 → 11.837 | −2.5% |

No measured headless regression exceeded 10% in either mode. Graphs are now the
Versius default; `--no-graph` retains eager stepping.

| Offscreen replay fixture | Physics before → after (ms) | Viewer before → after (ms) | Total loop before → after (ms) |
| --- | ---: | ---: | ---: |
| Block | 6.805 → 7.117 | 3.054 → 1.739 | 11.148 → 9.838 |
| Original chole | 12.325 → 12.670 | 4.010 → 2.437 | 17.965 → 16.470 |

Measured loop improvement was 11.8% for block and 8.3% for chole. The shared
viewer transforms tissue into persistent device buffers and publishes live
triangle indices on device. The internal-transfer probe observed Newton's
per-frame vertex readbacks for the three chole parts (1986, 2688 and 938
vertices), plus its packed-shape transform path through pinned host buffers.
`viewer-internal-transfers.json` records the observed NumPy calls; these are
source-level observations, not driver-level byte counts. Video framebuffer
readback also remains; see the guide for those limits. No 50 Hz guarantee
is made.

## Stack and repository checks

The isolated environment is `.venvs/isaaclab-current`, with explicit external
Lab/Newton locations. It copies the existing qualified third-party environment;
the repository-pinned optional SlangPy 0.43.1 was installed without dependencies
for full renderer tests. Root pins and existing environments are untouched.
The patched Warp native hash remains
`eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
Three external Newton files already differed from the historical `stack.json`
record before the initial benchmark; this run does not claim that old manifest
is a current source snapshot. No external file was edited by this work.

The initial default-environment suite had 146 failures, 942 passes and 237 skips.
A clean HEAD source snapshot in the same environment reproduced 145 of those
failures. The additional private-reach inventory failure introduced by moving
the task was fixed and rechecked. The shared failure set includes unconfigured
production-data paths and the unpatched default environment's HashGrid capture
failures. An initial isolated-stack run without production data had 141 failures,
1016 passes and 266 skips; this was followed by a run with explicit production
data and the pinned optional renderer dependency.

The final configured run completed with **1377 passed, 46 skipped**, exit 0,
in 1585.46 seconds. The 516 reported warnings were not failures. The focused
task/boundary/pin suite completed with **61 passed**, and the repaired
private-reach check also passed in the default environment. See
`configured-repository-tests.log` and `focused.log`.
`qualified-stack.json` records actual imports, source revisions, dirty-file
hashes and read-only asset hashes for qualification; it is never generated
by ordinary interactive startup.
The core synthetic CPU example passed. All six package builds passed, and wheel
contents were checked for the moved modules and complete/contact/deformation
case YAMLs. Import-boundary and private-reach checks passed with the focused
suite. Historical results and qualification environments were preserved.
