# Stage B v3 scenario qualification

B1 was accepted on the paired v2 evidence; see [the decision](stage-b.md).
The diagnostic six-run v2 envelope is not a new baseline. All v2 archives remain
untouched. This qualification changes scripted scenarios, not comparison rules
or numerical mechanics.

## Workload contract

`scripts/record_baseline.py` declares `physical-history-measurements-v3`.
It records the same explicit physical, history and measurement products as v2,
with the same complete-record slot canonicalization. Dtypes, shapes and product
sets must match. Zero-noise products must match byte for byte, including signed
zero. Other products use the measured six-run envelope multiplied by three;
both tolerance floors remain zero. Non-finite values fail. Every repeat runs in
a fresh process. Manifests include the fixed scenario schedule and reject a
replay with a different schedule or repetition contract.

The default workload has two worlds. Three cases retain the 60-step sinusoidal
script. The three revised cases use fixed schedules, without selecting checkpoints
based on whether their arrays happen to pass:

- `chole_tet_med`: leave other tools at episode defaults. Move the grasper 85%
  of the way from its initial root toward the authored gallbladder centroid over
  100 steps; close fully at step index 80. Require an engaged grasp from step
  120 onward. Retract 10% of the approach distance over 30 steps while keeping
  the jaws closed, then hold for 60 steps, recording at step 210. Failure to engage is a
  scenario failure, not permission to discard that repeat.
- `aorta_centreline`: drive straight insertion at `[0.3, 0, 0]` for 60 steps,
  hold zero commands for 1,800 steps, invalidate the containment reference cache,
  then run one normal physics step to rebuild it and hold for another 180 steps
  before recording. The reference
  therefore samples the relaxed vessel rather than the earlier build phase.
  Check that the rebuild flag was consumed. No history reset or velocity
  overwrite is used to create the checkpoint.

- `aorta_shell`: drive straight insertion at `[0.3, 0, 0]` for 60 steps, then
  hold zero commands for 360 steps. Require a nonzero containment count in every
  world at the checkpoint.

Regional parameter edits and initial/stepped/masked-reset products are retained.
Frozen v2 loaded each named case correctly before its regional-edit loop, then
unconditionally dispatched the same sinusoidal action loop for all six cases.
Its shadowed case variable changed the diagnostic case label, not that version's
action dispatch. V3 names the regional-edit local `parameter`, preserving the
case identity for v3's subsequently introduced case-specific scenario dispatch.

## Preserved trials

Evidence lives under `.tmp_tests/stage-b-v3/` (ignored local artifacts).
`final-b1-source/` preserves all six packages, the recorder and tests before host
replacement. `prechange/` contains the preserved pre-change packages with the
same v3 recorder; no original source archive was edited.

`trial-reference/` contains the first six-repeat trial of the two revised cases.
All six pre-change chole replays passed. Its catheter script used bending and
600 settling steps; one of six pre-change replays failed on masked-reset catheter
velocity. Position probes still showed appreciable relaxation at that phase.
The failed trial and every comparison are retained.

`settled-reference/` tests the final straight-insertion/1,800-step schedule.
All six pre-change replays passed. Vessel position and rebuilt-reference spreads
were about 0.00103 and 0.00117 case-distance units. Catheter velocities still
contain occasional contact-related spikes (six-run maximum spread 0.5073), so
passing this finite sample is not a claim of deterministic floating-point output.

`baseline-v3/` preserves the full six-case trial with an immediate post-rebuild
checkpoint. Three of its six full pre-change replays passed. The other three failed on:
catheter velocity (difference 0.5174 against tolerance 0.00893), catheter-control
moment (0.00679 against 0.0038), and one chole contact overflow counter
(`[784, 792]` against six `[792, 792]` references). Catheter positions and
reference samples passed.
The final scenario adds a fixed 180-step hold after rebuilding, so the checkpoint
also follows a settling period after that operation. Qualification of that
scenario is recorded below. The chole schedule also adds the
short retraction described above: the probe kept both worlds decisively grasped
throughout it. Contact-record capacity is already exceeded by this grasp workload
(the counter records dropped records, not a boolean); retraction does not remove
that limitation. These counters remain part of the unchanged recorder contract.

`baseline-v3-settled/` retains the four unchanged cases from their original six
B1 repeats and records six new repeats of each revised scenario. Package source
is identical throughout. Each case retains its original recorder hash; the only
recorder differences are the two explicit scenario schedules.

## Isolation policy

`test_chole_batched_matches_single` keeps the 0.2 own-world/other-world limit.
It performs three complete batched-versus-single comparisons and requires at
least two to satisfy that limit for every world. Every ratio is recorded in the
`chole_isolation` pytest property. The per-world phase separation and exact
integer topology checks still apply to all trials. The paired B1 follow-up
measured one failure in three pre-change runs; the policy addresses that measured
flake without weakening the numerical limit.

The six full pre-change replays against `baseline-v3-settled/` passed five times.
Both revised scenarios passed all six. The remaining failure was in the unchanged
`aorta_shell` script: masked-reset catheter-control impulse differed by 0.0003562
against a 0.0003000 tolerance. On 2026-11-04 the user explicitly selected revising
that scenario before host replacement. Host replacement did not begin on the
basis of this partially passing archive.

## Shell scenario and isolation verification

The revised shell scenario drives `[0.3, 0, 0]` for 60 steps and holds zero
commands for 360 steps. A probe retained one containment contact in each world
at every 60-step sample. Maximum velocity fell from 0.0279 at the end of insertion
to 0.000163 at the checkpoint; maximum motion over the final 60 steps was
0.0000705. The recorder requires a nonzero containment count in every world for
both settled catheter scenarios, preventing a vacuous no-contact baseline.

The majority isolation test passed (`isolation.log`, `isolation.xml`). It ran all
three comparisons, stored all nine individual ratios with `record_property`, and
also stored the complete comparison data in the `chole_isolation` property.

`baseline-v3-final/` contains six repeats per case from the unchanged final B1
packages: the three revised scenarios and the three unchanged scenarios. Earlier
trial archives remain available. The shell's new recordings and full pre-change
verification are in `record-shell-final.log` and `qualified-prechange-*.log`.

## Gate result

The final full pre-change replay passed all six cases (exit 0). Each revised
scenario also passed six pre-change checks against exactly the arrays retained
in the final archive. All cases have six final-B1 repeats and identical package
source hashes. `final-archive-index.json` records the immutable archive hashes.
The v3 prerequisite is met; host replacement starts after this result. The
diagnostic six-run v2 envelope remains unpromoted. Earlier failed trials above
are retained, and no floors, factor, slots, mechanics or pins changed.

The isolation ratios by trial were `[0.01997, 0.13897, 0.04640]`,
`[0.00889, 0.25940, 0.10079]`, and `[0.04995, 0.01859, 0.03332]`: two complete
trials passed, one failed, so the requested majority policy passed.

## Later host holdout

After replacement, a main holdout failed eight chole force/moment products. A
paired diagnostic reproduced those same eight fields (plus their masked-reset
counterparts) in preserved accepted B1. Six diagnostic runs per source passed
6/6 pre-change, 5/6 B1, and 6/6 main. The original main failure remains evidence;
this diagnostic is not a promoted envelope or an automatic gate acceptance.
See [the adapter report](stage-b-host.md). The frozen archive is unchanged.
