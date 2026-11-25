# Stage B — focused Lab 3.0 parity investigation

The subsequent [capture correction](warp117-hashgrid-capture.md) fixes the
reproduced native defect and is qualified in [Stage B closure](stage-b.md).
This document preserves the preceding investigation-only state, including its
failed samples and the limits of attribution to the original lost observation.

## Decision, 2026-11-05

**Investigation completed; original parity failure remains unresolved.** No
production, dependency, case, comparator or test source was changed. The released
Newton particle grid stays released. This work does not qualify the failed host
suite or begin Stage C.

A separate, relevant defect is **reproduced**: the pinned Warp 1.17.0 CUDA
HashGrid implementation shares capture metadata between distinct grids of the
same scalar type. Replaying an earlier graph after capturing another grid can
query the other grid's buckets. The minimal reproduction uses no Newton model,
so this defect does not require the released Newton particle grid. The i4h Fluid
path reproduces it too.

This is a concrete interference mechanism in the parity test, which alternates
host and standalone captured simulations in one process. It is **not proof of
the original failure's field or cause**: that sample's log truncates both tensors,
and no complete observations were saved for that failed invocation.

## Bounded comparisons

Evidence: `.tmp_tests/stage-b-parity-investigation-20261105/`. Source hashes were
saved before investigation. The saved pre-grid source and released-grid source
use identical qualified Lab 3.0 dependencies, the original seed-3 actions, two
worlds, 40 policy steps, decimation two and the unchanged final
`torch.allclose(..., atol=1e-4)` assertion (default `rtol=1e-5`). There was no
repeat-until-pass loop or outlier-rate estimate.

1. **Original test ordering:** one fresh process per source collected the same
   seven host test files and ran the prefix through the failing test. Both
   reported 24 passed, 4 skipped and 81 deselected. The final parity check passed
   in each, with maximum difference 2.384185791015625e-7 in
   `stretch_breaking.max_stretch`. Only final-array readback was added, after the
   original comparison. This did not reproduce a dependency on preceding tests.
2. **Per-policy-step trace:** one fresh process per source ran the original test
   with diagnostic readbacks after reset and each policy step. All 40 observation
   comparisons passed at the original tolerance. Commands, inverse masses and
   active flags were byte-equal throughout. Returned host observations matched
   direct live-state observations. Initial positions and velocities matched;
   first recorded differences were in block positions after policy step one.
   These are recorded differences, not an attribution of the original failure.
3. **Fluid data outside the policy's position views:** both traces showed large
   fluid-position divergence even though their final policy assertion passed.
   The observation contains integrated-body positions and fluid/suction counts,
   not fluid positions. The first fluid-position difference was tiny after step
   one; it exceeded 1e-4 after step two. Maximum differences over the trajectory
   were 0.6693629324436188 case units before grid release and
   0.6692991554737091 after. No count differences occurred in these two traces.
4. **Sequential diagnostic variant:** once per source, complete the host rollout
   before constructing and replaying standalone, retaining the same command
   sequence and final assertion. Both assertions passed, but final fluid
   positions still differed by 0.3876904547214508 / 0.35524219274520874 case units
   (pre-grid / released). Therefore merely serializing the parity test is not a
   demonstrated complete correction; the actual test was not changed. These
   runs do not isolate every source of fluid divergence.

All diagnostic readbacks are confined to saved scripts and excluded from
production and performance runs. The host's exact 1/60 timestep and standalone's
case timestep 0.016666667 were retained and recorded, not silently normalized.

## Analytic capture-isolation reproduction

`hashgrid_repro.py` creates two independent float32 grids, with two separated
pairs of particles per grid. Within each pair separation is 0.1; query radius is
0.2. The second grid is translated by 10. The exact expected result is **two
neighbors per particle, including itself**, regardless of another grid's capture.

- Both eager builds return `[2, 2, 2, 2]`.
- After both graphs are captured, graph A returns `[0, 0, 0, 0]`; graph B returns
  `[2, 2, 2, 2]`. Three fixed replays preserve this failure.
- `hashgrid_controls.py` additionally verifies each graph immediately after its
  own capture: both are correct. After the second capture, A fails as above.
- An eager build of the corresponding grid immediately before replaying its
  **existing** graph restores the correct counts. This is a diagnostic
  intervention only, not a production workaround or a timing result.

The installed `warp/native/hashgrid.cpp`, function
`hash_grid_update_device_impl`, lines 319–351, uses one
`static HashGrid_t<Type> grid`. It records the host-to-device descriptor copy from
that shared address. Capturing another grid overwrites that staging descriptor;
the earlier graph's descriptor upload subsequently refers to the later grid's
bucket and point arrays. The code and dependency hashes are in `dependency.json`.
No dependency source or binary was modified.

`fluid_grid_distinct_cells.py` exercises actual `Fluid.substep_start()` with two
independent i4h simulations, two worlds apiece, four active particles per world,
zero gravity/velocity and a 0.012 neighbor radius. Each group occupies a 0.003
span; the second simulation is translated by 0.12. Every particle must find its
four group members. Both eager queries return four. After both captures, the
first graph returns zero for every particle; the second returns four. The
invariant assertion fails and its log is preserved.

The earlier `fluid_grid_repro.py` used translation 10 and passed: with radius
0.012 and a 32-cell table, those spatial buckets wrap close enough to conceal the
alias in this particular query. That passing control is retained alongside the
explicit distinct-cell failure; it is not used as evidence of isolation.

## What closes the host gate

The original failed sample remains a failed sample. The investigation supplies
a minimal dependency-level defect and shows the current interleaved comparison
can encounter invalid neighbor-grid ownership, but it has not recovered the
original differing field or eliminated all fluid divergence.

A corrective follow-up must make captured grid metadata independent, or use a
properly isolated comparison with evidence that it preserves the intended
coverage. It must then pass the analytic reproduction, affected fluid/capture
checks and focused host qualification at unchanged tolerances. Simply rerunning
the existing test, serializing it, disabling capture, adding an eager rebuild per
step, or declaring the failure numerical noise is not a demonstrated fix.

Chole disposition, exclusive timing/cadence, dependency pins, contact capacity
256 and the accepted coupling limitation are unchanged. No final Stage B
acceptance or Stage C entry approval is inferred from this investigation.
