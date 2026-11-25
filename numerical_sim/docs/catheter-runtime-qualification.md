# Catheter runtime qualification, 2026-11-25

Capability implementation: physics PR #19, stacked on #17 at
`f2799ce998204495a4e629deb81fe3c59de114b3`.
Consumer: [Workflows PR #309](https://github.com/isaac-for-healthcare/i4h-workflows/pull/309),
stacked on Workflows #292. Source proposals #8 and Workflows #252 are unchanged.

## Tested source and environments

Published immutable capability candidate: `827dc426888b614d4e75d4cd309e66f1ae9ce753`.
Its package runtime sources are identical to `d970d8707739b396184e7accc651bb96bb91da89`;
827dc426 changes only tests. The final full suite loaded the pre-correction registry
test at d970d870. All six operational Workflows references select 827dc426.
Later documentation commits do not replace that tested pin.

The physics environment uses the unchanged checked-in lock: Newton 7f0c13af6
(1.9 development), stock Warp 1.17.0, Torch 2.14. The native Workflows consumer
uses its independently approved Lab 3.0 / Isaac Sim 6.2.0.0 host stack, Newton
d31603ff, stock Warp 1.17.0 and Torch 2.11. No dependency versions, case assets,
acceptance thresholds or lockfiles were changed. Warp HashGrid remains opt-in;
its known failures were not patched or enabled to obtain a passing result.

## Results

| Check | Result |
|---|---|
| Complete `tests/runtime tests/packages` | 1207 passed, 23 skipped, **2 failed**, 81 warnings, 1217.77 s |
| Final command/material/guide/contact/parity compatibility group | 78 passed |
| Registry expectation plus body-integrator and positional-constructor follow-up | 20 passed |
| Generated CPU rod/tet/contact/fluid example | Passed |
| Six package builds and 12 wheel/sdist artifact checks | Passed |
| Private import boundaries and pre-commit | Passed |
| Production asset hashes | 87 checked, 0 fetched |

The complete suite used:

```bash
I4H_DATA="$PWD/data" uv run pytest tests/runtime tests/packages -q -ra --durations=20
```

The live-registry test still expected only `damping` and `max_velocity`; it now
also expects the declared live `gravity_scale`. The focused follow-up passes.
The full suite was not rerun after that test-only correction, and is not reported
as passing.

The remaining `test_chole_batched_matches_single` failure is retained: the
phase-separation check measured `others = 0.0030141472816467285`, below its
unchanged strict `> 0.005` requirement. No threshold or case mechanics were
adjusted and no repeated trial was used to replace that observation.

The fixtures exercise direct and PBD rods at E=1/2/16, eager/captured execution,
xyzw roll, complete-batch NumPy/Warp/Torch commands, stream ordering, guide
exclusion, persistent identity, independent State/Control binding, selected reset,
portable continuation and rebuild. They also cover straight material under
curved placement, authored material rest curvature, wire-only interior pull,
two-way wall contact, pinned nodes, exact overflow fallback, mounted reset,
colored-wall capture and the unchanged bent-lumen bounds.

## Consumer gates and limits

The unchanged frozen mechanics campaign passes **47/50**, not promotion.
`zero_response`, `arrival_positive` and `arrival_negative` each fail force
relative L2: 1.3897889852523804 versus 0.1. Their moments pass the previously
authorized absolute 1e-9 N m policy; loaded force/moment limits remain 10%.
No force tolerance was added. Historical archives with mismatched source hashes
are retained and excluded from the comparison.

Production s0011 free navigation passes exact State/observation and subsequent
trajectory continuation, selected reset, reconstruction and native command limits.
Production arm snapshots restore exactly, but subsequent arm/rod trajectories
diverge. The qualification driver now treats those measurements as failures.
Finite PPO updates and recording/replay are separate checks and do not waive
these continuation or mechanics gates. See the consumer qualification report for
all native, UI, timing and replay results.

The vascular checkout is independently dirty and was preserved. Its original
environment has Newton 1.6.1, incompatible with the parent candidate, and tests
import `i4h_isaaclab.composition`, already retired by #17. A candidate-environment
CPU subset produced 5 passed, 4 skipped, 1 such import failure. Full vascular
regression remains blocked; no compatibility alias was added to hide the issue.

No numerical default-switch PR is opened. These drafts neither clear the
retained chole/consumer/vascular blockers nor authorize a merge.

## Raw evidence

Physics logs, builds and the initial/interrupted failure records are preserved at:

`/home/pkorzeniowsk/Projects/i4h/qualification-evidence/catheter-runtime-20261125/physics`

Native and mechanics evidence, including frozen inputs, legacy/current traces,
patient provenance, screenshots and process exit statuses, is preserved at:

`/home/pkorzeniowsk/Projects/i4h/i4h-workflows-worktrees/numerical-catheter-runtime/runs/endoluminal_navigation/20261125_134900`

Arm evidence uses the sibling `runs/endoluminal_navigation_arm/20261125_134900`.
The Workflows report records the exact runtime and qualification-driver revisions
for each final campaign. Raw evidence is local; it is not attached to the PR.
