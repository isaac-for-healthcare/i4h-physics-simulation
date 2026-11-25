# Stage C closure summary

**Stage C is closed (user, 2026-11-16), with the two review defects below deferred.** Newton owns the simulation data, an episode can be saved and resumed, and the final frozen campaign on `95c8f4c` meets the closure rule. This document states what is owned where, what the reset-like operations guarantee, what is excluded, and every failure and gap that is retained. Detailed evidence is in the checkpoint reports `stage-c30.md` to `stage-c44.md`.

## Final qualification

Subsequent work: the [2026-11-16 current-source assessment](current-stack-assessment-20261116.md) checks newer local Lab/Newton sources and records a focused compatibility patch. It is separate from the frozen qualification below; its results and limitations do not replace that evidence.

| Item | Result | Evidence |
| --- | --- | --- |
| Reference groups | 35 of 36; the failure is `aorta_shell` catheter impulse on Lab 3.0 eager | [stage-c44.md](stage-c44.md) |
| Chole trials | 17 of 18; every group at least 2 of 3 | same |
| Test suites | core 1,261, Lab main 123, Lab 3.0 122 passed; 0 failed | same |
| Step time | all eight hosted comparisons within +5% of the Stage B references (report-only by user decision); standalone `aorta_static` at one world is +5.8% over Stage A | same |
| Kernels | unchanged by AST across C4.0 to C4.4 | each checkpoint's preservation audit |
| Stack | Newton `7f0c13af` (1.9.0.dev0), Warp 1.17.0, MuJoCo and MuJoCo-Warp 3.17.0, Lab main `4cfdfcfa`, patched Lab 3.0 `e53cd288` | `qualification/c30-*.json` |

The closure rule (user, 2026-11-10): Stage C may close if the only failed reference groups are `aorta_shell` catheter-impulse products, with endpoints retained and the limitation stated.

## Ownership

Newton Model, State and Control hold the fields below, declared on the builder before finalize. Systems hold views of them. The table is by data family, not per system. Its chole rows were checked against the declarations of a bound `chole_tet_med` runtime; the rod and catheter row is taken from the C3.2 report.

| Data | Owner | Since |
| --- | --- | --- |
| Particle positions, velocities, forces, previous positions and live flags | State | C2.1 |
| Tet mechanics storage and topology masks | State | C2.3 |
| Surface live topology and support counters | State | C2.4 |
| Fluid pool particles | State | C2.2 |
| Instrument poses, jaw state and commands | State and Control | C3.1 |
| Rod frames, angular velocity, material tables; catheter commands | State, Model and Control | C3.2 |
| Mount flags and transforms | Model and Control | C3.3 |
| Centreline positions and counts; Hook, Grasp, VolumetricGrasp and Clipping slots; the six tool residue arrays | State | C4.1 |
| Heat, burn, coagulation, surface colour; fluid, bleeding, emitter, smoke and suction counters; graph containment candidates | State | C4.2 |
| Rest geometry, per-variant tables, parameters, initialization templates | Model | C2 onward |

Not Newton-owned, by decision or by kind:

- **TipEvidence and Procedure state** are host-owned assessment state (user decision, C4.2).
- **Scratch** rewritten before it is read inside a step: constraint deltas, contact records, containment `sides`, fluid neighbour lists and the hash grid, `heat_next`.
- **Outputs** of a completed step: impulses, moments, ToolForce force and jaw load, observables.
- **Search trees** of FluidContact, PartContact, ShellContainment and TetContainment: built at binding, refit while stepping, rebuilt by save and restore.

The allocation inventory (`docs/stage-c-inventory.json`, 454 sites in 192 files) records every allocation with its contract; no State site remains on a system owner.

## Three reset-like operations

| Operation | Call | Guarantee |
| --- | --- | --- |
| Episode reset | `Stepper.reset`, `PhysicsBinding.reset`, `I4hSolver.restore_episode` | Selected worlds return to authored state, history and command defaults; unselected worlds are bitwise unchanged; an empty selection does nothing |
| Cache reset | `I4hSolver.reset(state, mask, flags=0)` | Derived caches are refreshed from current state; history is kept |
| Continuation | `save_i4h` / `restore_i4h` and their binding and component forms | Every declared State and Control field and the scene's own particle rows are copied between containers of one Model; no reset or preparation runs |

Each system's reset keeps its exact earlier behaviour, including what it leaves alone: stale Grasp, VolumetricGrasp and Hook slot bytes, the drag node and target, mounted flags and the instrument's current pose. Three lifecycle writes outside episode reset remain as named exceptions: Hook's disabled `prepare` zeroes `active`, GraphContainment's `prepare` forces a rebuild, and Centreline rewrites its positions and counts from the current particles at binding.

## Continuation contract

- **Recipe:** bind, prepare, then restore.
- **Hosted order:** write the host's own state, call the simulation's `forward()`, and restore the i4h rows last. Otherwise Isaac Lab's cache reset follows the restore and replaces graph containment's saved rebuild request.
- **Side effect:** save and restore rebuild the contact search trees. A runtime that saves differs at rounding level from one that never saved.
- **Masks:** a masked call leaves unselected worlds' State and Control bytes unchanged. Bitwise resume is claimed for unmasked calls only.
- **Bitwise resume, tested:** on the CPU for the rod, mount, chole, `aorta_centreline` and `aorta_shell` cases; on CUDA for the rod and mount fixtures only. Chole and both aorta cases are not repeatable on CUDA, so there the tests check copy exactness.
- **Excluded:** a rigid solver's own history (including MuJoCo-Warp's), host time and episode counters, assessment, procedure and tip evidence, renderer state, and HDF5 recordings. With a rigid solver the host restores body state itself.

## Retained failures

None of these was attributed; the user declined attribution work for the first.

| Failure | Where seen | Evidence |
| --- | --- | --- |
| `aorta_shell` catheter impulse above three times its noise floor | Zero to two of six groups per campaign (none in C4.0 and C4.2; two in C4.1 and C4.3; Lab 3.0 eager in C4.4) | exact endpoints for C3.1 to C4.4 in the C4.4 certificate |
| Chole trial failures within the 2-of-3 rule | One to four of 18 trials per campaign; four in C4.3, with eight products not seen before or since | each certificate's `failures` list |
| Clipping count intermittent (`test_tracker_and_clip`) | Earlier campaigns; did not recur in C4.2 to C4.4 | `stage-c30.md` to `stage-c41.md` |
| `two_worlds` fluid count intermittent | Earlier campaigns; did not recur in C4.2 to C4.4 | same |
| `test_chole_batched_matches_single` over its 0.2 limit | Once in the C4.3 development selection; single trials over the limit in C4.0 and C4.1 | `stage-c43.md` |

Certificates and approved references are in the git-ignored `.tmp_tests`. The repository holds their SHA-256 lists: `qualification/approved-references.sha256`, `qualification/c43-certificate.sha256` and `qualification/c44-certificate.sha256`.

## Unverified

- **HDF5 recording and replay**: deferred by user choice; `h5py` is absent from the retained environments.
- **Tet containment** resume through a fresh runtime: no authored case uses it.
- **CUDA bitwise resume** for chole and the aorta cases.
- **Physical calibration**: forces, lengths and rod stiffness are in case units.
- **Devices, haptics, XR and rendering** are outside every Stage C campaign.

## Known limits of the code

- **Third-party private access** remains in the Isaac Lab 3.0 bridge, the env hooks, the storage layer (Newton custom-attribute internals) and Warp array views. Each is tied to a pinned revision; `scripts/check_private_reaches.py` lists them.
- **The standalone renderer** shares private names inside its own package.
- **Construction time** at 64 worlds is about 15 s for chole, of which about 12 s is authoring custom rows.
- **Memory:** chole retains 1.36 GB of Warp memory at 64 worlds, including initialization templates.

## Deferred beyond Stage C

- **P2: standalone binding settings bypass capture invalidation** (`physics/src/i4h_physics/bindings.py`). Setting `sim.physics.substeps`, `iterations` or `debug` writes the solver's value rather than the Stepper's. After preparing four substeps, setting `sim.physics.substeps = 5` permits stepping without recapture: a gravity-only probe produced `z=-0.006` eagerly and `z=-0.00625` captured, although both reported five substeps. Calling `prepare()` reverts the setting. Until fixed, change settings on the standalone Stepper (`sim.substeps`, `sim.iterations`, `sim.debug`) and call `sim.prepare()`.
- **P2: observable publishing retains removed systems** (`physics/src/i4h_physics/observables.py`). The publisher caches source instances when allocated. Removing an instrument-contact system and its force consumer, then calling `prepare()` and `step()`, still exported the removed source's stale force (approximately `-14,400` in the review probe). Until fixed, reconstruct the runtime when changing system membership. The fix should reject membership changes after binding or rebuild and validate the publisher's sources.
- External `particle_f` consumption.
- Contact geometry export and `soft_contact_capacity`.
- Physical rod units.
- Native Rod and VBD mechanics, and migration to Newton's coupled solver.
- A bulk-authoring request to Newton for the 64-world construction time.
- An archive location for the references and certificates.

## Decisions

On 2026-11-16 the user authorized closure of the refactor and a push to the current remote branch:

1. Stage C is closed under the 2026-11-10 closure rule, with all retained failures, exclusions and unverified items above preserved.
2. The hosted restore order remains the documented contract: write host state, call `forward()`, then restore i4h rows last.
3. The two P2 review defects above are explicitly deferred beyond Stage C; closure does not claim they are fixed.
4. Publish the current branch; merging to main is separate work.

The independent review on `ade243f` passed core **1,261** tests (**49** skipped), Lab main **123** (**7** skipped), and Lab 3.0 **122** (**8** skipped), with zero failures. The allocation inventory and stored C4.4 certificate/reference hashes also passed verification. These are review checks, not a new frozen reference or timing campaign. Closure changes documentation only.
