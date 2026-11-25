# C4.4 plan: private-name cleanup, final campaign, timing and closure

Implementer: Claude, with advisor review of the diff and the reports. Starts from commit `2e9b33b`. Approved by the user on 2026-11-12.

## Context

C4.3 delivered save and restore and left Stage C open on four stop conditions. Two of them are about code hygiene: `CoupledSolver` uses two private names of `I4hSolver`, and the host layer uses private names of the binding and solver. An inventory (2026-11-12) found 68 places where i4h code uses another i4h class's underscore-prefixed internals. C4.4 removes the ones in the simulator and its hosts, then runs the final frozen campaign and the timing campaign once on that code, and writes the closure report.

User decisions (2026-11-12):

- **Renames may touch hooks.** The cleanup may change step, reset and restore functions where the only change is a renamed attribute or method.
- **Scope: simulator and hosts.** Physics, endoluminal, surgical, assessment and the Isaac Lab adapter. The standalone renderer's internal sharing stays.
- **The final campaign and the timing campaign are the last steps of C4.4**, run once on the cleaned code.
- **Timing is report-only.** No gate; numbers against the Stage B references and against C4.0.
- **Archive: hashes only.** References and certificate stay in `.tmp_tests`; tracked SHA-256 lists are refreshed.
- Carried over: one campaign, no retries, failures retained; never widen tolerances; `aorta_shell` impulse failures retained without attribution; HDF5 deferred; no push.

Unchanged: every numerical kernel (AST-identical to `d3604e4`), hook order, behaviour of every hook, storage layout and byte counts, capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins, retained environments, historical certificates.

## Cleanup design

One public name per thing; no aliases kept for the old private names. Tests and scripts that use the old names are updated in the same commit.

| Owner | Private name used from outside | Replacement |
| --- | --- | --- |
| `I4hSolver` (`physics/src/i4h_physics/sim.py`) | `_check_state` (from `coupling.py`) | public `check_state(state, operation)` |
| | `_all` (from `coupling.py`) | existing `all_worlds` property |
| | `_command_views[id]` (from `bindings.py`, `component.py`) | public `command_view(system_id)` returning the record with `views` and `torch_views()` |
| | `_outputs` (from `stepper.py::signature`) | read-only `outputs` property |
| `OutputPublisher`, observables (`observables.py`, `coupling.py`, `newton_solver.py`, `newton_manager.py`) | `_identities`, `_rigid`, `_ours` | public fields `identities`, `rigid`, `ours` |
| `PhysicsBinding` (`bindings.py`) | `_runtime` | public `runtime` property |
| | `_output_arrays` (written by `newton_manager.py`) | method `register_outputs(arrays)` |
| | `_prepared` (written by `newton_manager.py`, read by `camera/tiled.py`) | `mark_prepared(signature)`, `invalidate_preparation()`, `prepared` property |
| | `_mask` (from `env.py`) and `Stepper._mask` (from `bindings.py`, `worker.py`, tests, `scripts/qualify_procedures.py`) | public `env_mask` on both |
| `Scene` (`scene.py`) | `_gravity` written by `newton_model.py` | method `set_gravity(...)` |
| | module names `_F32`, `_I32`, `_VEC3` | public module names |
| Rod modules | `material._membership`, `rod_pbd._qmul_np` | public module names |
| `CaseAuthoring` | `_mount_config` | `mount_config` |
| `CatheterBinding` (`endoluminal/bindings.py`) | `_sim` | `sim` |
| Assessment (`assessment/executor.py`, surgical and endoluminal `assessment.py`) | template hooks `_measure`, `_qualify`, `_bind_tool` | `measure`, `qualify`, `bind_tool` |
| `ConstructionContext` (`newton_manager.py`) | `_bind_main` attached by `newton_solver.py` | declared attribute `bind_main` |
| `I4hComponent` | `_order("in")` (from `manager.py`) | public `order_inputs()` beside the existing `sync_views()` |
| `I4hEnv` | `_case`, `_mid`, `_half` (from `recording.py`, standalone `devices.py`) | public `case`, `action_mid`, `action_half` |

Left as they are, and listed in the closure report:

- Every reach into third-party internals (79 sites: Isaac Lab 53, Warp 14, Newton 10, ctypes 2). Each one is tied to a pinned revision.
- The standalone renderer (`SlangRenderer`, `CudaDriver`; about 20 sites inside `standalone/render`).
- The `__i4h_context` and `_i4h_authoring` attributes that i4h stores on Isaac Lab and Newton objects.

## Steps

1. Save this plan as `docs/stage-c44-plan.md`, lint, commit. Evidence directory `.tmp_tests/stage-c44-<date>/`; record the starting commit.
2. **Checker first, own commit.** New `scripts/check_private_reaches.py` (AST-based): reports every `obj._name` where `obj` is not `self` or `cls`, private cross-module imports, and `getattr` with a private string, in the six package source trees. It carries an explicit allowlist with a reason per entry (third-party, renderer, stored attributes). At this commit the allowlist also holds the sites to be removed, so the checker passes; each later commit deletes entries. A test in `tests/packages` runs it.
3. **Cleanup in five commits**, each followed by the affected tests and the ownership suites in the core environment:
   1. solver, coupling and observables;
   2. binding, stepper and their host users (`newton_manager.py`, `newton_solver.py`, `env.py`, `camera/tiled.py`, `worker.py`), plus tests and `scripts/qualify_procedures.py`;
   3. scene, rod and authoring module names;
   4. assessment template hooks across the three packages;
   5. component, env and recording names.
4. **Preservation audit**, adapted from `.tmp_tests/stage-c43-20261111/preservation_audit.py`, against `d3604e4`:
   - all kernels AST-identical;
   - hooks AST-identical except a named list; each hook on that list must be identical after applying the rename table above. Any other difference is a stop.
5. Re-anchor the source hashes in `docs/stage-c-inventory.json`; `scripts/check_stage_c_inventory.py` must pass with 454 sites and no classification change.
6. **Auditor and launcher**, prepared before the freeze:
   - copy the C4.3 launcher and driver unchanged apart from paths;
   - the auditor derives the continuation byte count from the particle count and the row-domain counts in the sample, not from fitted constants;
   - construction samples must equal the C4.3 samples in every key except timings and device pointers;
   - **dry-run the auditor against the C4.3 evidence** and require it to pass there before freezing;
   - background waits watch a process id, not a command-line pattern.
7. Development selection in the three environments on the clean commit, then freeze and run **one final campaign** (15 jobs, as C4.3). No retries. Independent audit under the source guard.
8. **Timing campaign**, after the frozen campaign, on the same commit, GPU exclusive: `scripts/qualify_stage_b_timing.py` with the C4.0 options (16 workers, each once, `tissue_contact` and `aorta_static` at 1 and 64 worlds, both hosts, with and without exports), the cadence test, and `scripts/measure.py tissue_contact aorta_static --envs 1 64` in the same window. Report-only: every median against the Stage B references and against the C4.0 results. A busy GPU leaves timing pending and is reported; it does not block the rest.
9. **Closure documents:**
   - `docs/stage-c44.md`: cleanup table with what changed, audit result, campaign tables, failed endpoints, construction, timing tables, and the list of reaches left.
   - `docs/stage-c-closure.md` as outlined in the masterplan: ownership matrix per system, the three reset-like operations, continuation exclusions and the hosted call order, every retained failure with its evidence path, unverified items (HDF5), and the deferred list.
   - `docs/component-contracts.md` and the READMEs updated for the renamed public names.
   - `qualification/c44-certificate.sha256` for the final certificate; status prepended to `docs/stage-c.md`.
   - The reports state the disposition of each C4.3 stop condition: construction time (from the new samples), chole pattern (from the new campaign, compared product by product with C3.3 to C4.3), `CoupledSolver` reaches (removed), Lab call order (kept as a documented contract).
10. Advisor review, documents-only commit, full `pre-commit run --files $(git ls-files numerical_sim)`, manifest check against the final commit, no push. Stop: closing Stage C is the user's decision.

## Verification

- `scripts/check_private_reaches.py` passes with an allowlist that holds only third-party, renderer and stored-attribute entries.
- Preservation audit: zero kernel differences; hook differences only on the named list and only renames.
- Core suite `tests/runtime tests/packages` and both Lab six-module selections pass in the retained environments.
- Auditor recomputes suite, group, replay and chole counts; no production file differs between the freeze manifest and the final commit.
- Construction: bytes and row counts equal C4.3 exactly.
- Timing: 16 valid workers, no unrelated compute process in the observation log, references hashed, percentages recomputed from the JSON medians.

## Stop conditions (report and wait)

- A failed reference group other than `aorta_shell` catheter impulse, a chole group below 2 of 3, or a suite failure other than the two known intermittents.
- A hook or kernel difference in the preservation audit that the rename table does not explain.
- A construction sample whose bytes or row counts differ from C4.3.
- A rename that needs a behaviour change to keep a test passing.
- The auditor fails its dry run on the C4.3 evidence.

## Not in C4.4

Renderer internals, third-party private access, HDF5, physical rod units, external particle forces, contact geometry export, and any attribution of the chole or `aorta_shell` failures.
