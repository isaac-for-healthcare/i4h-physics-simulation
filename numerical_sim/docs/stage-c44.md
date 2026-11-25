# C4.4 — Private-name cleanup, final campaign and timing

**Implemented; the final campaign meets the closure rule, and all eight hosted timing comparisons are within 5% of the Stage B references.** The simulator and its hosts no longer use each other's underscore-prefixed names. The single frozen campaign on `95c8f4c` passed 35 of 36 groups, with the one failure in `aorta_shell` catheter impulse, and all three suites with zero failures. The timing campaign on the same commit was run report-only. Closing Stage C is the user's decision; the summary for it is [stage-c-closure.md](stage-c-closure.md). Nothing was pushed.

Unchanged: all 223 kernels, verified by AST against the C4.3 commit `d3604e4`; storage layout and every byte count; capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins and the retained environments.

## Deviations from the approved plan

1. **Groups 2 to 5 were tested together.** The plan said each cleanup commit is followed by its tests. The first commit was tested alone (665 tests); the other four were tested in one trial run of all three suites, then again in the development selection.
2. **Some names differ from the plan's table.** Delivered: `measure_step` (not `measure`, which is a module function), `resolved_case`, `adopt_gravity`, `region_membership`, `bind_callback`, `VEC3_BYTES`/`F32_BYTES`/`I32_BYTES`, and `mark_prepared()` without an argument.
3. **A standalone `PhysicsBinding` has no `env_mask`.** The stepper and the hosted binding have it; the standalone binding forwards to its stepper.
4. **The timing harness ran twice.** A full 16-worker run before the freeze, on the same commit, proved the renamed code still runs under it; an auditor self-comparison ran on the CPU at the same time, so it is not a clean measurement. The reported campaign ran once after the frozen campaign. Both sets of percentages are in the timing section.
5. **Standalone timing was bracketed, not monitored.** Two GPU process snapshots before and after show no compute process; the hosted campaign was monitored continuously.
6. **The auditor does not derive the continuation bytes from row-domain counts**, as the plan said. For C4.4 it requires every construction count to equal C4.3. In its dry run on the C4.3 evidence it accepts the byte count the code reports and checks only that it is consistent with the particle count.
7. **The preservation audit covers the physics, surgical and endoluminal packages only.** The assessment, Isaac Lab and standalone changes are covered by the suites alone.
8. **The certificate's status field reads "OPEN".** The inherited auditor writes "complete" only at 36 of 36 groups. The closure rule allows the one failure.

## Cleanup

| Owner | Private name used elsewhere | Public name now |
| --- | --- | --- |
| `I4hSolver` | `_check_state`, `_all`, `_command_views[id]`, `_outputs` | `check_state`, `all_worlds`, `command_view(id)`, `outputs` |
| Observable containers | `_identities`, `_rigid`, `_ours` | `identities`, `rigid`, `ours` |
| `PhysicsBinding` | `_runtime`, `_output_arrays`, `_prepared`, `_mask` | `runtime`, `register_outputs()`, `prepared` / `mark_prepared()` / `invalidate_preparation()`, `env_mask` |
| `Stepper` | `_mask`, `_prepared` | `env_mask`, `prepared` |
| `Scene` and its module | `_gravity` (written), `_F32`, `_I32`, `_VEC3` | `adopt_gravity()`, `F32_BYTES`, `I32_BYTES`, `VEC3_BYTES` |
| Rod modules | `_membership`, `_qmul_np` | `region_membership`, `qmul_np` |
| `CaseAuthoring` | `_mount_config` | `mount_config` |
| `CatheterBinding` | `_sim` | `sim` |
| Assessment targets | `_measure`, `_qualify`, `_bind_tool` | `measure_step`, `qualify`, `bind_tool` |
| `ConstructionContext` | `_bind_main` | `bind_callback` |
| `I4hComponent` | `_order("in")` | `order_inputs()` |
| `I4hEnv` | `_case`, `_mid`, `_half` | `resolved_case`, `action_mid`, `action_half` |

Old names were removed, not aliased. Tests and `scripts/qualify_procedures.py` use the new names.

**Checker.** `scripts/check_private_reaches.py` fails on any use of another object's private name that is not listed with a reason, and on any stale entry; `tests/packages/test_boundaries.py` runs it. It lists 50 remaining uses in 20 files:

| Reason | Where |
| --- | --- |
| Isaac Lab internals at the pinned revisions | `lab30.py`, `env.py`, `recording.py` |
| Newton internals at the pinned revision | `storage.py`, `newton_compat.py`, `observables.py`, `newton_solver.py` |
| Warp array lifetime reference and dtype layout | `storage.py`, `commands.py`, `continuation.py`, `rod_storage.py`, `instrument_storage.py` |
| ctypes structure layout | `xr/session.py` |
| i4h attribute stored on a host or Newton object | `authoring.py`, `lab30.py`, `newton_manager.py` |
| Own instance in the binding's alternate constructor | `bindings.py` |
| Standalone renderer sharing inside its package (out of scope) | `render/` |

The checker does not see private methods of third-party base classes that i4h overrides or calls through `self` (for example the Isaac Lab env hooks).

**Preservation audit** (`.tmp_tests/stage-c44-20261112/preservation-audit.json`): no kernel differs. Six hooks differ, and each is identical to its old version after applying the renames: `step`, `reset` and `restore_episode` of `I4hSolver` and of `CoupledSolver`.

Changes that are more than a renamed name, all without a change in behaviour:

- new accessors: `I4hSolver.outputs` and `command_view`, `PhysicsBinding.runtime`, `prepared`, `mark_prepared`, `invalidate_preparation` and `register_outputs`, `Stepper.prepared`, `Scene.adopt_gravity`, `I4hComponent.order_inputs`;
- `ConstructionContext.prepare` and `publish` call those binding methods where they assigned private attributes;
- `TiledCamera.prepare` reads `prepared`; `check_host_gravity` calls `adopt_gravity`;
- `detach` reads the declared `bind_callback` attribute where it used `getattr` with a default.

**Inventory:** 454 sites in 192 files, no classification change; two sites changed expression through a rename.

## Frozen campaign

Commit `95c8f4c`, 89,145 files, manifest `a88ce18f…72dd67ff`, 15 jobs, no retries, every job source-verified. Certificate: `.tmp_tests/stage-c44-20261112/certificate/frozen-campaign-certificate.json`; its files are hashed in `qualification/c44-certificate.sha256`. The auditor was dry-run on the C4.3 evidence before the freeze and reproduced that certificate.

| Gate | C4.4 | C4.3 |
| --- | --- | --- |
| Reference groups | 35 of 36 | 34 of 36 |
| Replays | 46 of 48 | 42 of 48 |
| Chole trials | 17 of 18; every group at least 2 of 3 | 14 of 18 |
| Core suite | 1,261 passed, 49 skipped, 0 failed | 1,260 passed |
| Lab main suite | 123 passed, 7 skipped, 0 failed | 123 passed |
| Lab 3.0 suite | 122 passed, 8 skipped, 0 failed | 122 passed |

- **Failed group:** Lab 3.0 eager `aorta_shell`, catheter impulse, stepped and masked reset: 4.998e-05 against a limit of 4.626e-05.
- **Failed chole trial:** standalone captured, trial 1, 19 comparisons, all in products that also failed in the C3.3 to C4.3 campaigns (impulse, jaw impulse and moment of Grasp, InstrumentContact and VolumetricGrasp; ToolForce force and jaw load). The eight products that were new in C4.3 did not fail.
- **Development selection** on the same commit: all three suites passed, including the chole isolation test that failed once before the C4.3 freeze.

### Construction (one sample each)

| Workload | Worlds | Seconds | C4.3 | Retained bytes |
| --- | ---: | ---: | ---: | ---: |
| chole | 1 | 1.478 | 0.930 | 30,054,199 |
| chole | 64 | 15.320 | 14.843 | 1,362,101,263 |
| `aorta_shell` | 1 | 0.691 | 0.094 | 3,374,898 |
| `aorta_shell` | 64 | 0.745 | 0.701 | 174,113,351 |
| `aorta_centreline` | 1 | 0.264 | 0.202 | 824,588 |
| `aorta_centreline` | 64 | 0.659 | 0.642 | 49,519,707 |

- **Bytes:** every count, shape, layout and row count equals C4.3.
- **Time:** chole at 64 worlds is +3.2% and `aorta_centreline` at 64 worlds +2.6%, inside the 10% stop limit.
- **One-world samples:** chole and `aorta_shell` at one world each spent about 0.5 s more in runtime binding (0.21 s to 0.73 s and 0.008 s to 0.593 s). Each is the first sample of its script after the freeze; the 64-world sample of the same script, run next, binds in 0.74 s and 0.03 s. That pattern fits a one-time kernel load. It was not investigated.

## Timing campaign (report-only)

Run once on `95c8f4c` after the frozen campaign: 16 workers, all valid; 387 process observations with no unrelated compute process; zero replay allocation in every run; cadence test passed. Evidence: `.tmp_tests/stage-c44-20261112/timing/`.

Median step time in ms, against the Stage B references and against C4.0:

| Workload | Worlds | Host | Stage B reference | C4.0 | C4.4 | Against Stage B | Against C4.0 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `tissue_contact` | 1 | Lab 3.0 | 0.362472 | 0.376695 | 0.373802 | +3.13% | -0.77% |
| `tissue_contact` | 1 | Lab main | 0.361239 | 0.378598 | 0.374735 | +3.74% | -1.02% |
| `tissue_contact` | 64 | Lab 3.0 | 0.529243 | 0.569134 | 0.542827 | +2.57% | -4.62% |
| `tissue_contact` | 64 | Lab main | 0.528342 | 0.569247 | 0.543481 | +2.87% | -4.53% |
| `aorta_static` | 1 | Lab 3.0 | 7.789783 | 7.405409 | 7.744950 | -0.58% | +4.59% |
| `aorta_static` | 1 | Lab main | 7.963070 | 7.409723 | 7.739654 | -2.81% | +4.45% |
| `aorta_static` | 64 | Lab 3.0 | 10.560836 | 9.819630 | 10.241552 | -3.02% | +4.30% |
| `aorta_static` | 64 | Lab main | 10.270203 | 9.821954 | 10.235698 | -0.34% | +4.21% |

- All eight comparisons are within +5% of the Stage B references. The two `tissue_contact` 64-world comparisons that exceeded 5% in C4.0 (+7.5%, +7.7%) are now +2.6% and +2.9%.
- `aorta_static` is 4.2% to 4.6% slower than in C4.0 and still at or below its Stage B references. The causes of either movement were not investigated. The two campaigns ran two days apart on a shared machine, and the OS kernel changed in between (`7.0.0-42` to `7.0.0-43`; GPU driver 615.71.09 at C4.4).
- The pre-freeze run on the same commit gave, against Stage B (Lab 3.0, Lab main): `tissue_contact` +3.42% and +3.91% (1 world), +2.91% and +3.17% (64 worlds); `aorta_static` +1.63% and -2.80% (1 world), -1.38% and +2.81% (64 worlds). The two runs differ by up to 3.2 points on `aorta_static`, which bounds what one run can resolve.
- Export overhead is at most +0.22%.
- Construction took 0.67 s to 1.37 s per hosted run.

Retained and peak Warp bytes per hosted run against the Stage B Lab 3.0 references (the Lab main references do not record memory):

| Workload | Worlds | Stage B retained | C4.4 retained | Stage B peak | C4.4 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tissue_contact` | 1 | 6,064 | 8,546 | 16,779,500 | 16,783,484 |
| `tissue_contact` | 64 | 310,669 | 424,346 | 16,899,767 | 17,056,967 |
| `aorta_static` | 1 | 5,855,154 | 7,143,540 | 16,822,188 | 18,938,172 |
| `aorta_static` | 64 | 18,163,527 | 21,198,588 | 20,486,429 | 24,239,237 |

The growth is the Newton-owned storage added in Stage C: initialization templates beside each State and Control field, row identity columns, and the continuation row maps.

Standalone (`scripts/measure.py`, p50 step time in ms, not gated):

| Workload | Worlds | Stage A | C4.0 | C4.4 |
| --- | ---: | ---: | ---: | ---: |
| `tissue_contact` | 1 | 0.351 | 0.369 | 0.370 |
| `tissue_contact` | 64 | 0.526 | 0.537 | 0.539 |
| `aorta_static` | 1 | 7.378 | 7.395 | 7.806 |
| `aorta_static` | 64 | 10.183 | 9.907 | 10.417 |

Standalone `aorta_static` is 5.8% (1 world) and 2.3% (64 worlds) above Stage A, and 5.6% and 5.1% above C4.0. Standalone timing has no gate and was not monitored.

## After the freeze

Committed after the campaign: this report, the closure summary, the roadmap status, `qualification/c44-certificate.sha256`, and the status line of the three `qualification/c30-*.json` records (they now say the final campaign ran in C4.4). Those three JSON files therefore differ from the frozen manifest; they hold no code and nothing reads the status. No source, test or script changed after the freeze.

## C4.3 stop conditions

| Stop condition | Outcome |
| --- | --- |
| `aorta_centreline` construction at 64 worlds +30% | Not repeated: 0.659 s against 0.642 s in C4.3 |
| New chole failure pattern | Not repeated: one failed trial, known products only. The C4.3 occurrence stays unexplained |
| Two private reaches of `CoupledSolver` | Removed |
| Lab cache reset can follow a restore | Unchanged: the documented call order is the contract. Awaits the user's acceptance |

## Open and unverified

- Whether the Lab call-order contract is accepted, and whether Stage C closes.
- `aorta_shell` catheter impulse failures and failed chole trials, retained without attribution.
- Third-party private access (79 sites by the earlier inventory) and the renderer's internal sharing.
- Tet containment fresh-runtime resume; CUDA bitwise resume for chole and both aorta cases.
- HDF5 recording and replay: deferred by user choice.
- References and certificates exist only in the git-ignored `.tmp_tests`; the repository holds their hashes.
