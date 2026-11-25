# C4.1 — Interaction history and tool-share residue as Newton State

**Implemented; the campaign is complete and its certificate status is open.** The only failed groups fall under the user's retained-failure rule, and progression to C4.2 is the user's decision. 36 history arrays of Centreline, InstrumentContact, Hook, Grasp, VolumetricGrasp and Clipping are Newton State. The single frozen campaign passed 34 of 36 groups and all three suites. Both failed groups are `aorta_shell` catheter-impulse products, a workload that holds none of the migrated State. **Stop for review before C4.2.** Nothing was pushed.

Unchanged: every numerical kernel (222) and every step, reset, refresh and prepare hook (151), verified by AST against `ebd86e2` ([audit][preserve]); capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins and the retained environments.

## Deviations from the approved plan

1. **Bitwise resume is claimed on the CPU only.** The plan asked for a bitwise resume comparison in CPU eager, CUDA eager and captured modes. On CUDA, two runtimes restored from the same checkpoint differ from each other after one step, in eager and captured execution alike ([eager][twin-eager], [captured][twin-captured]); stepping this case there is not bitwise repeatable, so no resume comparison can be bitwise. The CPU comparison is bitwise. On CUDA the tests check what can be checked deterministically: binding a fresh runtime changes no byte, its 36 arrays are views of the State it was bound to, and stepping one runtime leaves the other's containers untouched, also under capture.
2. **Two storage-layer extensions, not one.** Besides the typed vector view, `Declarations.add` accepts a uniform `default`. Such a field authors no per-row values and Newton fills it at finalization. Without it the six residue arrays would have added about 32 million Python values to chole's row authoring at 64 worlds.
3. **Centreline keeps one write at binding.** An existing test (`test_centreline_binding_uses_current_host_positions`) requires the tracker to follow a moved State at binding, so the plan's fallback applies: the constructor's reset is gone (templates carry the rest rows) and the `refresh` stays as a named exception. It writes only values derived from the bound State's particles; on a State saved after a completed step it rewrites identical bytes, which the tests check.
4. **VolumetricGrasp without a surface system authors its own particle rows** (user decision during implementation). Two existing test fixtures use that configuration. With a surface, as in chole, the existing rows are reused.
5. **Scratch poisoning covers contact records only.** The plan also listed `delta` and `delta_count`; they are self-clearing accumulators that must be zero between launches, not rewritten-before-read scratch, so poisoning them would rightly change results.
6. **Not done from the plan's test list:** two control systems in one scene with interaction systems on each, and moving the ownership test helpers of the three C3 modules into one shared module (the new tests have their own helper module).
7. **Grasp systems without instruments** were never constructible (they read the scene's instrument records) and still are not. The contact system without instruments is tested.

## What moved

| System | Arrays | Rows | Template |
| --- | --- | --- | --- |
| Centreline | `positions`, `active_count` | own node rows `(E,K)` | rest rows of each world's variant |
| InstrumentContact | `delta_k`, `delta_jaw` | surface particle rows `(E,Nb)`, tool width | zero |
| Hook | `verts`, `tet`, `bary`, `friction_coord`, `radial`, `active` | own anchor rows `(E,2)` | `verts`, `tet` -1, others zero |
| Grasp | nine slot fields; `count`, `pending`; `delta_k`, `delta_jaw` | own slot rows `(E,n,G)`; instrument rows; surface particle rows | `prim`, `tet` -1, others zero |
| VolumetricGrasp | four slot fields; `count`, `pending`, `claim`; `delta_k`, `delta_jaw` | own slot rows `(E,n,G)`; instrument rows; surface particle rows | `particle` -1, others zero |
| Clipping | `node`, `active`; `count`; `clipped` | own slot rows `(E,G)`; WORLD; centreline node rows | `node` -1, others zero |

- Declarations live in `surgical/src/i4h_surgical/interaction_storage.py` and use only the public storage API. Each system has a `declare_storage` hook; direct Scene callers use `declare_centreline`, `declare_instrument_contact`, `declare_hook`, `declare_grasp`, `declare_volumetric_grasp`, `declare_clipping`. Capacities come from the authored scene and the system settings.
- Slot and anchor rows reference the control system's instrument runtime rows; centreline nodes carry their world. Resets are byte-for-byte the previous behaviour, including the slot fields they leave stale; residue is never cleared by reset.
- Hook's column views (`entry_active`, `entry_tet`, contact `suppress_active`, `suppress_verts`) are registered identities. `Bleeding.clipped` still is `Clipping.clipped`. Hook's disabled-`prepare` zero of `active` stays, as a named exception.
- The allocation inventory lost the 36 State sites and one constructor reset mask and gained one view site: 475 sites in 188 files, checker passing.

## Tests

`tests/packages/test_interaction_owned_storage.py` runs on the real chole case. Pre-migration pins were committed first (`95db461`) on unchanged production code. As a post-campaign check, those 16 tests and their helper were checked out unchanged from `95db461` and run against `dbd5f3c`: 16 passed ([result](../.tmp_tests/stage-c41-20261110/pre-migration-pins/result.txt)). During the migration the module was extended and one pin was renamed and strengthened (binding a State saved after a step rewrites identical centreline bytes).

- Templates, 36 State names, pointers, shapes, identity columns and owners; live history never writes a template or another container.
- Episode reset clears exactly what it cleared before, in selected worlds only; cache-only preparation preserves history; replaced arrays (six cases, three modes) are rejected before any write; late, duplicate or dangling declarations and a changed capacity are rejected.
- **Resume, CPU, bitwise:** a scripted episode produces a surface grasp, a volumetric grasp and an applied clip by stepping. Two things are written into State directly before the save point, because the script does not reach them: a hook entry anchor on the fat triangle nearest the hook limb, and one sub-denormal residue value per share array. The hook is asserted active at the save point on the CPU; on CUDA the injected anchor is not always retained for two steps and is not asserted. State and Control are copied; a fresh runtime bound to the copy steps bitwise equal to the uninterrupted one for 12 steps.
- **Mounted tool, CPU, bitwise:** the grasper is carried two-way by a rigid body, holds tissue, and a reaction is pending on the body at the save point; the restored runtime steps bitwise equal for 10 steps. This closes the "mounted-tool episode restore remains partial" caveat for a rigid solver whose history is in native State. MuJoCo-Warp's private history stays excluded.
- Resume fixtures disable Thermal, Bleeding and the fluids (C4.2). Contact paint, cutting and stretch breaking stay enabled and write the system-owned appearance array; the comparison covers Newton containers only, so **appearance is not restored until C4.2**.

## Frozen campaign

The [certificate][cert] covers **89,103 files** at commit `dbd5f3c`, manifest SHA256 `f0ad8d130bdf0a0921d90a0e3ca60bfedf8b69872f31948b328c367164654d0b`. The independent auditor verified 85 startup and exit pairs and 104,557 file-backed import observations and recomputed every comparison from the retained arrays. One campaign, no retries, no reference promotion. It also qualifies the Lab 3.0 manager rename (`ebd86e2`).

| Suite | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Full core | 1,150 | 49 | 0/0 |
| Lab main six-module selection | 122 | 7 | 0/0 |
| Patched Lab 3.0 six-module selection | 121 | 8 | 0/0 |

| Environment / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-captured | pass | pass (3/3) | pass | **FAIL** | pass | pass |
| standalone-eager | pass | pass (2/3) | pass | **FAIL** | pass | pass |
| main-captured | pass | pass (3/3) | pass | pass | pass | pass |
| main-eager | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-captured | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-eager | pass | pass (3/3) | pass | pass | pass | pass |

**34/36 groups, 45/48 replays, 17/18 chole trials.**

- **Chole**, the only reference case that holds the migrated State, passed in all six groups. The one failed trial (standalone-eager trial 0) has eight failed products in the grasp, instrument_contact and tool_force families, the same eight products as the failed Lab 3.0 captured trial of C3.2.
- **aorta_shell** failed in both standalone modes, on stepped and masked-reset catheter impulse. The masked-reset rows equal the stepped rows of the unreset world, so each group is one excursion counted twice. The threshold is 4.63e-05.

| Mode | World | Reference | Candidate | Difference |
| --- | ---: | ---: | ---: | ---: |
| standalone-captured | 1 | -8.915209036786109e-05 | 2.846711504389532e-05 | 1.176e-04 |
| standalone-captured | 0 | -8.45981267048046e-05 | 1.504814008512767e-05 | 9.965e-05 |
| standalone-eager | 1 | -8.915209036786109e-05 | -4.1452385630691424e-05 | 4.770e-05 |

The world 0 value `1.504814008512767e-05` is identical to all digits to the failing value of C3.1 (Lab 3.0 captured) and of C2.3 (Lab main captured), read from their retained arrays. This is its third appearance, on three hosts and three source states. In this captured run both worlds deviate, where C3.1 and C3.2 showed one world per failing run. `aorta_shell` has no interaction system, but it does execute the changed storage layer; what bounds the change there is that its construction report is byte-identical to C4.0 and that all kernels and hooks are unchanged. These failures are retained without attribution, as the user decided; the certificate stores the exact endpoints for C3.1 to C4.1.

### Construction (one sample each)

| Workload | Worlds | Seconds | C4.0 seconds | Retained Warp bytes | C4.0 retained | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chole_tet_med | 1 | 0.874 | 1.129 | 30,049,229 | 27,823,129 | 2,226,100 |
| chole_tet_med | 64 | 14.935 | 15.000 | 1,361,783,246 | 1,219,312,846 | 142,470,400 |
| aorta_shell (rods) | 1 | 0.085 | 0.613 | 3,354,144 | 3,354,144 | 0 |
| aorta_shell (rods) | 64 | 0.612 | 0.652 | 172,785,410 | 172,785,410 | 0 |

The auditor explains the chole growth to the byte: at 64 worlds 140,979,968 bytes of templates for the 36 arrays plus 1,490,432 bytes of identity columns for 186,304 new slot, anchor and node rows. Live storage did not grow, because the arrays moved from system ownership into State. Each additional State container now carries the same 140,979,968 bytes. Construction time at 64 worlds did not grow. The six residue arrays keep their 129,300,480 bytes and now have a template each.

## Open and unverified

- Two `aorta_shell` groups failed; retained under the user's rule, cause unresolved.
- CUDA resume cannot be compared bitwise (deviation 1).
- Appearance, Thermal, fluids, Bleeding, GraphContainment and TipEvidence history remain system-owned until C4.2.
- HDF5 recording and replay: unverified, deferred by user choice. The clipping and fluid count intermittents did not recur in this campaign and remain unexplained; the fluid count one recurred once in the focused run after the Lab 3.0 rename.

[preserve]: ../.tmp_tests/stage-c41-20261110/preservation-audit.json
[twin-eager]: ../.tmp_tests/stage-c41-20261110/twin-resume-eager.txt
[twin-captured]: ../.tmp_tests/stage-c41-20261110/twin-resume-captured.txt
[cert]: ../.tmp_tests/stage-c41-20261110/certificate/frozen-campaign-certificate.json
