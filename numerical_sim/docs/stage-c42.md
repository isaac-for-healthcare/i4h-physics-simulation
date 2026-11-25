# C4.2 — Tissue, fluid and containment history as Newton State

**Implemented; the campaign passed every gate.** 25 history arrays of Thermal, the surgical surface colour, Fluid, Bleeding, Emitter, Smoke, Suction and GraphContainment are Newton State. The single frozen campaign passed 36 of 36 groups and all three suites. Two of the plan's stop conditions fired before the freeze and the user decided both; they are deviations 1 and 2. The certificate's `accepted` flag means the gates passed; closing C4.2 is the user's decision. **Stop for review before C4.3.** Nothing was pushed.

Unchanged: every numerical kernel (222) and every step, reset, refresh and prepare hook (151), verified by AST against `6c2dd2b` ([audit][preserve]); capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins and the retained environments.

## Deviations from the approved plan

1. **A fresh binding does not resume bitwise when fluid surface contact is active.** `FluidContact` builds its triangle BVH once in the constructor, from the positions at that moment, and only refits it afterwards. The tree's shape is therefore state that lives outside Newton. On the CPU a freshly bound chole runtime diverged from the uninterrupted one in the pool positions; two restored runtimes agreed with each other; with fluid contact disabled, or with the uninterrupted runtime's tree rebuilt at the save point, the resume is bitwise ([evidence][bvh]).
   The user chose to state this and examine it in C4.3. The resume test rebuilds the uninterrupted runtime's tree at the save point. The exclusion covers `chole_tet_med` and `two_worlds`, which both use fluid contact. The same construction-time BVH in TissueContact and in Shell and Tet containment was not examined.
2. **`aorta_centreline` construction at 64 worlds grew by more than the 10% limit in development.** Five alternating samples before and after the migration gave medians of 0.523 s and 0.583 s (+11.5%, 60 ms; [samples][repeat]). The cause is 65,280 new containment rows. The user accepted it. The single campaign sample is 0.493 s against a pre-migration sample of 0.516 s; one sample each cannot resolve a 60 ms difference, so the repeated measurement stands.
3. **TipEvidence was not migrated** (user decision before implementation). Its four arrays are host-owned assessment state, created after binding and read only by the assessment Procedure. They are excluded from the continuation claim.
4. **GraphContainment's `prepare` still forces a rebuild** (user decision). The measurement on unchanged code found no difference in particle or rod State on two trajectories whose cache arrays did differ; a third, faster trajectory rebuilt every step anyway and shows nothing ([measurement][neutral]); this is "no difference observed", not a proof, because the spline projection cannot be shown neutral from the code.
5. **`Layout` and the binding helper moved into the physics package** (`i4h_physics/history_storage.py`), because Fluid and GraphContainment are physics-package systems. The surgical module re-exports them.
6. **State written directly in the resume test.** The scripted chole episode reaches colour and irrigation by stepping. Hot fat tets, live smoke particles, the cut artery node (from which the blood sources follow), and non-zero `Suction.removed`, `Fluid.overflow` and `Smoke.cursor` are written into State before the save point; two more steps then let the kernels burn, paint and age smoke. `Smoke.acc` is zero at the save point, and smoke spawning and suction do not run after the resume.
7. **Not executed:** `scripts/parity/catheter.py`, which builds graph containment through a case and was only read.
8. **Not done from the plan's lists:** empty and repeated reset masks for the 25 arrays; a pin of Bleeding's `cut` carried from one step to the next (only Smoke's `serial` was pinned; the resume test covers `cut` after an injected value); a new test that constructs two Fluid systems (the existing `test_scene_storage` declares two through authoring and passes); and `docs/stage-c4-inventory.json`, which is unchanged because tracked JSON files are in the frozen manifest. The outcome is recorded in `stage-c4-inventory.md` only.
9. **Constructor memory prechecks** for the colour, smoke and coagulation arrays went away with their allocations.

## What moved

| System | Arrays | Rows | Template |
| --- | --- | --- | --- |
| Thermal | `heat`, `burn`; `coag` | its existing tet rows `(E,T)`; the surface's particle rows | zero |
| Surgical Surface | `appearance.visual` | the surface's particle rows | zero |
| Fluid | `cursor`, `overflow` | own pool rows `(E,nf)` | zero |
| Bleeding | `cut`, `source`, `acc`; `serial` | centreline node rows; WORLD | zero |
| Emitter | `acc`, `serial` | WORLD | zero |
| Smoke | `pos_age`, `vel_life`; `cursor`, `acc`, `serial` | own particle rows `(E,count)`; WORLD | zero |
| Suction | `removed` | the fluid's pool rows | zero |
| GraphContainment | `cand_idx`, `cand_count`, `cand_overflow`, `ref_samples`; `ref_tree`; `rebuild`, `built` | own sample rows `(E,n)`; own tree node rows `(E,Pt)`; WORLD | `cand_idx` -1, `rebuild` 1, others zero |

- Heat and burn were already declared as Newton State and unused; the kernels now use them and the system-owned duplicate is gone.
- Declarations: `physics/src/i4h_physics/fluid_storage.py`, `containment_storage.py`, and new sections of `surgical/src/i4h_surgical/interaction_storage.py`. Every migrated system has a `declare_storage` hook; direct Scene callers use `declare_fluid`, `declare_graph_containment`, `declare_thermal`, `declare_appearance`, `declare_bleeding`, `declare_emitter`, `declare_smoke`, `declare_suction`. The `lumen_containment` alias now forwards the hook.
- Sample rows are owned by the catheter particle that anchors each sample, tree rows by the tree particles. `cand_idx` stores its 64 slots as columns of a sample row, so the case authors 765 rows per world, not 48,960.
- Every painter (Thermal, Cutting, StretchBreaking, ContactPaint) and the renderers reach one colour array; Smoke reads the Newton-backed heat; `Bleeding.clipped` still is `Clipping.clipped`.
- A surgical Surface built on a Scene directly now needs `declare_appearance`; five test fixtures were updated.
- The allocation inventory lost 25 State sites and reclassifies the four TipEvidence sites: 450 sites in 191 files, checker passing.

## Tests

`tests/packages/test_tissue_owned_storage.py` runs on the real `chole_tet_med` and `aorta_centreline` cases. Sixteen pre-migration pins were committed first (`2f76e91`) on unchanged production code; that commit's test file and `tissue_helpers.py` were replayed against `bec96e6`, with `interaction_helpers.py` from `bec96e6` (a comment differs): 16 passed ([result](../.tmp_tests/stage-c42-20261110/pre-migration-pins/result.txt)).

- Episode reset clears the 25 arrays in selected worlds only; cache-only preparation preserves the 18 chole arrays; containment preparation changes `rebuild` and nothing else.
- All 25 arrays are views of the bound State, differ from the template and from a second State, and binding writes none of them. Shapes, strides and dtypes equal natively allocated arrays, so kernel signatures did not change.
- Replaced arrays (nine cases, three modes) are rejected; missing, duplicate and late declarations and a changed capacity are rejected.
- **Chole resume, CPU, bitwise** for 12 steps with heat, burn, coagulation, colour, blood, irrigation and smoke present, under the conditions of deviations 1 and 6.
- **Containment resume, CPU, bitwise** in two forms. A fresh binding equals the uninterrupted runtime prepared again at the save point. A State restored into an already prepared runtime reads the saved candidate lists (its first step does not rebuild), rebuilds in the same steps as the uninterrupted runtime and stays bitwise equal for 12 steps.
- On CUDA, eager and captured: binding changes no byte, and stepping one runtime leaves the other's containers untouched.

## Frozen campaign

The [certificate][cert] covers **89,120 files** at commit `bec96e6`, manifest SHA256 `e31af6862f84e6ec59e69749538d32ec0c4dbc7c00613dbccb80dea994e01fe2`. The independent auditor verified 87 startup and exit pairs and 105,658 file-backed import observations and recomputed every comparison from the retained arrays. One campaign of 15 jobs, no retries, no reference promotion.

| Suite | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Full core | 1,219 | 49 | 0/0 |
| Lab main six-module selection | 122 | 7 | 0/0 |
| Patched Lab 3.0 six-module selection | 121 | 8 | 0/0 |

| Environment / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-captured | pass | pass (3/3) | pass | pass | pass | pass |
| standalone-eager | pass | pass (3/3) | pass | pass | pass | pass |
| main-captured | pass | pass (3/3) | pass | pass | pass | pass |
| main-eager | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-captured | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-eager | pass | pass (2/3) | pass | pass | pass | pass |

**36/36 groups, 47/48 replays, 17/18 chole trials.**

- `two_worlds`, `chole_tet_med` and `aorta_centreline` hold the migrated State and passed in all six groups.
- The one failed chole trial (Lab 3.0 eager, trial 1) has the same eight failed products in the grasp, instrument_contact and tool_force families as the failed trials of C3.2 and C4.1.
- `aorta_shell` passed in all six groups, as it did in C4.0. Its catheter-impulse failures of C3.1, C3.2 and C4.1 remain unexplained, and a clean run here supports no conclusion about them. The certificate stores the exact endpoints for C3.1 to C4.2.
- The clipping and fluid count intermittents did not recur.

### Construction (one sample each)

| Workload | Worlds | Seconds | Prior seconds | Retained Warp bytes | Prior retained | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chole_tet_med | 1 | 0.852 | 0.874 | 30,031,729 | 30,049,229 | -17,500 |
| chole_tet_med | 64 | 14.137 | 14.935 | 1,360,663,246 | 1,361,783,246 | -1,120,000 |
| aorta_shell (rods) | 1 | 0.086 | 0.085 | 3,354,144 | 3,354,144 | 0 |
| aorta_shell (rods) | 64 | 0.614 | 0.612 | 172,785,410 | 172,785,410 | 0 |
| aorta_centreline | 1 | 0.196 | 0.281 | 822,790 | 600,418 | 222,372 |
| aorta_centreline | 64 | 0.493 | 0.516 | 49,404,950 | 35,173,142 | 14,231,808 |

Prior samples are the C4.1 campaign for chole and `aorta_shell`, and pre-migration samples on `6c2dd2b` for `aorta_centreline`. The auditor explains every delta to the byte.

- **Chole shrinks.** Per world, the 18 arrays hold 279,556 bytes. They leave system ownership (-279,556). Sixteen of them are new in State and in the templates (+128,972 each); heat and burn (150,584) were there already. Pool and smoke rows add 514 rows and 4,112 identity bytes.
- **`aorta_centreline` grows** by the containment cache template (214,212 bytes per world) and 8,160 identity bytes for 1,020 rows. Live storage is unchanged; each additional State container carries the cache.

## Open and unverified

- Fresh-binding resume with fluid surface contact (deviation 1); the construction-time BVHs of TissueContact, Shell and Tet containment are unexamined. Both go to C4.3.
- Bitwise resume claims are CPU-only, as in C4.1.
- TipEvidence and Procedure state are outside the continuation claim.
- HDF5 recording and replay: unverified, deferred by user choice.
- The `aorta_shell` impulse failures, the chole trial failures and the two intermittents remain unexplained.

[preserve]: ../.tmp_tests/stage-c42-20261110/preservation-audit.json
[bvh]: ../.tmp_tests/stage-c42-20261110/fluid-contact-bvh-resume.txt
[repeat]: ../.tmp_tests/stage-c42-20261110/construction-repeat
[neutral]: ../.tmp_tests/stage-c42-20261110/containment-neutrality.txt
[cert]: ../.tmp_tests/stage-c42-20261110/certificate/frozen-campaign-certificate.json
