# Stage C4 plan

Approved by the user on 2026-11-10. The first part details checkpoint C4.0; the second is the C4 masterplan. Implementer: Codex. Reviewer: Claude.

## C4.0 detailed plan: formatted baseline, qualification of the C3.3 follow-up, timing and inventory

Self-contained brief for Codex. The C4 masterplan follows; this section details its first checkpoint only. Implementer: Codex. Reviewer: Claude, against evidence on disk.

### Context

Commit `04d057e` holds stages A to C3.3 but was made with the pre-commit hooks bypassed, because the hooks fail on isort in 67 files and would rewrite a dependency patch and historical reports. The C3.3 follow-up source (public storage registration, `Control.clear` rejection) has only focused verification. No step-time measurement exists on the current dependency pins. C4.0 changes no ownership. It produces a hook-clean, fully qualified, timed and inventoried starting point, so that every later C4 difference is attributable to C4.

Outcomes: (1) hooks pass on every C4 commit without touching evidence files, (2) one frozen campaign on the formatted source, (3) one exclusive timing campaign against the Stage B references, (4) a verified list of the State sites C4.1 to C4.3 will migrate.

Unchanged: kernels, hook order, capacity 256, comparator, zero floors, factor three, approved references (`.tmp_tests/stage-b-chole-acceptance-20261105/approved-v3/`), chole 2-of-3, dependency pins, the three retained environments (`.tmp_tests/stage-c30-20261109/{core,main,lab30}-env`), all historical certificates. No retries, failures retained, no attribution work, no push.

### Step 0: preserve and record the plan

- Pre-edit audit as in C3: verify the tree against the C3.3 review manifest (`.tmp_tests/stage-c33-review-20261110/`, or the latest final-source manifest there), archive sources to `.tmp_tests/stage-c40-<date>/pre-c40-source.tar` with hashes.
- Save the C4 masterplan section as `numerical_sim/docs/stage-c4-plan.md`, written to pass markdownlint.

### Step 1: hook exclusions (commit 1, hooks passing)

Edit `../.pre-commit-config.yaml` only:

- `trailing-whitespace` and `end-of-file-fixer`: exclude `^numerical_sim/docs/.*\.patch$`. The whitespace hook corrupts context lines of `warp117-hashgrid-capture.patch`.
- `markdownlint`, `trailing-whitespace`, `end-of-file-fixer`: exclude the historical evidence documents, as an explicit list, not a wildcard that would cover future reports: `newton18-qualification.md`, `stage-b*.md`, `stage-c.md`, `stage-c-field-contracts.md`, `stage-c-scratch-audit.md`, `stage-c21-*.md`, `stage-c22.md` to `stage-c25.md`, `stage-c3-plan.md`, `stage-c30.md` to `stage-c33.md`, `warp117-hashgrid-capture.md`. `stage-c-field-contracts.md` is hashed by `scripts/check_stage_c_inventory.py`, so it must stay byte-identical.
- New documents from C4.0 on (`stage-c4-plan.md`, `stage-c40.md`, later reports) are not excluded and must pass markdownlint. `stage-c.md` stays the excluded append-only roadmap.
- Verify with `pre-commit run --files $(git ls-files numerical_sim)`, never `--all-files` and never a directory argument: only isort may fail at this point, and `git status` must be clean afterwards (no file modified by a fixer).
- Record in `docs/stage-c.md` that C3.3 is accepted for progression with retained gaps (user decision, 2026-11-10); it still says open.

### Step 2: formatting-only change (commit 2, hooks passing)

- Run isort with the repository configuration (`../.isort.cfg`: black profile, line length 120) from the repository root, **on tracked files only**: pass `git ls-files 'numerical_sim/*.py'` as the file list. Never pass a directory: that would rewrite Python under the git-ignored `.tmp_tests` (retained environments, archived sources, campaign scripts hashed by the manifests), `.venv` and `data`. Afterwards `git status --ignored` must show no change under `.tmp_tests`. Expected scope from a dry run on `04d057e`: 67 files, about 160 changed lines, import blocks only.
- Proof that it is formatting-only, kept as `.tmp_tests/stage-c40-<date>/format-audit.json`:
  1. For every touched file, the AST with import statements removed is identical before and after.
  2. Per scope (module, function, class), the multiset of imported names (module, name, alias, relative level) is identical before and after. Compare names, not statement text, because isort merges and splits statements.
  3. List every file where the relative order of two imports changed and at least one is a module with import-time side effects in this codebase: `warp` configuration, `isaaclab` app or simulation-context imports, `pxr`, `torch`, and any `i4h_*` module that registers kernels, schemas or gym ids at import. For each, state why the new order is safe, or pin the original order with `# isort: skip` or `# isort: off/on` and say so. Never reorder across an existing lazy (function-level) import boundary.
- Update the source hashes in `docs/stage-c-inventory.json` so `scripts/check_stage_c_inventory.py` passes, as C3.2 did after its correction. No site may change classification.
- Ruff and codespell must stay clean. No other edits ride in this commit.

### Step 3: development selection, freeze, one campaign

- Before the development selection, make commit 3 (hooks passing): the timing harness change of Step 4 and the promoted rod construction script. The campaign freezes a commit, not a dirty tree; record that commit hash in the manifest. Everything committed after the campaign is documents only.
- Development selection in all three environments before freezing: full `tests/packages`, `tests/runtime`, and the six-module host selection. Known intermittents (`test_tracker_and_clip` clip count, the `two_worlds` fluid count) are retained and reported if they recur, without rerun. Any other failure stops the checkpoint for review unless import order alone explains it and the fix is an order pin.
- Suite failures in the frozen campaign follow the same rule: a recurrence of those two known intermittents is retained and reported; any other suite failure is a stop condition.
- Freeze with the C3.3 launcher pattern (`.tmp_tests/stage-c33-20261109/certificate/freeze_and_launch.py`, `run_certificate.py`, `audit_certificate.py`): source manifest, startup and exit guards, import receipts, independent auditor.
- One campaign: full core suite; Lab main and Lab 3.0 six-module selections; 36 groups, 48 replays, 18 chole trials; one construction sample each at 1 and 64 worlds for chole (`scripts/measure_stage_c_construction.py`) and for rod-bearing `aorta_shell` (promote `.tmp_tests/stage-c33-review-20261110/measure_rod.py` into `scripts/` before the freeze so it is part of the frozen source).
- This campaign is the full qualification of the C3.3 follow-up source. State that explicitly and compare its construction and memory numbers with the C3.3 follow-up samples (chole 64 worlds: 16.028 s, 1,219,312,846 retained bytes; aorta_shell 64 worlds: 0.708 s, 172,785,410 retained bytes).
- Retain exact `aorta_shell` catheter-impulse endpoints for all six host and mode combinations, as since C3.2.

### Step 4: exclusive timing campaign

- Harness: extend `scripts/qualify_stage_b_timing.py` rather than writing a new one. It currently hard-codes the Stage B layout (a `prechange-lab30` version, `.tmp_tests/newton18` environments, Lab source paths). Add options for the environment root (`.tmp_tests/stage-c30-20261109`), for the reference directory, and to skip the `prechange` version. Keep the measured loop, the 20 warmup and 200 measured steps, captured execution, zero controls, the busy-GPU refusal, the continuous process monitor, the zero replay-allocation check and the cadence test unchanged. If the current Lab pins force an API change in the worker's setup code, list each change in the report and show the measured loop is identical. The harness change is in commit 3, so the campaign manifest covers it.
- Run timing **after** the frozen campaign, on the same frozen commit. A busy GPU leaves timing pending and does not block the campaign or inventory report.
- Workloads: `tissue_contact` and `aorta_static` at 1 and 64 worlds, on Lab 3.0 and Lab main, with and without exports. Sixteen workers, each run once.
- References and gate: median step time of each of the eight non-export runs against the **same references Stage B was gated on**, as recorded in `.tmp_tests/stage-b-exclusive-20261105/decision.json`: `prechange-lab30-<case>-<worlds>.json` in that directory for Lab 3.0, and `.tmp_tests/stage-b-host/perf-main-<case>-<worlds>.json` for Lab main. Limit +5.0% each. Do not gate against Stage B's own results, which would ratchet the limit; report the change against those results as a second column. Record reference paths and SHA256 in `decision.json`. Export overhead is reported, not gated.
- Standalone, reported and not gated: `scripts/measure.py tissue_contact aorta_static --envs 1 64` against the Stage A and B1 table in `docs/stage-b.md` (lines 1042 to 1052). `measure.py` has no busy-GPU check, so run it inside the same exclusive window, under the same process monitor.
- Also report construction seconds and retained and peak Warp bytes per run against the Stage B table in `docs/stage-b-exclusive-timing.md`.
- The GPU must be exclusive. Two unrelated compute processes of the user are running on it now (an `i4h-sim-rod-physical` and an `i4h-validation/catheter` Python process). Codex must not terminate them. If the harness reports a busy device, record the pending attempt and stop for the user. A contaminated or busy attempt is retained and is not a measurement; a valid completed campaign is never repeated.
- If any gated comparison exceeds +5%: do not tune, do not rerun. Report the numbers, which runs, and the per-run construction and memory deltas, then stop. The user decides.

### Step 5: inventory of remaining State sites

Deliverable `docs/stage-c4-inventory.md` plus a machine-readable companion, verified against source at the frozen revision. Starting point to confirm or correct (65 sites coded `S`):

| Family | Arrays | Checkpoint |
| --- | --- | --- |
| Centreline | `positions`, `active_count` | C4.1 |
| Residue | `delta_k`, `delta_jaw` of InstrumentContact, Grasp, VolumetricGrasp | C4.1 |
| Hook | `verts`, `tet`, `bary`, `friction_coord`, `radial`, `active` | C4.1 |
| Grasp | slot fields `prim`, `tet`, `t`, `radial`, `prev_axis`, `verts`, `bary`, `gap`, `active`; `count`, `pending` | C4.1 |
| VolumetricGrasp | `particle`, `sphere`, `rest`, `active`; `count`, `pending`, `claim` | C4.1 |
| Clipping | `node`, `active`, `count`, `clipped` | C4.1 |
| Thermal | `heat`, `burn` (Newton copies already declared), `coag` | C4.2 |
| Appearance | `visual` | C4.2 |
| Fluids | `Fluid.cursor/overflow`, `Bleeding.cut/source/acc/serial`, `Emitter.acc/serial`, `Smoke.pos_age/vel_life/cursor/acc/serial`, `Suction.removed` | C4.2 |
| GraphContainment | `cand_idx`, `cand_count`, `cand_overflow`, `ref_samples`, `ref_tree`, `rebuild`, `built` | C4.2 |
| TipEvidence | `positions`, `active`, `count`, `overflow` | C4.2 |

For each array: file and line, dtype, shape expression, row domain, owner reference, current reset (method, masked or not, which fields are deliberately not cleared), every alias and reader in other systems, and bytes per world for chole and aorta_shell. Separately list:

- Every lifecycle write that changes history outside episode reset. Known: Hook disabled-`prepare` `active.zero_()`, GraphContainment `prepare` `rebuild.fill_(1)`, Centreline constructor reset and refresh, TipEvidence `begin_action` clear.
- Every cache derived from history and how it is rebuilt today.
- Remaining private storage reaches: `authoring.py:150` (`storage._authored`), `scene.__dict__.setdefault` in `tet_storage.py`, `surface_storage.py` and `storage.py`, `scene._mount_declaration` in `mount_storage.py`.
- Confirm that containment `sides` and ToolForce accumulators are scratch and output, with the code evidence.

Fix the stale `certificate.site_count` (587, actual 511) in `docs/stage-c-inventory.json`. This step is read-only analysis plus documents; no production change.

### Step 6: report, commit, stop

- `docs/stage-c40.md` (markdownlint-clean): preservation audit, hook exclusions, format audit summary, campaign tables, failed endpoints, construction and memory, timing tables with decision, inventory summary, and what stays unverified (HDF5 by user choice). Prepend the status to `docs/stage-c.md`.
- Commit 4 with hooks passing, documents only: inventory documents, report, roadmap status. Do not push.
- Stop for review before C4.1. C4.1 gets its own detailed plan after this review.

### Verification (what the reviewer will check)

- `git show --stat` of commit 2 touches only Python import blocks and the inventory hashes; the format audit's two AST properties recomputed independently on a sample; every side-effect reorder accounted for.
- `git diff 04d057e -- numerical_sim/docs/*.patch` and the excluded reports is empty; `scripts/check_stage_c_inventory.py` passes.
- `pre-commit run --files $(git ls-files numerical_sim)` passes on the final tree and leaves `git status` clean; nothing under `.tmp_tests` created before C4.0 has changed.
- Suite counts recomputed from the frozen logs; group, replay and chole counts recomputed from the retained arrays; no production file differs between the freeze manifest and the final commit, or the difference is flagged as unqualified.
- Timing: sixteen valid workers, zero unrelated compute processes in the observation log, references hashed, each percentage recomputed from the JSON medians.
- Inventory: spot-check ten arrays against source for shape, reset behaviour and aliases.

### Stop conditions (report and wait for the user)

- A development-selection or frozen-suite failure other than the two known intermittents, unless import order alone explains it.
- Any gated timing comparison above +5%, or a GPU that cannot be made exclusive.
- A failed reference group other than `aorta_shell` catheter impulse, or a chole group below 2 of 3.
- An inventory finding that contradicts the masterplan's checkpoint order.

## Stage C4 masterplan: remaining history, continuation and Stage C closure

This section is self-contained and is the brief for Codex. The original stages A to C masterplan follows it unchanged.

### Masterplan context

Stages A to C3.3 are committed as `04d057e` on `i4h-isaaclab/manager-env` (local, not pushed, committed without pre-commit hooks so source stays byte-identical to the certificates). Particles, tets, surfaces, instruments, rods, catheter controls and mounts are Newton-owned. What remains on `System` objects is the episode history of 65 allocation sites coded `S` in `docs/stage-c-inventory.json`. Until they move, a fresh runtime bound to a saved State and Control cannot resume a surgical episode, which is the last ownership claim of Stage C. C4 migrates that history, proves continuation, runs the performance gate and closes Stage C.

User decisions binding on C4 (2026-11-10):

1. **C3.3 is accepted for progression with retained gaps.** Its follow-up source (public storage registration, `Control.clear` rejection) has focused verification only; C4.0's campaign is its full qualification.
2. **The six residue arrays migrate as Newton State, unchanged.** No kernel change. The extra template copy (about 129 MB at 64 worlds for chole) is accepted and must be reported.
3. **Feature work waits until after closure:** external `particle_f` consumption, `export_contact_geometry` / `soft_contact_capacity`, physical rod units. None of it is in C4.
4. **Closure rule:** Stage C may close if the only failed reference groups are `aorta_shell` catheter-impulse products, with endpoints retained and the limitation stated. No attribution work is authorized.
5. **Performance gate:** warmed median step time within 5% of the existing Stage B exclusive references (`.tmp_tests/stage-b-exclusive-20261105/decision.json`, `docs/stage-b-exclusive-timing.md`). One exclusive timing campaign runs at C4.0. If it exceeds 5%, stop and report; the user decides.
6. HDF5 recording and replay stay deferred by user choice and are excluded from every C4 claim.

### Design rules

- **Resets do not change.** Every per-system `reset` keeps its exact current behaviour, including what it deliberately leaves alone (stale Grasp, VolumetricGrasp and Hook slot bytes, Drag node and target, mounted flags, the instrument's current pose). C4 does not introduce a generic "restore every State field from its template" episode reset.
- **The generic piece is checkpoint copy.** Continuation means: copy every declared State and Control field (plus native particle fields) between preallocated containers, bind a fresh runtime with `CaseAuthoring.bind` (`physics/src/i4h_physics/authoring.py`), and resume with bitwise-identical results to uninterrupted execution. Episode reset, cache-only reset (`I4hSolver.reset(flags=0)`) and continuation stay three distinct operations.
- **Scope is exactly the 65 `S` sites.** Scratch (`X`), outputs (`O`), tables (`M`) do not migrate. Containment `sides` and ToolForce accumulators are scratch and output and stay where they are.
- **Declare through the public API only:** `defer_declaration`, `Declarations.domain/add/initializer/register_package`, `Views.register_package(identities=, checks=, replace=)` in `physics/src/i4h_physics/storage.py`. Follow `surgical/src/i4h_surgical/instrument_storage.py` and `physics/src/i4h_physics/mount_storage.py`. Remove the remaining private reaches while there: `authoring.py:150` (`storage._authored`), `scene.__dict__.setdefault` in `tet_storage.py`, `surface_storage.py`, `storage.py:555`, and `scene._mount_declaration` in `mount_storage.py`.
- **Row domains come from the indexing code:** per slot `(E,n,G)`, per instrument `(E,n)`, per world, per tet, per particle, per centreline node, per pool. Shared rows carry world -1, runtime rows real worlds, owners reference the instrument, tet, particle or pool domain.
- **Order follows aliases, not families.** An array that another system aliases migrates no later than its consumer.
- **Derived caches on fresh binding.** For each migrated system, state how every cache derived from history is rebuilt at bind or prepare without writing State. Where that is impossible, name the exclusion explicitly instead of weakening the test.
- Unchanged throughout: numerical kernels, hook order, capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3 policy, dependency pins, retained environments, historical certificates.

### Checkpoints

Each checkpoint follows the C3 process (`docs/stage-c3-plan.md`): pre-edit preservation audit and source archive, pre-migration test extension, development selection in all three environments, freeze, one campaign (full core suite, both six-module host selections, 36 groups / 48 replays / 18 chole trials, one construction sample at 1 and 64 worlds for chole and for rod-bearing `aorta_shell`), no retries, failures retained, report `docs/stage-c4N.md`, status prepended to `docs/stage-c.md`, stop for review. The pre-freeze selection must name every test that constructs a Scene directly or through a fixture factory.

#### C4.0 Baseline (no ownership change)

1. Formatting-only commit: run isort and the fixers so the hooks pass. First exclude `*.patch` and the historical `docs/stage-*.md` reports from the mutating hooks (trailing-whitespace, end-of-file-fixer, markdownlint `--fix`) in `../.pre-commit-config.yaml`, so no dependency patch or historical report is rewritten. Prove the change is formatting-only with an AST comparison of every touched Python file.
2. One frozen campaign on that source. This qualifies the C3.3 follow-up and the formatted tree together.
3. One exclusive timing campaign with `scripts/qualify_stage_b_timing.py` (tissue_contact and aorta_static at 1 and 64 worlds, standalone and both hosts, captured) against the Stage B references. Report, do not tune.
4. Inventory deliverable: confirm or correct the 65-site list below with file and line, row domain, current reset behaviour and aliases; fix the stale `certificate.site_count` in `docs/stage-c-inventory.json`; list every lifecycle write that destroys history outside reset.

#### C4.1 Interaction history and residue (surgical)

Order: Centreline, then InstrumentContact residue, then Hook, Grasp, VolumetricGrasp, Clipping.

- `Centreline.positions/active_count` (`surgical/.../centreline.py`), read by Clipping and Bleeding. Its constructor reset moves into a finalizer initializer on the Model templates, as instruments did in C3.1.
- Residue `delta_k/delta_jaw` of InstrumentContact, Grasp, VolumetricGrasp: State, `(E*Nb, max(n,1))` vec3, unchanged kernels (`tool_delta.py`), `tests/packages/test_storage_residue.py` stays green.
- Hook `verts, tet, bary, friction_coord, radial, active`; InstrumentContact's `suppress_*` views and Thermal's entry reads keep aliasing the same backing.
- Grasp slot fields and `count/pending`; VolumetricGrasp slot fields and `count/pending/claim`; Clipping `node, active, count, clipped`.
- This closes the "mounted-tool episode restore remains partial" caveat carried since C3.1; say so only if the continuation test below passes with an active grasp on a mounted tool.

#### C4.2 Tissue, fluids and endoluminal history

Order: Thermal, Appearance, Fluid, Bleeding, Emitter, Smoke, Suction, GraphContainment, TipEvidence.

- Thermal `heat/burn` already have Newton State copies from `surgical/src/i4h_surgical/tet_storage.py::declare_thermal_tet_fields`: repoint the kernels to them and delete the legacy arrays. Migrate `coag` (per particle).
- `Appearance.visual` (per particle vec4, `surgical/.../mechanics/surface.py`); writers Thermal, Cutting, StretchBreaking, ContactPaint and the renderer readers (`isaaclab/camera/tiled.py`, `standalone/render/snapshots.py`) keep one backing.
- Fluids are included: pool particle state is already Newton-owned and only counters remain. `Fluid.cursor/overflow`, `Bleeding.cut/source/acc/serial` (aliases `Clipping.clipped`), `Emitter.acc/serial`, `Smoke.pos_age/vel_life/cursor/acc/serial` (reads Thermal heat), `Suction.removed`. The unexplained fluid count intermittent stays retained and unattributed.
- `GraphContainment.cand_*/ref_samples/ref_tree/rebuild/built` and `TipEvidence.positions/active/count/overflow`. GraphContainment is the hard derived-cache case: decide and document whether candidates are history (migrated, exact resume) or a cache rebuilt at bind (then prove the rebuild is bitwise-neutral or state the exclusion).

#### Lifecycle writes to pin before migrating (both checkpoints)

Each gets a pre-migration test recording today's behaviour, then a decision recorded in the report on whether cache-only preparation may still perform it:

- Hook: disabled-`prepare` unmasked `active.zero_()`.
- GraphContainment: `prepare` unmasked `rebuild.fill_(1)`.
- Centreline: constructor reset and refresh.
- TipEvidence: unmasked `begin_action` clear of `count/overflow`.
- Partial slot resets in Grasp, VolumetricGrasp and Hook: stale bytes must survive binding and checkpoint copy exactly.

Binding and cache-only preparation must preserve all State and Control bytes, as required since C3.2.

#### C4.3 Continuation contract

- Public checkpoint copy on the physics binding (the roadmap's planned `save_i4h` / `restore_i4h`, `docs/stage-c.md` near line 1093): copy declared State and Control fields and native particle fields between containers of the same Model, validated by layout signature; masked variant for selected worlds; no allocation after construction.
- Generalize the existing per-test helpers (`authored`, `runtime`, `snapshot`, `copy_container` in `tests/packages/test_{instrument,rod,mount}_owned_storage.py`) into one shared test helper instead of a fourth copy.
- New `tests/packages/test_continuation.py`: save, advance, restore, replay and fresh-binding resume, bitwise against uninterrupted execution, on CPU eager, CUDA eager and captured, with at the save point: an active grasp and volumetric grasp, a set hook, an applied clip, a cut, nonzero heat/burn/coag and appearance, live fluid with bleeding and smoke, a catheter with drag and tip evidence, a mounted tool with a pending reaction (using the `ExposedRigid` test solver). Partial, empty and repeated world masks leave unselected worlds bitwise unchanged; foreign host particles and bodies are untouched; two Models from one builder stay independent; replaced arrays are rejected.
- Stated exclusions, in the report and in the docstring of the copy call: MuJoCo-Warp private solver history, host time and episode counters, assessment and procedure state, renderer state, HDF5 recording replay.
- After this checkpoint the inventory has zero `S` sites on system owners.

#### C4.4 Performance gate and Stage C closure

1. Exclusive timing campaign, same workloads and references as C4.0, on the frozen final source. Gate: every comparison within +5%. Also report peak and retained memory and construction time at 1 and 64 worlds against Stage B and C3.3, with causes (templates, identity columns, residue copy).
2. Final frozen campaign under the closure rule above.
3. Closure housekeeping: copy the approved references and the final certificate out of git-ignored `.tmp_tests` into a tracked location (or record their hashes and an archive path the user names); set `qualification/c30-*.json` from "candidate; not accepted" to the accepted status; replace absolute home-directory paths in the four committed files with repo-relative ones; rewrite `docs/component-contracts.md`, `docs/package-migration.md`, CONTRIBUTING.md and the READMEs from the implementation.
4. Closure report `docs/stage-c-closure.md`: ownership matrix for every system, the three reset-like operations and what each guarantees, continuation exclusions, every retained failure with its archive (aorta_shell impulses, clipping intermittent, fluid count intermittent, chole trials), unverified items (HDF5), and the deferred list (particle forces, contact export, rod units, native Rod/VBD, bulk-authoring request to Newton for the 64-world construction time).
5. Commit per checkpoint from C4.0 on, with hooks passing. No push without the user's instruction.

### Masterplan verification

- Per checkpoint: `uv run pytest tests/runtime tests/packages -q` in the core environment and the six-module host selection in main and Lab 3.0 (`.tmp_tests/stage-c30-20261109/{core,main,lab30}-env`), then the frozen campaign through `scripts/qualify_stage_b.py` and `scripts/record_baseline.py --compare` against `.tmp_tests/stage-b-chole-acceptance-20261105/approved-v3/`.
- Allocation: `scripts/check_stage_c_inventory.py` passes; `scripts/measure_stage_c_construction.py --worlds {1,64}` plus the rod-bearing sample; each migrated field appears with template, State and identity bytes; residue arrays now show template bytes.
- Preservation audit: kernel and hook ASTs unchanged against the pre-edit archive at every checkpoint.
- Reviewer checks I will apply to each report: counts recomputed from logs and arrays, post-freeze production changes flagged as unqualified, private-field reach, pre-freeze selection completeness, and whether any claim exceeds the stated exclusions.

### Risks

- **Memory.** Residue and per-slot templates add roughly 2.5 MB per world for chole before checkpoint copies; a checkpoint State doubles live State. Report measured totals; do not optimize in C4.
- **Captured graphs and aliases.** Hook and Clipping arrays are aliased across systems and baked into CUDA graphs; a wrong backing shows only under capture. Every alias gets an identity registration and a captured-mode test.
- **GraphContainment** may not resume bitwise from a rebuilt cache; the fallback is migrating its candidates as State.
- **Timing at C4.0 may already exceed 5%** because of C1 to C3 or the Newton pin; that is a stop-and-decide point, not something to fix silently.
