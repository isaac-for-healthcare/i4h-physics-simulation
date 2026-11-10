**C4.0 status — 2026-11-10: campaign clean, timing gate failed, stopped for decision.**
Three hook-passing commits (`c9339e9`, `7a5291b`, `02ea7f0`): hook exclusions for the
evidence files, import ordering only in 67 files, timing harness options. The single
frozen campaign on `02ea7f0` passed **36/36 groups, 46/48 replays, 16/18 chole trials**
and core **1,096**, Lab main **121**, Lab 3.0 **118** with zero failures; it is the full
qualification of the C3.3 follow-up source. Construction storage equals the C3.3
follow-up byte for byte. The one valid exclusive timing campaign **failed the +5% gate**:
`tissue_contact` at 64 worlds is +7.54% (Lab 3.0) and +7.74% (Lab main) against the
Stage B references; `aorta_static` is 4% to 7% faster. Nothing was tuned or rerun.
The remaining-State inventory confirms 65 sites and records five findings that change
C4.1 and C4.2. See [stage-c40.md](stage-c40.md) and
[stage-c4-inventory.md](stage-c4-inventory.md). **Stop before C4.1.** No push.

---

**C3.3 disposition and C4 entry — 2026-11-10.** The user accepted C3.3 for
progression with retained gaps: its follow-up source (public storage registration,
`Control.clear` rejection) has focused verification only, and HDF5 recording/replay
stays deferred by user choice. Stages A to C3.3 are committed as `04d057e`, made
without the pre-commit hooks so the source stayed byte-identical to the certificates.
[The C4 plan](stage-c4-plan.md) is approved. C4.0 (hook exclusions, formatting-only
import ordering, one frozen campaign that fully qualifies the C3.3 follow-up source,
one exclusive timing campaign and the remaining-State inventory) is in progress.
Historical statuses below remain unchanged.

---

**Current C3.3 review follow-up — 2026-11-10.** All four approved post-freeze
construction samples passed once each: corrected chole and rod-bearing
`aorta_shell`, each at 1 and 64 worlds. The original failed construction attempts
remain retained. Control.clear now rejects unsafe mount-pose clearing before
writes, and all six package families use public storage registration.
Final-source focused checks passed core **292 / 2 skipped**, main and Lab 3.0
**294 / 1 skipped each**, with matching worker/nested-process source guards.
The original campaign and construction measurements precede these production
follow-ups; no full-suite/reference or construction rerun qualifies the final
source. HDF5 is deferred by user choice; retained environments are unchanged.
**C3.3 remains open for review. No C4, commit or push.** See the updated
[report and separate certificates](stage-c33.md). Historical statuses below remain
unchanged, including the construction gap that was open before this approved pass.

---

**C3.2 disposition: accepted for progression with retained failures.** The frozen
campaign passed 34/36 groups, 43/48 replays and 15/18 chole trials; every chole
group passed the existing 2-of-3 policy. Standalone captured `aorta_shell` failed
stepped catheter impulse; patched Lab 3.0 captured failed stepped and masked-reset
catheter impulse. Causes remain unresolved. Frozen suites passed core 1,017,
main 117 and Lab 3.0 114, with their recorded skips. The subsequent production
alias-validation correction passed focused checks only (core 103, each host 104);
full qualification was not rerun. HDF5 recording/replay remains unverified because
the retained hosts lack `h5py`. Earlier clipping/fluid observations remain retained.
This disposition leaves all historical reports and certificates unchanged and
authorizes C3.3 only; its required gates are not waived. Stop for review before C4.

---

**Current C3.3 status — implemented, open for review.** Mounted flags, mount-pose
commands and pending reactions now use Newton-owned storage. The frozen campaign
passed 36/36 groups, 46/48 replays and 16/18 chole trials; all chole groups met the
unchanged 2-of-3 policy. Suites passed core 1,085, Lab main 121 and Lab 3.0 118.
Both authorized construction attempts failed an obsolete reporting assertion and
wrote no sample report. The archived harness was corrected and passed six synthetic
checks per environment; no construction or campaign retry occurred. Production
and tests remain unchanged from the freeze. C3.3 requires explicit disposition of
that qualification gap. HDF5 and rigid/unmigrated history boundaries remain.
See [stage-c33.md](stage-c33.md) for source certificates, arrays, accounting and
retained failures. **Stop before C4.** Historical roadmap/results below are retained.

# Latest status — C3.2 implemented; open after qualification, 2026-11-09

Rod, shared-control, regional-material, catheter, layout and Drag ownership is
implemented. Catheter named commands reuse the generic C3.1 API. Cache-only
preparation preserves history; tip-range shrink clears only retired slots.
The single frozen campaign passed **34/36 groups, 43/48 replays and 15/18 chole
trials**. Standalone captured and patched Lab 3.0 captured `aorta_shell` failed
catheter-impulse comparisons; causes remain unresolved. All chole groups pass
the existing 2-of-3 policy, with three failed individual trials retained.
Frozen suites passed: core **1,017 / 47 skipped**, main **117 / 5 skipped**,
Lab 3.0 **114 / 8 skipped**, with zero failures/errors.

After the frozen audit, a rod alias-validation correction passed focused checks:
core **103 / 2 skipped**, each host **104 / 1 skipped**. Full suites and references
were not rerun on that corrected production source. HDF5 catheter recording/replay
remains unverified because the retained host environments lack `h5py`.
[The C3.2 report](stage-c32.md) links the separate frozen/current certificates,
failed endpoints, preserved source patch and allocation accounting. C3.2 remains
**open**; C3.1 progression acceptance below does not waive these failed or unverified
gates. Mounted-tool episode restore remains partial until C3.3.
Stop for review before C3.3. No commit or push.

---

**C3.1 accepted for progression with retained failures (C3.2 implementation disposition).**
The frozen campaign passed **34/36 groups, 45/48 replays and 17/18 chole trials**.
Standalone eager and patched Lab 3.0 captured `aorta_shell` failed stepped
catheter-impulse comparisons; causes remain unresolved. All chole groups passed
the existing 2-of-3 policy; the failed individual trial remains retained.
The mounted-tool fixture's missing instrument declaration was corrected and all
12 affected checks subsequently passed; full suites were not rerun on that fixture.
Earlier clipping and fluid observations remain retained and unexplained.
This authorizes progression only; historical reports and certificates below are
preserved and no C3.2 failed gate is waived.

---

# Latest status — C3.1 implemented; open after frozen campaign, 2026-11-09

C3.0 is **accepted for progression with retained failures**, as explicitly directed
in the C3.1 implementation request. Its historical certificate and failed clipping,
catheter and individual chole observations remain unchanged; fluids remain deferred
with unresolved cause.

**C3.1 remains open.** Instrument Model/State/Control ownership and cached named
Warp/Torch command views are implemented. The single frozen campaign completed:
**34/36 groups, 45/48 replays, 17/18 chole trials passed**. Failed groups are
standalone eager and Lab 3.0 captured `aorta_shell`, on stepped catheter impulse.
All chole groups passed their existing 2-of-3 policy; one failed trial is retained.
Frozen suites: core **984 passed / 47 skipped / 4 failed**; main **113 / 5 / 4**;
Lab 3.0 **110 / 8 / 4**. All twelve failures came from one missed direct-Scene
fixture declaration. After the campaign audit, that fixture was fixed and all
four affected tests passed in each environment. Production source is unchanged;
full suites and references were not rerun, and frozen failures were not relabeled.

All 83 campaign process pairs and 103,380 file-backed import records match; all
48 replay results were independently recomputed from retained arrays. Construction
accounting matches the planned State/Control formulas and retains all six residue
arrays without new template/checkpoint copies. [The C3.1 report](stage-c31.md)
links the historical frozen certificate and the separate current review certificate.
C3.0 acceptance does not accept C3.1 failures. A new explicit disposition is required.
**Mounted-tool episode restore remains partial until C3.3.** Stop for review before
C3.2. No commit or push.

Prior status entries and historical results below are retained unchanged.

---

# Latest status — C3.0 frozen campaign completed, 2026-11-09

**C3.0 remains open; C3.1 has not started.** The authorized campaign completed
36 reference groups, 48 replays, exactly 18 chole trials, and one construction
sample at each of 1 and 64 worlds. **35/36 groups passed;
45/48 individual replays passed.**
core: **956 passed, 47 skipped, 1 failed, 0 errors**.
main: **117 passed, 5 skipped, 0 failed, 0 errors**.
lab30: **114 passed, 8 skipped, 0 failed, 0 errors**.
Both construction samples match C2.5's measured allocation accounting, including
templates, aliases, identity columns and all six legacy residue arrays. No
ownership migration, comparator change, retry or reference promotion occurred.
The historical fluid discrepancy remains deferred with unresolved cause.
The remaining failed gates are `lab30-eager/aorta_shell`, `core-suite`.
They require explicit user disposition; earlier progression decisions do not waive them.
[The C3.0 report](stage-c30.md) links the full matrix, numerical values, raw
arrays and frozen source certificate. Stop for C3.0 review before C3.1.

Prior status entries and historical results below are retained unchanged.

---

# Latest status — C3.0 item 2 complete, 2026-11-09

Both complete established host selections passed once after the compatibility
fixes: **117 passed, 5 skipped on main; 114 passed, 8 skipped on patched Lab 3.0**.
The user deferred the recurring fluid discrepancy for progression; historical
failed observations remain failures with unresolved cause. No tests were excluded.
The missing post-fix full-host verification gap is closed. [The C3.0 report](stage-c30.md)
records the runs and matching source checks. C3.0 remains open: the frozen campaign
and construction accounting are unrun, and C3.1 implementation has not started.
This item 2 request did not authorize starting that separate campaign.

Prior status entries and historical results below are retained unchanged.

---

# Current status — 2026-11-09

**C3.0 is open; C3.1 has not started.** The exact Lab/Newton pin update, fresh
qualification environments, main instance-adapter port and permitted finalizer
capability guard are implemented as a candidate. [The C3.0 report](stage-c30.md)
retains an unresolved Lab 3.0 stream-comparison failure: world 0 fluid alive
count 256 versus 255 and suction removal 0 versus 1. The frozen campaign and
both construction samples were not run because the development entry gate
failed. C2.5 acceptance does not waive this observation. Stop for review before
C3.1; no commit or push.

The [preceding C3 plan and binding revision](stage-c3-plan.md) are preserved as
separate verbatim entries. C3.0 isolates host/dependency qualification before
ownership migration; only an accepted C3.0 frozen baseline may become C3.1's
baseline. Continue supporting standalone, pinned Lab main and patched Lab 3.0.
MuJoCo/MuJoCo-Warp were already 3.17.0; this checkpoint changes Lab and Newton.

| Checkpoint | Scope | Required boundary |
| --- | --- | --- |
| C3.0 | Exact pins and host compatibility | Open; resolve/report gates and obtain acceptance before C3.1 |
| C3.1 | Instrument ownership and packed/named commands | Separate qualification and review; stop before C3.2 |
| C3.2 | Rods, shared controls and catheter controls | Separate qualification and review; stop before C3.3 |
| C3.3 | Mounts and pending coupling reactions | Separate qualification and review; stop before C4 |

Binding amendments for the later ownership checkpoints:

- Packed commands with named zero-copy views are the final design. Each control
  package supplies its view specification; `PhysicsBinding.command_views(system_id)`
  and component Warp/Torch exposure validate and cache it generically, without
  switching on concrete classes. Retain packed action and recording layouts.
  The typed-attribute/recording-format-v2 proposal is superseded; no such migration
  is added to C4. Masterplan item 10 was not found in this worktree and no
  unavailable source is invented or edited.
- Record actual instance-qualified frequency registration, world identities,
  owner references, shapes, strides and pointer relationships. Multiple control
  systems must not collide in a global instrument frequency.
- Finalizers must run after every required Model field and State/Control
  template exists and before containers clone those templates. Test two Models
  finalized from one retained builder, independent backing and valid subsequent
  clones. Ordinary live Controls may deliberately be shared by runtimes without
  mutating templates; explicitly opting into `control(clone_variables=False)`
  aliases templates and intentionally changes later clones. Keep these contracts
  and tests separate.
- C3.1 uninterrupted-execution comparisons must use fixtures with fully preserved
  instrument history. Retain integrated interactions, but do not claim fresh-runtime
  equivalence for unmigrated grasp, hook, Thermal or mounted state.
  **Mounted-tool episode restore remains partial until C3.3.**
- Before C3.2 edits, extend `test_rods.py::test_structural_and_live_settings` beyond
  its existing structural `tip_edges`/`prepare()` case. Distinguish cache-only
  preparation, supported structural recapture and reconstruction; verify commands
  before stepping and slots that cease to be controlled. Do not silently clear
  restored commands or keep unintended stale bend contributions. The inherited
  `aorta_shell` failures concern catheter-control impulse products; C3.2 migrates
  that ownership and must retain the exact failed endpoints in its comparisons.

All numerical kernels, ordering, capacities, tolerances, comparator policies and
approved references remain unchanged. Six legacy residue arrays remain explicit.
C4 and final performance qualification are later work. Historical results below
are preserved verbatim.

---

# Stage C — Newton-owned persistent data and controls

## C2.5 — C2 closed for progression; C3 not started, 2026-11-07

The user-supplied C2.5 plan accepts the C2.4 standalone captured `aorta_static`
masked-reset impulse and Lab 3.0 eager `aorta_shell` stepped/masked-reset impulse
failures for progression. This supersedes the historical open-checkpoint wording
below without changing any numerical result. Lab 3.0 captured chole trial 0
remains failed; the group retains its 2/3 majority pass.

[The C2.5 closeout and C3 handoff](stage-c25.md) records the completion matrix,
independent read-only audit and supplemental certificate. All 9,376 frozen
C2.4 files still match. Retained endpoints reproduce **34/36 case groups**,
**45/48 individual replays**, **17/18 chole trials**, three failed replays and
22 failed comparisons. The frozen suites remain inherited evidence:
**940/47 skipped** core, **109/8** Lab 3.0 and **112/5** main.

New nonzero-capacity post-rebind face emission passes all **six combinations in
each of standalone, Lab 3.0 and main**. Existing focused core checks pass
**279 tests, 1 skipped**. Provenance, accounting and the **587-site/181-file**
allocation inventory pass. The original C2.4 certificate, historical tables,
accepted C1 inventory and removal records remain unchanged. No production
ownership, numerical policy or dependency changes were made.

C2 is **closed for progression**. Commands, rods and coupling remain unmigrated;
**C3 has not started**. Remaining history, general continuation/save-restore and
final steady-step performance qualification remain later work. Work stops at
C2.5 without commit or push. The following sections are historical records.

## C2.4 — Physical Surface checkpoint, 2026-11-06

C2.3 is **accepted for progression** by the user-supplied C2.4 plan. Its two
certificates, all failed observations, construction samples and direct-caller
correction remain in [the C2.3 record](stage-c23.md); historical status text below
is preserved and superseded by that progression decision. No automatic C2.4
failure waiver applies.

[C2.4 implementation and verification](stage-c24.md) migrates all physical
Surface arrays to Newton, preserves selected State on binding/preparation, keeps
explicit selected-world reset and checked compatibility aliases, and removes the
three additive/runtime Surface duplicates. Appearance and Thermal history retain
legacy construction. No full fresh-runtime continuation is claimed.

The frozen campaign passes **940/47 skipped** core, **109/8** Lab 3.0 and
**112/5** main. **34/36** reference case groups and all **17/18** chole trials
pass (Lab 3.0 captured 2/3; other groups 3/3). Standalone captured `aorta_static` masked-reset impulse and Lab 3.0 eager
`aorta_shell` stepped/masked-reset impulse failures remain unaccepted, so
**C2.4 stays open**. Both one-time construction samples, all failed endpoints,
source guards, ownership accounting and unchanged historical tables are retained.
No next slice, attribution campaign, commit or push is performed.

The following sections preserve the earlier checkpoint reports.

## C2.3 — Tet mechanics migration, 2026-11-06

C2.2 is **accepted for progression**, including its certificate, numerical failures,
construction samples and accounting corrections. The historical records below
remain unchanged. That acceptance gives C2.3 no failure waiver.
[The C2.3 checkpoint report](stage-c23.md) records this bounded Tet migration and
its new verification separately. Final suites pass **878 / 47 skipped** core,
**109 / 8** Lab 3.0 and **112 / 5** main. References pass **33/36** case groups;
all **18/18** chole trials pass. New `aorta_shell` impulse failures in standalone
captured, main captured and main eager keep **C2.3 open**. Two construction
samples are retained once. The final certificate proves unchanged production,
reference and measurement source across the correction of one indirect direct-
Scene test caller; original failed suites remain preserved. Physical Surface and
Thermal remain legacy; C2.4 has not started. No full fresh-runtime continuation
is claimed.

## C2.2 implementation checkpoint, 2026-11-06

The user accepts C2.1, including its retained failed observations, certificate
and measured tables. That acceptance supersedes the historical open-checkpoint
wording below; no historical result or sample is removed or changed. It does
not waive C2.2 failures. [C2.2 implementation and verification](stage-c22.md)
records the private-pool and Scene-metadata migration. Final suites pass:
**823 passed / 47 skipped** core, **109 passed / 8 skipped** Lab 3.0,
**112 passed / 5 skipped** main. The frozen-runtime certificate retains two
C2.2 numerical failures: main captured aorta_shell impulse and main eager
aorta_static impulse. Standalone captured/eager chole are each **2/3** under
the unchanged approved majority policy; the other four groups are 3/3. Both
construction samples are retained once, including correct pool templates and
the 12-byte native-gravity subtotal correction. The original stale-filter core
failures and original reporting harness are preserved separately from the
corrected full-suite results. **C2.2 remains open for checkpoint review. C2.3
has not started.**

## C2.1 history and native rest follow-up, 2026-11-06

[Implementation, source certificate and all construction samples](stage-c21-closure.md)
records removal of `Model.i4h.particle_q0` in favor of native Model rest storage,
explicit selected-world history initialization, and completion after direct and
manager startup/reset events and fresh component rebuild callbacks. Binding and
cache-only refresh preserve restored history. Five history-boundary tests pass
on each host; exact ownership/layout/reset tests pass on CPU and CUDA. The final
certificate uses one source map with verified runtime provenance for retained
replays and corrected-suite guards: **815 passed / 47 skipped** core,
**109 passed / 8 skipped** Lab 3.0, **112 passed / 5 skipped** main. Two reference
groups fail: Lab eager aorta_static impulse and main captured aorta_shell impulse.
Chole is 3/3 in five groups and **2/3 standalone eager**, accepted by its existing
majority rule; every failed trial is retained. C2.1 remains open. C2.2 has not
started. Earlier measurements,
failed samples and residue instruction provenance below remain historical and
unchanged; the six legacy residue arrays remain 129,300,480 bytes at 64 worlds.

## C2.1 final-source follow-up, 2026-11-06

The accepted implementation is held at C2.1; C2.2 has not started. The
[final-source certificate and ownership/lifecycle review](stage-c21-followup.md)
records the frozen-source rerun, harness-only failure retention, residue instruction
provenance, position-write audit and planned removal of the redundant Model rest
copy. The source certificate records outcomes; it does not erase earlier failures.
All nine jobs verify one source hash file. Full core: **813 passed / 47 skipped**;
Lab 3.0: **106 passed / 8 skipped**; main: **109 passed / 5 skipped**. Captured
references pass on all hosts, but standalone and main eager aorta_shell impulse
comparisons fail. The reset-event audit also confirms live positions are changed
after history initialization; universal post-write history coverage is not
confirmed. C2.1 remains open. The measured construction and memory table below is
retained unchanged.


## C2.1 implementation and checkpoint, 2026-11-06

**C2.1 is implemented; its checkpoint is open. C2.2-C2.4 have not started.**
Residue instruction provenance: the user's implementation request supplied the
previous agent's plan and explicitly said, "Treat the plan as the source of user
intent." Its locked decision says, "retain all six `delta_k`/`delta_jaw` arrays as
State. Leave their kernels unchanged. Include their actual bytes in every memory
report; they remain on legacy owners until their systems migrate later."
This is the operative user-supplied implementation instruction. No earlier,
independent user selection of the C1 option-2 question has been established;
the historical C1 pending wording below remains historical. The implementation
retains the six arrays on legacy systems and counts their actual bytes.

Declarations now separate `register_schema(builder)` from `author_rows(builder)`.
The first freezes package fields before native particles are created; deferred
host identity resolvers run in the shared idempotent completion hook, and each
custom row is authored in one call. Native particle fields use per-world
`add_particles(custom_attributes=...)` values, sliced before list conversion.
The direct scene-only construction path uses the same particle helper.

Newton Model.i4h owns exact particle_q0, particle_inv_mass and particle_flags0;
State.i4h owns particle_q_prev and mutable particle_flags. Native State q/qd and
Model radius remain aliases. History initializes from authored positions rather
than the selected State. Binding installs and validates declared views before
Scene/system initialization, and rejects replaced namespaces, buffers or layouts.
Missing declarations and changed system/capacity definitions fail explicitly.

Four Scene clones and their post-finalization host readback/assignment path have
been removed: q0, inv_mass, flags0 and q_prev. The former Model flag alias is
replaced by independent State flags. Native Model flags remain untouched by
mutation and selected-world reset. Native rest/mass/default flag buffers remain
necessary Newton allocations; custom State initialization templates are additional
retained storage. Tet's five fields, Surface's three tables and Thermal heat/burn
still have additive declarations with legacy runtime owners. Pools, body variant
arrays, commands, rods, thermal history, mounts and full save/restore remain
outside this checkpoint. Full fresh-runtime continuation is not claimed.

The accepted C1 inventory and source snapshot are preserved and still verify:
627 sites in 178 files. The current linked inventory verifies 623 sites in 179
files, with the four removed sites and native owner assignments recorded. See
[accepted C1 snapshot](stage-c-inventory-c1-accepted.json) and
[current field contracts](stage-c-field-contracts.md).

Evidence directory: `.tmp_tests/stage-c21-20261106/`. It retains all initial
failures, qualification workers, arrays, logs, source hashes, binary checks and
construction samples. No dependency pin, patched Warp binary, capacity 256,
reference, tolerance, ordered mechanics kernel or coupling limitation changed.
The ordinary workspace `.venv` failed collection because its Newton SolverBase
lacks ObservableKind; qualification uses the preserved patched environments.
Initial sandbox CPU/CUDA attempts and the obsolete memory-estimate failures are
retained separately and are not successful checks.

Reference checkpoint (all exports enabled; eager and captured groups retained
separately, exactly three fresh chole trials in each group, without retries):

| Host | Captured six references | Eager six references | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | PASS | PASS | 3/3 / 3/3 |
| Lab 3.0 | PASS | PASS | 3/3 / 3/3 |
| Main | **FAIL: aorta_shell impulse**; five other cases pass | PASS | 3/3 / 3/3 |

Early captured workers preceded the final capacity-validation and memory-estimate
correction; each worker retains its exact source hashes. These passing samples
are bounded evidence, not certification of a final integrated source revision.

Main's captured `stepped/system/catheter_control/impulse` differs by
6.864e-05 against tolerance 4.6e-05. This one retained failed product keeps the
checkpoint open regardless of subsequent diagnostics. No majority policy is
applied to aorta_shell and no passing eager result substitutes for capture.

Fresh-process chole construction samples, using the qualified patched binary
(each sample retained once, with no timing retry):

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.475132 | 5.807029 |
| Schema registration (s) | 0.003785 | 0.024237 |
| Native world authoring (s) | 0.122318 | 0.969090 |
| Custom row authoring (s) | 0.137547 | 3.794909 |
| Finalization (s) | 0.048464 | 0.594328 |
| Binding and runtime construction (s) | 0.062663 | 0.312229 |
| Construction Warp peak (bytes) | 26,781,701 | 1,139,762,456 |
| Retained Warp storage (bytes) | 26,575,457 | 1,126,557,170 |
| Newton Model buffers (bytes) | 2,505,949 | 41,782,669 |
| Newton initialization templates (bytes) | 450,356 | 28,822,784 |
| Live Newton State (bytes) | 517,700 | 33,132,800 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 517,700 | 33,132,800 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

Each of the six residue arrays contains **336,720 bytes at 1 world** and
**21,550,080 bytes at 64 worlds**. Their per-instance names and bytes are retained
in each JSON report. They remain legacy system State, already included in retained
Warp storage; they have zero Newton template/checkpoint bytes because these fields
have not migrated. Additional checkpoint containers are allocation measurements,
not a complete save/restore implementation. Newton Control bytes exclude commands
still on legacy system owners. Container bytes exclude BVH/solver/legacy-system
buffers; measured Warp retained/peak totals include those allocations. Do not add
the residue subtotal to retained storage again.

Both samples recorded Solvecraft PID 65109 using 1,071 MiB of GPU memory. These
are construction measurements under contention, distinct from an exclusive
steady-step performance campaign. Host peak RSS and the complete compute-process
listing are also retained. Native counts remain 5,612 / 359,168 particles and zero
native tets/triangles. Earlier estimates and samples below remain historical.


### C2.1 focused and runtime evidence

- CPU/CUDA ownership/lifecycle selection: **49 passed**. This includes exact
  inverse-mass bytes with a reciprocal-round-trip counterexample, authored
  history, independent States, shared/default Controls, host offsets/foreign
  ranges, selected/empty/repeated resets, replaced buffers and retained builders.
  Added empty-native padding/missing-schema and late-capacity checks also passed
  in the focused correction selection.
- Full core package/runtime sample: **808 passed, 47 skipped, 3 failed**. The
  failures were a test caller missing `complete(builder)` and two memory-estimate
  ownership checks. The caller is converted; migrated defaults/history/templates
  now appear in the Newton estimator rather than Scene-only categories. The
  initial correction passed seven focused checks but retained a chole estimate
  closeness failure. The final estimate accounts for grid/State nonconcurrency:
  `build_model` releases the grid before `model.state()` allocates native State
  and declared history. The qualified temporary/BVH allowance and unchanged test
  bounds are preserved. Final focused estimate checks: **3 passed**. All three
  failures from the full core sample are corrected in focused tests; a second
  full suite was not run, and the original sample remains failed.
- Lab 3.0 focused host suite: **105 passed, 8 skipped, 1 failed** at
  `test_user_stream_ordering[newton]`. Roots/action ordering passed; the final
  observation comparison failed. Original assertion/log retained.
- Main focused host suite: **108 passed, 5 skipped, 1 failed** at
  `test_standalone_matches_host[two_worlds-newton]`. Differing products are world
  0 `fluid.alive_count` and `suction.removed`. The original full observations are
  retained as `main-suite-host-parity.npz` in the evidence directory.
- One fixed pre-change/current main aorta_shell diagnostic pair passed on both
  sources. One fixed pre-change/current Lab 3.0 stream diagnostic pair also
  passed on both sources, with original comparison bounds. Neither pair
  reproduced or explained the original failure, and neither replaces a failed
  qualification/suite sample. No further acceptance retries were performed.
- Static AST comparison finds all **266 Warp kernels/functions** and **120
  ordered phase/reset definitions** unchanged from the preserved C1 source.
  Ruff and whitespace checks pass. Current inventory, four inventory negative
  checks, preserved C1 inventory, all indexed approved references and qualified
  patched binary hashes verify. Source manifests and all correction attempts
  remain retained; estimates are not presented as new construction samples.

**C2.1 checkpoint remains failed/open.** The main captured aorta_shell reference,
Lab 3.0 stream observation assertion and main standalone/host parity assertion
require investigation/disposition. Their causes are unresolved; passing bounded
diagnostics do not establish that these are pre-existing or harmless. There is
no final integrated-source qualification or C2 exit claim. C2.2 private pools and
metadata, C2.3 Tet storage and C2.4 physical surface remain pending as required by
the locked first-checkpoint gate.

## C2 entry replay and decision gate, 2026-11-05

**Standalone C2 entry passed. No C2 migration or kernel change has started.**
The user accepted the C1 source inventory and confirmed the residue finding.
Before migration, the unchanged qualification harness replayed all six approved
references against current source with declarations active, captured standalone
execution and all i4h exports enabled.

| Approved case | Result | Retained fresh-process replays |
| --- | --- | ---: |
| two_worlds | PASS | 1/1 |
| chole_tet_med | PASS | 3/3 |
| aorta_static | PASS | 1/1 |
| aorta_shell | PASS | 1/1 |
| tissue_contact | PASS | 1/1 |
| aorta_centreline | PASS | 1/1 |

Chole alone retains the approved majority-of-three rule, with exactly three
fresh processes and every result retained. All comparisons reported zero failed
products. No tolerance, capacity, scenario, recorder payload or reference changed.
The qualification workers check imported source paths and retain full source
hashes; each used current `load_case`, which calls `authoring.complete(builder)`
before finalization. The C1 inventory checker also passed before replay.

Evidence: `.tmp_tests/stage-c2-entry-20261105/` contains `entry-result.json`,
`standalone/summary.json`, per-case results/arrays/logs, all three chole trial
directories, source/reference hashes and verified corrected native binary hashes.
Both the parent harness and all eight replay workers exited successfully. Source
and approved reference hashes matched after the run. Host replay qualification
is deferred to the **first C2 checkpoint**, as requested; prior C1 host tests are
not relabeled as C2 reference checks.

### Pending residue decision

The user must choose one of the following before C2 migration:

1. Unconditional clearing in `apply_average_tools`, with an explicit numerical
   kernel-change record and replay evidence qualifying that change.
2. Keep delta_k/delta_jaw as State and include their bytes in **every memory
   report**, including initialization templates, live States and checkpoints.

The choice was requested and is still pending. The inventory's conservative State
classification remains provisional for implementation; no default/preselected
answer is treated as authorization. The entry replay above qualifies the
existing kernel, not an unconditional-clear change.

### C2 reporting contract and entry ownership

Scope remains particles and mutable topology through native authoring hooks,
removing replaced allocations and qualifying exact reset parity, independent
States and untouched foreign host ranges. No solver fusion, new mechanics,
native elasticity replacement or coupling migration is included.

Once particle/topology families are declared, **each C2 report must provide
measured construction totals**, not reuse projected row timings as measurements.
Report construction wall time and its registration/finalization components,
construction peak, retained Model bytes, initialization templates, live
State/Control containers, intentional sharing and additional checkpoint storage
separately. Keep the historical chole64 baseline/probe samples and estimates as
historical evidence. No new construction measurement was taken for this entry
replay, and no estimate is promoted to a measured current total.

At this entry point:

- **Persistent allocations removed: none.**
- Existing runtime arrays remain authoritative. Ten fields have an additional
  declared Newton copy: Tet tets/active0/rest_volume/active/tet_part;
  Surface tet_edges/face_neighbor/face_tri; Thermal heat/burn. Newton also retains
  initialization templates for declared State fields; report those separately.
- The reserved native_tet=-1 mapping and domain reference metadata are new
  declared data, not replacements for a legacy allocation.
- Native live particle q/qd already alias Newton State. Exact rest/history,
  mutable flags and private pool arrays remain on their legacy paths; the C2
  replacement and foreign-range isolation checks have not yet occurred.
- The six per-tool correction arrays remain allocated on their legacy systems.
  They have not been migrated or given additional declared copies.

Every C2 checkpoint must update this removed/still-duplicated list rather than
claiming ownership transfer merely because a declaration exists.

## C1 source-inventory exit, 2026-11-05

**C1 is qualified for the recorded source. C2 migration has not started; Stage C
is not complete.** The remaining source-inventory gate is closed by the explicit
[site inventory](stage-c-inventory.json), [field contracts](stage-c-field-contracts.md)
and [scratch lifecycle audit](stage-c-scratch-audit.md). Earlier entries below
retain their historical incomplete statuses.

All **627 allocation/helper sites in 178 source files** have contract links,
including scoped direct fields, nested material dictionaries, local allocation
helpers, live-field expansion, optional arrays, aliases, renderer boundaries and
Newton's empty-native backing allocator. The inventory retains exact source
expressions/dtype/shape expressions and source SHA-256 hashes. The 627 count is
sites, not unique fields or allocated-buffer count. One helper can allocate many
fields; aliases do not imply another owner. Model initialization templates and
additional State/Control containers remain separately accounted.

The source audit corrected three ownership/layout details:

- Smoke lifetime/speed/upward are float32 `(E,2)` ranges, not scalar `(E,)` fields.
- Smoke.centre is a stale-capable output when hot=0, not overwritten scratch.
- **InstrumentContact, Grasp and VolumetricGrasp delta_k/delta_jaw are State.**
  `apply_average_tools` conditionally clears these arrays only when squared
  vector length is positive. A nonzero `(1e-30,0,0)` residue survives on CPU and
  CUDA while delta_count clears. Their existing reset/disabled-prepare paths
  leave them intact. Preserve those bytes and reset semantics during migration;
  no numerical kernel was changed to force unconditional clearing.

The residue result disproves the previous scratch classification; it does not
claim a measured clinical-workload error or change Stage B reference acceptance.
These six State arrays must contribute to retained/template/checkpoint storage
reports when migrated. Ownership changes do not promise lower memory.

`python3 scripts/check_stage_c_inventory.py` verifies current source coverage,
contract references and review-document hashes without importing simulation
packages. Four negative checks reject missing sites, missing contracts, stale
source and stale review documents. The checker enforces provenance/coverage;
it does not automatically prove the semantic judgments in the source audit.

C1 evidence carried forward:

- Preservation of pre-C source, approved references, qualified patched binaries
  and workload definitions remains intact.
- Declaration machinery, independent State/default Control allocations,
  intentional shared controls, reference remapping, empty padding and view
  identity/layout checks passed in prior C1 suites.
- Existing per-tet package helpers are connected to CaseAuthoring.complete,
  standalone and both adapter builder hooks; additive declared storage survives
  binding byte-for-byte. Normal runtime storage is still authoritative until C2.
- Prior real adapter results remain **Lab 3.0: 9 passed / 3 skipped; main:
  12 passed**. No host/runtime source changed in this inventory round.
- Final package suite: **190 passed**. The four CPU/CUDA residue regression cases
  cover retained tiny vectors and cleared ordinary vectors. Ruff passes.
- Inventory and negative checks pass; evidence is retained in
  `.tmp_tests/stage-c-inventory-20261105/`. Earlier census/queue evidence remains
  unchanged, including its unresolved statuses at those historical revisions.

C2 starts with particles and mutable topology: declare exact defaults through
native authoring hooks, bind cached owner views, remove duplicate persistent
allocations, and qualify reset parity/independent states/foreign-host isolation.
The measured construction cost and row estimates remain binding evidence.
Scratch initialization/rebuild and removal of history-changing prepare/constructor
writes are recorded requirements, not implemented fresh-runtime continuation.
Full poison replay, save/restore, packaging and the exclusive final numerical/
performance campaign remain pending in later milestones.

## C1 completion-hook integration and cost investigation, 2026-11-05

**C1 remains open: inventory certification is not complete.** Existing per-tet
helpers are now connected to CaseAuthoring, standalone construction and both
host adapters. Runtime physics still uses its previous arrays; declarations are
additive until C2 makes them authoritative. C2 migration has not started.

`CaseAuthoring.complete(builder)` runs after all worlds/host entities exist and
before finalization. It calls explicit package `declare_storage` hooks in the
authored system list; Tet, Surface and Thermal provide these hooks. It neither
constructs runtime systems nor discovers arrays. Surface resolves its declared
Tet domain through the earlier authored record. Completion is idempotent on a
retained builder and rejects changed system specifications. Binding a case with
these declarations without completion fails before runtime allocation.

Standalone `load_case` calls completion before finalize. Main's pure solver
builder hook calls the same method, and its coupled hook first delegates rigid
preparation. Lab 3.0's existing finalization hook delegates rigid preparation
before the shared completion call. Host-dependent mount-field declarations
remain future C3 work; no current coupling metadata or reaction storage changed.
`scene.attr(domain, key)` returns the cached declared view. Binding validates
container ownership before allocating existing runtime buffers and leaves
registered history unchanged; this is not full fresh-runtime continuation.

### Registration investigation

The pinned public `add_custom_values_batch` loops over `add_custom_values`; it
has no bulk column implementation. A retained CPU microprobe used 100,000 rows
with explicit world/particle references, active and native_tet columns:

| One sample per implementation | Time |
| --- | ---: |
| Current row loop | 0.108448 s |
| Zip-based row construction | 0.111441 s |
| Public batch API | 0.211041 s |

A separate profile attributes 0.506 of 0.555 profiled seconds to
`add_custom_values`. Profiling overhead makes that profile unsuitable as a
throughput estimate. Neither alternative was adopted; no Newton internals,
dependency pins or row-authoring contract changed. The prior chole measurements
remain intact. These CPU samples are not the exclusive final performance gate.
The construction script now exercises `complete` and reports completion time
(including host values), rather than labeling it pure row-registration time.

### Inventory and validation

Explicit Tensor method copies/transfers, cat/stack, graphics factories and sensor
utility allocations now join the source checklist. The [field contracts](stage-c-field-contracts.md)
expand their boundary ownership and alias relationships. The census contains
626 sites across 178 files. A retained textual reconciliation queue has 246
single-candidate links, 199 ambiguous links and 181 sites needing expansion or
explicit mapping. **These are review candidates, not certified field coverage**;
all still require source-specific sign-off, including aliases and first use.
The remaining work is explicit in the coverage checklist.

Evidence directory: `.tmp_tests/stage-c-c1-integration-20261105/`.

- Core package and physical-surface suite: **190 passed**.
- Final focused authoring tests: **11 passed**, including rejected omitted
  completion, native prefixes, shared domain identity, independent State defaults,
  byte-preserving declared storage binding and changed-spec rejection.
- Real Lab 3.0 adapter suite: **9 passed, 3 skipped**; main: **12 passed**.
  Includes eager/captured paths, hard rebuild, output routing and native coupled
  force ownership. These do not qualify checkpoint continuation.
- Ruff passes on changed Python modules. AST comparison against preserved pre-C
  sources confirms all **221 Warp kernels/functions in 48 files unchanged**.
- The initial added authoring test omitted the required particles system and
  failed; that log is retained alongside the corrected passing fixture.

No approved reference, capacity, numerical threshold, source archive or prior
measurement was replaced. Full inventory certification, remaining field
migration, scratch poisoning, continuation, packaging and final performance
qualification remain pending.

## C1 shared-domain and sizing follow-up, 2026-11-05

Tet and Surface now share the referenced Tet's variant domain when they resolve
to the same body. Surface fields retain `i4h:{surface.id}_{field}` names and
separate lookup keys on that domain; repeated Surface instances cannot collide.
The helper rejects a different body before adding fields. There is one frequency
and one world/owner column pair for that logical row set. The probe caller now
resolves and passes the declared Tet domain explicitly. Normal CaseAuthoring
integration remains pending.

For chole64 this removes 18,823 registration rows and 150,584 Model metadata
bytes. The previous construction samples are unchanged and remain evidence for
the previous layout; no new timing sample was taken. Private Newton helper
names, purposes and verified rc1 revision are recorded in the
[dependency notes](newton18-qualification.md#stage-c-private-helper-dependencies).

### Estimates before edge, face and particle declarations

Host-only authoring of the unchanged chole configuration gives one tissue
variant, 18,823 tets, 30,038 edges, 18,456 allocated face slots (including cut
capacity), and 5,612 particles. Evidence and the calculation script are in
`.tmp_tests/stage-c-shared-domain-20261105/estimate.json` and `estimate.py`.
No edge, face or particle-family declarations have been added in this round.

| Proposed row family | Formula, E=64 and V=1 | Custom rows | Estimated registration |
| --- | --- | ---: | ---: |
| Edge variants, live masks/support, shared edge_part | (V+E+1)*30,038 | 1,982,508 | 3.086 s |
| Face variants, live tris/tri_tet, shared tri_part | (V+E+1)*18,456 | 1,218,096 | 1.896 s |
| Particle support defaults, with live support on native PARTICLE | V*5,612 | 5,612 | 0.009 s |
| Alternative custom particle support including live rows | (V+E)*5,612 | 364,780 | 0.568 s |

The native-particle option adds about **3,206,216 custom rows / 4.991 s** for
these families, plus bulk authoring of 359,168 native particle rows. The custom
particle alternative totals **3,565,384 rows / 5.551 s**. Both assume shared
logical edge/face row sets across Tet and Surface and retain existing shared
part-index layouts. They exclude part/world domains, fluid pools, other systems,
finalization and native bulk-authoring cost. Multiple fields on a row do not
multiply row counts.

Times extrapolate the preserved measured 1.556861 microseconds per row. Different
column widths/counts can change that rate; these are estimates, not fresh
measurements or performance acceptance. This cost must remain visible when
choosing the construction layout; it does not authorize a different indexing
contract or an automatic rebuild.

The inventory checklist now includes construction-only skin binding, omitted
Torch constructor/conversion sites and interop aliases. The revised census has
573 sites: 482 allocations, nine acceleration structures, three explicit pointer
views, 55 live helpers, 13 conversion-or-allocation sites and 11 interop views.
Conversion entries are potential allocation sites, not claims of unique device
ownership. Per-site certification and CaseAuthoring wiring remain open.

Verification: corrected-core package suite **185 passed**; focused shared-domain
suite **31 passed**. A subsequent repeated-Surface regression verifies separate
attribute names on the same domain (`repeated-surface.log`). Ruff passes.
The three installed helper modules match the verified rc1 checkout byte-for-byte.

## C1 schema review follow-up, 2026-11-05

**Schema corrections passed; complete inventory and runtime migration remain
pending.** This update supersedes the foundation counts below, without closing
C1. Evidence is retained in `.tmp_tests/stage-c-schema-20261105/`, including
failed development tests and both construction samples.

The declaration layer now uses `i4h:{instance}_{field}` names, with scene
`particle_*` and `pool_*` prefixes. Domains carry genuine Newton native
frequencies or one custom frequency per domain instance. Custom rows include
world and owner reference columns, and all fields of a row are authored in one
`add_custom_values` call. Native particle values are supplied through
`entity_values` for `add_particles(custom_attributes=...)`; there is no post-hoc
particle authoring. Enum, normalized string and custom-frequency references are
accepted and tested through two `add_builder` merges. Local topology columns
remain local; `native_tet` must contain only unreferenced -1 values.

Cached zero-copy views, buffer and container identity checks, empty padding,
independent states/default controls, intentionally shared controls, native host
prefix/stride placement and separate byte accounting remain covered. View
lifetimes now retain their backing arrays. Replaced metadata buffers and modified
cached-view layouts are rejected. Unknown references fail before builder mutation.

Explicit package helpers declare all eleven persistent per-tet fields from Tet,
Surface and Thermal, preserving shared variant data, six/four-column local
indices, world-local state and native topology counts. These helpers are **not
yet connected to normal CaseAuthoring**. The [field contracts](stage-c-field-contracts.md)
now describe scene/pool, topology, rods/material tables, controls, mounts,
contact caches, surgical history, fluids/emission, appearance and excluded
renderer/host fields. Allocation-site reconciliation remains open: this document
is not yet a certified complete inventory.

Verification:

- Corrected core full package suite: **184 passed** (`packages.log`, XML).
- Schema plus per-tet suite in pinned Lab 3.0: **30 passed**; main: **30 passed**.
  These are declaration tests, not host runtime or continuation qualification.
- A subsequent zero-world per-tet regression is recorded separately in
  `empty-tet.log`; it preserves shared Model rows without inventing native entities.
- Existing physics constructors, kernels and ordered system loop remain unchanged.
  State migration, checkpoint copying, scratch poisoning and fresh binding are
  still pending; no numerical or final performance gate is claimed.

### Chole 64-world construction cost

One fresh process used preserved pre-C (Stage B) package source; one used the
same stack with the full eleven-field per-tet declaration probe. Both used the
corrected Warp binary and capacity 256. Samples were retained without timing
retries. The probe is additive: existing runtime buffers have not been removed.

| Measurement | Preserved Stage B | Added declarations |
| --- | ---: | ---: |
| Total construction | 1.129076 s | 5.275111 s |
| Domain-row registration | none | 3.838928 s |
| Newton finalization | 0.208890 s | 0.458105 s |
| Retained device bytes | 1,064,485,942 | 1,119,524,394 |
| Peak construction device bytes | 1,077,691,228 | 1,132,729,680 |
| Additional State/Control pair | 12,930,048 | 27,386,112 |

**The per-row Python authoring cost is the main cause:** 2,465,813 custom rows
cost 1.556861 microseconds each, totaling 3.838928 seconds. The 55,038,452-byte
retained increase consists of 26,126,324 Model bytes, 14,456,064 initialization
template bytes and 14,456,064 bound-State bytes; declared Control bytes are zero.
Changing ownership alone does not establish a memory reduction.

An unrelated GPU process was present in both samples and is recorded in the
reports. These construction measurements do **not** satisfy the exclusive final
5% step-performance gate. Exact reports are `chole64-stage-b.json` and
`chole64-declarations.json`, with process logs beside them. Later changes only
add declaration validation and zero-world handling; the measured chole row
layout and registration path remain the same.

## C1 implementation progress, 2026-11-05

**Stage C is incomplete. C1 is in progress; C2–C4 and the final campaign have
not started.** The declaration infrastructure below is isolated from the current
runtime. No existing persistent allocation has yet changed ownership. The
inventory gate is still open, so this is not a C1 exit or a continuation claim.

Before adding the infrastructure, the exact dirty working tree was preserved in
`.tmp_tests/stage-c-entry-20261105T130848Z/`:

- `pre-c-source.tar` and `source-index.json`: all 528 tracked and nonignored
  working-tree files, including source, assets, tests, workload scripts and pins;
  `archive-sha256.txt`, `git-status.txt` and `working-tree.patch` retain provenance.
- `approved-references/`: copied approved six-case references, including the
  capacity-256 chole manifest with its exactly-three-replays policy.
- `stage-b-timing/`: the completed exclusive campaign and workload definitions.
- `qualified-stack/`: native binary/source provenance, launcher, environment
  paths/configuration and actual native-library hashes for corrected core,
  Lab 3.0 and main. The existing qualified environments were left in place.

The [explicit declaration layer](../physics/src/i4h_physics/storage.py) accepts
package-supplied fields keyed by owner-instance domain and field name. It
registers Newton custom frequencies and Model/State/Control attributes, stores
flat backing arrays, and caches views of the declared shapes. It neither discovers
runtime arrays nor constructs or schedules systems. Authored input arrays are
copied on the host; binding only creates aliases and checks ownership. Registering
freezes the declarations; additions require reconstruction.

The pinned Newton finalizer omits zero-count custom attributes. Empty logical
fields therefore explicitly reserve one inaccessible backing element, while
their cached views retain their exact empty shape. Accounting includes this
padding and reports Model fields, initialization templates, each State/Control,
and the overlap of intentionally shared controls separately. These are declared
buffer totals, **not** construction peaks or total runtime memory measurements.

The [source census tool](../scripts/inventory_stage_c.py) records allocation and
helper sites for offline review; it does not generate runtime declarations.
`allocation-census.json` contains 558 sites from 176 source files: 482 allocation
calls, 65 live-helper calls, nine acceleration-structure constructors and two
explicit pointer views. This is **not a complete field inventory**: helper and
loop expansion, nested material/container fields, alias relationships, reset
semantics and first-reader/writer contracts remain to be reviewed. Its entries
explicitly retain pending ownership rather than guessing scratch eligibility.

Verification using the corrected core environment:

- Focused declaration tests: **19 passed**, including CPU/CUDA bidirectional
  Torch aliases, independent States/default Controls, intentionally shared
  Controls, repeated-system domains, empty domains, scalar/vector/matrix dtypes,
  byte-preserving rebinding, unchanged foreign native particles, native topology
  counts, the local `native_tet=-1` sentinel, incompatible buffers, nested
  replacement and separate finalized-model rejection. Logs/JUnit are
  `storage-tests.log` and `storage-tests.xml` in the entry evidence directory.
- Full package suite: **173 passed** (`packages.log`, `packages.xml`). Ruff
  passes for all three new Python files. Preservation checks verify all 528
  archived file contents, unchanged approved references/native binaries and
  unchanged pre-existing runtime sources.
- Existing runtime/host replay, scratch poisoning, checkpoint continuation,
  packaging and final exclusive performance gates: **not run for Stage C**.

Next required gate: finish the per-field allocation contracts and extend the
declarations into package authoring, including host-dependent completion. Only
then qualify C1 and begin migration. `scene.attr`, command-view APIs, checkpoint
copying, tip-evidence declarations and fresh-runtime continuation remain pending.
Stage B references, kernels, dependencies, capacity and accepted limitations are
unchanged.

## Entry decision, 2026-11-05

**Stage C prerequisites are satisfied for the qualified stack. Runtime migration
has not started.** [Stage B's final closure record](stage-b.md#final-closure-record-2026-11-05)
records the completed exports/lifecycle, host grid release, numerical regression,
chole reference disposition and exclusive timing/cadence gates.

The first complete valid timing campaign passed all eight comparisons against
the unchanged +5% median limit (largest slowdown +2.392%). All 20 workers had zero
replay allocation growth; exports-on overhead, construction time, retained bytes
and peak bytes were measured separately. The cadence test passed once in that
same exclusive window. No measured sample was replaced or retried.

Two explicit limitations carry forward:

- Chole uses the user-approved capacity-256 reference and majority-of-three
  policy. The measured 1/36 failure and all prior samples remain evidence.
- Coupling normal refinement still **fails** at 5.194896832108498% /
  5.309711396694183% against 5%; the user accepted that limitation.

The qualified runtime uses the [local Warp 1.17.0 capture correction](warp117-hashgrid-capture.md),
not merely any binary reporting version 1.17.0. Preserve its provenance and the
original environments. Dependency pins, numerical thresholds, capacity 256,
q0/inv_mass sidecars and persistent-state ownership remain unchanged at entry.

Stage B closure is complete. The ownership plan below is the next stage's scope;
this entry decision does not implement it. Earlier entries retain historical
gate states and are superseded by this decision.

## Stage B chole disposition, 2026-11-05

The user approved chole's majority-of-three policy. The unchanged capacity-256
candidate is now the approved reference, with the measured 1/36 failure rate and
policy in its manifest. Pre-change verification passed 3/3 at the same capacity;
all three fresh-process isolation trials passed. Corrected standalone, Lab 3.0
and main each passed 3/3 exports-on replays. The numerical thresholds and the
other five reference cases are unchanged; every earlier failure is retained.
See [Stage B closure](stage-b.md#chole-reference-accepted-2026-11-05).

The host grid and chole gates are closed. **Stage C entry remains pending only
exclusive timing, export overhead and the cadence test in that same window.**
The latest exclusivity guard found unrelated GPU workloads and launched no
measurement. Coupling remains visibly failed against 5%, with the user's
accepted limitation. This remains Stage B closure work; persistent-state
migration has not started. Earlier entries below are historical reviews.

## Stage B capture correction follow-up, 2026-11-05

The [Warp capture correction](warp117-hashgrid-capture.md) is implemented and
passes the analytic reproductions, 47 upstream tests and both focused host
suites (Lab 3.0: 109 passed / 9 skipped; main: 112 passed / 6 skipped).
All five approved numerical cases pass on all three hosts with exports enabled;
chole passes one replay per host against its still-unpromoted candidate. Memory
snapshots at 1/64 worlds are unchanged from the grid release, with zero growth.
The final package/runtime suite passed: 760 passed, 47 skipped, 1 deselected,
after the keyboard-only test's display-isolation correction. Cadence remains
reserved for an exclusive window. The final preflight still found unrelated GPU
compute workloads and launched no timing or cadence worker. Historical failures
are preserved and are not relabeled as passing.

The host grid-release/qualification gate is passed for this corrected build.
**Stage C entry remains gated** by chole's explicit reference disposition and
exclusive timing/export overhead/cadence. The coupling assertion remains failed
and explicitly accepted. Dependency pins, contact capacity, `q0`/`inv_mass`
sidecars and persistent ownership are unchanged. No Stage C migration or final
Stage B acceptance has started. Earlier reviews below retain the pre-correction
state; see [current Stage B evidence](stage-b.md#captured-grid-correction-2026-11-05).

## Stage B closure follow-up, 2026-11-05

The prior closure round is accepted; the current follow-up is still **Stage B**.
The host particle grid is now released, while `q0` and `inv_mass` remain i4h-owned.
Memory snapshots passed on both hosts at 1/64 worlds. Main's focused suite passed
103 tests; Lab 3.0 retained one parity failure (99 passed, 9 skipped). A fixed
one-before/one-after diagnostic pair passed without reproducing that failure;
it does not qualify the failed suite sample.

The subsequent [focused parity investigation](stage-b-parity-investigation.md)
reproduced a separate captured-HashGrid metadata alias in the pinned dependency,
including through i4h Fluid. It did not reproduce or identify the original
suite failure's differing field. No correction or acceptance is claimed; the
host qualification gate remains unresolved and runtime sources remain unchanged.

Chole awaits the user's reference-policy decision. Majority-of-three has not
been selected: no candidate promotion, manifest update, pre-change capacity-256
verification, isolation trials or chole-only comparator policy was performed.
Exclusive timing/export overhead and the authorized one-shot cadence rerun
remain pending a free GPU. The campaign's busy-device guard launched neither.

The final Stage B summary and Stage C entry acceptance remain deferred until
these three gates close. This file records the pending prerequisites and retains
the future ownership plan; **no Stage C migration has begun**. See
[Stage B closure](stage-b.md#host-particle-grid-release-2026-11-05) for measurements
and the retained host-suite failure.

## Entry review, 2026-11-05

**Status: preparation only; runtime migration has not started.** The supplied
Stage C plan permits planning now and requires the remaining Stage B exports,
exclusive timing, chole-reference disposition and coupling-refinement
disposition to be completed or explicitly waived before runtime migration.
The Stage B closure instruction carries that gate forward. It explicitly accepts
the coupling refinement limitation, but does not accept the chole reference or
waive exports and exclusive timing. No further waiver is inferred.

The current worktree already contains the Stage B implementation and substantial
uncommitted changes. The current changes implement Stage B exports and qualification. They are not
Stage C runtime migration.

| Prerequisite | Current repository evidence | Entry disposition |
| --- | --- | --- |
| Chole reference | [Stage B outlier qualification](stage-b.md#outlier-rate-qualification): capacity 256 has zero overflow but 1/36 runs fails 19 force/moment/jaw products; maximum force difference 16.60479736328125 versus tolerance 7.38482666015625. The candidate is unpromoted; conditional pre-change verification and isolation trials were not reached. | Open; acceptance of the capacity fix is not promotion of its reference. |
| Exclusive timing | [Stage B timing](stage-b.md#exclusive-timing-awaiting-a-free-gpu): exclusive campaign pending. Earlier shared-GPU samples do not close the 5% median gate. | Open; this review does not infer current GPU availability from historical PIDs. |
| Coupling refinement | [Paired refinement](stage-b.md#paired-coupling-refinement-accepted-limitation-assertion-failed): world 0 fails at 5.194896832108498% pre-change and 5.309711396694183% replacement against 5%. | **Accepted limitation** by the 2026-11-05 instruction; assertion remains **failed**, both samples preserved. |
| Exports | [Export contract](component-contracts.md#stage-b-solver-observables) and [current Stage B qualification](stage-b.md#closure-work-2026-11-05): raw wrench/contact-depth and optional surgical geometry implemented; focused tests and both qualified host suites pass. | Output/lifecycle and five approved numerical references passed on all three hosts; chole candidate passed one replay per host without promotion. Export overhead remains pending. |
| Full runtime regression | Stage B full suite retains a cadence failure: 81 physics steps for 17 render frames, requiring >85. The new neutral-name boundary failure was corrected and separately verified. | **Failed**; no retry or waiver inferred. |

`chole_tet_med.yaml` currently specifies `instrument_contact.records: 256`.
Preserve that value, every failed sample, all archives and dependency pins.
Do not restore the absolute 0.5 cutoff, promote an envelope automatically, or
change byte equality, slot canonicalization, factor three or zero floors.

The preceding closure-round evidence is recorded in [the closure report](stage-b.md#final-numerical-and-allocation-results). Nine captured exports-on allocation checks passed with zero growth. Stage C runtime migration remains **gated** by chole reference disposition, exclusive timing/export overhead, and the retained cadence failure. No Stage C implementation was started.

## Ownership contract

Declarations belong to each package. A small shared binding layer maps declared
storage to Newton attributes and cached views; it must not discover arbitrary
arrays, schedule systems or introduce another numerical implementation.

| Owner | Responsibility |
| --- | --- |
| Model | Topology, authored defaults, exact inverse masses, variant data, live parameter/material tables and indexing metadata. |
| State | All mutable i4h physics history, topology changes, particle flags, accumulated appearance and pending reactions. |
| Control | Existing packed actions and externally written commands. |
| Solver | Scratch overwritten before consumption, self-clearing accumulators and rebuildable acceleration structures. |
| Observables | Stage B outputs, retaining raw/conditioned distinctions and their existing ownership. |

Continuation is a copy between preallocated State/Control containers for the
same finalized Model, unchanged parameters and unchanged structural settings,
at completed physics-step boundaries. It includes all i4h-owned history, but
does not restore host time, assessment, renderer state or rigid-child private
history. Pure output measurements become current after the next completed step.

`author_case`, `add_world` and `bind` remain the construction boundary. Declare
storage before finalization, author exact values for each world, and bind aliases
without initializing or resetting their contents. Keep existing variant sharing,
padding, local indices, native Newton tet/triangle counts and five-hook order.

## Source-reviewed ownership map

This is a preparatory map, **not the completed C1 field inventory**. C1 must
expand every group into explicit declarations with dtype, shape, indexing,
initialization, episode-reset policy and consumers. No group below establishes
that all its runtime allocations have migrated or that continuation passes.
`E` denotes worlds, `V` variants; other capacities are local to each source.

| Source / fields | Intended owner and layout | Initialization, reset and consumption constraints |
| --- | --- | --- |
| [Scene particle blocks](../physics/src/i4h_physics/scene.py): `q`, `qd`, `q_prev`, `flags`; `q0`, `inv_mass`, `flags0`, `radius`, body variants | State for live positions, velocities, history and flags; Model for authored rest, exact masses, flags/defaults, radius and variant lookup. Position fields are vec3, masses/radii float32, flags int32. Main rows remain world-major; private fluid pools remain body-major. | `q_prev` is currently initialized from bound host positions; migrated binding must preserve restored history instead. Copy only authored particle ranges, including nonzero host offsets; leave foreign rows untouched. |
| [Tet](../physics/src/i4h_physics/systems/mechanics/tet.py): connectivity/rest/default masks and part indices | Model: `tets` vec4i `(V,T)`, `edges` vec2i `(V,S)`, `rest_volume`, `edge_rest` float32; `active0`, `edge_active0`, `tet_part`, `edge_part` int32. | Preserve variant lookup and exact authored topology. Reserve logical `i4h:tet/native_tet`, int32 all `-1`, without reference metadata. |
| Tet live masks and materials | State: `active` int32 `(E,T)`, `edge_active` int32 `(E,S)`. Model: `spring`, `spring_damping`, `volume`, `volume_damping`, float32 `(E,parts)`. | Episode reset selects each world's variant masks; live material edits persist. Constructor currently calls reset and must stop doing so during binding. `delta`, `delta_count`, per-substep XPBD lambdas remain scratch. |
| [Physical Surface](../physics/src/i4h_physics/systems/mechanics/surface.py) | Model: variant tet-edge/face tables, `tris0`, `tri_tet0`, support defaults, part and cut-capacity metadata. State: `tris` vec3i `(E,F)`, `tri_tet` int32 `(E,F)`, `support` int32 `(E*count,)`, `edge_support` int32 `(E,S)`, accumulated deletion/overflow bookkeeping. | Preserve cut-face slot semantics and selected-world reset from variant defaults. Resolve shared tet arrays through `scene.attr`; refs still establish order and authored relationships. Classify `dirty` and `cut_count` by next-use analysis, not their names. |
| [Regional properties and rod controls](../physics/src/i4h_physics/regions.py), [material providers](../physics/src/i4h_physics/systems/rods/material.py) | Model: `RegionProperties.edge` and `tables` float32 `(E,regions)`, immutable rest data and radius baseline. State/Control declarations must explicitly cover `RodControls.stretch_scale`, `bend_scale` float32 and `bend_offset` vec3. | Tables persist across episodes. Control fields may be written by systems or externally; classify each supported writer. Derived material tables are regenerated at the existing first-substep phase. Audit nested dictionaries and namespaces, not only system attributes. |
| [RodDirect](../physics/src/i4h_physics/systems/rods/rod_direct.py), [RodPbd](../physics/src/i4h_physics/systems/rods/rod_pbd.py) | State: frames, angular history and persistent rod controls. Model: rest geometry, exact angular inverse masses, topology, inertia data and live materials. | Preserve each solver's current reset policy and PBD multiplier meanings. Direct assembly, factorization, prediction and correction workspaces remain scratch only where next-use overwrite is established. |
| [InstrumentControl](../surgical/src/i4h_surgical/systems/instruments/instrument_control.py) | Control: `actions` float32 `(E,instruments,10)`. State: root/orientation/jaw/trigger history, held/pending state and sphere history. Model: authored geometry, default poses and live parameters. | Constructor reset currently seeds valid poses. Move initialization to authoring; binding must not replay it. Instrument channels remain position `0:3`, quaternion xyzw `3:7`, grip `7`, energy `8`, fire `9`. Preserve mounted pose writers and source order. |
| [CatheterControl](../endoluminal/src/i4h_endoluminal/systems/rods/catheter_control.py) | Control: `actions` float32 `(E,3)` insertion, rotation, bend. State: integrated `targets` vec3 `(E,)`, root/mount history. Model: track/rest defaults, bounds, rates and other live parameters. | Episode reset still uses existing zero commands/targets and valid rest poses. Retain root reaction accounting and the shared rod-control aliases. |
| [Mounts and MountGroup](../physics/src/i4h_physics/coupling.py) | Model: `target` int32 `(M,)`, `offset` transform `(M,)`, `body` int32 `(E,M)` and environment transforms. State: pending `force`, `torque` vec3 `(E,M)`. | Pending buffers initialize/reset to zero, but continuation restores their actual values. Apply previous reactions before rigid stepping; drive mounts, step i4h, then fill next reactions. Keep component/world frame conversion and root-to-COM moment shift. |
| [GraphContainment](../physics/src/i4h_physics/systems/contact/graph_containment.py) | State as one group: `cand_idx` int32 `(E*n*capacity,)`, `cand_count`, `cand_overflow` int32 `(E*n,)`, `ref_samples` vec3 `(E*n,)`, `ref_tree` vec3 `(E*Pt,)`, `built` vec2 `(E,)`, `rebuild` int32 `(E,)`. Model: tree topology and live settings. | Candidates initialize to `-1`, counts/references/built to zero and rebuild to one. Save/restore preserves them together. Do not call `refresh`, which changes rebuild timing. Displacement scratch and self-clearing correction arrays need next-use verification; contact outputs retain observable ownership. |
| [Fluid](../physics/src/i4h_physics/systems/fluids/fluid.py) and Scene pool block | State: pool particle history, `cursor` int32 `(E,nf)` and accumulated overflow. Model: `const` tables, pool indexing and live viscosity/cohesion/relaxation. | Emitters consume ring cursors across steps. Pool/span/body views must alias one owner. Neighbours, grid, density, lambdas and velocity/correction scratch are rebuilt in existing phases; distinguish pure alive-count measurements from history. |
| [Bleeding](../surgical/src/i4h_surgical/systems/fluids/bleeding.py) | State candidates: `cut`, `source` int32 `(E,K)`, fractional `acc` float32 `(E,K)`, `serial` int32 `(E,)`; preserve clipped-state aliasing. | Source discovery precedes spawning; prove which source fields can be reconstructed before excluding them. Reset clears selected emission history. Sealing/source measurements remain outputs. |
| [Emitter](../surgical/src/i4h_surgical/systems/fluids/emitter.py) | State: `acc` float32 `(E,)`, `serial` int32 `(E,)`; Model/Control classification for live source commands. | Preserve fractional emission and spawn sequence across continuation. `spawned` is a step measurement. |
| [Smoke](../surgical/src/i4h_surgical/systems/fluids/smoke.py) | State: `pos_age`, `vel_life` vec4f `(E*count,)`, `cursor`, `serial` int32 `(E,)`, `acc` float32 `(E,)`. | Preserve particle lifetime and serial history even when not visible. Source-search scratch is rewritten; hot/alive measurements stay outputs. |

Remaining mandatory inventory families include attachments, connectors, shell,
bounds, both stretch-breaking systems; static/part/fluid/instrument contacts and
rod/shell/tet containment; surgical Surface/Appearance, Centreline, Thermal,
Cutting, Grasp, VolumetricGrasp, Hook, Clipping, Suction, ContactPaint, ToolForce;
ControlLayout, Drag and TipEvidence. Include `System._live` allocations, nested
containers, optional fallback arrays, all renderer-consumed accumulated
appearance and cached aliases. Use [the Stage B refresh audit](stage-b-refresh-audit.md)
as a next-use checklist, not as proof that these fields are already declared.

## Binding and lifecycle work

- `scene.attr(domain, key)` caches zero-copy declared-attribute views. Domain
  identity includes the owning body/system instance; repeated types cannot
  collide. Surface, TetContainment, Thermal, Cutting, StretchBreaking, Grasp,
  VolumetricGrasp, Hook and Centreline stop obtaining tet storage from a Tet
  instance while retaining reference ordering and relationship metadata.
- Binding `command_views(system_id)` returns Warp aliases; the component returns
  cached Torch aliases of the same packed commands. Verify both write directions,
  bounds, system/instrument ordering and recording payload compatibility.
- `save_i4h(dst_state, dst_control, mask=None)` and
  `restore_i4h(src_state, src_control, mask=None)` copy declared i4h fields and
  owned particle ranges. No separate system snapshot, host sync or per-step
  allocation. Empty masks do nothing; unselected worlds and host entities remain
  unchanged. Validate source/destination model compatibility before copying.
- Keep continuation distinct from episode reset and cache-only reset. Bind a
  new runtime to restored containers without constructor resets, general refresh
  or prepare paths that overwrite history. In particular, `ControlLayout.prepare`
  currently fills rod scales and clears bend offsets; this needs an explicit
  construction/prepare disposition before claiming fresh-runtime continuation.
- Extend identity validation beyond current top-level arrays to declared Newton
  namespaces, containers, nested material/mount fields and cached views. Reject
  replacement before captured execution; invalidate stale bindings on rebuild.
- Account for Newton initialization templates, every host State/Control,
  retained live allocations and construction peaks separately. Remove old
  persistent owners when their declarations migrate; aliases are not copies.

## Milestones and evidence required

| Milestone | Implementation / qualification status | Required focused evidence |
| --- | --- | --- |
| C1: declarations and ownership | Not started; source-reviewed map above is partial. | Complete field inventory; multiple instances/variants; padding/empty domains; exact world values; foreign particle offsets; independent allocations and intentional shared Control; nested identity rejection; template/instance memory accounting. |
| C2: particles, tets, topology | Not started. | Exact rest/inverse masses and flags; native counts unchanged; `native_tet=-1` with no refs; all named tet consumers use domains; cuts/support/reset parity. |
| C3: commands, rods, coupling | Not started. | Named/packed bidirectional aliases; valid reset poses; recording replay; rod motion/material persistence; mounted controls and pending reaction replay with unchanged delay. |
| C4: remaining history and continuation | Not started. | Save/advance/restore/replay, including fresh binding, active grasp/hook/clip, thermal/fluid/emission/appearance and containment history; partial/empty/repeated restoration and untouched foreign entities. |
| Final campaign | Not run. | Eager/captured standalone and both Lab hosts; lifecycle, decimation/assessment, native rigid `CONTACT_F` and pure rejection; six-case approved-reference replay; allocation/synchronization audit and exclusive pre-C timing within unchanged 5% median limit. |

Before the first runtime edit, preserve a complete pre-C source and workload
snapshot in a fresh evidence directory and record the Stage B closure/waiver.
The capacity-256 chole reference disposition must be explicit before claiming a
six-case approved-reference numerical pass. A waiver to begin work alone does
not promote the candidate or turn any failed numerical/performance check into
a pass.

Verification for this preparatory change is limited to source/document review,
local-link resolution and whitespace checks. No runtime tests or GPU campaigns
were run, since no runtime code changed. Native elasticity/topology replacement,
iteration callbacks, native coupling migration, particle-force input,
rod-material conversion, disk checkpoint formats and case/recording schema
changes remain outside Stage C.
