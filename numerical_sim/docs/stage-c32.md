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

# C3.2 — Rod ownership, shared controls and catheter command views

**C3.2 is implemented and remains open.** The single frozen campaign passed
**34/36 groups, 43/48 replays and 15/18 chole trials**. Two `aorta_shell` groups
failed. A subsequent alias-validation correction passed focused checks in all
three environments; full qualification was not rerun on the corrected source.
HDF5 recording/replay verification is unavailable in the retained environments.
**Stop for review before C3.3.**

C3.1 was explicitly accepted for progression with retained failures. Its frozen
campaign passed 34/36 groups, 45/48 replays and 17/18 chole trials. Standalone eager
and patched Lab 3.0 captured `aorta_shell` failed stepped catheter-impulse
comparisons; causes remain unresolved. Every chole group passed the existing
2-of-3 policy, with the failed individual trial retained. The missing mounted-tool
fixture declaration was corrected and its 12 affected checks passed afterward;
full suites were not rerun. Earlier clipping/fluid observations remain unexplained.
The [historical C3.1 report](stage-c31.md) and certificates remain intact. Its
progression acceptance does not waive any C3.2 failure.

## Preservation and scope

The pre-edit audit matched **89,030 files**, manifest SHA256
`a12c665ced02bd82fc281ea133c5c6cd77538d2f8e7162c3e4e2d3ac6022d99e`.
All 569 tracked and untracked nonignored source files were archived with hashes,
status and binary diff in [the evidence directory][evidence]. Every prior manifest
entry is preserved in place or in `pre-c32-source.tar`. Environments, references,
endpoint arrays, historical certificates and foreign logs remain in place.

The retained pins are Newton `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`, Lab main
`4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1`, patched Lab 3.0
`e53cd288727a5ca3dae7187d1d415b4d32b67e11`, patched Warp 1.17.0,
MuJoCo/MuJoCo-Warp 3.17.0 and Python 3.12.12. No sync or upgrade occurred.
Warp's imported native binary remains SHA256
`eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.

The preservation audit compares rod/endoluminal numerical-kernel ASTs and every
step/substep/iteration/reset hook with the archive. They are unchanged. Clipping
assertions and its failure-only exporter are unchanged. No execution-order,
capacity, comparator, tolerance or approved-reference changes were made; no
numerical attribution campaign, clipping fix, C3.3 implementation, commit or push.

## Ownership, identity and initialization

[Physics declarations](../physics/src/i4h_physics/rod_storage.py) own rods and
regional materials. [Endoluminal declarations](../endoluminal/src/i4h_endoluminal/rod_storage.py)
own catheter, ControlLayout and Drag fields. Physics never imports endoluminal.
Each system supplies `declare_storage`; direct callers use the matching
`declare_rod_direct`, `declare_rod_pbd`, material-provider,
`declare_catheter_control`, `declare_control_layout` and `declare_drag` helpers.
Missing, duplicate, late or incompatible declarations fail explicitly. Indirect
fixture factories and host composition paths declare storage before allocation.

| Subsystem | Model | State | Control |
| --- | --- | --- | --- |
| Direct rod | Rest frames, exact angular inverse masses, rest length/Darboux, local inertia, live parameters | frame, omega | Shared RodControls |
| PBD rod | Segment/pair topology, rest frames/Darboux quaternion, rest length, exact angular inverse masses, unit_twist, live parameters | frame, omega | Shared RodControls |
| Regional material | Membership, tables, radius baseline, shared rest/topology aliases | — | Shared RodControls aliases |
| Catheter | Track defaults, base orientation, live parameters | targets, root | Packed actions `(E,3)` |
| ControlLayout | Configuration, topology, hops, arc offsets, world-to-layout map | — | Shared RodControls aliases |
| Drag | max_delta | weights, baseline | active, node, target |

Shared rod geometry and controls use body-qualified `body_<index>_rod` domains.
System-specific domains use the declared system id. Shared rows have world -1;
runtime rows carry actual worlds. Native particle owners resolve after host
creation, including foreign prefixes. Nonparticle-owned rows and padding use -1.
Only identity columns carry Newton references; kernel topology stays local.

Each body and selected live Control has exactly one stretch_scale, bend_scale and
bend_offset allocation. Solver, provider, catheter and layout consumers alias it.
Direct material rest arrays and PBD material pair topology alias the same declared
rod Model fields. Nested containers, tables, routed solver aliases and Newton
buffers are checked; replacing them requires a fresh binding. Topology, capacity,
backend, inertia allocation mode and storage layouts are fixed. Existing chain,
tree and single-variant restrictions remain.

Templates contain exact rest frames, zero angular history, unit rod scales, zero
bend offsets, zero catheter actions/targets, authored catheter root and Drag
defaults. The existing generic trailing finalizer computes local inertia from the
Model being finalized, using its exact i4h inverse masses, native radii and declared
rest lengths through the unchanged float64 computation. Initialization completes
before runtime template clones. New declaration records retain host data and weak
scene references, not finalized device arrays. Two Models from one retained builder
remain independent.

Predicted frames are scratch copies of the selected State frames. Derived regional
outputs, solver workspaces and Drag.ref remain scratch. Self-clearing accumulators
still start at zero. Catheter impulse/moment remain completed-step outputs.
Mounted flags, mount poses and pending reactions retain their existing ownership.

## Preparation, reset and commands

Binding and cache-only preparation preserve commands and history byte-for-byte.
ControlLayout defaults moved into construction templates; its legitimate per-step
writes and provider-before-solver order are unchanged. Catheter tip recapture
validates the new range first. Shrinking old to new clears only `[B-old, B-new)`
across worlds, preserving actions, targets, retained tip slots and other slots.
Unchanged/increased ranges and repeated preparation write no commands. The existing
step kernel fills the active tip range. PBD sweeps, tip axis/mapping and supported
scalar routing settings remain recapturable.

Explicit resets preserve their prior behavior: rod rest frames/history and channel
defaults, catheter actions/targets and reactions, and only Drag.active. Drag node,
target, weights and baseline remain intact; live Model edits persist. No additional
catheter-root reset was introduced.

| Catheter view | Slice | Shape |
| --- | --- | --- |
| actions | 0:3 | `(E,3)` |
| insertion | 0:1 | `(E,1)` |
| rotation | 1:2 | `(E,1)` |
| bend | 2:3 | `(E,1)` |

Mappings are cached/read-only and their Warp/Torch values are writable zero-copy
views. Named Warp strides are `(12,4)`, Torch strides `(3,1)`, and column offset
is `4*c` bytes. The C3.1 generic validation, backing retention, invalidation and
stream ordering apply. Commands remain normalized rates with unchanged bounds,
scaling, integration, step ordering and recording formats. Drag has no invented
packed action layout; typed commands were not added.

## Verification and accounting

The pre-migration structural/live-settings test passed and retained its observation:
actions/targets survived preparation while all bend slots were cleared. The migrated
test and ownership suite check preservation and selective retirement instead.
Development logs retain every failed attempt and correction. Final development
results: core focused **231 passed, 1 skipped**; main and patched Lab 3.0 final
focused **77 passed each**; package suite **383 passed**. The dedicated ownership
suite had 29 tests at freeze across its mode/backend matrix; the corrected-source
suite has 61 cases, with availability-dependent skips described below. Ruff and the reviewed inventory
pass. Earlier development failures were missing fixture declarations, a flat-view
authoring shape error and assertions using old preparation/scratch expectations;
these were corrected before freezing. Focused coverage
includes CPU eager, CUDA eager and capture; unit/physical Direct and PBD chain/tree;
regional/nonregional routing; multiple bodies/ids/worlds and host prefixes; empty
bend domains; pointer identity; packed/named Warp/Torch commands; partial/empty/
repeated resets; persistent materials; active Drag continuation; independent Models,
States and Controls; ordinary live-Control sharing and explicit template aliasing;
replacement/foreign-container rejection; scratch poisoning and zero accumulators.

Resumed/uninterrupted comparisons use fixtures whose required history is fully
migrated. Integrated mounted-tool, contact, containment and host lifecycle tests
remain without claiming complete restoration of their unmigrated history.
The new HDF5 catheter recording/replay check is retained but skipped because the
required optional dependency is absent; this remains an unverified gate.

[Focused allocation accounting][allocations] is separate from the established
construction benchmark. For nonempty domains, each rod State uses `28*E*F` bytes;
shared rod Control uses `4*E*S+16*E*K`; catheter State/Control use `24*E`/`12*E`;
Drag State/Control use `16*E*P`/`20*E`. Matching templates are counted separately.
Model fields, identities, padding, live/checkpoint containers, deliberate sharing
and finalization temporaries are separate. Aliases deduplicate by maximum backing
extent. Temporary host readbacks, float64 arithmetic and float32 inertia upload
leave no additional persistent finalizer buffer; the focused report includes
traced host transient peaks rather than inferring them from persistent totals.

| Synthetic fixture, 3 worlds | Model data | Templates | Each State | Each Control |
| --- | ---: | ---: | ---: | ---: |
| Direct unit + catheter | 564 | 1,056 | 660 | 396 |
| Direct physical regional + catheter | 660 | 1,056 | 660 | 396 |
| PBD regional chain + catheter | 688 | 924 | 576 | 348 |
| PBD regional tree + layout + Drag | 424 | 636 | 444 | 192 |
| Empty bend-pair tree + layout + Drag, including padding | 260 | 268 | 180 | 88 |

Identity-column bytes for the five focused fixtures above are respectively 800,
872, 1,000, 776 and 384, separate from Model data. Live/checkpoint containers have
the same sizes but independent storage. Sharing an ordinary live Control adds zero
bytes. Explicit template aliasing overlaps the listed Control bytes and affects
later clones. The empty-pair fixture includes 36 Model padding bytes (24 geometry,
4 unit_twist, 8 layout pairs), 16 live/checkpoint Control padding bytes and 16
matching template padding bytes. Direct finalization traced host peaks are 16,344
and 15,288 bytes for unit and physical-regional fixtures, respectively; explicit
readbacks use 80 bytes and the float32 inertia upload uses 84 bytes for seven
frames. These transient measurements are not persistent Model/State accounting.

The reviewed inventory contains **523 sites in 186 source files**: 51 old allocation
sites removed and five view/scratch sites added relative to C3.1. The established
`chole_tet_med` construction samples remain exactly one at one world and one at
64 worlds. All six residue arrays remain **2,020,320 bytes** and **129,300,480 bytes**
respectively, with no new residue template/checkpoint copies.

## Frozen campaign and disposition

The [frozen campaign certificate][frozen] covers **89,042 files**, manifest SHA256
`cc58fbedbfa541aebbe67220009080c8fb5aa3aa983e137e7f673f8003aeda2b`.
All **83 startup/exit process pairs** and **103,439 file-backed import observations**
match. Every worker and nested process was guarded. All 48 replay comparisons
were independently recomputed from retained endpoint arrays. Exactly 18 fresh-process
chole trials and two established construction samples ran. Capacity 256, zero floors,
factor-three comparison, approved references and the existing 2-of-3 chole policy
were unchanged. There were no qualification retries or reference promotions.

| Frozen suite | Passed | Skipped | Failed/errors | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Full core | 1,017 | 47 | 0/0 | 467.271 |
| Lab main six-module selection | 117 | 5 | 0/0 | 137.63 |
| Patched Lab 3.0 six-module selection | 114 | 8 | 0/0 | 134.355 |

Both host selections cover adapters, solver, component, Isaac Lab environment,
manager environment and coupling. All suite processes exited zero. The unchanged
clipping test passed this run; that observation does not explain or close its
historical failure. Earlier clipping and fluid observations remain retained.

| Environment / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-eager | pass | pass (2/3) | pass | pass | pass | pass |
| standalone-captured | pass | pass (3/3) | pass | **FAIL** | pass | pass |
| main-eager | pass | pass (3/3) | pass | pass | pass | pass |
| main-captured | pass | pass (2/3) | pass | pass | pass | pass |
| lab30-eager | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-captured | pass | pass (2/3) | pass | **FAIL** | pass | pass |

There are **49 failed product comparisons** across five failed replays. All chole
groups pass their existing policy, while these individual failures remain retained:

- [standalone-eager/trial0](../.tmp_tests/stage-c32-20261109/certificate/standalone-eager/chole_tet_med.trial0/chole_tet_med.json): 19 failed products; [endpoint arrays](../.tmp_tests/stage-c32-20261109/certificate/standalone-eager/chole_tet_med.trial0/chole_tet_med.npz).
- [main-captured/trial1](../.tmp_tests/stage-c32-20261109/certificate/main-captured/chole_tet_med.trial1/chole_tet_med.json): 19 failed products; [endpoint arrays](../.tmp_tests/stage-c32-20261109/certificate/main-captured/chole_tet_med.trial1/chole_tet_med.npz).
- [lab30-captured/trial0](../.tmp_tests/stage-c32-20261109/certificate/lab30-captured/chole_tet_med.trial0/chole_tet_med.json): 8 failed products; [endpoint arrays](../.tmp_tests/stage-c32-20261109/certificate/lab30-captured/chole_tet_med.trial0/chole_tet_med.npz).

The two failed groups are standalone captured `aorta_shell` (stepped catheter
impulse) and patched Lab 3.0 captured `aorta_shell` (stepped and masked-reset
catheter impulse). Causes remain unresolved; the retained failures are not
attributed to a particular subsystem or numerical change.

### Exact historical and new catheter-impulse observations

These tables show the maximum-error component for each failed comparison. The
[frozen certificate][frozen] stores complete stepped and masked-reset arrays for
**all six environment/mode combinations**, separately under `C3.1` and `C3.2` in
`aorta_shell_exact_catheter_impulse_endpoints`. Historical certificates and arrays
are unchanged. For the failures below, approved noise is
`1.5421312127728015e-05` and the factor-three threshold is
`4.6263936383184046e-05`.

C3.1 historical failures:

| Environment / mode | Endpoint | Index | Reference | Candidate | Absolute difference |
| --- | --- | --- | ---: | ---: | ---: |
| standalone-eager | stepped | `[0,0,0]` | -8.45981267048046e-05 | -0.00019178928050678223 | 0.00010719115380197763 |
| lab30-captured | stepped | `[0,0,0]` | -8.45981267048046e-05 | 1.504814008512767e-05 | 9.964626678993227e-05 |

C3.2 frozen failures:

| Environment / mode | Endpoint | Index | Reference | Candidate | Absolute difference |
| --- | --- | --- | ---: | ---: | ---: |
| standalone-captured | stepped | `[0, 0, 0]` | -8.45981267048046e-05 | 6.616961036343127e-05 | 0.00015076773706823587 |
| lab30-captured | masked_reset | `[1, 0, 0]` | -8.915209036786109e-05 | -6.382550054695457e-06 | 8.276954031316563e-05 |
| lab30-captured | stepped | `[1, 0, 0]` | -8.915209036786109e-05 | -6.382550054695457e-06 | 8.276954031316563e-05 |

Current failed-group reports and full arrays: [standalone captured report][sa-report],
[standalone arrays][sa-array], [Lab 3.0 captured report][lab-report],
[Lab 3.0 arrays][lab-array]. The earlier standalone eager failure passing in this
campaign does not establish that its unresolved cause was fixed.

### Established construction samples

Exactly one `chole_tet_med` sample ran at each world count. All allocation totals
match C3.1; the benchmark workload has no rod subsystem, so focused rod accounting
above remains separate. The benchmark's historical C3.1 variant label is retained.
These samples measure construction only and make no steady-state performance claim.

| Quantity | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model buffers, bytes | 6,172,433 | 93,446,585 |
| Initialization templates, bytes | 1,511,136 | 96,712,704 |
| Each live/checkpoint State, bytes | 1,578,280 | 101,009,920 |
| Each live/checkpoint Control, bytes | 200 | 12,800 |
| Retained Warp, bytes | 27,822,949 | 1,219,303,846 |
| Construction peak Warp, bytes | 28,029,193 | 1,232,509,132 |
| Additional State + Control, bytes | 1,578,480 | 101,022,720 |
| Host peak RSS, KiB | 794,372 | 2,749,740 |
| Six retained residue arrays, bytes | 2,020,320 | 129,300,480 |
| Construction time, seconds | 0.9287251039640978 | 15.561810837010853 |

Each residue array retains `336720*E` bytes; none gained a template or checkpoint
copy. Ordinary live-Control sharing adds zero bytes. Explicit template aliasing
has 200/12,800 bytes of overlap; it is not counted as another backing allocation.
The [one-world sample][construction1] and [64-world sample][construction64] retain
per-field, domain identity, padding, pointer, temporary and timing breakdowns.

### Correction after the frozen campaign

Read-only inspection found that replacing the rod's `variant` alias could be
silently adopted on preparation. After every frozen process and the independent
campaign auditor had finished, the affected source was archived in
[the frozen correction-source archive][fix-archive]. Binding now records the
variant identity. PBD effective stretch/bend/twist aliases are also registered at
construction so replacement before the first preparation is rejected. Numerical
kernels, per-step/reset hooks and execution order remain unchanged.

Exactly four frozen paths changed: physics `rod_storage.py`, `rod_pbd.py`, the
ownership test module and the reviewed inventory's source hashes. The
[retained source patch][fix-patch] shows all Python changes. The final tests add
30 alias-replacement cases across Direct/PBD, before/after preparation, CPU eager,
CUDA eager and capture, plus explicit catheter host lifetime and optional HDF5
recording/replay coverage. This is a production-source correction, so the frozen
full-suite and reference results **do not qualify the corrected source**.

The [current review certificate][current] covers **89,043 files**, manifest SHA256
`bf8ef3f8cbb390a1a605c16e50eb661dfb37b53d33cc817dafffc58704b8bec8`.
The corrected-source driver and three test processes have four matching
startup/exit pairs and 6,092 matching file-backed import observations. Its separate
auditor also passed startup/exit and import checks. Report and roadmap prose are
outside the source manifests, allowing final results to be recorded afterward.

| Corrected-source focused selection | Passed | Skipped | Failed/errors |
| --- | ---: | ---: | ---: |
| Core | 103 | 2 | 0/0 |
| Lab main | 104 | 1 | 0/0 |
| Patched Lab 3.0 | 104 | 1 | 0/0 |

The selection is ownership, runtime rods and runtime regions. All 30 new alias
regressions passed in each environment. Named Torch-command views across a host
step, hard rebuild and close passed in both hosts. Core skips the two host-only
checks. Both hosts skip the HDF5 recording/replay check because `h5py` is absent;
no dependency was installed or synchronized. That required check remains
unverified. The initial corrected-check launcher error occurred before workers
started and is retained in `post-campaign/launcher-error.txt`.

Full suites, reference replays and construction samples were **not rerun** after
the correction. Historical failures were not relabeled. Ruff, allocation-inventory
validation, source preservation and numerical-hook/clipping AST checks pass.

C3.2 remains **open**, with two failed reference groups, an unverified HDF5 gate
and only focused verification of the corrected production source. C3.1 progression
acceptance does not waive these gaps. No new disposition has been granted.
**Mounted-tool episode restore remains partial until C3.3.**
**Stop for review before C3.3.** No commit or push was made.

[evidence]: ../.tmp_tests/stage-c32-20261109/
[allocations]: ../.tmp_tests/stage-c32-20261109/focused-allocation.json
[frozen]: ../.tmp_tests/stage-c32-20261109/certificate/final-source-certificate.json
[current]: ../.tmp_tests/stage-c32-20261109/post-campaign/review-source-certificate.json
[fix-archive]: ../.tmp_tests/stage-c32-20261109/post-campaign/frozen-fix-sources.tar
[fix-patch]: ../.tmp_tests/stage-c32-20261109/post-campaign/source-change.patch
[sa-report]: ../.tmp_tests/stage-c32-20261109/certificate/standalone-captured/aorta_shell.json
[sa-array]: ../.tmp_tests/stage-c32-20261109/certificate/standalone-captured/aorta_shell.npz
[lab-report]: ../.tmp_tests/stage-c32-20261109/certificate/lab30-captured/aorta_shell.json
[lab-array]: ../.tmp_tests/stage-c32-20261109/certificate/lab30-captured/aorta_shell.npz
[construction1]: ../.tmp_tests/stage-c32-20261109/certificate/construction-1.json
[construction64]: ../.tmp_tests/stage-c32-20261109/certificate/construction-64.json
