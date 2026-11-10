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

# C3.1 — Instrument ownership and named command views

**C3.1 is implemented and remains open. Stop for review before C3.2.**
The single frozen campaign completed with **34/36 groups, 45/48 replays and
17/18 chole trials passing**. All three full suites failed the same four missing-
declaration fixture tests; that fixture was corrected after the frozen audit and
all four tests then passed in each environment. The two catheter-reference failures
remain unresolved. No full suite, reference replay or construction sample was rerun.

C3.0 was explicitly **accepted for progression with retained failures** by
the implementation request. This disposition does not change its historical
certificate or accept any C3.1 failed gate.

The retained C3.0 observations are: the second clipping trigger produced one clip
instead of two; patched Lab 3.0 eager `aorta_shell` exceeded the approved threshold
on stepped and masked-reset catheter impulses; both main chole groups passed the
existing 2-of-3 policy but retain their failed individual trials. The earlier
fluid discrepancy remains deferred with unresolved cause. [The unchanged C3.0
report](stage-c30.md) retains the exact values and endpoint archives.

## Preservation and pins

Before edits, all **89,012** C3.0 manifest entries matched SHA256
`35ddaf42183dfbcd3281624185391ce7a22b13f28cf627dcf1f252802b0f2a7a`.
The complete tracked and untracked nonignored source (**565 files**) was preserved
in [pre-c31-source.tar][snapshot], with entry hashes, status and binary diff.
[The preservation audit][preservation] verifies changed source against that archive
and every other C3.0 manifest entry at its original path. Existing environments,
certificates, references, arrays and foreign `FollouDeviceLogs` remain in place.

The existing C3.0 environments are reused without sync or upgrade:

| Dependency | Exact retained version |
| --- | --- |
| Newton | `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`, `1.9.0.dev0` |
| Lab main | `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1` |
| Patched Lab 3.0 | `e53cd288727a5ca3dae7187d1d415b4d32b67e11`, unchanged cloner patch |
| Warp | patched `1.17.0`; imported `warp.so` SHA256 `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb` |
| MuJoCo / MuJoCo-Warp | `3.17.0` / `3.17.0` |
| Python | `3.12.12` |

Fresh import/source identity checks passed for [core][core-id], [main][main-id]
and [Lab 3.0][lab30-id]. Instrument kernel ASTs and clipping assertions match the
preserved source. No step execution-order, capacity, comparator, tolerance, approved
reference, catheter ownership or clipping-mechanics change was made. No commit
or push was made.

## Ownership and initialization

[Instrument storage](../surgical/src/i4h_surgical/instrument_storage.py) declares
50 fields for each control-system instance. Shapes, dtypes, indexing and local
geometry indices are unchanged.

| Owner | Fields |
| --- | --- |
| Control | `actions`, float32 `(E,N,10)` |
| State, `(E,N)` | `root` (vec3); `shaft_q`, `wrist_q`, `jaw_q` (quat); `jaw_angle`, `jaw_delta` (float32); `held`, `grasp_pending`, `released`, `energized`, `fire_on`, `fired` (int32) |
| State, `(E,S)` | `sphere_q`, `sphere_q_prev` (vec3); `sphere_active` (int32) |
| Model, shared instrument geometry | `trocar`, `has_trocar`, `initial_position`, `initial_rotation`, `jaw_open`, `jaw_closed`, `jaw_response`, `grasp_threshold`, `wrist_pivot`, `jaw_anchor`, `wrist_end`, `has_wrist_end`, `wrist_end_bind`, `shaft_spacing`, `prim_start`, `prim_counts`, `jaw_root_a`, `jaw_root_b` |
| Model, shared sphere tables | `sphere_instrument`, `sphere_role`, `sphere_jaw_sign`, `sphere_local`, `sphere_bind`, `sphere_radius`, `sphere_chain`, `sphere_chain_index`, `sphere_chain_count` |
| Model, shared primitive tables | `prim_a`, `prim_b`, `prim_instrument`, `prim_kind` |
| Model, WORLD `(E,)` | live `max_velocity`, `energy_threshold`, `fire_threshold` |
| Unchanged | legacy `mounted` and `mount_pose`; host geometry records; rebuildable `_all` mask; other systems' state |

Each instance has its own `i4h:<system_id>_<kind>_rows` frequencies. Shared
`instrument`, `sphere` and `primitive` rows have world `-1`; root instrument
owners are the absent-owner sentinel `-1`. Sphere/primitive owner columns refer
to that instance's shared instrument domain. `runtime` and `sphere_runtime`
rows carry their actual world ids and refer to their shared entity rows.
No global instrument frequency is introduced. Primitive geometry has no mutable
per-world table. Kernel indices remain local integers without Newton remapping.
Empty custom domains retain one inaccessible backing row, separately accounted.

Case authoring uses `InstrumentControl.declare_storage`. Direct Scene callers
use `declare_instruments(scene, system_id=..., ...)` before schema registration
or allocation. A generic deferred hook keeps physics independent of surgical
imports. All direct constructors, including the composition example, were updated.
Missing/late declarations and changed layouts/settings fail explicitly.

`Declarations.initializer` registers package callbacks in declaration order.
A trailing ONCE Model marker, after all declared storage fields, invokes them
through the existing Newton capability guard. The surgical callback resolves
only the Model being finalized, including its State/Control templates. It uses
the existing reset-then-place kernel launches through shared helpers. Temporary
unmounted flags, default mount poses and mask are not retained by that callback;
its scene reference is weak. Initialization synchronizes before finalize returns.

Binding and repeated binding only validate and alias storage. Preparation does
not reset commands, trigger history, jaw relaxation or sphere history. Explicit
masked episode reset still uses the current Model defaults and existing kernels.
Two Models from one retained builder have independent backing and templates;
finalizing the second cannot modify the first or its selected live containers.

## Commands and lifetime

`System.command_view_spec` is unsupported by default. A package supplies named
half-open final-axis slices of its declared Newton Control `actions` field.
`PhysicsBinding.command_views(system_id)` returns cached Warp views;
`I4hComponent.command_views(system_id)` returns cached Torch views.

| View | Columns | Shape |
| --- | --- | --- |
| `actions` | `0:10` | `(E,N,10)` |
| `position` | `0:3` | `(E,N,3)` |
| `quaternion` (xyzw) | `3:7` | `(E,N,4)` |
| `grip` | `7:8` | `(E,N,1)` |
| `energy` | `8:9` | `(E,N,1)` |
| `fire` | `9:10` | `(E,N,1)` |

Mappings are read-only; values are writable zero-copy aliases. For nonempty
instruments the Warp byte strides are `(40*N,40,4)` and Torch element strides
are `(10*N,10,1)`. A slice starting at `c` has pointer `actions.ptr + 4*c` and
retains its backing reference. The full `actions` view is the packed array.

Generic code validates specifications, Control ownership, dtype, device, shape,
strides and backing identity, with no concrete control-class branch. Unknown ids
raise `KeyError`, unsupported systems raise `NotImplementedError`, invalid
specifications raise `ValueError`. Both Newton-owned arrays and system/nested
aliases are checked before execution and preparation. Replacement requires a
fresh binding. Stop, close and hard rebuild invalidate access through the old
binding; callers must reacquire views. Existing Torch/Warp stream boundaries,
packed action APIs, bounds, ordering and recording formats are preserved.

Packed commands plus named views remain the final design. No typed-command or
recording-format-v2 migration is introduced.

## Development verification

These checks precede the freeze and are not the frozen campaign certificate.

| Check | Result |
| --- | --- |
| Initial new ownership suite | 25 passed |
| Existing instrument suite with assets | 23 passed |
| Core focused integration after import fix | 123 passed, 1 skipped |
| Main focused ownership/host lifecycle/adapter selection | 80 passed |
| Patched Lab 3.0 focused selection | 77 passed, 3 skipped |
| Package development suite | 349 passed |
| Final ownership suite, each of core/main/Lab 3.0 | 31 passed in each environment |
| Ruff and reviewed inventory | passed; 569 sites in 184 source files |

The initial core focused selection failed collection because a newly inserted
local import was incorrectly indented; that source error was fixed before the
123-pass run. An initial instrument run without the data environment skipped
23 asset tests; the subsequent asset-enabled run passed all 23. Logs/XML are
retained under [C3.1 evidence][evidence]. No clipping observation was rerun in
these development selections.

The ownership suite covers CPU eager, CUDA eager and CUDA capture; multiple ids
and worlds; empty domains/padding; foreign host particles/bodies; normalized
initial poses, jaw geometry and sphere history; exact pointer offsets/strides;
packed/Warp/Torch writes; an external control class; partial/empty/repeated resets;
trigger edges and jaw relaxation; nondefault byte-preserving binding/preparation;
independent Models and live Controls; two runtimes deliberately sharing one
ordinary live Control; explicit `clone_variables=False` and later-clone effects;
foreign containers and replaced nested arrays/layouts. Host tests check command
view invalidation on hard rebuild, close and detach. Integrated contact, grasp,
hook, cutting, mounted-tool, lifecycle and recording coverage remains in place.

Uninterrupted-execution equality is asserted only for the instrument-control
fixture whose required history is fully migrated. No fresh-runtime equivalence
claim is made for integrated systems with unmigrated state.

Clipping retains its original assertions, reach and timing. Future failed runs
export current instrument/command/trigger, clipping and centreline arrays,
particle positions and host centreline tables to `clipping-failure.npz` in the
pytest artifact directory. This does not reconstruct or retry the C3.0 failure.

## Frozen campaign

The [manifest][manifest] covers **89,025 files**, SHA256
`69e8c2ab0adf8c87d519ace61c1637b73e7326002fa17de5e664b8047378880d`:
source, tests, harnesses, hosts, exact dependencies, native binaries, interpreter,
inputs and all approved references. Each worker and nested Python process must
have matching startup/exit guards and imported-source records.

The [audited frozen certificate][certificate] records **83 matching process pairs**
and **103,380 matching file-backed import observations**. Its separately guarded
auditor independently recomputed all 48 results from retained arrays. It checked
reference hashes, zero floors, factor three, capacity 256, all 18 fresh-process
chole workers, and exactly two construction samples. No reference replay was retried and
no reference was promoted.

| Frozen suite | Passed | Skipped | Failed | Errors | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| core-suite | 984 | 47 | 4 | 0 | 436.639 |
| main-suite | 113 | 5 | 4 | 0 | 132.535 |
| lab30-suite | 110 | 8 | 4 | 0 | 121.788 |

Each suite's four failures are `test_mounted_probe_follows_its_body_in_the_component_frame`,
`test_two_way_reaction_is_the_tissue_force_on_the_holder`,
`test_two_way_reaction_moves_the_holder_and_reset_clears_it`, and
`test_mount_configuration_errors` in `test_coupling.py`. The `held_probe` fixture
constructs controls indirectly through `build_systems`; its missing pre-allocation
`declare_instruments(scene)` call was missed during the initial direct-constructor
migration. Every failure raises the explicit missing-storage error before the
mounted mechanics assertions execute. The logs/XML retain those failed observations.
The unchanged `test_tracker_and_clip` passed in this campaign; this does not
reclassify or explain the retained C3.0 clipping failure.

| Host / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-captured | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |
| standalone-eager | PASS | PASS (3/3) | PASS | **FAIL** | PASS | PASS |
| main-captured | PASS | PASS (2/3) | PASS | PASS | PASS | PASS |
| main-eager | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |
| lab30-captured | PASS | PASS (3/3) | PASS | **FAIL** | PASS | PASS |
| lab30-eager | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |

Totals: **34/36 groups, 45/48 replays, 17/18 chole trials**. All six chole groups
pass the unchanged 2-of-3 policy. The failed main-captured trial remains retained
with 19 failed product comparisons. The two failed `aorta_shell` groups each fail
one stepped catheter-impulse comparison. Their masked-reset products pass in this
campaign. Matching product names do not establish a common cause with earlier
C2/C3.0 observations; C3.0 progression acceptance does not waive these C3.1 failures.

The failed individual replay artifacts are:

- [standalone-eager/aorta_shell (standalone-eager) report](../.tmp_tests/stage-c31-20261109/certificate/standalone-eager/aorta_shell.json), [endpoint arrays](../.tmp_tests/stage-c31-20261109/certificate/standalone-eager/aorta_shell.npz).
- [main-captured/chole_tet_med (chole_tet_med.trial1) report](../.tmp_tests/stage-c31-20261109/certificate/main-captured/chole_tet_med.trial1/chole_tet_med.json), [endpoint arrays](../.tmp_tests/stage-c31-20261109/certificate/main-captured/chole_tet_med.trial1/chole_tet_med.npz).
- [lab30-captured/aorta_shell (lab30-captured) report](../.tmp_tests/stage-c31-20261109/certificate/lab30-captured/aorta_shell.json), [endpoint arrays](../.tmp_tests/stage-c31-20261109/certificate/lab30-captured/aorta_shell.npz).

| Failed replay / product | Difference | Approved noise | Threshold (3 × noise) |
| --- | ---: | ---: | ---: |
| `standalone-eager/aorta_shell/stepped/system/catheter_control/impulse` | 0.000107191153802 | 1.54213121277e-05 | 4.62639363832e-05 |
| `main-captured/chole_tet_med/masked_reset/system/grasp/impulse` | 0.000268598552793 | 3.27799934894e-05 | 9.83399804682e-05 |
| `main-captured/chole_tet_med/masked_reset/system/grasp/jaw_impulse` | 0.000268598552793 | 3.27799934894e-05 | 9.83399804682e-05 |
| `main-captured/chole_tet_med/masked_reset/system/grasp/moment` | 4.05373575632e-05 | 4.75205888506e-06 | 1.42561766552e-05 |
| `main-captured/chole_tet_med/masked_reset/system/instrument_contact/impulse` | 0.000831579789519 | 0.000134021043777 | 0.000402063131332 |
| `main-captured/chole_tet_med/masked_reset/system/instrument_contact/jaw_impulse` | 0.00087477453053 | 7.57351517677e-05 | 0.000227205455303 |
| `main-captured/chole_tet_med/masked_reset/system/instrument_contact/moment` | 8.92861862667e-05 | 1.29390973598e-05 | 3.88172920793e-05 |
| `main-captured/chole_tet_med/masked_reset/system/tool_force/force` | 16.9577026367 | 2.46160888672 | 7.38482666016 |
| `main-captured/chole_tet_med/masked_reset/system/tool_force/jaw_load` | 4.82373046875 | 0.249389648438 | 0.748168945312 |
| `main-captured/chole_tet_med/masked_reset/system/volumetric_grasp/impulse` | 0.00037707388401 | 5.09545207024e-05 | 0.000152863562107 |
| `main-captured/chole_tet_med/masked_reset/system/volumetric_grasp/jaw_impulse` | 0.00037706643343 | 5.0961971283e-05 | 0.000152885913849 |
| `main-captured/chole_tet_med/masked_reset/system/volumetric_grasp/moment` | 3.32379713655e-05 | 7.41332769394e-06 | 2.22399830818e-05 |
| `main-captured/chole_tet_med/stepped/system/grasp/impulse` | 0.000268598552793 | 6.91708410159e-05 | 0.000207512523048 |
| `main-captured/chole_tet_med/stepped/system/grasp/jaw_impulse` | 0.000268598552793 | 6.91708410159e-05 | 0.000207512523048 |
| `main-captured/chole_tet_med/stepped/system/grasp/moment` | 4.05373575632e-05 | 1.00482138805e-05 | 3.01446416415e-05 |
| `main-captured/chole_tet_med/stepped/system/instrument_contact/impulse` | 0.000831579789519 | 0.000215418636799 | 0.000646255910397 |
| `main-captured/chole_tet_med/stepped/system/instrument_contact/jaw_impulse` | 0.00087477453053 | 0.000186955556273 | 0.00056086666882 |
| `main-captured/chole_tet_med/stepped/system/instrument_contact/moment` | 8.92861862667e-05 | 2.41808593273e-05 | 7.25425779819e-05 |
| `main-captured/chole_tet_med/stepped/system/tool_force/force` | 16.9577026367 | 2.46160888672 | 7.38482666016 |
| `main-captured/chole_tet_med/stepped/system/tool_force/jaw_load` | 4.82373046875 | 0.863037109375 | 2.58911132812 |
| `lab30-captured/aorta_shell/stepped/system/catheter_control/impulse` | 9.96462667899e-05 | 1.54213121277e-05 | 4.62639363832e-05 |

### Fixture correction after the campaign

After all frozen guards exited and the independent audit completed, the exact
[frozen fixture source][frozen-fixture] was preserved. The only subsequent change
to an existing frozen source file adds the surgical import and pre-allocation
`declare_instruments(scene)` call in `held_probe`. **Production source is identical
to the frozen campaign.** The frozen manifest and certificate were not rewritten.

[The current review certificate][review-certificate] and [current source manifest][review-manifest]
record the corrected fixture separately: **4/4 affected tests passed in each of
core, main and patched Lab 3.0**, with eleven matching startup/exit pairs covering
the check driver, three test workers, the synthetic exporter probe and six nested
Python processes. The probe ran no clipping physics: it verified no export on a passing fixture and all expected
command, trigger, clipping and centreline arrays on an artificial fixture failure.
[Its validation record][retention-validation] labels those arrays as synthetic.
No new reference replay or construction sample ran.

The current review manifest contains **89,030 files**, SHA256
`a12c665ced02bd82fc281ea133c5c6cd77538d2f8e7162c3e4e2d3ac6022d99e`.
It changes one existing test file and adds the preserved fixture plus follow-up
harness files; production, dependencies, references and numerical inputs match
the frozen campaign.

The single full campaign was not repeated. **The frozen suite failures remain
failures; these targeted checks do not constitute a clean full-suite certificate
for the corrected fixture source.** C3.1 remains open for the two numerical group
failures and a new explicit disposition on the retained suite gates. The earlier
fluid discrepancy remains deferred with unresolved cause.

## Construction accounting

The two samples completed with matching source guards. The chole workload has
`N=5`, `S=376`, `P=17`. Nonempty logical ownership per instance is:

- State: `92*E*N + 28*E*S` bytes per container, plus an equal initialization template.
- Control: `40*E*N` bytes per container, plus an equal initialization template.
- Model fields: `184*N + 104*S + 16*P + 12*E` bytes.
- Identity columns: `8*(N+S+P+E*N+E*S)` bytes, counted separately.
- Finalizer-only temporary arrays: `4*max(N,1) + 28*E*N + E` bytes.

| Instrument accounting, bytes | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model fields | 40,308 | 41,064 |
| Each live/checkpoint State | 10,988 | 703,232 |
| Each independent live/checkpoint Control | 200 | 12,800 |
| Initialization templates, State + Control | 11,188 | 716,032 |
| Identity columns | 6,232 | 198,256 |
| Empty field/template padding | 0 | 0 |
| Temporary finalization arrays | 161 | 9,044 |
| Migrated legacy live allocations removed | 51,496 | 757,096 |
| Net retained increase over accepted C3.0 | 17,420 | 914,288 |

Independent checkpoint containers add the same State/Control bytes as live
containers. Ordinary deliberate live-Control sharing adds zero bytes and leaves
templates unchanged. Explicit template aliasing overlaps 200 / 12,800 bytes
with the Control initialization template and intentionally affects later clones.
The raw historical key `intentionally_shared_control` still denotes this explicit
template alias; `ordinary_shared_live_control` reports the distinct live-sharing
case. Pointer records retain template/live/checkpoint/template-alias identities.

Aliases are deduplicated by maximum backing extent. In particular, native gravity
retains its global-world row; the shorter Scene gravity alias is not counted as
a second allocation. Empty-domain backing is tested separately; these chole
construction samples have no instrument padding.

All six residue arrays remain explicit: `InstrumentContact.delta_k`,
`InstrumentContact.delta_jaw`, `Grasp.delta_k`, `Grasp.delta_jaw`,
`VolumetricGrasp.delta_k`, `VolumetricGrasp.delta_jaw`. They total **2,020,320 bytes
at one world** and **129,300,480 bytes at 64 worlds**, with **zero new residue
initialization-template or checkpoint copies**. Each retains 336,720 /
21,550,080 bytes. Previous pool, Tet and Surface accounting remains unchanged.

| Complete construction measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model buffers, bytes | 6,172,433 | 93,446,585 |
| Initialization templates, bytes | 1,511,136 | 96,712,704 |
| Live / checkpoint State, each, bytes | 1,578,280 | 101,009,920 |
| Independent live / checkpoint Control, each, bytes | 200 | 12,800 |
| Retained Warp storage, bytes | 27,822,949 | 1,219,303,846 |
| Peak Warp storage, bytes | 28,029,193 | 1,232,509,132 |
| Additional checkpoint State + Control, bytes | 1,578,480 | 101,022,720 |
| Host peak RSS, KiB | 792,448 | 2,741,260 |
| Total construction time, seconds | 0.896779 | 15.344015 |

These are single construction observations, not performance qualification.
The raw samples and complete reconciliation against the accepted C3.0 samples
are embedded in the frozen certificate. Both retained and peak Warp storage
increased by exactly the new initialization templates plus identity columns.
The finalizer temporaries were released before the later construction peak.

[The inventory delta][inventory] preserves historical C1/C2/C3.0 contracts and
removed sites, records 23 removed helper/allocation sites and 5 added view or
temporary sites, and updates source hashes. The new storage fields are declared
explicitly, not inferred from this census.

**Mounted-tool episode restore remains partial until C3.3.** C3.1 remains open
pending a new explicit disposition on the failed gates and post-campaign fixture correction.
C3.2/C3.3 implementation, catheter migration and clipping mechanics are out of
scope. Stop for review before C3.2.

[evidence]: ../.tmp_tests/stage-c31-20261109/
[snapshot]: ../.tmp_tests/stage-c31-20261109/pre-c31-source.tar
[preservation]: ../.tmp_tests/stage-c31-20261109/preservation-audit.json
[core-id]: ../.tmp_tests/stage-c31-20261109/core-identity.json
[main-id]: ../.tmp_tests/stage-c31-20261109/main-identity.json
[lab30-id]: ../.tmp_tests/stage-c31-20261109/lab30-identity.json
[manifest]: ../.tmp_tests/stage-c31-20261109/certificate/source-manifest.json
[inventory]: ../.tmp_tests/stage-c31-20261109/inventory-delta.json

[certificate]: ../.tmp_tests/stage-c31-20261109/certificate/final-source-certificate.json
[frozen-fixture]: ../.tmp_tests/stage-c31-20261109/post-campaign/frozen-test_coupling.py
[review-certificate]: ../.tmp_tests/stage-c31-20261109/post-campaign/review-source-certificate.json
[review-manifest]: ../.tmp_tests/stage-c31-20261109/post-campaign/source-manifest.json
[retention-validation]: ../.tmp_tests/stage-c31-20261109/post-campaign/retention-validation/validation.json

[The C3.1-only source delta](../.tmp_tests/stage-c31-20261109/c31-source-delta.patch)
is based on the preserved starting snapshot. [The final preservation audit](../.tmp_tests/stage-c31-20261109/preservation-final.json)
includes the corrected fixture and verifies that every C3.0 manifest entry remains
available at its original path or in the pre-C3.1 archive.
