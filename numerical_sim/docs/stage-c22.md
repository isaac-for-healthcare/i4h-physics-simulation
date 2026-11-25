# C2.2 — Newton-owned private pools and Scene metadata

**Accepted for progression to C2.3 by the 2026-11-06 implementation request.**
The certificate, all failed observations, construction samples and accounting
corrections below remain historical and unchanged. Acceptance supersedes the
historical open-checkpoint wording; it gives C2.3 no failure waiver.

The user accepts C2.1 and authorizes this bounded migration. Its failed samples,
[certificate and measured tables](stage-c21-closure.md), earlier follow-ups and
accepted C1 inventory remain unchanged. C2.1 acceptance does not waive C2.2 failures.
C2.3 has not started. No dependency pin, patched Warp binary, contact capacity
256, coupling limitation, reference, tolerance or comparator is changed.

`scene_storage.py` supplies the declaration/completion helpers for CaseAuthoring
and direct Scene construction. The eight private pool fields retain the exact
flat body-major order and author together on one custom row domain. Its world
column follows each body's worlds, including non-fluid gaps; native-particle
owners are -1. No native particles or topology are added. Model owns q0,
exact inverse mass, radius and default flags (24 bytes per logical particle).
Each State owns positions, velocity, independent history and live flags (40 bytes
per logical particle), and Newton retains another 40 bytes per row in initialization
templates. Identity columns cost 8 bytes/row; zero-capacity custom domains retain
one inaccessible backing row for every field and identity column. Removing the
legacy allocations does not imply lower retained memory.

Each body has one WORLD-frequency Model variant table, identified by declaration
order independently of system ids. Every system uses the same Body.variant alias.
Static meshes declare Model points and flattened mesh-local indices once per
mesh, without Newton reference offsets. `static_wp_mesh` creates/caches only
the acceleration structure over those aliases; independent runtimes share Model
geometry while rebuilding their own acceleration structure.

Component gravity is registered before world creation. A Scene-specific finalizer
uses completed transforms and Newton's finalized builder gravity: unrotated scenes
alias native gravity without allocating a duplicate; rotated scenes allocate the
derived values once on Model. Explicit `scene.refresh_gravity(model)` refreshes a
finalized Model before binding; `scene.refresh_gravity()` and model-property
notifications refresh it in place after binding. Host edits require that explicit
boundary. Binding preserves derived values without refreshing them.

Direct callers use `newton_model.complete_scene(builder, scene)` after all worlds
and before finalize. CaseAuthoring.complete uses the same helper. Completion is
idempotent; retained builders can finalize again. Changed capacities, mappings,
geometry layouts or transforms require reconstruction, and incomplete world
transforms fail. Bind installs checked aliases before systems, preserves all
restored pool/history bytes, and rejects incompatible buffers. Stale storage
bindings fail through the existing solver guards and direct Scene reset/history
operations. Cached body/part, Fluid span/body and Scene.attr views remain aliases.
Selected-world pool reset remains full even when main q/qd reset is disabled.

Fluid cursors, overflow, live settings, constant tables, Tet/Surface, commands,
rods, mounts, thermal history and the six residue arrays retain their existing
owners. The residue remains 2,020,320 bytes at one world and 129,300,480 bytes at
64 worlds, with zero new Newton templates in C2.2. Full fresh-runtime continuation
is not claimed. The declined attribution campaign is not run.

Development logs and all new samples are retained in
[the C2.2 evidence directory](../.tmp_tests/stage-c22-20261106/).
The first focused run retained two failures (20 passed): duplicate accounting of
the native gravity alias and an obsolete test permitting finalization before
completion. They were corrected. A mistyped focused test path produced a retained
no-tests invocation. The first new fixture omitted its required particles system
(4 failures, 3 passed); that fixture was corrected. No numerical reference is
rerun to replace a failed observation.

## C2.1 accepted failures, reported separately

C2.1's final-source record retained Lab eager aorta_static impulse and main
captured aorta_shell impulse failures. Its standalone eager chole group was
2/3 (17/18 individual trials across the final campaign). Earlier campaigns and
all their failed observations remain in that record. The user's acceptance
applies to those historical outcomes; C2.2 has its own new results below and
receives no automatic failure waiver.

## Focused C2.2 verification

The qualified CPU/CUDA focused selection passed **77 tests**, including exact
pool ownership, independent States/Controls, main-particle history, direct Scene
construction, world transforms, fluid emission/motion/suction, static contact,
and eager/captured behavior. A final **30-test ownership selection** additionally
covers nonzero foreign prefixes/strides through pool stepping and reset, and
runs after the authoring-copy optimization that shares immutable declarations.
Package preflight passed **205 tests**. Ruff and the allocation inventory checker
pass. No existing Warp kernel or numerical system source changed; the source
audit verifies all four Scene kernels against the pre-change source and records
zero dependency/binary changes. New `scene_storage.py` contains host declaration
and binding-lifecycle code only.

## Frozen-source reference campaign

Final source manifest SHA256:
`7af0d8695092866256776f8794ae19787ba9ecc9fff95cd64f043ade1bb49a9f`.
Original full-campaign manifest SHA256:
`4cebeaf779996973feabc1cf125268bc6b901bd175948c493633adffc25ff485`.
The [manifest](../.tmp_tests/stage-c22-20261106/certificate-reviewed/source-manifest.json)
covers production, harnesses/tests, qualified host sources, installed Newton/Warp
and the patched native binaries. The original construction, reference and suite workers verify the original
map at startup and exit. The final core suite verifies the final map. Only
`tests/runtime/test_two_worlds.py` and the post-allocation subtotal helper in
`scripts/measure_stage_c_construction.py` change between those maps; every
production, numerical scenario, reference and focused-host test source matches
the final manifest. The
[provenance proof](../.tmp_tests/stage-c22-20261106/certificate-reviewed/inherited-runtime-proof.json)
records this boundary and retains both original harness files. No production
source change, numerical acceptance retry or additional construction sample
occurs. Tolerances, references and scenarios remain unchanged.

| Host | Captured six-case group | Eager six-case group | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | PASS under existing chole policy | PASS under existing chole policy | **2/3 / 2/3** |
| Lab 3.0 | PASS | PASS | 3/3 / 3/3 |
| Main | **FAIL: aorta_shell impulse** | **FAIL: aorta_static impulse** | 3/3 / 3/3 |

Exactly three fresh chole processes per host/mode are retained: **16/18**
individual trials passed. Standalone captured trial 0 failed 32 product
comparisons; standalone eager trial 0 failed 19. Their full arrays, failure lists
and process receipts are preserved in each group's `chole_tet_med.trial0/`
directory. Each group passes the approved majority-of-three rule; neither is
reported as 3/3, and no trial is removed or repeated. The other four chole groups
are 3/3. Across all six references, 34/36 case groups pass under the approved
policy; the two failed single-replay cases remain C2.2 checkpoint failures.

| C2.2 failed comparison | Location | Absolute difference | Unchanged tolerance |
| --- | --- | ---: | ---: |
| Main captured aorta_shell, stepped impulse | World 0, tool 0, x | 4.990682646166533e-5 | 4.6263936383184046e-5 |
| Main eager aorta_static, masked-reset impulse | World 1, tool 0, y | 1.862645149230957e-9 | 1.7462298274040222e-9 |

Both failed cases have byte-identical initial products, and each preserves its
unselected world's completed impulse through reset. Those retained endpoint
observations establish neither the cause nor harmlessness of the reference
failures. No attribution distribution or new diagnostic replay is run. The
read-only endpoint analyses and all failed samples are retained separately from
C2.1's accepted failures.

## Accounting harness corrections after the original campaign

The first full core suite retained **821 passed, 47 skipped, 2 failed**. Both
failures are the existing fresh-process memory probes: they filtered estimate
keys with `startswith("newton")`, excluding all migrated pool and Scene-metadata
fields even though Newton now allocates them. The test now compares its measured
Newton Scene peak and steady storage with the complete declared Scene budget.
No memory limit, estimate coefficient or assertion threshold is relaxed. The
corrected focused probes pass **2 tests**. Separately, the measurement helper
now takes the maximum extent for each shared pointer, fixing the 12-byte native
gravity subtotal undercount without a new sample.

These two harness corrections are made only after all original campaign workers
finish with matching source receipts. Production source is unchanged. The full
core suite runs again on the final manifest because it executes the changed
probe. Both focused host suites already pass (**109 passed / 8 skipped** Lab,
**112 passed / 5 skipped** main); neither executes either changed harness file,
so their original guarded results remain applicable through the explicit source
proof. All numerical replays and construction samples are retained once, with
their original harness hashes. The original failed core-suite log remains in
`certificate/core-suite.log`.

## Final certificate and checkpoint disposition

The [final certificate](../.tmp_tests/stage-c22-20261106/certificate-reviewed/final-source-certificate.json)
verifies one final runtime source map spanning **9,372 files**, the 48 retained
reference workers' explicit package source maps, **82** final/inherited Python
process startup/exit receipts and **504** file-backed import observations. The
non-file module observation is retained explicitly rather than represented as
a verified path. The twelve original core-suite process receipts remain separate,
for **94** retained process receipts across the original and corrected harnesses.
The accepted C1 inventory and all C2.1 certificates, failures and historical
tables are preserved.

| Suite | Final result |
| --- | --- |
| Full core packages/runtime | **823 passed, 47 skipped** |
| Lab 3.0 focused host suite | **109 passed, 8 skipped** |
| Main focused host suite | **112 passed, 5 skipped** |

**C2.2 is implemented; its checkpoint remains open.** All suites pass, but main
captured aorta_shell and main eager aorta_static fail the unchanged numerical
references. The standalone chole groups are each accepted **2/3** under the
existing majority policy, with their failed trials retained. C2.1's acceptance
does not waive these new failures. Stop here for checkpoint review; C2.3 has not
started. No full fresh-runtime continuation or performance qualification is
claimed, and no declined attribution campaign is run.

## New C2.2 construction samples

Exactly one fresh process per world count, under the frozen source and qualified
patched binary. These samples are never retried for timing. They measure
construction under contention, without a steady-step performance claim.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.668543 | 9.587114 |
| Schema registration (s) | 0.004977 | 0.042819 |
| Native world authoring (s) | 0.008574 | 0.749086 |
| Custom row authoring (s) | 0.403617 | 7.445778 |
| Finalization (s) | 0.066183 | 1.164136 |
| Binding and runtime construction (s) | 0.086903 | 0.074281 |
| Construction Warp peak (bytes) | 27,856,429 | 1,173,753,424 |
| Retained Warp storage (bytes) | 27,650,185 | 1,160,548,138 |
| Newton Model buffers (bytes) | 3,759,621 | 63,567,033 |
| Newton initialization templates (bytes) | 941,876 | 60,280,064 |
| Live Newton State (bytes) | 1,009,220 | 64,590,080 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 1,009,220 | 64,590,080 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

The Model-buffer row above includes the full native gravity allocation. The
frozen measurement helper's pointer-keyed dictionary keeps the shorter alias's
extent (E vec3 rows) instead of the native allocation's extent (E+1 rows),
undercounting that raw subtotal by 12 bytes in each JSON. The raw reports remain
unchanged: they retain both native extents (24/780 bytes) and alias extents
(12/768 bytes), proving the correction. The
[accounting correction](../.tmp_tests/stage-c22-20261106/construction-accounting-correction.json)
adds those 12 bytes without another sample. Initialization-template, State,
Control, peak and retained totals are unaffected. The original helper's frozen source
remains archived as evidence of the raw counter's limitation. The current helper
now counts the maximum extent among arrays sharing a pointer; it reports the
correct full native allocation in future samples.

Both entry listings record the concurrent vascular Python process PID 1078020
using 791 MiB. Peak host RSS, phase timings, full array bytes, and binary hashes
are retained in each JSON. The additional checkpoint State/Control containers
are measured after retained construction storage, separately from its peak.
Intentionally shared Control and shared-template overlap are both zero bytes for
this case, because its commands still use legacy owners.

| Pool accounting | 1 world | 64 worlds |
| --- | ---: | ---: |
| Logical particles | 12,288 | 786,432 |
| Model fields (bytes) | 294,912 | 18,874,368 |
| Each State (bytes) | 491,520 | 31,457,280 |
| Initialization templates (bytes) | 491,520 | 31,457,280 |
| World/owner identity columns (bytes) | 98,304 | 6,291,456 |

The static-geometry row identities add 552,248 Model bytes at either world count.
Relative to the historical C2.1 sample, retained bytes increase by 1,142,072 at
one world and 38,300,984 at 64 worlds: exactly the pool templates, pool identity
columns and static-geometry identity columns. The former live pool/variant/
geometry allocations are replaced, and their ownership changes; deleting those
sites does not remove the need for their data. Native particle counts remain
5,612 / 359,168, with zero native tets/triangles. The six residue arrays remain
legacy State, included once in retained storage and absent from the Newton
initialization templates and checkpoint containers.

## Historical C2.1 final-source construction sample (carried unchanged)

The following earlier sample is historical. All earlier intermediate tables
remain unchanged in [the C2.1 record](stage-c21-closure.md).

One fresh process at each world count on the complete startup/reset/rebuild
implementation, under the same final source manifest as all integrated gates.
Every sample is retained; these timings are not distributions or a performance
verdict. Buffer totals match both intermediate sources.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.476171 | 5.442159 |
| Schema registration (s) | 0.003752 | 0.021898 |
| Native world authoring (s) | 0.008576 | 0.718893 |
| Custom row authoring (s) | 0.254328 | 3.771190 |
| Finalization (s) | 0.046083 | 0.540506 |
| Binding and runtime construction (s) | 0.060326 | 0.279208 |
| Construction Warp peak (bytes) | 26,714,357 | 1,135,452,440 |
| Retained Warp storage (bytes) | 26,508,113 | 1,122,247,154 |
| Newton Model buffers (bytes) | 2,438,605 | 37,472,653 |
| Newton initialization templates (bytes) | 450,356 | 28,822,784 |
| Live Newton State (bytes) | 517,700 | 33,132,800 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 517,700 | 33,132,800 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

Both final samples record Solvecraft PID 65109 using 1,071 MiB at entry; no
other compute process appears in those two entry listings. They remain
construction measurements under contention, not an exclusive steady-step
campaign. JSON reports retain per-array counts, peak RSS, source and binary
hashes. Residue templates/checkpoint copies remain zero because those six
legacy fields have not migrated; their 129,300,480 bytes are included in
retained Warp storage. Model particle_q is counted once among initialization
templates because Newton uses it to initialize State positions, and the new
q0 alias adds no bytes. Native counts and the remaining duplicates are unchanged.

