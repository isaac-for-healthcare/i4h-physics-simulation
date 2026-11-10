# C2.1 history initialization and native rest alias

This follow-up implements the user's instruction to fix the particle-history
boundary, remove the duplicate rest allocation, investigate the retained eager
failures and repeat construction and integrated verification. C2.2 has not started.
Evidence is retained in `.tmp_tests/stage-c21-closure-20261106/`; the preceding
certificate, all failed samples and the accepted C1 inventory remain unchanged.

## Implementation and ownership

`Scene.q0`, main/body/part rest views and `scene.attr(particle_domain, "q0")`
now alias native `Model.particle_q`. Declarations describe this native alias but
register no `i4h:particle_q0` attribute, author no duplicate column and allocate
no second rest array. Binding validates the native array's identity, shape,
dtype, device and layout. Replacement invalidates the binding. In-place edits to
native Model rest positions intentionally become the selected reset source;
the old test's independently preserved custom rest copy no longer exists.
New State history still comes from the authored `q_prev` template. Editing native
rest does not rewrite that template or existing history: use explicit reset or
post-write initialization when changing an episode's initial conditions.

Exact inverse mass and i4h-semantic default flags remain Model custom attributes;
live flags and history remain State custom attributes. Native State q/qd and
Model radius retain their aliases. Empty native rest views have zero backing
bytes; custom empty native attributes keep their required inaccessible padding.
Pools and all six residue arrays remain on their existing owners. No topology,
material, command, coupling, dependency, capacity or reference policy changes.

`Scene.initialize_particle_history(mask)` is a selected-world device operation
that copies live q into q_prev for owned main rows and private pool bodies only.
It does not reset velocities, flags, topology, materials or commands. It excludes
foreign native prefixes/strides and unselected worlds. Binding and
`I4hSolver.reset(flags=0)` continue to preserve independent/restored history.
`PhysicsBinding.initialize_particle_history(ids, mask=...)` exposes explicit
selection before preparation as well as during runtime; supply ids or a mask.
`I4hComponent` adds Torch-to-Warp ordering for host writes.

| Position-writing path | Completion contract |
| --- | --- |
| Fresh native State creation | Native q and authored history initially match the authored rest rows. |
| Explicit physical episode reset | Existing selected reset restores q, then q_prev=q. |
| Direct I4hEnv and manager-based I4hHostMixin startup events | After the base constructor applies startup events, order Torch writes and initialize history for all owned rows before returning the fresh environment. Tests inspect positions/history before any reset or step. |
| Direct I4hEnv and manager-based I4hHostMixin reset events | Physical reset, Warp-to-Torch ordering, events, Torch-to-Warp ordering, selected history initialization, then procedure consumers. |
| Fresh component construction/rebuild callbacks | Order construction before callback reads; after callbacks complete current-stream position edits, order and initialize owned history before prepare/capture. Callbacks using another stream explicitly complete edits on that stream before returning. |
| Public position views or raw native State edits before/after binding | Caller explicitly completes a teleport with the new history method before stepping. Raw writes cannot be detected; restoring a State is not a teleport. |
| Native Lab DeformableObject/MPM asset writers | Their imported asset rows are foreign to the manually authored i4h component. Geometry dirty alone is not an i4h history update. A caller targeting i4h-owned rows must use the explicit completion method. |
| Cache-only host reset/FK and render proxy buffers | Cache-only reset preserves history; FK affects body/joint fields. Render proxy positions are separate buffers. |

Restoring independent history belongs after fresh construction callbacks. No
unconditional history clone has been reintroduced into binding, preparation or
generic geometry invalidation. This is an explicit supported write protocol,
not a claim to intercept arbitrary external writes.

The source audit against the accepted C2.1 snapshot finds exactly one new Warp
kernel, `initialize_particle_history`; all existing kernels/functions and ordered
system phase/reset definitions are unchanged. Current inventory: 624 sites in
179 source files, including one new zero-capacity native alias with zero backing
bytes. The custom rest allocation removal is separately recorded because its
backing was allocated inside Newton rather than at a Python allocation site.

## Focused verification

CPU/CUDA ownership, exact rest/inverse-mass bytes, independent States, native
buffer replacement, foreign prefix/stride preservation, partial/empty/repeated
history initialization and reset, variant rows, inactive padding, empty bodies,
unequal private pool sizes and rotated host-frame tests pass: **51 tests**.
Reset-event and component position-edit tests pass on both hosts, including
side-stream writes and preserving unselected history. A subsequent main-host
component/memory selection passes **27 tests**. Final integrated suites below
also include the last callback-ordering change.

Development failures are retained: the first core run exposed slicing an empty
native alias and the intentionally changed native-default test expectation
(3 failed / 42 passed); both were corrected. The first new Lab history assertion
compared all main particles against one observed body's positions (1 failed /
1 passed); its scope was corrected, and the unchanged position comparison plus
new per-body history assertions passed. Inventory refresh also retained a
Python Counter setup error before its corrected verification. None of these
runs substitutes for the frozen-source campaign.

## Retained eager failure investigation

`retained-analysis.json` compares all six already-retained aorta-shell samples
from the previous certificate against the approved reference, without running
new numerical samples or modifying any comparator. Every initial product is
byte-identical in all six groups. The two failed groups have only the catheter
impulse beyond tolerance; other numerical products also differ within their
recorded envelopes. This does not establish a harmless or pre-existing failure.

| Previous failed group | Reference x impulse | Actual x impulse | Absolute difference | Unchanged tolerance |
| --- | ---: | ---: | ---: | ---: |
| Standalone eager | -8.915209036786109e-5 | -2.122869591403287e-5 | 6.792339445382822e-5 | 4.6263936383184046e-5 |
| Main eager | -8.915209036786109e-5 | -1.82274125108961e-5 | 7.092467785696499e-5 | 4.6263936383184046e-5 |

Both are world 1, tool 0, component x. Masked reset selects world 0 and retains
world 1's completed impulse, so its repeated failed comparison is not a second
independent sample. The recorded products omit `RodDirect.lambda_sum` and
intermediate substep contributions. Source tracing identifies `root_hold` as
the consumer that subtracts those root multipliers into impulse after each
substep; `track_guidance` also accumulates positional corrections. Shell and
containment use floating-point atomic reductions that feed the rod's subsequent
solve. These are paths to investigate, not proof of the saved samples' cause.
The retained endpoints do not justify changing a physics kernel or a tolerance.
No attribution distribution campaign is run: the user's earlier "ignore it and
continue" instruction remains in force. Earlier failures are not waived by a
later passing invocation.

## Fixed diagnostic traces on the final runtime

After the integrated campaign, the predeclared `trace-protocol.json` runs exactly
one standalone captured trace and one eager trace, each following the unchanged
60-drive/360-settle scenario, plus its warm-up/reset, with all exports enabled.
Readbacks after each completed step retain particle state, rod frames/multipliers,
controls and containment products. All 421 step records per process are retained;
source guards match the final runtime. No pre-change distribution sampling occurs.

Initial checkpoint products and warm-up records match exactly between these two
traces. Control targets match throughout. The first completed-step differences are:

| Product | Scenario step | Maximum absolute difference at that step |
| --- | ---: | ---: |
| Catheter impulse | 56 | 1.0231815394945443e-12 |
| Catheter moment | 56 | 8.185452315956354e-12 |
| Shell q and q_prev | 70 | 1.1920928955078125e-7 |
| Shell qd | 70 | 5.7220458984375e-5 |
| Catheter q | 120 | 9.110663086175919e-7 |
| Rod lambda_sum | 120 | 1.1698527941916836e-6 |

The shell's completed-step trajectory diverges before the rod's recorded
positions/frames/multipliers. The earlier impulse difference is very small and
precedes any difference in the recorded physical state. This locates useful
boundaries for a future substep trace; it does not identify a specific defective
kernel or explain the historical failed invocation. A completed-step lambda_sum
is only the last substep's value, so it cannot reconstruct all eight root_hold
contributions. Readbacks also synchronize execution and can change reduction
schedules. Both diagnostic endpoint comparisons pass the unchanged reference;
these passing diagnostics do not replace or waive the failed qualification runs.
Raw arrays, source receipts and analysis are under `trace-*` and
`trace-analysis.json`.

## Frozen-source certificate

The final single source map is
`certificate-reviewed/source-manifest.json`, SHA256:

`9e36c4282fea8187b50f3868df9d65df55a49b803d10432eafcf4027d7279dde`

The [source manifest](../.tmp_tests/stage-c21-closure-20261106/certificate-reviewed/source-manifest.json)
and [machine-readable certificate](../.tmp_tests/stage-c21-closure-20261106/certificate-reviewed/final-source-certificate.json)
record the scopes and every retained result. Final runtime hashes:
`scene.py = 62b705494cd0a307be1d6234ae7114ef8d1cfe24ddbf376f86cdb2f702837c74`;
`newton_model.py = 6a6cbc8cf27d292ce64b33c907e4c90063cd5d0bc8b985ed02ae4de220531801`.

It covers 9,368 files. The final suites are complete:

| Suite | Result |
| --- | --- |
| Full core packages/runtime | **815 passed, 47 skipped** |
| Lab 3.0 focused host suite | **109 passed, 8 skipped** |
| Main focused host suite | **112 passed, 5 skipped** |

The final certificate verifies 48 reference-worker source maps, 62 inherited
reference/construction process receipts and 20 corrected-suite process receipts.
All 504 file-backed package-root observations resolve to covered files, including
installed-package symlinks. One core-suite `isaaclab` module observation has no
`__file__` and is retained separately, not represented as a verified file path.
The original and final test harness hashes are explicit; there is no mixed
production revision.

| Host | Captured six-case group | Eager six-case group | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | PASS | PASS under existing chole policy | 3/3 / **2/3** |
| Lab 3.0 | PASS | **FAIL: aorta_static impulse** | 3/3 / 3/3 |
| Main | **FAIL: aorta_shell impulse** | PASS | 3/3 / 3/3 |

Chole uses the already-approved **majority-of-three** rule (two complete passing
replays required, exactly three retained fresh processes). The final standalone
eager trial 2 failed 32 product comparisons, including grasp/contact products,
slot geometry and tool-force output; its complete arrays and failure list remain
in `certificate-complete/standalone-eager/chole_tet_med.trial2/`. The group is an
accepted 2/3 result, not 3/3. The earlier commentary claiming all 18 final chole
trials passed was incorrect and is superseded by this trial-level audit: **17/18**
passed. No failed trial was removed or retried.

The two stopped campaigns also retain failed chole trials despite passing their
group policy: `certificate/lab30-eager` is 2/3 and
`certificate-final/standalone-eager` is 2/3; their other completed chole groups
are 3/3. `all-chole-trial-dispositions.json` records every completed group and
links each failed trial. These are separate source revisions, not a pooled
failure-rate estimate.

| Failed group comparison | Location | Absolute difference | Unchanged tolerance |
| --- | --- | ---: | ---: |
| Lab eager aorta_static, masked-reset impulse | World 1, tool 0, y | 2.444721758365631e-9 | 1.7462298274040222e-9 |
| Main captured aorta_shell, stepped and masked-reset impulse | World 1, tool 0, x | 8.587960292061325e-5 | 4.6263936383184046e-5 |

For the static-aorta sample, the same impulse difference passes the stepped
threshold of 4.889443516731262e-9. Reset preserves the unselected world's impulse
exactly; its masked-reset reference has a tighter recorded envelope. The other
59 products match exactly; the four differing impulse/moment endpoint products are
listed in `static-failure-analysis.json`. The shell sample also retains the
unselected world's completed impulse across reset, and all its initial products
match the reference exactly. These are observations, not a tolerance waiver or
proof of the cause of the divergent trajectory.

**C2.1 remains open.** The ownership/history corrections and all suites pass, but
two numerical reference groups fail. No failure waiver or full continuation claim
is made, and C2.2 has not started.

The numerical replays and
construction measurements are retained from `certificate-complete/`; its
runtime files match the final manifest exactly. `inherited-runtime-proof.json`
checks the guarded reference/construction records against the 9,278-file runtime
scope, and each of the 48 reference workers' explicit package source maps is
checked against the final manifest. Reference/construction entry points do not
execute pytest test files.

After that complete campaign, two obsolete assertions in `test_solver.py` and
`test_lab_adapters.py` were corrected from requiring a distinct rest buffer to
requiring `scene.q0 is model.particle_q`. The final manifest differs from the
replay manifest only in those two test files and the additional suite-audit
harness files. Production, reference/comparison scripts, scenarios, dependencies
and native binaries are unchanged. The original suite results remain retained:
core **814 passed / 47 skipped / 1 failed**, Lab **107 passed / 8 skipped /
2 failed**, main **110 passed / 5 skipped / 2 failed**, all failures being those
obsolete alias assertions. Full core and both host suites run again on corrected
tests with startup/exit source guards. Numerical replays are not repeated for a
more favorable outcome. This distinction is explicit: all final jobs match one
runtime source map; the original and corrected suite harnesses have different
recorded hashes.

The first campaign (`certificate/`) was stopped after a final path audit found
the manager-based `I4hHostMixin._reset_idx` also needed post-event history
initialization. The current worker finished before any source edit; all 52
started Python processes have matching source receipts. Construction and five
reference groups completed, all passing; main eager and the three suites did
not start. `campaign-stopped.json` records the boundary and `intermediate-source/`
preserves its workspace files against its manifest. This partial campaign is
not a final-source certificate. The manager-corrected intermediate campaign is `certificate-final/`.
The manager path now orders and copies selected history before procedure
consumers, tested on both hosts with initial and side-stream partial resets.
That second campaign was stopped after auditing startup events in both native
base constructors. Its 32 Python workers finished with matching source receipts;
construction and three reference groups completed, all passing, before any edit.
Its workspace is retained in `manager-corrected-source/`. The complete source
adds startup completion in both environment classes and has five focused history
boundary tests passing on each host. The complete runtime-source campaign is `certificate-complete/`;
each construction sample is run once again on that source. Every intermediate
measurement and stopped-campaign receipt is retained. The final construction samples and qualification jobs use
one final manifest with startup/exit verification in every Python worker. The manifest
also covers the twelve approved reference files, both host source trees and the
three qualified Newton/Warp installations, including patched native binaries.
No source edits occur during the campaign. Six cases run in captured and eager
mode on each host, with exactly three fresh retained chole replays per group;
then the full core suite and both focused host suites run once. No acceptance
retry, tolerance change or reference replacement is permitted.

## Final-source construction sample

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

## Manager-corrected intermediate construction sample

One fresh process per world count, once each on the manager-corrected intermediate
source. Every memory subtotal matches the intermediate sample; timing values
are independent retained observations, not distributions or a performance verdict.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.481218 | 5.451909 |
| Schema registration (s) | 0.003786 | 0.020119 |
| Native world authoring (s) | 0.008463 | 0.714193 |
| Custom row authoring (s) | 0.255249 | 3.781756 |
| Finalization (s) | 0.047329 | 0.540966 |
| Binding and runtime construction (s) | 0.064518 | 0.285480 |
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

Both manager-corrected samples record Solvecraft PID 65109 (1,071 MiB) and simulation
processes 993313 (793 MiB) and 993341 (656 MiB) at entry. Full process listings, peak RSS, source/binary hashes and per-array bytes are retained
in their JSON reports. Native counts remain 5,612 / 359,168 particles, zero native
tets and triangles. State templates, private pools and residue ownership are
unchanged; additional checkpoint containers do not imply full save/restore.

## Intermediate source construction sample (before manager-path correction)

Each of the following samples is a separate fresh process on the intermediate source,
run once before the numerical suites. The source guard runs outside the measured
construction interval. Times include actual construction, including any first-use
kernel preparation, and are not distributions or exclusive steady-step results.
The 1-world log records 485.56 ms compiling a Scene module; the 64-world process
loads that module from cache. Both samples retain that real cache condition.
Both record three other GPU processes (Solvecraft and two simulation workloads);
none were stopped or excluded.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.979465 | 5.597095 |
| Schema registration (s) | 0.003872 | 0.021195 |
| Native world authoring (s) | 0.008552 | 0.724892 |
| Custom row authoring (s) | 0.253934 | 3.877584 |
| Finalization (s) | 0.052439 | 0.553490 |
| Binding and runtime construction (s) | 0.558732 | 0.313124 |
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

The removed custom rest buffer accounts for **67,344 bytes at 1 world** and
**4,310,016 bytes at 64 worlds**; both measured retained totals fall by precisely
those amounts relative to the preceding samples. Other measured container totals
are unchanged. This is an allocation accounting result, not a timing claim.
Native Model particle_q remains counted once in initialization templates because
Newton also uses it to initialize State q; the q0 alias adds no bytes.

Compute processes recorded at entry (raw nvidia-smi output):

1 world(s):

```text
65109, /tmp/.mount_solvecCKlfCk/usr/bin/solvecraft, 1071 MiB
983357, /home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-internal-worktrees/i4h-sim-rod-physical/physics_simulation/i4h-sim/.venv/bin/python, 938 MiB
983384, /home/pkorzeniowsk/Projects/i4h/i4h-validation/catheter/.venv-newton18/bin/python, 652 MiB
```

64 world(s):

```text
65109, /tmp/.mount_solvecCKlfCk/usr/bin/solvecraft, 1071 MiB
983357, /home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-internal-worktrees/i4h-sim-rod-physical/physics_simulation/i4h-sim/.venv/bin/python, 938 MiB
983384, /home/pkorzeniowsk/Projects/i4h/i4h-validation/catheter/.venv-newton18/bin/python, 652 MiB
```

## Carried historical construction and memory evidence

The following measured table and accounting paragraphs are carried forward
verbatim from the previous C2.1 record. They are historical samples, not current
measurements. The six residue arrays remain legacy State, totaling 129,300,480
bytes at 64 worlds in both sets. Once they migrate, their actual Newton templates
and each additional State/checkpoint allocation must be reported separately.

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


