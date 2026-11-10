# C3.0 — exact dependency pins and host compatibility

**C3.0 remains open.** The authorized frozen campaign completed all 36 reference groups,
48 individual replays and 18 fresh-process chole trials, plus one construction
sample at each of 1 and 64 worlds. **35/36 groups passed.**
Both construction samples match C2.5's ownership and allocation accounting.
The historical fluid discrepancy remains deferred by the user, with unresolved
cause and failed observations retained. **Stop before C3.1 implementation.**
No commit, push, tolerance change, reference promotion or retry was made.

## Preservation and exact targets

Before edits, all **10,130** C2.5 manifest entries matched SHA256
`2932e8e9a7fc006d90a810bf5ed17aaa4bf320550205a7d6edc138263151daca`.
[Preflight evidence][preflight] records this check. The complete tracked and
untracked nonignored worktree source (**557 files**) was saved in
[pre-c3-source.tar][snapshot], with hashes, the worktree status and a binary diff.
The uninitialized `neural_sim/cosmos_h_dreams` submodule was not populated.
Existing environments, source checkouts, patches, references, certificates and
failed endpoint archives were left in place. The [final preservation audit][preservation]
resolves changed worktree entries against the archive and checks every other
C2.5 manifest entry against its original path.

[The preceding C3 plan and its revision](stage-c3-plan.md) are preserved verbatim,
as separate entries, from planning session
`01a1b4f9-301d-7af2-aef0-29de43b2f1f0`. The revision supersedes the earlier
C3.1 entry point and dependency conditions. The packed-command contract and
separately gated C3.1/C3.2/C3.3 boundaries remain binding.

| Component | Candidate target, verified from imports/source |
| --- | --- |
| Lab main | `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1` |
| Patched Lab 3.0 | `e53cd288727a5ca3dae7187d1d415b4d32b67e11` + unchanged [cloner patch](newton18-lab30-cloner.patch) |
| Newton, all three environments | `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`, `1.9.0.dev0` |
| Warp | Patched `1.17.0` wheel; hashes below |
| MuJoCo / MuJoCo-Warp | `3.17.0` / `3.17.0`, already present before C3.0 |
| Python | CPython `3.12.12`, preserved qualification interpreter |

Warp wheel SHA256:
`dff334c1589510f4374807eb3fbdc85cfa5e20ab8a0e05548dc54e0b64990546`.
Imported `warp.so` SHA256 in all three environments:
`eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
The [Warp patch provenance](warp117-hashgrid-capture.md) remains applicable.

Fresh `core-env`, `main-env` and `lab30-env` are under [C3.0 evidence][evidence].
Newton was installed from the exact git requirement; its installed Python
sources were compared byte-for-byte with an isolated checkout of that commit.
Lab checkouts were independently cloned and detached at the exact commits;
the preserved patch was applied only to the new Lab 3.0 checkout. Imported Lab
paths, revisions, package versions and patch state were checked. The preserved
checkouts and environments were not upgraded.

**C3.0 did not modify the supplied checkout at
`/home/pkorzeniowsk/Projects/newton/newton-1.8`.** C3.0's independent Newton clone
came from `.tmp_tests/newton18/newton` with `--no-hardlinks`; the environments
installed the exact Git revision into their own site-packages, as their import
records show. [The read-only checkout check][supplied-newton] finds a clean
worktree at `5a2a7deca118522c30012afa1016f44e849da018`, with the latest HEAD
reflog entry at `2026-11-09 12:07:51 +0100`, before C3.0 work began. This confirms
the action/provenance account; no pre-C3 whole-tree hash of that supplied checkout
was recorded, so it is not a historical byte-for-byte certificate.

Host-specific manifests keep the mutually exclusive Lab pins separate:
[core](../qualification/c30-core.json), [main](../qualification/c30-main.json),
[Lab 3.0](../qualification/c30-lab30.json). Each records exact Lab package git
requirements where applicable, resolved-environment hashes and dependency
changes. Actual package/import records are
[core][core-identity], [main][main-identity] and [Lab 3.0][lab30-identity].

The physics dependency and workspace `uv.lock` now pin Newton by full git
revision. The fresh environments resolve the previous third-party versions;
main additionally requires **`ovstorage==0.3.0`** for the pinned host imports.
Host package metadata dependencies are satisfied, including `isaaclab-contrib`.
The unused `isaaclab-ov` package is not installed. The source paths used by the
headless test harness are retained in its runner. This is the established
headless Newton selection, not qualification of the full Lab development
workspace, training extras or Isaac Sim rendering. `uv pip check` passes for
all three final environments. Standalone installs the i4h adapter distribution
for existing shared utilities, but imports no Isaac Lab package; its environment
has no `isaaclab` installation.

## Compatibility implementation

[The main adapter](../isaaclab/src/i4h_isaaclab/newton_solver.py) now implements
the pinned instance-method contract with `self.cfg` and `self.native`.
Pre-model hooks and `fixed_tendon_control` remain classmethods as required by
upstream. A coupled adapter retains its rigid child adapter instance and
delegates capabilities, contacts, pre/post hooks, masked reset, FK and diagnostics.
`CoupledSolver` still owns the original rigid-substep and coupling order.

The manager continues reading its upstream `_cfg`, checking prepared bindings
before stepping and ordering Torch views after the inherited manager call.
Solver-init callbacks, stop/rebuild invalidation and cleanup remain on the
existing host boundary. The test helper now inspects main's `backend.graphs`
cache instead of its removed `backend.graph` field.

Main can supply a separate `CONTACT_F` container for a contact sensor on the
last decimation substep. The adapter caches a non-owning combined output
container before capture: host rigid outputs and the component's i4h outputs
are both populated by the same step. When both consumers request contact forces,
the adapter copies the completed contact output into the component's existing
buffers. No physical ownership, kernel, solve order, action layout or recording
format changes. Tests cover real nonzero sensor forces, component-output
freshness, stable output pointers, eager/captured stepping and decimation three.
Adding a sensor after construction requires the existing explicit `prepare()`
structural-edit boundary. Pure i4h sensor registration remains explicitly rejected.

`lab30.py`, the host selector and the cloner patch are retained unchanged.
`QUALIFIED` names the target. The host-specific manifests retain their frozen
candidate labels; the campaign certificate and current status here record the
qualification outcome. No frozen manifest was rewritten after execution.

The only physics production change is the permitted
[capability guard](../physics/src/i4h_physics/newton_compat.py), called before
registering the existing Scene-gravity finalizer. It requires a callable bound
`_add_custom_attribute_model_finalizer(key, finalizer)` and names the exact
Newton commit and callback contract `finalizer(builder, model, custom_attribute)`.

[The finalizer probe](../tests/packages/test_newton_finalizer.py) verifies CPU
and CUDA ordering and template availability. The hook runs **in custom-attribute
registration order**, not automatically after all Model fields. An initializer
sentinel must therefore follow every prerequisite declaration. It verifies two
Models from one retained builder, independent templates, unchanged first-Model
live containers, valid later State/Control clones, ordinary live-Control sharing,
and the intentional template writes of `control(clone_variables=False)`.
No instrument migration was implemented.

The reference harness now accepts only an explicit `--newton-revision` equal to
the pinned C3.0 commit, verifies installed VCS provenance and the precise
`1.8.0rc1` → `1.9.0.dev0` transition, and retains both software identities.
Other software and hardware identity checks remain exact. This is preparation
for the C3.0 campaign; it does not alter the comparator, zero floors, factor
three, reference products, scenario, capacity 256 or 2-of-3 chole policy.
Nested chole workers receive the same revision argument. Tests reject drift in
commit, version, other software or GPU identity. The installed-task check also
understands and verifies git pins instead of assuming Newton uses `==version`.

## Development results and unresolved failure

These are **development checks**, without frozen-campaign startup/exit guards.
Separate environment identity probes do not constitute per-worker source receipts.
Complete logs, JUnit and process exit records are retained in [the evidence directory][evidence].

| Check | Result |
| --- | --- |
| Core finalizer + C2.5 Surface + HashGrid preflight | 21 passed |
| Lab 3.0 initial established host selection + those preflights | 130 passed, 8 skipped |
| Main expanded development run before helper/fixture fixes | 125 passed, 13 failed, 5 skipped |
| Lab 3.0 expanded development run | **134 passed, 1 failed, 8 skipped** |
| Final core package suite, after pin-check fixes | **323 passed** |
| Final main compatibility selection addressing helper/fixture failures | **24 passed** |
| Final Lab 3.0 adapter selection | **14 passed, 3 skipped** |
| Ruff / reviewed allocation inventory | Passed / 587 sites in 182 source files |

The three environments each passed the **six** C2.5 nonzero-capacity Surface
variants (PBD/XPBD × CPU eager/CUDA eager/CUDA captured), the **nine** retained
HashGrid capture regressions and the **six** finalizer tests. Existing host
coverage includes construction, single-state stepping, eager/captured execution,
decimation, partial reset, hard rebuild, callback cleanup, observables and
coupling. These checks are not a clean frozen full-suite certificate.

The unresolved C3.0 failure is
`test_isaaclab_env.py::test_user_stream_ordering[newton]` on Lab 3.0.
**This is a recurrence of C2.1 stream/parity symptoms, not a newly observed
failure class; a common cause remains unproven.** The retained history is:

- C2.1's [Lab 3.0 suite log][c21-lab30-suite] already fails the same stream
  observation assertion while its roots pass. That original test predates full
  observation retention; its truncated log does not establish which two products
  differed, and no full arrays from that particular suite failure were retained.
- C2.1 main's `test_standalone_matches_host[two_worlds-newton]` fails on exactly
  `fluid.alive_count/0` and `suction.removed/0`. The [retained full parity arrays][c21-main-arrays]
  and [suite log][c21-main-suite] show world 0 host/standalone values **256/255**
  and **0/1**, respectively—the same two products and values seen in C3.0.
- C2.1 also retains [full stream arrays from retention validation][c21-retention-arrays].
  This was an **injected harness failure**, not another numerical failure:
  [its validation record][c21-retention-record] says `actual_result=true` before
  forcing the assertion result to false. Rechecking those arrays gives no values
  outside the unchanged tolerance (maximum absolute difference
  `3.5762786865234375e-07`). They do not substantiate a claim that the same two
  products failed numerically during retention validation.

[The historical-evidence check][history-check] records the array hashes and exact
values without rerunning simulations. The recurrence does not identify a cause
or waive the C3.0 failure.
The test compares identical action sequences on default and side Torch streams:

| Failed endpoint, world 0 | Default stream | Side stream |
| --- | ---: | ---: |
| `fluid.alive_count/0` | 256 | 255 |
| `suction.removed/0` | 0 | 1 |

The root comparison passed with maximum absolute difference **0**. The observations have shape `(2, 414)`, with exactly
two entries outside the unchanged `atol=1e-4`, `rtol=1e-5` comparison; maximum
absolute difference is 1. Full arrays and labels are preserved in
[the failure archive][stream-arrays] and [exact value extraction][stream-values].
The cause is unresolved: these endpoints do not establish whether the source is
stream synchronization, numerical sensitivity, or Newton. Other GPU jobs were
present; no exclusive-performance conclusion is made. At the review checkpoint,
the failed stream test had not been retried; the later adapter-only checks did
not supersede it. The user subsequently deferred fluids and authorized item 2:
one complete host-selection run per host, including this test. Its new passing
result below does not explain or erase the retained failure.

Other retained development failures are: the initial core environment missing
the i4h adapter distribution required by standalone; main's missing `ovstorage`;
nine main assertions using removed graph introspection; four new sensor fixtures
missing explicit preparation after registering an observable; and two package
tests that still assumed the old version-only Newton pin and old Lab target.
Their compatibility fixes and subsequent focused checks are listed above.

**Second entry-gate gap, recorded at review:** no full established host selection
had a clean run after the fixes addressing thirteen main compatibility failures.
The 24-pass main and 14-pass/3-skip Lab 3.0 targeted checks were insufficient for
that claim. [Their commands][final-commands] remain unchanged.

**Item 2 follow-up — completed on 2026-11-09.** The user said "we can ignore fluids
for now" and then "ok run the item 2". This defers the fluid discrepancy for
progression and authorizes the host-suite verification; it does not mark the old
failure as passing or close C3.0. Each host ran the complete established selection
once: `test_lab_adapters.py`, `test_solver.py`, `test_component.py`,
`test_isaaclab_env.py`, `test_manager_env.py`, and `test_coupling.py`.
No test was excluded; fluid and stream tests remained included.

| Item 2 host | Full selection result | Duration | Process exit |
| --- | --- | ---: | ---: |
| Main | **117 passed, 5 skipped** | 102.34 s | 0 |
| Patched Lab 3.0 | **114 passed, 8 skipped** | 99.21 s | 0 |

Both `test_user_stream_ordering[newton]` invocations passed on these new runs.
**Item 2's missing full-selection verification is now closed.**
[The results, commands and summary][item2] retain fresh logs, JUnit, pytest
artifacts and imported-source receipts. All 88,850 review-manifest entries
matched before and after the two runs; every recorded file-backed simulation
import matched the same manifest. The harness source hashes are retained in
`invocation.json`. This is a bounded development follow-up, not the prescribed
frozen campaign: it has no startup/exit receipts for every nested process.
No reference replay or construction sample ran, and no runtime source changed.

No tolerance was changed and no numerical failure was relabeled as a pass.

## Frozen campaign — completed on 2026-11-09

After deferring the recurring fluid discrepancy and completing item 2, the user
explicitly authorized this separate campaign with "ok do it". The earlier
[preflight certificate][certificate] is retained as historical evidence, including
its incomplete-campaign flags; it is superseded for current campaign status by
[the audited frozen certificate](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/final-source-certificate.json).
The complete report before this run is preserved as
[stage-c30-before.md](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/stage-c30-before.md).

The [frozen manifest](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/source-manifest.json) covers **89,012 files**:
source, tests, harnesses, pinned host code, installed dependency files and native
binaries, the Python executable, input assets and all approved references.
SHA256: `35ddaf42183dfbcd3281624185391ce7a22b13f28cf627dcf1f252802b0f2a7a`.
Every Python worker and nested process has matching startup/exit guards and
imported-source records: **83 process pairs**, including the
driver, and **103,288 file-backed import observations**.
The audit matches each replay to its own guarded process and independently
recomputes every comparator result from the retained endpoint arrays. The
post-run auditor's own hash is recorded separately in the certificate.

The review manifest covers **88,850 files**, including installed dependency files
and native binaries, with SHA256
`cc7e068141af048df01bacf71e623e69d89fee42ad9e6b1bfcb8c2e4987d38cf`.
[The independent review-snapshot audit](../.tmp_tests/stage-c30-20261109/final-review-audit.json)
checks every file and the explicitly incomplete certificate status. The initial
review manifest/certificate are also retained: their active report-log hash was
finalized after its writer closed, before this audit. This reporting correction
changed no runtime result or numerical sample.

The review follow-up preserves the preceding manifest, certificate and audit in
[its evidence directory][review-followup], adds historical-evidence and supplied-
checkout records, and records `numerical_sim/FollouDeviceLogs/` explicitly as
**foreign and excluded** in the manifest's `excluded_foreign` metadata. It already
had zero entries in `files`; none of that directory is qualification evidence,
and the directory was left untouched.


| Frozen suite | Passed | Skipped | Failed / errors | Test duration |
| --- | ---: | ---: | ---: | ---: |
| core-suite | 956 | 47 | 1 / 0 | 419.774 s |
| main-suite | 117 | 5 | 0 / 0 | 126.815 s |
| lab30-suite | 114 | 8 | 0 / 0 | 120.299 s |

**Failed test:** `tests.runtime.test_clipping::test_tracker_and_clip`. [Complete log](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/core-suite.log), [XML and traceback](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/core-suite.xml).

The core clipping failure occurs at the second trigger press in
`tests/runtime/test_clipping.py:110`: the test expects the clip count to become
**2**, but observes **1**. Earlier first-clip attachment, pinch and hold checks
passed; later masked-reset assertions in that test were not reached. The cause
is unresolved, and this failure is separate from the deferred fluid discrepancy.
The existing fixture retained the assertion value and full traceback in the
suite log/XML, but does not export full clipping-state endpoint arrays. No retry
was made to reconstruct or replace the failed observation.

The core selection is `tests/packages tests/runtime`. Each host runs the full
established six-module selection: `test_lab_adapters`, `test_solver`,
`test_component`, `test_isaaclab_env`, `test_manager_env` and `test_coupling`.
No tests were excluded to accommodate the fluid deferral. Suite XML, complete
logs and all nested process receipts are retained in the campaign directory.

| Host / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-captured | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |
| standalone-eager | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |
| main-captured | PASS | PASS (2/3) | PASS | PASS | PASS | PASS |
| main-eager | PASS | PASS (2/3) | PASS | PASS | PASS | PASS |
| lab30-captured | PASS | PASS (3/3) | PASS | PASS | PASS | PASS |
| lab30-eager | PASS | PASS (3/3) | PASS | **FAIL** | PASS | PASS |

Totals: **35/36 groups**, **45/48 individual
replays**, **16/18 chole trials**. Chole uses the unchanged
2-of-3 whole-replay policy, contact capacity 256, zero floors and factor-three
comparator. Exactly three fresh processes ran for each host/mode. Every sample,
including failed trials inside passing groups, is retained without retries.

### Retained numerical failures

These are failed observations, with unresolved cause. Earlier C2 progression
decisions and the fluid deferral do not waive these reference failures.
All endpoint arrays and exact comparator values are linked below and retained
in the certificate; no reference or tolerance was changed.

- `main-captured/chole_tet_med`: [report](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/main-captured/chole_tet_med.trial2/chole_tet_med.json), [arrays](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/main-captured/chole_tet_med.trial2/chole_tet_med.npz); 8 failed product comparisons.
- `main-eager/chole_tet_med`: [report](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/main-eager/chole_tet_med.trial0/chole_tet_med.json), [arrays](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/main-eager/chole_tet_med.trial0/chole_tet_med.npz); 8 failed product comparisons.
- `lab30-eager/aorta_shell`: [report](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/lab30-eager/aorta_shell.json), [arrays](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/lab30-eager/aorta_shell.npz); 2 failed product comparisons.

| Replay / product | Difference | Approved noise | Threshold (3 × noise) |
| --- | ---: | ---: | ---: |
| `main-captured/chole_tet_med.trial2/stepped/system/grasp/impulse` | 0.0003001538570970297 | 6.917084101587534e-05 | 0.00020751252304762602 |
| `main-captured/chole_tet_med.trial2/stepped/system/grasp/jaw_impulse` | 0.0003001538570970297 | 6.917084101587534e-05 | 0.00020751252304762602 |
| `main-captured/chole_tet_med.trial2/stepped/system/grasp/moment` | 4.498127964325249e-05 | 1.0048213880509138e-05 | 3.0144641641527414e-05 |
| `main-captured/chole_tet_med.trial2/stepped/system/instrument_contact/impulse` | 0.0007744245231151581 | 0.00021541863679885864 | 0.0006462559103965759 |
| `main-captured/chole_tet_med.trial2/stepped/system/instrument_contact/jaw_impulse` | 0.0009021461009979248 | 0.0001869555562734604 | 0.0005608666688203812 |
| `main-captured/chole_tet_med.trial2/stepped/system/instrument_contact/moment` | 9.848951594904065e-05 | 2.4180859327316284e-05 | 7.254257798194885e-05 |
| `main-captured/chole_tet_med.trial2/stepped/system/tool_force/force` | 13.71234130859375 | 2.46160888671875 | 7.38482666015625 |
| `main-captured/chole_tet_med.trial2/stepped/system/tool_force/jaw_load` | 4.9031982421875 | 0.863037109375 | 2.589111328125 |
| `main-eager/chole_tet_med.trial0/stepped/system/grasp/impulse` | 0.00029426137916743755 | 6.917084101587534e-05 | 0.00020751252304762602 |
| `main-eager/chole_tet_med.trial0/stepped/system/grasp/jaw_impulse` | 0.00029426137916743755 | 6.917084101587534e-05 | 0.00020751252304762602 |
| `main-eager/chole_tet_med.trial0/stepped/system/grasp/moment` | 4.4082989916205406e-05 | 1.0048213880509138e-05 | 3.0144641641527414e-05 |
| `main-eager/chole_tet_med.trial0/stepped/system/instrument_contact/impulse` | 0.0008328147232532501 | 0.00021541863679885864 | 0.0006462559103965759 |
| `main-eager/chole_tet_med.trial0/stepped/system/instrument_contact/jaw_impulse` | 0.0009201513603329659 | 0.0001869555562734604 | 0.0005608666688203812 |
| `main-eager/chole_tet_med.trial0/stepped/system/instrument_contact/moment` | 9.673851309344172e-05 | 2.4180859327316284e-05 | 7.254257798194885e-05 |
| `main-eager/chole_tet_med.trial0/stepped/system/tool_force/force` | 16.2633056640625 | 2.46160888671875 | 7.38482666015625 |
| `main-eager/chole_tet_med.trial0/stepped/system/tool_force/jaw_load` | 4.7845458984375 | 0.863037109375 | 2.589111328125 |
| `lab30-eager/aorta_shell/masked_reset/system/catheter_control/impulse` | 4.769970473716967e-05 | 1.5421312127728015e-05 | 4.6263936383184046e-05 |
| `lab30-eager/aorta_shell/stepped/system/catheter_control/impulse` | 4.769970473716967e-05 | 1.5421312127728015e-05 | 4.6263936383184046e-05 |

All six chole groups meet the existing 2-of-3 policy. Their retained individual
trial failures do not add failed group gates or require a new policy exception.

The Lab 3.0 eager `aorta_shell` failure recurs on the same two catheter-control
impulse products as the [C2.5-inherited C2.4 report](../.tmp_tests/stage-c24-20261106/certificate/lab30-eager/aorta_shell.json)
and [retained arrays](../.tmp_tests/stage-c24-20261106/certificate/lab30-eager/aorta_shell.npz).
Both current differences are `4.769970473716967e-05`, against the unchanged
threshold `4.6263936383184046e-05`; the prior differences were
`6.471141023212112e-05`. The current values still fail. Matching products do
not establish a common cause. [The exact recurrence comparison](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/aorta-shell-recurrence.json)
retains both observations. C3.2 is the planned catheter-control ownership
migration, and must retain these exact failed endpoints in its comparison
report; that work has not started.

The failed gates are: `lab30-eager/aorta_shell`, `core-suite`.
**C3.0 remains open pending explicit user disposition on these results.**

## Ownership/accounting reconciliation

The allocation inventory preserves its existing sites and ownership contracts:
**587 sites, zero additions/removals, zero ownership migrations**. The extra source
file contains only the capability guard. Historical C1/C2 inventories and removed
sites remain intact. [Inventory reconciliation][inventory] records this result.

Exactly one construction sample ran at [1 world](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/construction-1.json)
and one at [64 worlds](../.tmp_tests/stage-c30-20261109/certificate-20261109T163226Z/construction-64.json). Both match the C2.5-inherited
C2.4 samples exactly for Model buffers, native per-buffer extents, templates,
live/checkpoint containers, declared fields, identities, padding, residue,
retained storage and peak Warp storage. **No Newton-induced allocation
difference was observed in this construction workload.** The certificate
records the full comparison by field; this is not a universal allocation claim.

| C3.0 measured accounting, bytes (unchanged from C2.5) | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model buffers | 6,125,893 | 93,207,265 |
| Initialization templates | 1,499,948 | 95,996,672 |
| Live State / independent checkpoint State, each | 1,567,292 | 100,306,688 |
| Live / checkpoint / template-alias Control | 0 / 0 / 0 | 0 / 0 / 0 |
| Control/template overlap | 0 | 0 |
| Pool / Tet / Surface identity columns | 98,304 / 1,172,688 / 532,760 | 6,291,456 / 25,800,144 / 12,663,032 |
| Retained Warp storage | 27,805,529 | 1,218,389,558 |
| Six legacy residue arrays, total (already included) | 2,020,320 | 129,300,480 |

The six residue arrays remain explicit: `InstrumentContact.delta_k`,
`InstrumentContact.delta_jaw`, `Grasp.delta_k`, `Grasp.delta_jaw`,
`VolumetricGrasp.delta_k`, `VolumetricGrasp.delta_jaw`. Each retains
336,720 / 21,550,080 bytes at 1 / 64 worlds, under its legacy system State owner.
They have **zero** new Newton template/checkpoint copies.

Retain maximum backing extent when aliases share a pointer: native
gravity is 24 / 780 bytes while its Scene alias is 12 / 768 bytes. Empty-domain
padding remains covered by focused tests; the C3.0 chole construction samples
have no such backing rows. Thermal heat/burn retain additive and legacy copies,
Appearance remains legacy, and native/exact inverse-mass and flags duplicates
remain. See [C2.5's complete reconciliation](stage-c25.md) for per-family Model,
State, template, identity and padding breakdowns. Construction timings and peaks
are recorded below.


The raw harness key `intentionally_shared_control` denotes the existing explicit
`model.control(clone_variables=False)` template alias, not ordinary live-Control
sharing. All Control allocations are zero bytes in this workload. The separate
finalizer preflight tests retain the distinction between ordinary cloned live
Controls, deliberately sharing one live Control, and opting into a template alias.
The raw harness's C2.4 variant label describes the unchanged ownership layout;
these new samples use the C3.0 dependency pins and frozen source certificate.

| Construction measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Retained Warp bytes | 27,805,529 | 1,218,389,558 |
| Peak Warp bytes | 28,011,773 | 1,231,594,844 |
| Additional checkpoint State/Control bytes | 1,567,292 | 100,306,688 |
| Host peak RSS, KiB | 831,880 | 2,736,080 |
| Construction time, seconds | 1.117490 | 15.562639 |

These are single construction observations, not steady-state performance
qualification. Phase timings and the GPU process list at entry are preserved
in each raw sample. C4 and final performance qualification remain later work.

C3.0 remains open. C3.1 may adopt this frozen baseline only after C3.0 acceptance;
no C3.1 implementation was started.
**Mounted-tool episode restore remains partial until C3.3.** C4 history, a typed
command/recording-format-v2 migration, and final performance work were not added
to this checkpoint.

[evidence]: ../.tmp_tests/stage-c30-20261109/
[preflight]: ../.tmp_tests/stage-c30-20261109/c25-preflight.json
[snapshot]: ../.tmp_tests/stage-c30-20261109/pre-c3-source.tar
[preservation]: ../.tmp_tests/stage-c30-20261109/preservation-final.json
[core-identity]: ../.tmp_tests/stage-c30-20261109/core-identity.json
[main-identity]: ../.tmp_tests/stage-c30-20261109/main-identity.json
[lab30-identity]: ../.tmp_tests/stage-c30-20261109/lab30-identity.json
[stream-arrays]: ../.tmp_tests/stage-c30-20261109/lab30-stream-failure/user-stream-ordering.npz
[stream-values]: ../.tmp_tests/stage-c30-20261109/stream-failure-values.json
[certificate]: ../.tmp_tests/stage-c30-20261109/preflight-certificate.json
[inventory]: ../.tmp_tests/stage-c30-20261109/inventory-delta.json

[c21-lab30-suite]: ../.tmp_tests/stage-c21-20261106/lab30-suite.log
[c21-main-suite]: ../.tmp_tests/stage-c21-20261106/main-suite.log
[c21-main-arrays]: ../.tmp_tests/stage-c21-20261106/main-suite-host-parity.npz
[c21-retention-arrays]: ../.tmp_tests/stage-c21-final-certificate-20261106/retention-validation/pytest/test_user_stream_ordering_newt0/user-stream-ordering.npz
[c21-retention-record]: ../.tmp_tests/stage-c21-final-certificate-20261106/retention-validation/validation.json
[history-check]: ../.tmp_tests/stage-c30-20261109/review-followup-20261109/historical-evidence-check.json
[supplied-newton]: ../.tmp_tests/stage-c30-20261109/review-followup-20261109/supplied-newton-checkout.json
[review-followup]: ../.tmp_tests/stage-c30-20261109/review-followup-20261109/
[final-commands]: ../.tmp_tests/stage-c30-20261109/final-check-results.json

[item2]: ../.tmp_tests/stage-c30-20261109/item2-20261109T162250Z/summary.json
