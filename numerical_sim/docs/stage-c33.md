# C3.3 review follow-up — 2026-11-10

**The approved construction pass and requested code follow-ups are complete.
C3.3 remains open for review; C4 has not started.** HDF5 is explicitly deferred
by the user. No retained environment was changed, no dependency installed, and
no commit or push made. The original frozen results and failures below are
historical evidence, not qualification of the later production changes.

## Approved post-freeze construction pass

Before editing, all **89,065 files** matched the current review manifest
`fda523d726d54303c24be10d3f8e11cdb43b28c4a691e701c07d3e490968c522`.
The 576 tracked/untracked nonignored source files were preserved in
[pre-review-source.tar](../.tmp_tests/stage-c33-review-20261110/pre-review-source.tar), with
[entry hashes](../.tmp_tests/stage-c33-review-20261110/pre-review-source-hashes.json).

The user's approval was **construction only; defer HDF5**. The corrected,
established chole reporter ran once at each world count. A separate rod-bearing
`aorta_shell` reporter ran once at each count in the same pass, using its direct
rod, catheter controller and deformable shell. All four fresh-process CUDA
construction samples passed, without stepping or capture. No suite or reference
campaign was rerun. The original two failed construction attempts remain retained.

| Workload | Worlds | Construction seconds | Retained Warp bytes | Peak Warp bytes | Additional State + Control bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| chole_tet_med | 1 | 0.924 | 27,823,129 | 28,029,373 | 1,578,620 |
| chole_tet_med | 64 | 16.028 | 1,219,312,846 | 1,232,518,132 | 101,031,680 |
| aorta_shell | 1 | 0.089 | 3,354,144 | 18,032,112 | 275,952 |
| aorta_shell | 64 | 0.708 | 172,785,410 | 205,359,656 | 17,658,408 |

These are individual construction measurements, not statistical performance
claims. Other GPU processes present at entry are recorded in each report.
Additional checkpoint containers were allocated after the construction peak.
Pointer accounting takes the maximum extent for each backing pointer and
separates Model buffers, initialization templates, live/checkpoint containers and
explicit template-alias Controls. The six chole residue arrays remain exactly
**2,020,320 bytes at one world and 129,300,480 bytes at 64 worlds**, with zero new
residue template/checkpoint copies.

The rod case has 128 nodes per world and 5,187 total native particles per world.
Its shared rod record has 4,592 Model bytes and `2,540*E` Control bytes; the rod
solver record has `1,536 + 20*E` Model bytes and `3,584*E` State bytes. The catheter
record has `44 + 48*E` Model bytes, `24*E` State bytes and `40*E` Control bytes
(the latter includes its 28-byte mount pose). Matching templates, identity
columns and other scene buffers are separately reported; these rod records have
no padded fields. This rod sample has no configured host mappings; synthetic
mount allocation checks remain separate from construction measurements.

[Construction certificate](../.tmp_tests/stage-c33-review-20261110/construction-certificate.json),
[chole 1](../.tmp_tests/stage-c33-review-20261110/chole-1.json), [chole 64](../.tmp_tests/stage-c33-review-20261110/chole-64.json),
[aorta 1](../.tmp_tests/stage-c33-review-20261110/rod-1.json), [aorta 64](../.tmp_tests/stage-c33-review-20261110/rod-64.json).
The measurement manifest contains **89,068 files**, SHA256
`7daf9bfa6e849c26cfc3c05f33c59aaf441c6d08d3a2b63c58b1ebaf6c328cf3`.
All **5 startup/exit pairs** and **1,975 imported-file observations** match.
Production/test source was unchanged from the prior C3.3 review during this pass.

## Control.clear and public registration

A mount-capable Model's returned Controls now reject **`clear()` and
`clear(model)` before mutation**, with an error explaining that blanket clearing
would zero mount-pose quaternions. The pinned Newton version provides no custom
clear hook, so finalization wraps only that Model's Control factory; Newton's
classes and dependency source stay unchanged. The policy covers pre-binding,
independent and template-alias Controls, including separate Models finalized from
one retained builder. Models without mountable controls retain ordinary clearing.
Packed/named action views remain clearable and retain the current mount poses.
Episode reset semantics and recording formats remain unchanged.

All six requested package families now use **`register_package`** on authoring or
bound storage: tets, surfaces, instruments, rods, endoluminal and mounts.
Authoring publishes host domain records and rejects duplicate/late registration.
Bound storage registers identity/layout checks, including array backing metadata,
and validates old identities before explicit alias routing. `container(owner)`
provides public access to selected Newton containers. Package code no longer
mutates storage's private identity, array or check collections. The detailed
contract is in [field contracts](stage-c-field-contracts.md#c33-review-follow-up-registration-and-control-clearing).

The allocation inventory remains **511 sites / 187 source files**, with no added
or removed device allocations. Numerical kernels, Stepper/host solver code,
capacities, references and clipping assertions remain unchanged. Original source
and dependencies are accounted for in the [preservation audit](../.tmp_tests/stage-c33-review-20261110/preservation-audit.json);
the [source patch](../.tmp_tests/stage-c33-review-20261110/review-source-change.patch) preserves the change against
the review baseline.

## Focused final-source verification and qualification boundary

| Environment | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Core | 292 | 2 | 0/0 |
| Lab main | 294 | 1 | 0/0 |
| Patched Lab 3.0 | 294 | 1 | 0/0 |

Each environment passed ten new registration/clear-policy regressions. Existing
ownership, alias rejection, reset, sharing and continuation regressions passed on
CPU eager and CUDA eager/captured. Actual MuJoCo-Warp coupling runs on CUDA;
no CPU host-coupling support is claimed. Both hosts passed mounted-tool rebuild,
command-view invalidation and close tests. HDF5 remains unverified/deferred.

The first core launcher incorrectly included a host-only node selector and exited
4 during collection, before running tests. That log and result are retained.
Only the applicable core selection was then run on the **same frozen source**;
neither host run was repeated. The earlier development selection passed 266 tests
with 2 skips. None of these runs was a full-suite or reference campaign.

The final focused manifest has **89,072 files**, SHA256
`77c1739648e5d6f5be218be53b6f69cba29e7c7ddd90408e125a5c691e57378c`.
All **11 startup/exit pairs**, including six nested test processes, match, with
**7,728 matching imported-file observations**. The independent auditor recomputed
sample accounting, pointer relationships, report counts, manifest hashes and
receipts: [follow-up certificate](../.tmp_tests/stage-c33-review-20261110/review-followup-certificate.json),
[auditor](../.tmp_tests/stage-c33-review-20261110/audit_followup.py),
[core results](../.tmp_tests/stage-c33-review-20261110/core-corrected-focused.xml),
[main results](../.tmp_tests/stage-c33-review-20261110/main-focused.xml), [Lab 3.0 results](../.tmp_tests/stage-c33-review-20261110/lab30-focused.xml).

**The original campaign and the new construction pass precede the registration
and Control.clear production changes. Final production source has focused
qualification only.** No second campaign or final-source construction repeat was
run. The construction reporting gap is now measured, but progression still needs
review disposition. Rigid private solver history and unmigrated grasp/hook/contact
history remain outside the continuation claim. **Stop before C4.**

---

The following original C3.3 report is preserved as historical evidence, including
its then-open construction gap and then-current source qualification.

# C3.3 — Mount ownership and pending coupling reactions

**Mounted flag, mount-pose and pending-reaction ownership are implemented and
passed their checks. C3.3 remains open.** Both authorized construction attempts
failed in reporting; the corrected harness has focused synthetic verification
only. Stop for review before C4. No commit or push was made.

C3.2 was accepted for progression with retained failures: 34/36 frozen groups,
43/48 replays and 15/18 chole trials passed, with every chole group passing its
2-of-3 policy. Standalone captured `aorta_shell` failed stepped catheter impulse;
patched Lab 3.0 captured failed stepped and masked-reset catheter impulse. Causes
remain unresolved. Frozen suites passed core 1,017, main 117 and Lab 3.0 114,
with recorded skips. The later production alias-validation correction passed
focused checks only (core 103, each host 104); full qualification was not rerun.
HDF5 recording/replay remains unverified because both retained hosts lack `h5py`.
Earlier clipping/fluid observations and all historical certificates remain intact.
That disposition does not waive any C3.3 gate.

## Preservation and scope

The pre-edit audit verified all **89,043 files** against manifest SHA256
`bf8ef3f8cbb390a1a605c16e50eb661dfb37b53d33cc817dafffc58704b8bec8`.
All 573 tracked/untracked nonignored source files were archived with hashes,
status and binary diff in [the evidence directory][evidence]. Existing source,
environments, certificates, arrays and approved references remain in place or,
for edited source, are preserved in `pre-c33-source.tar`.

Retained dependencies: Newton `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`, Lab main
`4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1`, patched Lab 3.0
`e53cd288727a5ca3dae7187d1d415b4d32b67e11`, patched Warp 1.17.0,
MuJoCo/MuJoCo-Warp 3.17.0 and Python 3.12.12. No synchronization, upgrade or
optional dependency installation. Numerical kernels, execution order, capacities,
tolerances, comparison policies, references and clipping assertions/export behavior
are unchanged. No numerical attribution, clipping correction, C4 work, commit or push.

## Ownership and authoring

| Fields | Newton owner | Initialization / behavior |
| --- | --- | --- |
| Instrument/Catheter `mounted` | Model | Configured flags or zero; explicit `mount(k)` edits persist through reset |
| Instrument/Catheter `mount_pose` | Control | Existing authored root/orientation; host/external writers; reset preserves |
| MountGroup `target`, `offset`, `body` | Model | Local target indices, normalized offsets, genuine native BODY references |
| MountGroup `force`, `torque` | State | Zero templates; pending reactions preserved through binding/preparation |
| Mounts `env_xform` | Model | Completed component-to-world transforms |

[Physics mount declarations](../physics/src/i4h_physics/mount_storage.py) consume
small frozen package-supplied `Mountable` descriptors: system ID, target names,
field declarations and root-reaction capability. Physics imports neither surgical
nor endoluminal implementations. Instrument and catheter declarations publish their
own interface fields; `CaseAuthoring` publishes coupling after system declarations.
Direct Scene callers use `declare_mounts` after target declarations and before
schema registration. Optional `mounts` and `coupling_mode` authoring arguments are
passed through existing host configurations and hard rebuilds; the public mapping
format is unchanged. `Mounts(...)` validates and binds matching declarations,
allocates no migrated storage and never calls `control.mount()`.

Shared mount domains derive from control IDs, carry world -1, and reference the
declared target-control domain. Their target field remains a local kernel index.
Runtime mount domains carry real worlds and native BODY owners; the body field is
also a genuine BODY reference. Nonentity and padding owners are -1. Deferred
custom field values resolve with the completed builder before any custom rows are
emitted, alongside deferred identities. Exact-label/suffix resolution requires
one match per world, including foreign body prefixes and bodies added after i4h's
world hook. Finalization rechecks host body mappings, including retained builders.
Missing, duplicate, ambiguous, late and incompatible configurations fail explicitly;
two-way PBD catheter and unsupported rigid solver prohibitions remain.

Instrument initialization reads finalized Model mounted flags and mount-pose
templates before any State/Control clone. The former temporary mounting buffers
are removed; authored initial poses remain intact and no initial host drive occurs.
The new declaration records retain host data and weak references, not finalized
device arrays.

## Binding, reset and continuation

All runtime aliases point to the selected Newton containers. Identity checks cover
native buffers, mount groups and their nested fields, control aliases, source
impulse/moment arrays and root arrays. Coupling rejects invalid selected containers
or replaced storage before writing reactions or poses. Binding, ordinary preparation
and cache reset preserve Model/State/Control bytes.

The numerical sequence is unchanged: **apply previous reactions → rigid substeps
→ drive mounts → i4h step → accumulate next reactions**. Existing sign/scaling,
component/world conversion, root-to-COM moment shift, multiple-source accumulation
and shared-body atomic addition kernels are unchanged. Explicit episode reset
clears selected pending forces/torques; empty/repeated masks preserve unselected
worlds. Mounted flags and pose commands persist; instrument reset uses the current
pose and catheter reset adds no root reset.

Runtime `control.mount()` remains an explicit in-place Model flag edit for an
externally driven target, with unchanged index validation and no recapture. It
creates no host mapping/reaction group. Host mapping changes require reconstruction.
Packed/named actions and recording formats are unchanged; mount poses are excluded.

Exact resumed/uninterrupted equivalence is verified using a deterministic rigid
test solver whose entire integration history is exposed in native State body pose
and velocity. Independent States/Controls, ordinary live-Control sharing, explicit
template aliasing, later clones and two Models from one retained builder are tested.
Real MuJoCo-Warp integration is checked on CUDA eager/captured paths; CPU tests cover
storage and kernels, not CPU host coupling. These results do **not** establish
restoration of MuJoCo-Warp private history or unmigrated grasp/hook/contact history.

## Focused evidence and allocation accounting

The extended pre-migration mounted-tool suite passed 17 checks. The final mount
ownership/coupling selection passed **45 checks in each environment**, covering CPU
eager, CUDA eager/captured, instrument/catheter mounts, multiple controls/worlds,
shared bodies, transformed environments, offsets, foreign prefixes, padding,
byte preservation, rejection before writes, masked reset and continuation.
The final broader focused selections passed core **216/2 skipped**, Lab main
**221/1 skipped**, and patched Lab 3.0 **221/1 skipped**, including all C3.2 alias
regressions and host named-command/rebuild/close checks. The host skips require
absent `h5py`; core skips the two host-only tests. Development failures/logs remain
separate: missing fixture declarations/configuration and new synthetic-test
assumptions were corrected before freezing. Catheter root motion retains its
existing per-substep limit; no numerical correction was made.

For E worlds, N instrument targets per controller, C catheter controllers and M
configured mounts:

| Storage | Bytes |
| --- | ---: |
| Mounted Model flags | `4*max(N,1)` per instrument controller, plus `4*C` |
| Mount-pose Control, per container | `28*E*(N+C)` plus matching template |
| Pending-reaction State, per container | `24*E*M` plus matching template |
| Target/offset Model fields | `32*M` |
| Native BODY mapping Model fields | `4*E*M` |
| Environment Model transforms | `28*E` per Mounts aggregate |

[Focused synthetic allocation evidence][allocations] separately accounts for custom
identities, padding, scratch, temporary finalization allocations and alias overlap.
Custom identity domains use `8*max(rows,1)` bytes; participating domains include
reused control runtime identities. Empty instrument flags use four bytes and
empty mount-pose backing uses 28 bytes per container/template. The coupling adds
no scratch arrays. Instrument runtime/reset mask scratch and finalization mask
temporary each use E bytes. Mapping validation has host readbacks, no additional
device temporaries; traced host peaks are reported. Maximum backing extents are
used when deduplicating aliases. Ordinary shared Control adds zero bytes; explicit
template aliases overlap template storage.

The reconciled inventory contains **511 allocation/helper sites in 187 source
files**, with 12 legacy allocation sites removed and no new allocation sites.
Synthetic examples (E=3; bytes; each Control/State has a matching template):

| N / C / M | Flags | Pose Control | Pending State | Target/offset | BODY map | Env transforms | Identities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / 0 / 2 | 8 | 168 | 144 | 64 | 24 | 84 | 152 |
| 2 / 2 / 4 | 16 | 336 | 288 | 128 | 48 | 84 | 280 |

The deduplicated field extents across Model, templates, live/checkpoint States and
Controls, shared live Control and template aliases are 1,116 and 2,148 bytes
respectively, excluding the separately counted identities. Empty and manual-only
fixtures are also retained. Mapping-validator traced host peaks were 1,440 bytes
for both populated fixtures; it adds no device temporary. These synthetic tests
are separate from the established chole benchmark.

All six legacy residue allocation paths remain unchanged and add no template or
checkpoint copies. The retained C3.2 measured totals are **2,020,320 bytes at one
world** and **129,300,480 bytes at 64 worlds**. The two C3.3 construction attempts
failed before writing reports, so new construction measurements, including
residue totals and peak/retained bytes, are **unverified**. The historical measured
values are not presented as successful C3.3 samples.

## Frozen campaign and independent audit

The [frozen campaign certificate][frozen] covers **89,059 files**, manifest SHA256
`27bc10867fe4b657fe358914a3f20e287fc9ee2bb9aa5382c7c5882967243812`.
All production, tests and campaign inputs stayed unchanged through every worker's
exit. The auditor verified **83 matching startup/exit pairs**, **103,521 matching
file-backed import observations**, guarded parent/child relationships and every
endpoint comparison independently from retained arrays.

| Frozen suite | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Full core | 1,085 | 49 | 0/0 |
| Lab main six-module selection | 121 | 5 | 0/0 |
| Patched Lab 3.0 six-module selection | 118 | 8 | 0/0 |

All **36/36 reference groups passed**, with **46/48 individual replays** and
**16/18 chole trials** passing. Exactly 18 fresh-process chole trials ran. The
retained failed trials are standalone captured trial 1 and standalone eager trial
0; both groups passed the unchanged 2-of-3 rule. Their 27 failed product
comparisons remain in the certificate. No retry or reference promotion occurred.
Capacity 256, zero floors and factor-three comparisons are unchanged.

Historical and new `aorta_shell` endpoints are retained separately:

| Mode | Historical C3.2 | New C3.3 |
| --- | --- | --- |
| Standalone eager | [pass / arrays](../.tmp_tests/stage-c32-20261109/certificate/standalone-eager/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/standalone-eager/aorta_shell.npz) |
| Standalone captured | [failed stepped impulse / arrays](../.tmp_tests/stage-c32-20261109/certificate/standalone-captured/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/standalone-captured/aorta_shell.npz) |
| Lab main eager | [pass / arrays](../.tmp_tests/stage-c32-20261109/certificate/main-eager/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/main-eager/aorta_shell.npz) |
| Lab main captured | [pass / arrays](../.tmp_tests/stage-c32-20261109/certificate/main-captured/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/main-captured/aorta_shell.npz) |
| Lab 3.0 eager | [pass / arrays](../.tmp_tests/stage-c32-20261109/certificate/lab30-eager/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/lab30-eager/aorta_shell.npz) |
| Lab 3.0 captured | [failed stepped/masked-reset impulse / arrays](../.tmp_tests/stage-c32-20261109/certificate/lab30-captured/aorta_shell.npz) | [pass / arrays](../.tmp_tests/stage-c33-20261109/certificate/lab30-captured/aorta_shell.npz) |

The certificate also records exact stepped and masked-reset catheter-impulse
values for C3.1, C3.2 and C3.3. New passing endpoints do not establish a cause or
resolution of the historical failures. Clipping assertions and failure-only
export behavior are unchanged.

## Construction failure, reporting correction and qualification gap

Exactly one established construction attempt ran at each required world count.
Both [one-world][construction1] and [64-world][construction64] runs exited 1 at
`instrument_accounting`: its obsolete assertion expected `40*E*N` Control bytes,
although migrated mount poses add `28*E*N`. Both processes passed their source
and import guards. Neither wrote a construction JSON report; no construction
attempt was retried.

After all frozen workers exited, the reporting harness was preserved in
[an archive][fix-archive] and its assertion alone was corrected to `68*E*N` in
[this patch][fix-patch]. **No production or test source changed after freezing.**
The corrected helper passed **6 synthetic checks in each environment**: populated,
mixed-controller and empty mounts on CPU and CUDA, with container bytes preserved.
These checks ran zero chole construction samples. Full suites, references and
construction samples were not rerun. The frozen campaign does **not** qualify
the corrected reporting harness or supply the missing construction measurements.

The [current review certificate][current] covers **89,065 files**, manifest SHA256
`fda523d726d54303c24be10d3f8e11cdb43b28c4a691e701c07d3e490968c522`.
The separate correction driver, three workers and independent auditor have five
matching startup/exit pairs and 1,945 matching file-backed import observations.
Every frozen production/test file is unchanged; the sole changed frozen path is
the archived construction reporter. The auditor recomputed all comparisons and
explicitly retained both construction failures. `stage-c33.md` and `stage-c.md` prose are outside these manifests, allowing final
results to be written afterward.

**C3.3 remains open pending an explicit disposition of the missing construction
reports and the corrected harness's qualification gap.** HDF5 recording/replay
remains unverified without `h5py`; rigid private history and unmigrated interaction
history remain outside the ownership claim. No C4 work, commit or push was made.
**Stop for review before C4.**

[evidence]: ../.tmp_tests/stage-c33-20261109/
[allocations]: ../.tmp_tests/stage-c33-20261109/focused-allocation.json
[frozen]: ../.tmp_tests/stage-c33-20261109/certificate/frozen-campaign-certificate.json
[current]: ../.tmp_tests/stage-c33-20261109/post-campaign/review-source-certificate.json
[fix-archive]: ../.tmp_tests/stage-c33-20261109/post-campaign/frozen-fix-sources.tar
[fix-patch]: ../.tmp_tests/stage-c33-20261109/post-campaign/source-change.patch
[construction1]: ../.tmp_tests/stage-c33-20261109/certificate/construction-1.log
[construction64]: ../.tmp_tests/stage-c33-20261109/certificate/construction-64.log
