# C2.1 final-source certificate follow-up

C2.1 implementation remains accepted as described. Its checkpoint remains open;
C2.2-C2.4 have not started. No production runtime source changed in this follow-up.
The test harness adds failure retention to `test_user_stream_ordering` with the
same root `allclose(atol=1e-5)`, observation `allclose(atol=1e-4)` and final
`torch.equal` comparisons. On failure it stores both full observations, both root
arrays, ordered observation labels and comparison settings; the final step check
also stores its full observation/position/copy arrays. A forced observation-failure
harness validation verified retention, separate from qualification. An earlier
validation setup error (missing artifact parent) is retained in its own log.

## Certificate protocol and attribution scope

Evidence: `.tmp_tests/stage-c21-final-certificate-20261106/`. One immutable
`source-manifest.json` covers executable workspace sources, cases, scripts, tests,
qualified host sources, installed Newton/Warp Python and their patched native
binaries. Its digest is in `source-manifest.sha256`. Inherited startup/exit guards
verify the same manifest and retain per-process receipts and imported package
paths. The parent checks the manifest before and after each job. Thus the
certificate directly includes final `scene.py` and `newton_model.py`; it does not
borrow earlier mixed-revision qualification.

The fixed campaign is six cases, captured and eager, on standalone, Lab 3.0 and
main, with all exports enabled; chole uses exactly three retained fresh trials in
each host/mode group. Then the full core package/runtime suite and both focused
host suites run once. Each failed sample, source receipt and pytest artifact is
retained; there are no acceptance retries or tolerance changes. The existing
Solvecraft workload is not removed; this is not an exclusive performance campaign.

The requested additional attribution distribution campaign is **not run**. The
pre-declared protocol was unavailable in the visible conversation/Stage C evidence;
when asked for its fixed counts, scope and authorization, the user replied,
"ignore it and continue." This supersedes item 3 for this follow-up. The previous
one-before/one-after diagnostics remain bounded observations, not distributions
or evidence of equivalence, harmlessness or pre-existing failure.

## Final-source results

The nine-job campaign is complete against this one source-manifest digest:

`2c796419091f0f2517e1931aa387b6d4073d4524b96d577414d8c9a48f597ef6`

The manifest covers **9,294 files**. All nine jobs passed source verification;
**80 Python processes** retained matching startup/exit receipts, and **494 imported
package-root observations** resolve to files in that same manifest. All **48
reference workers** independently recorded source maps matching it. The final
`scene.py` hash is `27a409fb00c05dbe18c6631dc2d5ed3f459d4e477996e9c1d9cb79f354449ddc`;
`newton_model.py` is `d938f16d7965c2f908b548e7a1673df2174d990b3ac7b9a2df26bf0fa2574b3c`.
The 207 production source/resource files are unchanged from the accepted C2.1
runtime. The test-retention change is part of the frozen certificate manifest.

| Host | Six captured references | Six eager references | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | PASS | **FAIL: aorta_shell impulse**; five other cases pass | 3/3 / 3/3 |
| Lab 3.0 | PASS | PASS | 3/3 / 3/3 |
| Main | PASS | **FAIL: aorta_shell impulse**; five other cases pass | 3/3 / 3/3 |

| Suite on final source | Result | Source verified |
| --- | --- | --- |
| Full core package/runtime | **813 passed, 47 skipped** | Yes |
| Lab 3.0 focused host suite | **106 passed, 8 skipped** | Yes |
| Main focused host suite | **109 passed, 5 skipped** | Yes |

Standalone eager's aorta_shell impulse difference is
**6.792339445382822e-05**; main eager's is **7.092467785696499e-05**. Both exceed the
unchanged **4.6263936383184046e-05** tolerance in the stepped and masked-reset
products at index `(world=1, tool=0, component=0)`. The masked-reset comparison
retains this unselected world's previous impulse. These are four failed product
comparisons in two failed case runs. Full arrays/logs and exact values are in
`reference-failure-products.json` and the respective eager worker directories.
No retry, threshold change, failure waiver or attribution is applied.

Machine-readable certificate: `final-source-certificate.json`. Source checks are
in `certificate-results.json`, `campaign-complete.json`,
`reference-worker-source-match.json` and `receipts/`. Original failing suite and
reference samples in `.tmp_tests/stage-c21-20261106/` remain unchanged. Passing
this final-source suite invocation does not explain or erase those samples.

**C2.1 remains open:** eager reference failures and the position-write history gap
below remain unresolved. The redundant rest-copy removal is planned. C2.2 has not
started. No all-pass integrated numerical or fresh-runtime continuation claim is
made.

## Residue instruction provenance

The source is the user's first implementation message in this conversation:
"Treat the plan as the source of user intent," followed by the locked decision:

> retain all six `delta_k`/`delta_jaw` arrays as State. Leave their kernels unchanged.
> Include their actual bytes in every memory report; they remain on legacy owners
> until their systems migrate later.

No earlier independent answer to C1's pending option-2 question was found or is
claimed. Preservation and accounting were authorized by this later user-supplied
implementation instruction. The implementation does not change the residue kernel
or retroactively relabel the old pending entry as an earlier user decision.

## Particle-position writes and history audit

**The universal confirmation requested in item 5 cannot be given for the current
source.** `I4hSolver.reset(flags=0)` preserves history by the accepted cache-refresh
contract; it calls system `refresh` methods without copying q into q_prev.
`Particles` has no history-copy refresh override. Binding intentionally preserves
selected State history and initializes new States from authored templates.

| Path | Position writer / ordering | History result |
| --- | --- | --- |
| Case creation and host backend State creation | `CaseAuthoring.add_world` authors before finalize; Newton `model.state()` copies native Model positions after finalize | Authored history initially equals authored positions. This is not an arbitrary post-finalization teleport. |
| Explicit i4h episode reset, including component reset | `PhysicsBinding.reset_physics` -> `restore_episode` -> `Scene.reset` -> `reset_particles` | Selected q restored, then q_prev=q in the same selected-world operation; foreign rows excluded. |
| Main/Lab 3.0 FK/reset delegates, pure and coupled | Main backend `forward`/solver reset; Lab 3.0 `_reset_solver_internals`; coupled delegates to `ours.reset(flags=0)` | Cache-only refresh preserves q_prev; FK affects bodies/joints, and is not a history-copy guarantee for particle edits. |
| `I4hEnv._reset_idx` with reset events | Physical reset first, then `super()._reset_idx` events may write `obs_views`; `_torch_to_warp` orders streams | **Gap:** no subsequent live-position history initialization. Existing `ShiftEvents` changes particle positions by 0.01 after history was reset. |
| Native DeformableObject nodal position/state/kinematic-target setters, both qualified hosts | Index/mask scatter into native State positions; `_mark_particles_dirty` invalidates geometry/BVH timestamps | No i4h history refresh. USD-imported asset ranges are distinct from manual CaseAuthoring component ranges; not qualification of edits to i4h-owned rows. |
| Native MPMObject reset and state/position index/mask writers, both qualified hosts | Scatter into native State positions, then invalidate geometry caches | Same limitation: geometry dirty is not a live-position history copy. |
| Component rebuild callbacks / public position views | `I4hComponent._bind` invokes callbacks; callbacks can write `component.positions()` before the first step; `ConstructionContext.publish` then prepares | **Gap:** preparation does not initialize q_prev from those live writes. |
| Direct host writes to selected `state.particle_q` before or after binding | Caller controls the write; binding is not a teleport/reset operation | **Gap:** authored/restored q_prev is preserved. Neither binding nor cache-only reset initializes history from the edited q. |
| Tiled-camera proxy Model/State points | Render-only buffers; not the bound physics Model/State | Outside particle-physics history; do not apply a blanket copy to this path. |

Source anchors: `isaaclab/src/i4h_isaaclab/env.py` (`_reset_idx`),
`physics/src/i4h_physics/{sim.py,scene.py,systems/base.py}` and both qualified
`isaaclab_newton` trees under `.tmp_tests/` (`assets/deformable_object`,
`assets/mpm_object`, `physics/newton_backend.py` or `newton_manager.py`). The exact
host-source hashes are included in the certificate manifest. After the certificate,
a read-only wrapper around the unchanged reset-event test retained q, q_prev and
q0 on both hosts. The original position-visibility test passed on both, while the
additional history observation showed this gap:

| Host | Reset selection | Maximum selected abs(q_prev - q) | Unselected history |
| --- | --- | ---: | --- |
| Lab 3.0 | Worlds 0 and 1, before the first step | 0.01000000536441803 | Unchanged |
| Lab 3.0 | World 1 after the test's rollout | 0.010000001639127731 | Unchanged |
| Main | Worlds 0 and 1, before the first step | 0.01000000536441803 | Unchanged |
| Main | World 1 after the test's rollout | 0.010000001639127731 | Unchanged |

Each selected-history equality check is false. Raw arrays and reports are retained
in `history-lab30/` and `history-main/`. This is a deterministic lifecycle audit of
the existing shift-event example, not a pre-change/final-source attribution sample
or an explanation of the reference failures.

The pre-bind case is specifically changed by C2.1: the old Scene binding cloned
selected live q, whereas the new binding correctly preserves authored/restored
State history. A host teleport and a restored State are different intents and
cannot be inferred from q bytes alone. This requires an explicit write/initialization
boundary, not restoring unconditional cloning in bind.

Required closure before claiming this property: introduce an explicit selected-
world/owned-row post-write history initialization operation, distinct from binding
and cache-only reset. Call it after supported owned-particle teleport/reset events
and before pre-step consumers. Preserve unselected worlds, foreign native rows,
flags, topology, commands and material edits. Add partial/empty/repeated cases and
non-default-stream coverage. Do not initialize history on every bind (which would
overwrite restored State) or on every native geometry-dirty event (which can concern
only foreign particles). This remains an open C2.1 lifecycle follow-up; no causal
attribution of the earlier numerical failures is made.

## Redundant particle rest copy: removal plan

`Model.i4h.particle_q0` is a duplicate in normal authored construction; the required
rest source is native `model.particle_q`. Exact inverse masses and i4h-semantic
flags have distinct numerical/ownership reasons; a normally byte-identical rest
copy does not. It is **not justified as a permanent allocation**. Keep the accepted
implementation unchanged for this certificate, then remove it as a bounded C2.1
follow-up before C2.2:

1. Compare exact native Model rest bytes against authored positions across variants,
   padding, empty domains, transforms and nonzero host prefixes/strides.
2. Bind Scene/main/body/part q0 views directly to `model.particle_q`. Preserve the
   declared/native lookup API with a cached native alias and identity validation;
   do not allocate another custom q0 backing array.
3. Remove the q0 custom Field, host authoring column, allocation links and duplicate
   Model bytes. Preserve State q_prev and its initialization template; binding
   still must not copy a restored State's live q over history.
4. Update the test that deliberately changes native Model defaults after finalize:
   reset will now use the native rest source, rather than the independent copy.
   Define default-edit/rebuild behavior explicitly; do not call this an unchanged
   behavior for that exceptional test.
5. Requalify ownership/reset/foreign-range/independent-State behavior and repeat the
   required frozen-source gates after the removal. Record new measurements as new
   samples without replacing the table below.

At the measured particle counts, the removable q0 Model buffer is **67,344 bytes at
1 world** and **4,310,016 bytes at 64 worlds** (N*12; allocation size, not a new
measured retained-memory reduction). It has no extra q0 State template because it
is Model-owned. q_prev initialization templates remain and must still be counted.

## Carried construction and memory evidence

The following table and accounting paragraphs are copied verbatim from the accepted
C2.1 record. They are not fresh measurements of the certificate harness. The six
legacy residue arrays remain **129,300,480 bytes (129.3 MB) at 64 worlds**. Once those
fields migrate, report each actual Newton template, live State and checkpoint
allocation separately; do not retain zero template bytes as a future assumption.

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


