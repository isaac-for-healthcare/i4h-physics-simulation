# Stage B closure

## Final closure record, 2026-11-05

**Stage B is closed for the qualified stack, with the explicit chole and coupling
limitations retained.** The first complete exclusive timing campaign and its
single cadence-test execution passed. All prerequisites are now passed or
explicitly accepted; [Stage C entry is eligible](stage-c.md#entry-decision-2026-11-05).
No Stage C runtime migration has started.

| Gate | Final disposition | Evidence |
| --- | --- | --- |
| Raw wrench, contact depth and optional surgical geometry exports | **Passed** | Solver-owned outputs, routing, lifecycle/capture checks and exports-on numerical comparisons qualified. Surgical geometry remains opt-in and disabled by default. |
| Hosted particle-grid release | **Passed** | Grid released on both hosts; 1/64-world snapshots and focused suites pass. q0/inv_mass sidecars remain until Stage C. |
| Captured-grid dependency correction | **Passed** | Local Warp 1.17.0 patch: analytic regression 9 passed; upstream suite 47 passed. Original binary/environments and every failed reproduction retained. |
| Chole reference disposition | **Accepted limitation** | User approved capacity-256 reference and chole-only majority-of-three. Historical 1/36 failure remains recorded. |
| Chole verification and isolation | **Passed** | Pre-change at the same capacity: 3/3; isolation: 3/3; corrected standalone/Lab 3.0/main exports-on replays: 3/3 each. |
| Numerical regression | **Passed** | Five other approved cases pass on all three hosts. Byte equality, canonicalization, factor three and zero floors unchanged. |
| Package/runtime and host qualification | **Passed**, documented skips retained | Broad corrected suite: 760 passed, 47 skipped, cadence deselected then qualified separately below. Latest package suite: 154 passed. Lab 3.0: 109 passed/9 skipped; main: 112 passed/6 skipped. |
| Packaging and CPU example | **Passed** bounded checks | Six package wheels/sdists, installed boundaries and CPU example qualified; corrected local Warp wheel integrity verified. Inherited profile metadata issues remain documented; no clean `pip check` claim. |
| Exclusive steady-state timing | **Passed** | All eight host/workload/world comparisons meet the unchanged +5% median limit; largest slowdown +2.392%. |
| Export overhead and allocation stability | **Passed** for the prepared workloads | Eight exports-on measurements; overhead -0.482% to +0.139%. All 20 workers show zero replay allocation growth. No production host synchronization added. |
| Construction, retained memory and peak memory | **Passed** reporting/ownership checks | Reported separately for every worker; released grid remains absent and expected sidecar differences remain. |
| Physics/render cadence | **Passed** | One uncontaminated execution of `test_physics_rate_independent_of_render`: 1 passed, unchanged assertions. |
| Coupling normal refinement | **Failed — accepted limitation** | 5.194896832108498% pre-change / 5.309711396694183% replacement against the unchanged 5% assertion. Both failures retained; explicitly accepted by the user. |

### First exclusive campaign

After the user's explicit termination instruction, the two remaining identified
catheter processes exited on SIGTERM; the vascular process had already exited.
The prepared runner then completed **20 workers**, with captured execution,
**20 warmup + 200 measured steps**, unchanged zero controls, tissue/catheter
workloads and **1/64 worlds**. Its **478 process observations** found no unrelated
compute process. No measured run or cadence test was retried.

| Workload | Worlds | Lab 3.0 vs paired pre-change | Main vs existing host reference |
| --- | ---: | ---: | ---: |
| tissue_contact | 1 | -0.013% | +0.024% |
| tissue_contact | 64 | +0.221% | -0.001% |
| aorta_static | 1 | -0.067% | -2.113% |
| aorta_static | 64 | -0.294% | +2.392% |

Main's reference remains historical and host-specific. Export overhead is a
separate measurement; negative overhead values represent measured variation,
not a demonstrated speedup from exporting. The prepared workloads bound this
overhead result; they do not represent every surgical geometry workload.

The [full timing report](stage-b-exclusive-timing.md) includes all medians,
exports-on overhead, construction times, retained bytes and peak bytes. Raw
samples, source/native hashes, process observations, cadence log/JUnit and the
passed decision are preserved in `.tmp_tests/stage-b-exclusive-20261105/`.
Earlier busy-device attempts and failed samples remain unchanged. Runtime
sources, dependency pins, case capacity and numerical thresholds did not change
during this campaign.

Stage B qualification applies to the [corrected local Warp build](warp117-hashgrid-capture.md)
and its isolated core/Lab 3.0/main environments. Stock Warp 1.17.0 still contains
the reproduced capture defect. The historical host parity failure's exact cause
remains unproven, and general bitwise fluid reproducibility is not claimed;
the documented corrected-build qualification passed.

The sections below preserve the sequence of prior decisions and their original
failures. This final record supersedes their pending gate states without erasing
their evidence.

## Chole reference accepted, 2026-11-05

The user explicitly approved the **chole-only majority-of-three policy**. The
capacity-256 candidate is promoted as an unchanged copy at
`.tmp_tests/stage-b-chole-acceptance-20261105/approved-v3/`. Its manifest records
the approval, the exact candidate hashes, capacity 256, the measured **1/36**
failure rate (run 31), and the replay policy. The candidate's original manifest,
failed promotion decision, all six recording samples and the outlier campaign
remain untouched. The other five case references are byte-identical copies.

Every chole qualification now runs exactly three fresh-process replays against
that approved manifest and requires at least two **complete replays** to pass.
It does not vote separately by product, stop early, replace failures or retry for
a better sample. All products, per-replay reports and worker logs are retained.
Worker/integrity errors invalidate qualification rather than being hidden by a
numerical majority. The other five cases retain one replay; historical manifests
without the explicit approved policy retain their original behavior.

`record_baseline.py --compare` and `qualify_stage_b.py` both apply the policy.
The numerical `compare()` function, byte equality for zero-noise products, slot
canonicalization, factor three, zero floors, scenario, snapshots and noise
envelope are unchanged. This accepts measured variability; it does not resolve
the outlier's cause or establish a population failure probability.

| Stage B gate | Current disposition |
| --- | --- |
| Host particle-grid release and qualification | **Passed** for the corrected Warp build; preceding measurements remain valid. |
| Chole reference variability | **Accepted limitation**, explicitly approved by the user. Capacity-256 reference promoted; historical 1/36 failure retained. |
| Pre-change verification against promoted chole | **Passed: 3/3**, fresh processes, preserved pre-change runtime with only the authorized capacity-256 case change. |
| Chole isolation | **Passed: 3/3**, fresh processes, unchanged 0.2 own/other ratio and exact integer checks. |
| Corrected replacement chole, exports enabled | **Passed: 3/3 each** on standalone, Lab 3.0 and main. |
| Exclusive timing, export overhead and cadence | **Pending free GPU.** Latest guard launched no workers; unrelated compute workloads were left untouched. |
| Coupling normal refinement | **Failed, accepted limitation:** 5.194896832108498% / 5.309711396694183% against the unchanged 5% assertion. |

Isolation ratios, in world order, were `[0.01355, 0.13379, 0.02236]`,
`[0.00803, 0.16771, 0.02286]`, and `[0.03683, 0.05853, 0.08608]`.
All phase-separation checks passed and every integer comparison matched.
The complete batched and single-world arrays are retained for each trial.
These bounded verification replays do not replace or extend the measured
36-run outlier campaign.

Validation: **154 package tests passed**, including 15 focused numerical-contract
and replay-policy tests. Integrity checks confirm all 984 indexed original inputs,
all candidate arrays and all five other references remain unchanged. There are
no runtime-source, dependency-pin or case changes in this acceptance round.

The evidence directory retains `approved-reference-index.json`,
`preserved-input-index.json`, `prechange-source-differences.json`,
`isolation-summary.json`, all four numerical replay groups, and
`final-policy-reassessment.json`. The final error-handling guard was added during
host qualification; both exercised harness revisions are preserved in
`harness-revisions.json`. Every saved replay was re-evaluated under the final
policy with the unchanged numerical comparator; no GPU trial was repeated.

For a future approved-reference replay, from `numerical_sim`:

```bash
python3 .tmp_tests/stage-b-hashgrid-fix-20261105/run.py core \
  scripts/qualify_stage_b.py --host standalone \
  --baseline .tmp_tests/stage-b-chole-acceptance-20261105/approved-v3 \
  --case chole_tet_med --exports --out /path/to/fresh-results
```

Chole's reference and verification gate is closed. Stage B's final closure
summary and Stage C entry acceptance remain deferred until exclusive timing,
export overhead and cadence qualify. **No Stage C migration has started.**
Earlier sections retain their historical dispositions and failed samples.

## Captured-grid correction, 2026-11-05

The confirmed Warp capture defect is corrected by the
[local Warp 1.17.0 source patch](warp117-hashgrid-capture.md). Dependency versions,
`uv.lock`, mechanics, capacity 256 and the numerical comparator remain unchanged.
The original environments and every failed sample are preserved. The corrected
core, Lab 3.0 and main environments are isolated under
`.tmp_tests/stage-b-hashgrid-fix-20261105/`.

| Remaining Stage B gate | Current disposition |
| --- | --- |
| Host grid release and qualification | **Passed** for the corrected build: both focused host suites and all four 1/64-world memory snapshots pass. The original failed suite sample remains evidence; its missing full tensors prevent definitive attribution of that exact invocation. |
| Chole reference disposition | **Accepted limitation; verification passed.** Superseded by the explicit approval and three-replay results above. |
| Exclusive timing, exports overhead and cadence | **Pending free GPU.** The prepared runner selects the patched replacement environments and preserves the original pre-change runtime. No timing result is inferred from shared-GPU checks. |

This is Stage B correction and qualification, not Stage C migration or final
Stage B acceptance. The final closure summary and Stage C entry acceptance remain
deferred until the pending gates close. The coupling refinement assertion remains
**failed but accepted**, at 5.194896832108498% / 5.309711396694183% against 5%.

### Corrective change and validation

Each grid build now publishes its descriptor by value from the existing final
native kernel. This removes the shared captured host staging address, including
when multiple builds of the same grid use different cell widths. It adds no
populated-grid kernel, steady-state allocation or production host synchronization.
The released Newton particle grid stays absent; `q0` and `inv_mass` stay as
independent i4h sidecars. No i4h runtime source changed in this corrective round.

| Check | Result |
| --- | --- |
| Analytic regression on stock Warp | **Failed as reproduced:** 8 failed, 1 passed. Covers float16/32/64, grouped/ungrouped and same-grid capture metadata. |
| Same regression on corrected Warp | **Passed:** 9 passed, including empty rebuild/query behavior. |
| Original three analytic reproductions | **Passed unchanged** with corrected binary. |
| Upstream Warp HashGrid suite | **Passed:** 47 CPU/CUDA tests; CPU-only native stub also compiles. |
| Lab 3.0 focused host suite | **Passed:** 109 passed, 9 skipped, including the original alternating standalone/host parity test. |
| Main focused host suite | **Passed:** 112 passed, 6 skipped. |
| Final package/runtime suite | **Passed:** 760 passed, 47 skipped, 1 deselected. Cadence remains excluded for its exclusive-window execution. |
| Five approved numerical references | **Passed** on standalone, Lab 3.0 and main, with exports enabled and unchanged comparison rules. |
| Chole candidate comparison | **Passed** once per host; candidate remains unpromoted and this is not an outlier-rate campaign. |
| Aorta memory snapshots | **Passed:** all four match the grid-release measurements exactly; zero step growth. |
| Captured fluid workload allocations | **Passed:** standalone/Lab 3.0/main, 20 warmup + 200 steps, zero allocation growth and stable output pointers; no timing claim. |
| Local native wheel and CPU example | **Passed:** 592 wheel RECORD hashes/lengths verified, correct local Linux tag, CPU example passed. |
| Timing-runner control flow | **Passed:** 9 fake-worker tests; no GPU timing evidence. Selected implementation identity was also verified for preserved and current sources. |

The initial broad run retained **757 passed, 47 skipped, 1 failed, 1 deselected**.
Its keyboard-symbol test opened an unnecessary pyglet X11 shadow window and
failed with `XlibScreen.is_primary` missing. One unchanged isolated run failed on
each original/patched environment. The keyboard-only test now temporarily
disables that shadow window while checking the same key mapping; it passes on
both environments. The original failure logs are retained. The parity test's
failure-only reporting also now saves full observations and names differing
fields, without changing its predicate or simulation sequence.

In the corrected diagnostic trajectory, fluid positions match through the first
six policy steps. The first recorded difference moves from step 1 to step 7,
and the first difference above 1e-4 moves from step 2 to step 10. Later fluid
positions still diverge (maximum 0.3246283233165741 case units); this fix does
**not** establish general bitwise fluid reproducibility. All 40 policy-observation
comparisons pass at the original tolerance; maximum observed difference is
6.020069122314453e-6. Commands match and returned host observations match live
state. The historical failed observation remains unavailable, so no exact causal
attribution or failure-rate claim is made for that old sample.

Both hosts retain the same memory values as the earlier grid release:

| Worlds | Live bytes | Peak bytes | Post-close bytes | Step growth |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5,858,226 | 16,822,188 | 16,895 | 0 |
| 64 | 18,360,135 | 20,683,037 | 16,895 | 0 |

The 16,777,304-byte live reduction versus the prior grid-retaining Lab 3.0 build
is preserved at both sizes. No sidecar migration occurred.

`uv pip check` is not claimed clean: the copied qualified profiles retain exactly
the original profile's metadata/optional-dependency issues (one on core, two on
Lab 3.0, three on main). The before/after logs distinguish those inherited issues
from native-wheel integrity. No resolver changes or extra dependencies were
introduced to suppress them.

The timing runner records the native Warp hash and accepts
`--replacement-env-root .tmp_tests/stage-b-hashgrid-fix-20261105`; the pre-change
worker continues using `.tmp_tests/newton18/lab30-env`. Its source-identity lookup
now loads the recorder belonging to the selected implementation, so the
preserved worker is not mistakenly checked against current source paths.
The cadence test uses the selected replacement core environment in that same
exclusive window. Construction/retained/peak memory and exports overhead remain
separate outputs; the 5% gate and no-favorable-retry policy remain unchanged.

The final exclusivity preflight still found unrelated compute PIDs 1509249 and
1509341. It returned **pending** and launched no timing or cadence worker; see
`.tmp_tests/stage-b-hashgrid-fix-20261105/exclusive-attempt/decision.json` and its
process observations. Both workloads were left untouched. The first complete
valid timing campaign remains outstanding.

Earlier sections below preserve the preceding measurements and unresolved
investigation state; this section gives the current corrected-build disposition.



## Host particle-grid release, 2026-11-05

The user accepted the preceding closure round: exports, lifecycle tests, the
outlier campaign and the memory analysis are complete and correctly recorded.
This follow-up remains **Stage B closure**. The final Stage B summary and Stage C
entry acceptance are deferred until the three remaining gates below close.

| Remaining gate | Current status |
| --- | --- |
| Release the unused host particle grid and qualify it | Release implemented; memory checks **passed** on both hosts at 1/64 worlds. Main host suite **passed**. Lab 3.0 host suite **failed** one standalone/host parity assertion; a bounded diagnostic pair did not reproduce it. Do not treat the suite as passed. |
| Chole reference disposition | **Pending user decision.** Capacity-256 candidate remains unpromoted; measured outlier rate remains 1/36. No majority-of-three policy has been selected or implemented. |
| Exclusive timing and cadence | **Pending free GPU.** Prepared campaign includes exports-on overhead, construction/retained/peak memory and one cadence-test execution in the same monitored window. The busy-GPU guard launched no measurement or cadence worker. |

### Released grid and measured memory

`ConstructionContext.bind()` now sets `model.particle_grid = None` before
`authoring.bind()` allocates the i4h runtime and constructs its static mesh/BVH.
This matches standalone. Both adapters bypass Newton's collision pipeline;
MuJoCo-Warp continues to own rigid contacts. Main's host code has no consumer of
this grid. `q0` and `inv_mass` remain independent i4h sidecars until Stage C.
No physics kernel, parameter, dependency pin or case capacity changed.

Four fresh processes reran the existing allocation-snapshot harness for
`aorta_static`, at 1 and 64 worlds, on Lab 3.0 and main. All four report the grid
absent after binding, **zero growth** between construction and 20 captured steps,
and 16,895 retained Warp bytes after close/GC. Both hosts have identical results:

| Worlds | Prior replacement Lab 3.0 live / peak bytes | Grid released, either host live / peak bytes | Live reduction vs prior Lab 3.0 | Remaining sidecars vs original pre-change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 22,635,530 / 24,959,314 | 5,858,226 / 16,822,188 | **16,777,304** | 3,072 bytes |
| 64 | 35,137,439 / 37,460,341 | 18,360,135 / 20,683,037 | **16,777,304** | 196,608 bytes |

The remaining differences equal `q0 + inv_mass`: 2,304 + 768 bytes at one world,
and 147,456 + 49,152 bytes at 64 worlds. One-world peak remains the earlier model
finalization peak; releasing the grid does not erase that high-water mark.
These are allocation measurements, not timing qualification on the shared GPU.

### Focused host qualification and retained failure

The focused suites cover environment/component/manager behavior, recording,
teleoperation, both adapters, native rigid `CONTACT_F`, custom outputs, eager and
captured stepping, and hard rebuilds. Assertions now also verify that the grid
stays released after rebuild while `q0`/`inv_mass` remain independent buffers.

- Main: **103 passed, 6 skipped**.
- Lab 3.0: **99 passed, 9 skipped, 1 failed**:
  `test_isaaclab_env.py::test_standalone_matches_host[two_worlds-newton]` failed
  its unchanged `torch.allclose(..., atol=1e-4)` final-observation comparison.

The full-suite failure log is preserved. One fixed diagnostic pair ran the same
unchanged test, once on the saved pre-grid source and once on the released-grid
source, retaining both final observation arrays before the assertion. Both
passed. Their largest differences were respectively 2.384185791015625e-7 and
1.1920928955078125e-7, both in `stretch_breaking.max_stretch`. The paired report
confirms grid retained before and released after. This does **not** erase the
failed suite sample, identify its differing field, or establish its cause. No
further retries, tolerance changes or numerical fixes were made.

Evidence lives in `.tmp_tests/stage-b-grid-release-20261105/`: the pre-edit source
archive/index, four memory JSON/log pairs and exact arithmetic summary, both host
suite logs, and the diagnostic pair's raw observations, reports and scripts.
The accepted prior closure evidence remains unchanged.

### Focused parity investigation follow-up

The [focused investigation](stage-b-parity-investigation.md) reproduced a
pre-existing captured-HashGrid ownership defect in pinned Warp 1.17.0 and in the
i4h Fluid path. The minimal test has an analytic neighbor-count expectation;
an earlier graph queries the later grid after a second capture. Both pre-grid
and released-grid traces expose fluid divergence, but the original final policy
assertion did not fail in the bounded diagnostic runs. Its original failing
field remains unavailable. Merely serializing the two rollouts also left fluid
divergence, so no test rewrite, mechanics change or dependency patch was made.
The host qualification gate remains **failed / unresolved**, with all original
samples and new diagnostic evidence preserved.

### Pending disposition and exclusive campaign

Await the user's chole decision. **Only if majority-of-three is chosen**: promote
the capacity-256 candidate with the measured 1/36 rate and policy in its manifest,
verify pre-change at the identical capacity, retain all three isolation trials,
and change replay policy for chole alone. The numerical comparator, manifest and
candidate are unchanged now; the other five cases retain their current policy.

The prepared timing runner now also executes
`test_app_loop.py::test_physics_rate_independent_of_render` exactly once, in a
fresh core process after the measurements, under the same exclusivity monitor.
It preserves the 20-warmup/200-step, 1/64-world, zero-control workloads and 5%
median limit, and reports exports-on overhead separately. The cadence outcome is
recorded independently from the timing comparisons; a skipped or contaminated
cadence run cannot pass. There are no automatic retries. Seven fake-worker
control-flow tests passed; they produced no GPU timing evidence. Ruff and
whitespace checks passed.

`exclusive-attempt/decision.json` records **pending**, with unrelated PIDs
1509249 and 1509341 still present. No timing or cadence trial was run in that
attempt, and no unrelated workload was altered. The prior cadence failure
(81 steps, required >85 for 17 frames) remains preserved pending its authorized
single execution in an exclusive window. Use a fresh result directory when a
window is available.

## Closure work, 2026-11-05

The current instruction accepts the normal-refinement limitation, preserves the
capacity-256 fix, and limits chole work to saved-evidence analysis and focused
reproductions. It supersedes the earlier open coupling disposition and any
suggestion to launch another outlier-rate campaign. Stage C remains gated.

| Gate | Status | Current evidence |
| --- | --- | --- |
| Chole investigation | **passed** (bounded investigation completed) | Saved runs analyzed; causal result **inconclusive**, with no mechanics patch. |
| Chole reference acceptance | **pending** | Capacity-256 run 31 remains failed; candidate remains unpromoted. Explicit disposition still required. |
| Coupling normal refinement | **accepted limitation; assertion failed** | 5.194896832108498% / 5.309711396694183% remain above the unchanged 5% assertion. No rerun or tolerance edit. |
| Output interfaces | **passed** | Custom raw wrench, containment depth/count, optional surgical records and counts implemented and tested. |
| Output and lifecycle qualification | **passed** focused/host checks | Standalone eager/capture, analytic invariants, reset/subsets, and both qualified Lab hosts pass. Numerical results below remain separate. |
| Exclusive timing | **pending** | Guarded campaign prepared; unrelated GPU workloads still active. No timing samples taken. |
| Full runtime regression | **failed** | Cadence assertion: 81 physics steps versus required >85 in 17 render frames. Boundary naming failure corrected separately. |
| Packaging / CPU example | **passed** | Six wheels and sdists, artifact boundaries, installed roots and CPU example passed. |
| Five approved numerical references | **passed** | All five passed with exports enabled in standalone, Lab 3.0 and main. |
| Chole candidate replay | **passed** single replay per host | Does not promote the candidate or erase the preserved 1/36 failure. |
| Allocation / synchronization | **passed** bounded checks | Nine fresh captured runs: zero Warp allocation growth, stable output pointers; publication adds no host synchronization. |
| Export overhead | **pending** | Prepared as a separate exports-on measurement in the exclusive campaign. |

The pre-export worktree is preserved in
`.tmp_tests/stage-b-closure-20261105/pre-exports-source.tar.gz` with SHA256
`source-index.json` (471 files), the original git status, and an index of 1,153
prior JSON evidence files. Earlier archives and failed samples are untouched.

### Chole saved-evidence diagnosis

`chole-analysis.json` and its script under that closure directory compare every
remaining capacity-256 run with run 0, plus capacity-8 outliers 14, 20 and 26.
No new chole outlier-rate trials were run.

Run 31 first differs in **recorded accumulated pair count at scenario step 79**:
world 0 grasper count 169 versus 167. The first retained final-event capsule count
difference is **step 84, substep 0, world 1, capsule 1**, 34 versus 30. Overflow
is byte-zero throughout and grasp counts are byte-identical. Engagement remains
step 82 and the decisive assertion remains step 120. These are first observed
measurement differences, not proven causal divergence points. The 34 other
passing comparisons first differ in pair counts at steps 77/79/82 and in recorded
capsule counts at steps 83–86. Early count differences therefore do not uniquely
identify the failed run.

At the final stepped phase, the world-1 grasper force difference (run 31 minus
run 0) decomposes as follows, in case force units. These paired differences are
not the gate's differences against its candidate reference.

| Source | x | y | z |
| --- | ---: | ---: | ---: |
| Contact | 23.286670 | 11.404645 | 5.945545 |
| Surface grasp | -7.242212 | -7.856063 | -1.672163 |
| Volumetric grasp | -0.800244 | -7.440019 | -11.136532 |
| Hook | 0 | 0 | 0 |
| Sum | 15.244214 | -3.891436 | -6.863149 |

The observed x difference is 15.244202; summed source differences reproduce all
force components within 0.000225 of float32 addition/scaling. The unchanged gate
still records maximum **16.604797** versus tolerance **7.384827**. Moment and jaw
impulse differences likewise occur in contact and both grasps; hook remains zero.
The saved analysis retains their vectors rather than assigning a single causal
source. Jaw load is the norm of the summed jaw shares and is not an additive
per-source scalar.

The source audit finds one atomic record per capsule/triangle pair, capacity
clamping at consumers, deterministic penetration/capsule/triangle tie-breaking
for surface grasp, duplicate-anchor suppression, and per-frame reaction clearing.
Volumetric capture claims slots atomically and can depend on arrival order when
full; saved capacity-256 run 31 does not fail its zero-noise slot products.
Contact and grasp reductions use the displacement actually applied after
averaging, with reaction accounting gated by active flags and positive inverse
mass. Producer and apply launches are adjacent; the evidence does not establish
an inactive/pinned transition leaving stale channel contents between them.
The analytic averaging check covers mass, root moment and scratch clearing.
No saved intermediate coordinates/corrections identify a minimal causal input,
so **no reproducible defect was established and no physics kernel was patched**.
Capacity-8 outliers also show count differences without changed grasp-count
timelines; their truncation cannot explain the byte-zero-overflow run 31.

### Verification of exports and lifecycle

Evidence is under `.tmp_tests/stage-b-closure-20261105/`. The focused physics run
passed **146 tests, 1 skipped**, covering the new outputs plus contact, grasps,
haptics, catheter root/guide balance and coupling. The final host suites include
all 15 output tests (analytic sign/scaling/root moment; applied averaging;
translated/rotated root-to-COM reaction; raw/conditioned separation; final-event
records, exact capacity, overflow, cuts and degenerate slots; multiple sources
and worlds; disabled/empty sources; partial/empty resets; captured lifetimes).

| Run | Result |
| --- | --- |
| Full standalone/package suite | **741 passed, 47 skipped, 2 failed** in 517.15 s |
| Corrected package boundaries | **24 passed** |
| Final Lab 3.0 host suite | **100 passed, 9 skipped** |
| Final main host suite | **103 passed, 6 skipped** |
| Wheel/sdist builds and artifact checks | **All six passed** |
| Installed roots and CPU synthetic example | **Passed** from outside the checkout |
| Focused Ruff and whitespace | **Passed** |

The full suite's first failure was the neutral-package identifier check on the
new export field `tissue_point_completed`. The public field is now
`surface_point_completed`; the corrected boundary suite and both final host
suites, including all export tests, passed. The second failure is
`test_app_loop.py::test_physics_rate_independent_of_render`: 81 steps across 17
render frames fails the unchanged `steps > 5 * frames` assertion. This sample
remains **failed**. No retry, threshold change, skip or causal attribution was
used to close it. The shared GPU is not valid exclusive performance evidence.

The first export-only development run also retained four empty-domain metadata
failures in `outputs-first.log`; handling absent zero-row Newton label attributes
fixed them, and `outputs-second.log` passed 10/10 before the additional invariant
and lifecycle tests were added. These earlier logs have not been rewritten.

The wheel smoke environment uses freshly built packages and the existing
qualified wheel environment's pinned third-party dependencies through a `.pth`
entry. An attempted entirely offline dependency installation could not obtain
the pinned `glfw==2.10.2` wheel; its failed log is retained. No dependency was
repinned or downloaded, and the prior environment was not modified. Installed
root checks verify that all six i4h packages load from the new wheel environment.

`preservation-check.json` verifies all **48 original v3 reference files**, **573
outlier-campaign files**, **256 capacity-qualification files**, **34 investigation
files**, and **1,153 prior JSON evidence files** unchanged. The original recorder,
coupling qualification script, lockfile and capacity-256 YAML are byte-identical
to the pre-export snapshot. `final-source-index.json` freezes the 193 implementation
source/case/qualification/lock files for the final comparisons.

### Final numerical and allocation results

The unchanged comparator passed all five approved references with all custom
exports enabled on **standalone, Lab 3.0 and main**. Each host also passed one
chole replay against the explicitly **unpromoted capacity-256 candidate**. All
18 products/comparison reports and logs are retained in the three `*-numerical/`
directories; their reports label each reference's disposition. No reference was
promoted, rewritten or enlarged. The preserved 1/36 failure remains the reason
chole reference acceptance is pending.

Nine fresh allocation-only processes (three hosts × `two_worlds`,
`tissue_contact`, `aorta_static`, each at two worlds) ran captured execution with
all exports, 20 warmup and 200 additional steps. All measured **zero retained
Warp allocation growth** and stable output pointers. They are not timing or
1/64-world performance qualification. The publication path contains only
preallocated output clears and launches; it introduces no production host
readback/synchronization and writes no physics buffer. Construction/peak memory
and export overhead remain part of the pending exclusive campaign.

`gpu-final.txt` confirms the two unrelated PIDs **1509249** and **1509341** are
still active after our verification processes exit. They were left untouched.
No timing pass is claimed. From `numerical_sim`, the prepared command is:

```bash
python3 scripts/qualify_stage_b_timing.py --out .tmp_tests/stage-b-exclusive-20261105
```

Use a fresh output directory; the runner refuses a busy GPU and preserves any
contaminated attempt as invalid. It never retries automatically.

**Entry decision: Stage C runtime migration remains gated.** Explicit chole
reference disposition, exclusive timing/export overhead and disposition or
qualification of the retained cadence failure remain open. Coupling refinement
is an accepted limitation with its failed 5% assertion still visible.
`closure.json` is the machine-readable gate record.

### Export implementation and timing preparation

The [output contract](component-contracts.md#stage-b-solver-observables) defines
four custom i4h kinds, stable world/system/tool/capsule/slot mappings, separate
raw and conditioned forces, recording-event versus completed-step geometry,
capacity/overflow, reset and subset behavior. Geometry is disabled by default.
Native `CONTACT_F` retains its rigid owner. Only output-layout metadata is added
to Model; no Stage C persistent-state migration is included.

`scripts/qualify_stage_b_timing.py` preserves the 1/64-world tissue/catheter
workloads, zero controls, capture, 20 warmup and 200 measured steps. It monitors
compute-process exclusivity every 250 ms and at run boundaries; any contamination
invalidates the campaign and is retained, without automatic retry. It compares
pre-change Lab 3.0 with replacement, reports main against its preserved host
reference, and records exports-on overhead separately. Construction, retained
memory, peak memory and replay allocation growth are separate products.
`timing-window-check/decision.json` records a busy-GPU refusal before any timing
worker launched. Both unrelated workloads were left untouched.


The following prior qualification record retains earlier decisions and measurements.
The closure table above gives the current gate disposition.

**The adapter migration is accepted numerically by the user (2026-11-04).**
Acceptance rests on the user's two criteria:

1. Deterministic products are byte-identical in every run.
2. Failure patterns are indistinguishable from accepted B1 under the frozen v3 gate.

This accepts the adapter migration, not all of Stage B. All failed samples,
earlier archives and dependency pins are preserved. The user explicitly
authorized increasing only `instrument_contact.records` in the shipped
`chole_tet_med.yaml`, using measured per-capsule counts. **The capacity-256 change
is accepted as a defect fix.** The previous absolute-noise stop rule was
miscalibrated: 2.46 against a recorded force of about 2800 is approximately 0.09%
relative noise, accepted by the user as atomic-ordering noise, not bistability.
The six-run envelope was never the failure; the outliers outside it were.

The replacement qualification runs 36 fresh processes per capacity: 256 against
its unpromoted six-run candidate and 8 against the original frozen v3 reference,
with the existing zero floors and factor three. Zero failures at 256 together
with reproduction of the historical outliers at 8 authorizes chole promotion,
pre-change verification with the identical capacity, then three fresh-process
isolation trials. If 256 still has approximately 16-unit outliers, overflow is
not their cause; retain the accepted capacity fix and compare each outlier's
grasp slot count, tool-force contact count and first decisive-grasp step with a
passing run. The other five case references remain fixed. Continue the ordered
investigations below after recording this result; the old noise stop rule no
longer applies.

Stage B remains open, in this order: chole overflow/noise qualification and
three fresh-process isolation runs; memory ownership explanation; one exclusive
GPU timing campaign; the bounded coupling-error and v2 documentation fixes;
paired coupling refinement for the user's disposition; then the requested raw
wrench/contact-depth and optional surgical geometry outputs and final regression
campaign. Stage C remains separate and has not begun.

## Outlier-rate qualification

The fixed campaigns completed **36 fresh processes per capacity**, with no
discarded or replacement runs. Capacity 256 replayed against the unpromoted
six-run candidate; capacity 8 used the preserved original YAML and frozen v3
reference. Both used the unchanged recorder comparator, zero floors and factor
three. The two imported package snapshots differed only in that YAML. Copies of
the existing `record_six.py` and `audit_chole.py` performed the loop, adding the
requested count/engagement diagnostics without changing the scenario or products.

| Capacity | Failing runs | Observed rate | Outlier maximum force differences against its reference |
| --- | ---: | ---: | --- |
| 256 | 1 / 36 | 2.78% | run 31: 16.60479736328125 |
| 8 | 3 / 36 | 8.33% | run 14: 14.197265625; run 20: 14.57916259765625; run 26: 15.4107666015625 |

All 36 capacity-256 trajectories had byte-zero overflow. Run 31 nevertheless
failed 19 force/moment/jaw products; its force tolerance was 7.38482666015625.
The control reproduced the historical force-outlier scale. Control run 14 also
failed the zero-noise volumetric-grasp sphere slots and other slot/visual
products; all of these failures remain recorded. The capacity-256 failure did
not include any zero-noise product.

**Overflow is not the cause of the remaining approximately 16-unit outliers.**
The accepted capacity-256 defect fix stays. The candidate is not promoted under
the zero-failure criterion, so conditional pre-change verification and isolation
trials were not reached. The measured rates do not establish a precise population
rate or prove that the capacities have different rates. Subsequent ordered
investigations proceed as explicitly requested.

For the sole capacity-256 outlier, compare run 31 with passing run 0 (world order
0, 1; only the grasper is nonzero):

| Diagnostic | Passing run 0 | Outlier run 31 |
| --- | --- | --- |
| Final grasp slot count | [2, 2] | [2, 2] |
| Final tool-force contact count | [3166, 3166] | [3170, 3312] |
| First positive grasp count, per world | [82, 82] | [82, 82] |
| First scheduled decisive-grasp assertion | 120 | 120 |

The grasp-count timelines were byte-identical, and both retained engagement
through the required hold. The outlier's second-world contact count increased
by 146 relative to that passing run. This identifies a contact-count difference,
not a demonstrated cause or a grasp-engagement timing difference. Step numbers
here are one-based scenario steps; initialization is stored separately as step 0.

All 72 products, trajectories, audit identities, comparisons and logs are under
`.tmp_tests/stage-b-outliers-20261104/`; `campaign-summary.json` records the fixed
counts and `outlier-details.json` records the paired diagnostics. Integrity
checks verified the original v3 archive's 48 indexed files and all 256 indexed
files from the earlier capacity qualification unchanged. No envelope, threshold,
mechanics or dependency pin was changed.

## Construction-memory cause

This is the preserved pre-release measurement. The Stage B follow-up above
releases the identified grid; it retains the sidecars and the original evidence.

Eight fresh hosted Lab 3.0 processes measured pre-change and replacement at 1
and 64 worlds for aorta and tissue. Snapshots record current and peak Warp
allocation after host authoring, model finalization, state allocation, scene
binding, static-mesh/BVH construction, capture, 20 steps, garbage collection and
close. These are allocation measurements, not timing qualification. Sources,
pins and the patched Lab 3.0 host match the earlier campaign; no allocation
policy or mechanics was changed.

The retained owner is **the host Newton model's `particle_grid`**. Newton creates
`wp.HashGrid(128, 128, 128)` during finalization: two cell-index buffers consume
16,777,216 bytes, and the device descriptor consumes 88 bytes. Measured total:
**16,777,304 bytes** in every workload. Pre-change `finish_model` released that
host-owned grid when i4h owned all particles. Replacement binding deliberately
leaves the host allocation intact. It persists through capture, stepping and
GC and is released with the model at close.

Replacement also owns `scene.main.q0` (12 bytes/particle) and
`scene.main.inv_mass` (4 bytes/particle), which pre-change aliased from the
model. Their 16 bytes/particle plus the retained grid explain **every byte** of
the measured live-memory increase. The tissue control has the same retained grid;
its earlier nearly unchanged peak did not mean unchanged live allocation.

| Case | Worlds | Live MB before / replacement | Peak MB before / replacement | Exact live increase, bytes |
| --- | ---: | --- | --- | ---: |
| tissue_contact | 1 | 0.006064 / 16.783496 | 16.779500 / 16.783820 | 16,777,432 |
| tissue_contact | 64 | 0.310669 / 17.096165 | 16.899767 / 17.122571 | 16,785,496 |
| aorta_static | 1 | 5.855154 / 22.635530 | 16.822188 / 24.959314 | 16,780,376 |
| aorta_static | 64 | 18.163527 / 35.137439 | 20.486429 / 37.460341 | 16,973,912 |

The static aorta mesh is unchanged between sources: 22,539 points (270,468
bytes) and 134,064 indices (536,256 bytes), plus 4,831,716 bytes owned by the
Warp mesh/BVH. Its construction temporarily uses another **2,323,800 bytes**,
which are gone by `Mesh.__init__` return. This transient overlaps the retained
grid only in replacement. At one world the pre-change peak was instead its
earlier grid/finalization peak, explaining why the reported peak increase was
8.137 MB even though the live increase is 16.780 MB. At 64 worlds the mesh build
sets both peaks, and their difference equals the live increase, 16.974 MB.

Selected aorta stages below use **current / peak bytes**; static mesh creation
is nested inside runtime binding. Full traces also retain tissue's particle and
contact BVH stages (tissue has no static mesh).

| Stage | Before, 1 world | Replacement, 1 world | Before, 64 worlds | Replacement, 64 worlds |
| --- | --- | --- | --- | --- |
| Host authoring | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| Model finalized | 16,808,362 / 16,818,250 | 16,808,362 / 16,818,250 | 18,758,772 / 19,397,400 | 18,758,772 / 19,397,400 |
| Scene arrays bound | 47,960 / 16,822,188 | 16,828,336 / 16,828,336 | 3,063,133 / 19,643,573 | 20,037,045 / 20,037,045 |
| Static mesh/BVH complete | 5,855,138 / 16,822,188 | 22,635,514 / 24,959,314 | 18,162,629 / 20,486,429 | 35,136,541 / 37,460,341 |
| Runtime systems bound | 5,855,150 / 16,822,188 | 22,635,526 / 24,959,314 | 18,163,397 / 20,486,429 | 35,137,309 / 37,460,341 |
| Captured, 20 steps | 5,855,154 / 16,822,188 | 22,635,530 / 24,959,314 | 18,163,527 / 20,486,429 | 35,137,439 / 37,460,341 |
| Closed and collected | 16,895 / 16,822,188 | 16,895 / 24,959,314 | 16,895 / 20,486,429 | 16,895 / 37,460,341 |

No live Warp growth occurred during capture/20 steps or after GC. After closing,
both sources return to identical residual use: 16,895 bytes for aorta and 1 byte
for tissue. The cumulative peak remains high after release; it is a high-water
mark, not retained runtime memory or a measurement of the allocator's reserved
VRAM. Evidence and exact arithmetic are in
`.tmp_tests/stage-b-investigations-20261104/memory-*.json` and
`memory-summary.json`. This is an explanation only; no broad memory refactor was
performed.

## Exclusive timing: awaiting a free GPU

The memory investigation is complete. The next timing campaign cannot yet start:
PIDs **1509249** and **1509341** remain active GPU catheter workloads, and a
current device query reports 100% utilization. They have been left untouched.
The user was asked asynchronously when an exclusive window will be available;
no window has been supplied. No shared-GPU timing sample has been substituted
for the requested exclusive campaign, and no 5% timing pass is claimed.

The pending campaign retains tissue and catheter at 1/64 worlds, 20 warmup and
200 measured steps, captured execution and zero commands, comparing preserved
pre-change 3.0 with replacement 3.0 and recording main as its host-specific
reference. Construction and peak memory remain separate. It must verify GPU
exclusivity throughout and use the first complete campaign against the unchanged
5% median limit, without rerunning to obtain a favorable sample. Current process
evidence is retained in `stage-b-investigations-20261104/gpu-before-timing.txt`
and `timing-window-observations.jsonl` under `.tmp_tests/`.

The independent small fixes and numerical refinement comparison below were
completed while the external timing prerequisite remained pending. Timing is
still an open Stage B gate.

## Bounded adapter and recorder fixes

Main's pure and coupled adapters now return an explicit reason for unsupported
`in_place=False` coupling. For supported in-place coupling, the coupled adapter
forwards validation to the configured rigid child, preserving its CPU-reset
restriction. Three focused checks reproduced the previous incorrect acceptance;
after the fix, all **8 main-adapter tests passed**, including actual hard
rebuilds, native rigid forces and observable ownership. Release 3.0 passed its
**5 adapter tests**, with the **3 main-only checks skipped**; the unchanged
recorder contract passed **4/4**. Focused Ruff and
`git diff --check` passed. The before/after logs are retained under
`.tmp_tests/stage-b-investigations-20261104/`.

Frozen v2 loaded each named case correctly before the local variable was
shadowed and unconditionally ran the same sinusoidal action loop for all six
cases. Shadowing changed its diagnostic label, not its action dispatch. The
[v3 documentation](stage-b-v3.md) now states this explicitly. No frozen v2 source
or archive was edited.

## Paired coupling refinement: accepted limitation (assertion failed)

The preserved pre-change and replacement qualification scripts ran once each in
fresh processes using the same patched Lab 3.0 environment. Their scenario
inputs and software inventories were saved and compared equal. The script
changes between implementations are the relocated coupling import and selection
of the bound control object; the workload, timestep, targets, material settings,
solver counts, averaging window and 5% bound are unchanged. Returned arrays were
saved before display rounding; no mechanical change or threshold adjustment was
made to pass this row.

| Evidence | World 0 normal refinement | World 1 normal refinement | Unchanged 5% bound |
| --- | ---: | ---: | --- |
| Historical pre-change | approximately 5.1% | approximately 4.5% | fail |
| Historical replacement | approximately 5.2% | approximately 4.5% | fail |
| New paired pre-change | 5.194896832108498% | 4.485569894313812% | **fail** |
| New paired replacement | 5.309711396694183% | 4.485549777746201% | **fail** |

Actual last-second mean normal forces at dt / (dt/2), in the qualification's
existing units:

| Source | World 0 | World 1 |
| --- | --- | --- |
| Pre-change | -2.901095390319824 / -3.051804304122925 | -1.568318486213684 / -1.6386665105819702 |
| Replacement | -2.9008631706237793 / -3.0548906326293945 | -1.5683176517486572 / -1.6386653184890747 |

Both runs exited 1 for that row; the other seven probe/catheter rows passed.
The new numbers are not represented as identical to each other or to historical
samples. The 2026-11-05 instruction explicitly accepts this limitation; the row remains
failed under its unchanged assertion. Raw
values, exact inputs, scripts' hashes and software identities are retained in
`coupling-prechange.json`, `coupling-replacement.json`, their `.rows.json` and
logs, and `coupling-pair-validation.json` under the investigation directory.

## Initial capacity qualification: superseded absolute-noise stop

The initial experiment eliminated contact-record overflow and measured the
remaining six-run force spread. It stopped under the now-retired absolute-noise
rule; it did not measure the outlier rate or establish the outliers' cause. Its
candidate and failed decision are preserved without rewriting their manifests.
The user's updated acceptance and outlier-rate criterion above supersede that
decision. Stage B remains open.

The only shipped numerical change is `instrument_contact.records: 256` in
`surgical/src/i4h_surgical/cases/chole_tet_med.yaml`, previously omitted and
therefore defaulting to 8. A complete v3 trajectory at capacity 8 measured a
maximum of 136 records on one capsule and a per-world step overflow as high as
1,115 dropped records. Capacity 256 is the next power of two above that observed
maximum. A second complete probe at 256 retained the maximum of 136 and had
byte-zero overflow. No other case setting or numerical mechanism changed.

The source was then frozen and six new processes recorded the unchanged v3
trajectory: two worlds, initialization step and full reset, 210 scripted steps,
and the masked reset. The decisive-grasp assertion remained enabled from step
120 onward and passed in every repeat. Each process retained all eight
substeps' per-capsule record counts for every step, plus the overflow counter
after construction, every step and both resets (214 observations per process).
The counts were copied into separately allocated diagnostic device buffers after
the original contact `substep_end`; only the qualification harness performs host
readback after complete steps/resets. No production instrumentation was added.
The retained substep counts reproduce the original overflow accumulator exactly,
including the overflowing capacity-8 probe. Reset count tables were also saved
and checked for zero in the restored worlds.

| Fresh repeat | Maximum records per capsule | Maximum overflow throughout trajectory |
| --- | ---: | ---: |
| 0 | 136 | 0 |
| 1 | 136 | 0 |
| 2 | 136 | 0 |
| 3 | 137 | 0 |
| 4 | 136 | 0 |
| 5 | 136 | 0 |

The unchanged recorder assembled the six repeats and computed its existing
maximum pairwise absolute-difference noise metric. Its product set,
canonicalization, zero floors and factor three were preserved. The historical
decision below used the noise itself without converting force units; the 0.5
column is retained as evidence of the retired, miscalibrated criterion:

| Recorded phase, `system/tool_force/force` | Six-run noise | Strictly below 0.5 |
| --- | ---: | --- |
| initial | 0.0 | yes |
| stepped | 2.46160888671875 | **no** |
| masked_reset | 2.46160888671875 | **no** |

Evidence is retained under `.tmp_tests/stage-b-finalize-20261104/`:

- `capacity8-probe/` and `capacity256-probe/`: complete probe products, trajectories
  and per-capsule maxima; the original YAML is also preserved separately.
- `frozen-source/` and `frozen-source-index.json`: the six package sources,
  unchanged v3 recorder, diagnostic harness and dependency lock.
- `repeat0/` through `repeat5/`: every candidate's products, full trajectory and
  source identity; corresponding logs retain process output.
- `candidate-chole-v3/`: unpromoted six-repeat archive. Its manifest records old
  and new YAML contents, SHA256 hashes, capacities, source identity, products,
  trajectory hashes and the failed promotion decision.
- `decision.json`, `evidence-validation.json` and their logs: the cutoff result
  and independent integrity/trajectory checks. `audit_chole.py`, `record_six.py`
  and `validate_evidence.py` preserve the qualification procedure.

Validation passed: all 390 product schemas and noise values were recomputed;
all 338 zero-noise products were byte-identical across the six repeats; all
products were finite. The focused recorder contract tests passed **4/4**.
The original frozen v3 archive's **48 indexed files are unchanged**, including
the original chole archive and all five other case references. All earlier failed
samples remain retained; dependency pins and the recorder are unchanged.
`git diff --check` passed. These checks validate this bounded experiment; they
do not constitute the deferred final Stage B regression campaign.

## Historical B1 acceptance

**B1 accepted by the user on the paired evidence (2026-11-04).** Acceptance rests on two criteria:

1. All 887 deterministic products are byte-identical in six B1 runs.
2. Holdout failure patterns under the frozen v2 gate are indistinguishable from pre-change.

The diagnostic six-run v2 envelope is **not promoted**. All v2 archives remain
untouched. Before host replacement, qualify `physical-history-measurements-v3`
using decisive chole grasp actions and settled containment references, with six
repeats from final B1, zero floors and factor three, then verify pre-change
against that same gate. The isolation comparison retains its 0.2 limit and uses
a majority of three trials, recording every ratio. The measured pre-change
flake rate was one failure in three trials.

The v3 prerequisite passed: six B1 repeats per case are frozen in
`.tmp_tests/stage-b-v3/baseline-v3-final/`; all three revised scenarios passed six
pre-change checks, and the final full six-case pre-change replay passed. The
isolation test passed two of three comparisons at the unchanged 0.2 limit.

Main and patched 3.0 now use the shared authoring, binding and numerical solver.
The alternate backend and `NewtonSim` have been removed. Stage B remains in
progress: the broader output and performance gates below are separate from this
adapter migration. A later main v3 holdout failed on eight chole force/moment
products, and the same pattern reproduced in preserved accepted B1. The host
numerical migration is now explicitly accepted on the criteria above; the paired
evidence has not been used to enlarge the archive. Stage C has not begun. Historical B1 evidence below and in
the follow-up report is retained, with its original failures.

The [v3 scenario qualification](stage-b-v3.md) records the revised workload,
retained trials and unchanged comparison policy.

The [2026-11-04 B1 follow-up](stage-b-b1-followup.md) records the paired replay,
four mechanism experiments, additional Lab cloner patch and latest verification.
The [refresh audit](stage-b-refresh-audit.md) found and fixed stale Centreline
geometry after host state edits. The frozen archive and comparison rules remain
unchanged; pre-change sources also fail some of its measured envelopes.

## Implemented construction and reset boundary

- `i4h_physics.author_case` loads host records and ordered system specifications.
  Instrument sphere/capsule layout preparation is host-only. `add_world` authors
  each world's exact variant rows before Newton finalization and records the
  particle ranges and component transforms on the retained builder's authoring
  object. Matching declarations are idempotent; duplicate worlds and conflicting
  case declarations fail.
- `bind(model, state, control)` creates fresh runtime systems, validates ownership,
  device, world count and equal-stride layout, and aliases current host particle
  positions and velocities without overwriting host state or model parameters.
  Runtime position history starts from the bound positions. Authored episode rest
  positions and flags remain separate from mutable model defaults.
- Exact authored inverse masses are private i4h parameters in the existing flat
  layout. Newton computes its native inverse masses from float32 masses; that
  reciprocal round trip changes 657 values in the two-world `aorta_shell` workload.
  Keeping the authored values removes this physical parameter regression without
  writing the host's finalized model. The memory estimate includes these buffers.
- Standalone `load_case` follows authoring, finalization, state/control allocation,
  binding, then Stepper construction. Host authoring/binding creates no Stepper.
  `I4hSolver` can be constructed unbound and bound once before configuration/use.
- `reset(flags=0)` preserves physical history and commands while refreshing or
  invalidating derived caches. Supported particle flags affect only owned rows;
  `None` selects those particle fields, and pure i4h rejects rigid/joint flags.
  `restore_episode` restores physical history, topology, command defaults and
  selected pending coupling reactions. Existing persistent parameter policies remain.
  Standalone reset captures call episode restoration explicitly.
- The five-hook loop and case order remain unchanged. Pure stepping rejects
  external collision input; foreign state/control and replaced bound buffers fail.

## Hosted ownership and adapter migration

Main uses distinct `I4hNewtonSolver` and `I4hCoupledNewtonSolver` adapters with
fixed capabilities. A component-owned `ConstructionContext` authors exact world
rows, retains immutable authoring with the builder, subscribes to lifecycle
events, and binds once after native contacts exist and before capture. Hard
rebuilds reuse authoring on the retained builder but create fresh systems and
bindings. Stop, failure, rebuild and close invalidate old handles. Only that
component's subscriptions are removed; there is no pending-scene registry.

Lab owns model/state/control allocation, stepping, contacts, capture and time.
`PhysicsBinding` exposes the selected state/control, scene, systems, measurements
and episode restoration; it has no hosted `step`, reset graph or clock. Main's
small manager subclass brackets Lab's inherited step with buffer validation and
host stream ordering because adapter hooks run inside capture. Assessment runs
in adapter `post_step`, once per physics sample, including decimation batches.

`lab30.py` isolates release-specific class manager access. Its builder, solver
construction, contact allocation, stepping, reset and cleanup hooks call the same
construction context and numerical solver. Binding occurs during solver creation;
output allocation finishes after native contacts and before capture. Assessment
runs in the solver-step hook. Outer solver substeps remain one on both releases.

`CoupledSolver`, mounts, reaction buffers and kernels now live in `i4h_physics`.
Their numerical order remains previous reaction, rigid internal substeps, mounted
poses, i4h substeps/iterations, next reaction. Config translation stays in the
host package. Pure i4h rejects native force-contact requests and external
collision inputs. Coupled adapters delegate native rigid contacts and legacy
`update_contacts` to MuJoCo-Warp. Requested Newton `CONTACT_F` observables use
separate wrapper-owned and child-owned containers, allocated before capture.
Replacing captured state, output arrays or containers requires rebinding.

Direct and manager environments, component handles, teleoperation, cameras and
recording use the binding. The alternate `physics.py` backend, `NewtonSim`, and
backend CLI switches are removed. Explicitly recorded legacy `backend: i4h`
metadata can be translated only with the existing complete physical-action
layout metadata; insufficient metadata fails. Packed action order and recording
payloads remain unchanged. Component position views remain the observation path;
native `DeformableObject` access is disabled for component-frame particles.

Current qualification and remaining gates are in
[the adapter evidence](stage-b-host.md).

## Dependencies and isolated installation

The declarations and lockfile now pin Newton `1.8.0rc1`, Warp `1.17.0`, MuJoCo
`3.17.0`, and MuJoCo-Warp `3.17.0`. MuJoCo is the physics package's `coupled` extra,
selected by the host package. Python remains 3.12. The isolated install's Newton
Python files match the qualified source at
`91fb115d685a96311b2f1bac5527754ee4e9c21b` byte for byte.

From `numerical_sim/`, create a new environment without changing a supplied one:

```bash
UV_PROJECT_ENVIRONMENT=.tmp_tests/stage-b/install-env uv sync --locked --group dev
I4H_DATA="$PWD/data" .tmp_tests/stage-b/install-env/bin/python -m pytest tests/runtime tests/packages -q
.tmp_tests/stage-b/install-env/bin/python examples/core_synthetic.py --device cpu
uv build --all-packages --out-dir .tmp_tests/stage-b/dist
.tmp_tests/stage-b/install-env/bin/python scripts/check_artifacts.py .tmp_tests/stage-b/dist
```

Use a fresh destination for a new qualification run. The test installation also
includes the reference optional packages `slangpy==0.43.1`, `meshio==5.3.5`, and
`pyglet==2.1.14`. Its inventory and imported source identity are saved privately in
`.tmp_tests/stage-b/`. The supplied core, Lab main and Lab 3.0 environments were
not synchronized or repinned.

## Preserved references and upstream finding

The current migration regression gate is frozen RC1
`physical-history-measurements-v3`, qualified before replacing host code. Stage
A's v2 archive and every diagnostic/trial archive remain untouched. Zero floors, factor three, and slot canonicalization are unchanged.
All five Stage A cross-stack physical differences remain unresolved; a Stage B
pass would not establish Newton 1.6.1/RC1 parity.

Before implementation, package sources, scripts and tests were archived under
`.tmp_tests/stage-b/prechange/sources.tar.gz`. Coupling qualification against
patched Lab 3.0's environment reproduced the existing normal-reaction refinement
failure: approximately 5.1% at half timestep versus the unchanged 5% limit.
The catheter checks passed. This was the direct coupling workload, not a Lab
host-timing qualification.

An actual pre-change Lab 3.0 environment test exposed an **additional upstream
compatibility requirement** beyond the committed patch at
`e53cd288727a5ca3dae7187d1d415b4d32b67e11`:

```text
isaaclab_newton/cloner/replicate.py:274
AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'
```

That cloner still enumerates `("cloth", "soft")` and accesses
`_{family}_label`; RC1 exposes `surface_label` / `volume_label` and corresponding
particle ranges. The Stage A probes exercised direct builder construction, not
this environment-cloner branch. The supplied and isolated qualified upstream
checkouts remain unchanged. The follow-up isolates this additional delta in
[`newton18-lab30-cloner.patch`](newton18-lab30-cloner.patch) on a new worktree based
on e53cd288. The failing environment test, actual hard rebuild and close/reopen
tests pass there. The replacement-adapter checks use this explicitly patched
worktree; they do not claim that unmodified e53cd288 passes the cloner path.

Main remains pinned at `dc01ec356052f2acd04246752a92a55162f6cd68`. There is no
pre-change main i4h adapter reference. No such comparison is claimed.

## Historical B1 qualification evidence

This section preserves the initial B1 results before the acceptance decision at
the top of this document. Current adapter checks are recorded separately above.

Raw logs and result archives are private under `.tmp_tests/stage-b/`. The first
B1 replay detected the reciprocal-mass issue above. After correcting it,
`aorta_shell` passed; chole measurement and centreline-reference fields still
exceeded the frozen envelopes. A replay from the preserved pre-change sources
passed all six cases. Those observed candidate failures remain evidence; they
are not removed by a later passing sample.

The initial full isolated-installation suite finished with **724 passed, 46 skipped,
1 failed** (`b1-suite-final.log`). The failure is
`test_clipping.py::test_tracker_and_clip`: the second press left the clip count
at one instead of two. Three subsequent fresh-process repetitions each on the
preserved source and B1 source all passed (`clip-comparison.json`). This does not
turn the failed full-suite run into a pass or establish its cause.

After the final declaration-validation adjustment, the focused construction and
ordering tests passed **17/17**. The CPU synthetic example passes. All six wheels
and source distributions build and pass the package/asset-boundary inspection.
The wheels were installed into a second isolated environment synchronized to the
workspace lock; all six installed-root checks and the CPU synthetic example pass
from outside the checkout. Focused Ruff checks and `git diff --check` pass.

All three initial B1 numerical replays used one identical source/software/hardware
identity and preserved all failures (`baseline-summary.json`,
`final-verification-exits.json`). Failure counts against the unchanged frozen
RC1 envelopes:

| Case | Replay 0 | Replay 1 | Replay 2 |
| --- | ---: | ---: | ---: |
| two_worlds | 0 | 0 | 0 |
| chole_tet_med | 11 | 1 | 0 |
| aorta_static | 0 | 0 | 0 |
| aorta_shell | 0 | 0 | 0 |
| tissue_contact | 0 | 0 | 0 |
| aorta_centreline | 5 | 0 | 5 |

Chole failures include grasp coordinates, contact/grasp reactions, moments and
conditioned tool forces. Centreline failures include particle positions/history
and reference geometry. These remain physical regression candidates; their
cause is not isolated. No tolerance, assertion or baseline was changed.
At that point the frozen gate had not passed and host replacement had not
begun. The later paired evidence and user acceptance supersede that decision;
the original failed samples remain preserved.

Standalone performance was measured sequentially from the preserved pre-change
sources and the fixed B1 sources, using the same RC1 environment, RTX PRO 6000,
driver 615.71.09, captured stepping, zero commands, 20 warmup steps and 200 timed
steps. Every final array was finite. Median milliseconds:

| Case | Worlds | Before | B1 | Change | Construction seconds before/B1 | Peak Warp MB before/B1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tissue_contact | 1 | 0.351 | 0.358 | +2.0% | 0.276 / 0.271 | 16.8 / 16.8 |
| tissue_contact | 64 | 0.526 | 0.535 | +1.7% | 0.271 / 0.272 | 16.9 / 16.9 |
| aorta_static | 1 | 7.378 | 7.386 | +0.1% | 0.314 / 0.318 | 16.8 / 16.8 |
| aorta_static | 64 | 10.183 | 10.153 | -0.3% | 0.323 / 0.436 | 20.0 / 20.2 |

These standalone timing samples meet the 5% limit. They do not qualify hosted
performance or waive the numerical/surgical failures. The private authored
buffers increase the 64-world catheter peak by approximately 0.2 MB. Construction
time is reported separately and has not been given a passing performance claim.
No new per-step device allocation or host synchronization was added.

## Stage C boundaries

No elasticity replacement, native surgical topology, iteration callbacks, native
coupling migration, persistent-state schema migration, named commands, particle
force input, rod-material conversion, or case/recording schema conversion was
introduced. Carry forward `scene.attr(domain, key)` for tet consumers,
`i4h:tet/native_tet = -1` without reference metadata, exact per-world rest
authoring, five-hook ordering, and the existing raw wrench sign/frame/origin and
applied-correction contract. Wrench observables and analytic output qualification are now implemented and
covered by the closure evidence above; exclusive timing and explicit chole
reference disposition remain outstanding.
