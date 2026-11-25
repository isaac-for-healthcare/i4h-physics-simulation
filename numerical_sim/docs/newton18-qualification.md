# Newton 1.8 qualification gate

Stage A of the Newton ownership/Isaac Lab adapter migration, 2026-11-04.
**The follow-up Stage A exit condition is met:** main and patched Lab 3.0 probes pass, and the rescoped unchanged-stack baselines replay cleanly. Stages B and C have not started. The original gate failure below remains historical evidence; see the follow-up results at the end. Package
pins, the lockfile, numerical kernels, and the existing hosted runtime remain
unchanged. No replacement baseline is designated authoritative.

## Reproduced compatibility blocker

Isaac Lab 3.0 at `44044b0963001962abe5701133427e37b003fffe` fails during
`sim.reset()` against Newton `91fb115d685a96311b2f1bac5527754ee4e9c21b`
(`1.8.0rc1`). Its `NewtonBackend.__init__`, in
`source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:180`, snapshots
deformable metadata through `_{cloth,soft}_label` and matching particle ranges.
Newton 1.8 exposes `surface_label`/`volume_label` and the corresponding
`_{surface,volume}_particle_{start,end}` instead. The first failure is:

```text
AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'
```

This happens with two worlds containing ordinary particles and no native
elements, before a custom i4h solver could bind. It reproduces on CPU and CUDA,
both with and without an empty-element deformable observation range. Changing
i4h's solver configuration alone cannot repair this backend construction path.
A compatibility change to this Lab integration boundary must be qualified before
the proposed 3.0 adapter can be considered supported. The supplied Lab checkout
and its environment were not patched; a second supported Newton version was not
introduced.

## Implemented qualification runner

[`scripts/qualify_newton18.py`](../scripts/qualify_newton18.py) checks exact
source revisions, clean tracked sources, package versions, imported locations,
and the installed Newton Python source against the checkout. It records the
interpreter, full package inventory, device, capture setting, and GPU/driver
identity. Each probe runs in a fresh process. Reports include exit codes and
timeouts; a success message followed by a failed process exit is a failure.

Every run requires a new output directory and saves the exact probe source and
its hash. Earlier evidence is never replaced. Unit checks exercise failure
propagation, abnormal exits, timeouts, source checks, source snapshots, and
refusal to overwrite results.

The probes cover:

- Native XPBD particles, independent allocations, pinned particles, the discrete
  gravity trajectory, a masked cache reset, and CUDA graph replay.
- Native MuJoCo rigid stepping; on CUDA, MuJoCo-Warp graph replay, selected-world
  warm-start reset, preservation of unselected history, and an empty reset mask.
  MuJoCo CPU uses one world because its separate-world path simulates only the
  template world.
- Actual Lab simulation construction, allocation, XPBD stepping, and hard reset
  of a retained builder without appending entities again.
- An actual native `DeformableObject` over three-particle ranges with zero tet,
  triangle, and bending-edge counts. Two worlds have translated and rotated
  authoring transforms. Shape, ordering, world-space reads, stepping, asset reset
  preservation, and hard rebuild are checked.

These are dependency probes. They do not qualify the planned i4h adapter,
coupler, surgical contact output, force sensors, or assessment lifecycle. The
native observation probe authors world-space particle values; it does not prove
that the existing component-local i4h storage satisfies that contract. Existing
explicit component views remain the observation path until that binding is
qualified.

## Isolated environments and evidence

The three supplied reference checkouts were clean and matched their requested
revisions. Detached qualification worktrees and separate Python 3.12.12
environments live under `numerical_sim/.tmp_tests/newton18/`:

| Role | Worktree | Environment |
| --- | --- | --- |
| Newton RC1 standalone qualification | `newton` | `core-env` |
| Lab main `dc01ec356052f2acd04246752a92a55162f6cd68` | `lab-main` | `main-env` |
| Lab 3.0 `44044b0963001962abe5701133427e37b003fffe` | `lab-3.0` | `lab30-env` |

The candidate environments use Newton `1.8.0rc1`, Warp `1.17.0`, MuJoCo
`3.17.0`, and MuJoCo-Warp `3.17.0`. Lab environments use Torch `2.11.0`; the
standalone candidate uses Torch `2.14.0`, matching the existing standalone
environment. The original standalone environment remains at Newton `1.6.1`.
Full inventories are preserved; these are explicit isolated dependency
overrides, not claims that the unmodified package metadata resolves to RC1.

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver `615.71.09`.
Private raw evidence is retained in the directory above, following the existing
contribution guide's qualification-artifact policy:

- `evidence-index.json`: SHA-256 hashes of 182 retained evidence files.
- `original-identity.json`, `original-requirements.txt`, and installation logs.
- `candidate-prealignment-packages.json`, `aligned-requirements.txt`, and
  `verified-aligned-core-cuda/`: the initial standalone candidate inventory,
  subsequent alignment of all shared original package versions except Newton,
  and fresh native probes after alignment. Additional MuJoCo-family dependencies
  remain explicit. The initial numerical recordings precede this alignment.
- `original-baseline/` and `newton18-baseline/`: all six existing baseline cases,
  two environments, 60 steps, three fresh processes each; original product
  hashes and measured envelopes remain intact.
- `dependency-drift.json`, `dependency-drift.log`, and `compare_stacks.py`:
  comparisons authenticate every repetition against its manifest, verify source,
  hardware and workload equality, and use the original zero floors/factor 3.
- `original-selfcheck/` and `original-selfcheck.log`: independent old-stack
  replay checks on a separate copy of the baseline directory.
- `verified-{main,lab30}-{cpu,cuda}/`: exact probe source, stack identity,
  per-probe logs, and machine-readable reports.
- `original-suite.log`, `original-configured-suite.log`, `newton18-suite.log`,
  `aligned-extra-tests.log`, `original-synthetic.log`, and `candidate-synthetic.log`.
- `original-performance.{json,log}` and `newton18-performance.{json,log}`:
  sequential performance samples, taken after the earlier probes and suites
  finished. The standalone candidate shared packages were aligned first.
- `main-native-asset.log`: an attempted upstream asset test could not collect
  because its default launcher requests Isaac Sim/Kit, absent from these
  isolated kitless environments. This is unavailable coverage, not a passed
  asset suite. The explicit Newton observation probe above ran successfully.

## Results

| Check | Lab main CPU/CUDA | Lab 3.0 CPU/CUDA |
| --- | --- | --- |
| Exact dependency/source identity | Pass | Pass |
| Native particles | Pass | Pass |
| Native rigid solver | Pass | Pass |
| Lab particle stepping and hard rebuild | Pass | Fail at backend construction |
| Empty-element native observation probe | Pass | Fail at backend construction |

Original standalone tests: **509 passed, 227 skipped** without external-data
configuration; **689 passed, 47 skipped** with `I4H_DATA` set to the checkout's
data directory. Both original and RC1 CPU synthetic rod/tet/fluid/contact
examples passed. Qualification-runner tests: **7 passed**.

The initial RC1 standalone suite reports **682 passed, 50 skipped, 5 failed**:

- Three `test_component_layout.py` fixtures construct a builder with scalar
  `gravity[2]`. RC1 requires a three-vector and raises
  `ValueError: Expected gravity with shape (3,), got ()`. These fixtures are
  unchanged; the new dependency probes use the supported vector API.
- `test_chole_batched_matches_single` fails its existing isolation threshold:
  environment 1 differs by `0.0025036335` from its single-world run, against
  `0.2 * 0.0117864609` allowed by that test. No threshold was changed.
- An XR validation test failed because the initial isolated candidate omitted
  SlangPy. Matching the original optional packages (`slangpy`, `meshio`,
  `pyglet`, and their dependencies) resolves that omission. The formerly missing
  XR check plus the newly available VTU, containment, and GL-key checks report
  **17 passed** in `aligned-extra-tests.log`. The earlier full-suite failure is
  retained, and a fully passing RC1 runtime suite is not claimed.

Numerical comparisons against the original frozen envelopes:

| Case | Arrays per repetition | RC1 repetition failure counts |
| --- | ---: | --- |
| `aorta_centreline` | 489 | 0, 0, 0 |
| `aorta_shell` | 327 | 0, 0, 0 |
| `aorta_static` | 207 | 0, 0, 0 |
| `tissue_contact` | 303 | 0, 0, 0; all arrays bitwise unchanged |
| `two_worlds` | 981 | 2, 0, 2 |
| `chole_tet_med` | 1323 | 5, 14, 17 |

The failing fields include fluid corrections, grasp/volumetric-grasp history,
and some surgical reaction/centreline outputs. **The unchanged Newton 1.6.1
replay also fails the frozen envelopes for both `two_worlds` and
`chole_tet_med`.** These comparisons do not isolate Newton-induced drift from
existing run-to-run variability. No tolerances were widened, failures hidden,
or baselines replaced. Resolving this baseline repeatability issue is a second
prerequisite for asserting Stage B numerical parity.

## Performance sample

The existing `scripts/measure.py` ran each workload in a fresh process on
`cuda:0`, with CUDA graphs enabled, 20 warmup steps, 200 measured steps, zero
commands, and the existing distance units and solver settings. Original and
candidate measurements ran sequentially, without other qualification jobs
using the GPU. This covers the small tissue-contact fixture (8 particles per
world) and the static-aorta catheter (192 main particles per world), not
large-case or hosted performance.

| Case | Worlds | Original median ms | RC1 median ms | Change | Peak Warp mempool MB, original/RC1 | Construction s, original/RC1 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `tissue_contact` | 1 | 0.358 | 0.358 | 0.0% | 16.8 / 16.8 | 0.449 / 0.286 |
| `tissue_contact` | 64 | 0.539 | 0.529 | -1.9% | 16.9 / 16.9 | 0.442 / 0.288 |
| `aorta_static` | 1 | 8.060 | 7.740 | -4.0% | 16.8 / 16.8 | 0.484 / 0.333 |
| `aorta_static` | 64 | 10.768 | 10.893 | +1.2% | 20.0 / 20.0 | 0.502 / 0.340 |

These step-time samples meet the 5% limit; they do not waive the compatibility
and numerical blockers. Rounded mempool peaks are unchanged. Construction
samples include lazy initialization and cache/order effects; their improvement
has not been isolated to a particular Newton change. Large custom-domain
authoring does not exist yet and is unmeasured. No runtime implementation was
changed, so these measurements do not qualify a refactored runtime.

## Reproduction

From `numerical_sim/`, use the existing isolated environments without running
`uv sync` against the production workspace:

```bash
.tmp_tests/newton18/main-env/bin/python scripts/qualify_newton18.py \
  --stack main --newton-source .tmp_tests/newton18/newton \
  --lab-source .tmp_tests/newton18/lab-main --device cuda:0 \
  --out .tmp_tests/newton18/recheck-main-cuda

.tmp_tests/newton18/lab30-env/bin/python scripts/qualify_newton18.py \
  --stack 3.0 --newton-source .tmp_tests/newton18/newton \
  --lab-source .tmp_tests/newton18/lab-3.0 --device cuda:0 \
  --out .tmp_tests/newton18/recheck-lab30-cuda
```

The first command should pass the dependency probes; the second must currently
exit nonzero with the backend error above. Use another output directory on a
subsequent run. Substitute `--device cpu` for CPU qualification. `--stack core`
omits Lab probes and requires no Lab source path.

## Accepted gate repair and Stage C constraints

The follow-up authorizes a compatibility delta only in the isolated `lab-3.0`
worktree. Commit `e53cd288727a5ca3dae7187d1d415b4d32b67e11` on
`i4h-qualification-newton18` has parent `44044b0963001962abe5701133427e37b003fffe`.
The complete compatibility diff is saved in [newton18-lab30.patch](newton18-lab30.patch). Its code delta is:

```diff
- for family, kind in (("cloth", "surface"), ("soft", "volume"))
+ for family, kind in (("surface", "surface"), ("volume", "volume"))
- getattr(builder, f"_{family}_label")
+ getattr(builder, f"{family}_label")
  getattr(builder, f"_{family}_particle_start")
  getattr(builder, f"_{family}_particle_end")
- JointType.CABLE
+ JointType.ROD
```

The commit also updates the adjacent comment and adds the package's required
changelog fragment. Ruff lint and format checks passed for both touched Python
files. `patched-lab30-{cpu,cuda}/` records passing identity, particle, rigid,
host, and observation probes. This explicitly replaces the unpatched 3.0
qualification revision; the supplied checkout remains at its original pin.
These probes do not exercise a cable asset, so the enum edit has source and
lint validation rather than a claimed cable runtime qualification.

The baseline contract is now `physical-history-measurements-v2`. The explicit
`SYSTEM_FIELDS` list in `record_baseline.py` is the reviewed output contract;
unknown system types fail rather than silently losing coverage. It retains
particle state, rod frames/angular velocity, topology state, instrument/control
history, fluid cursors, thermal/coagulation history, visual paint/blood history,
and published measurements and reactions. Solver corrections, linear-system
scratch, contact candidate lists, BVH bounds, and self-clearing accumulators
are excluded. Complete live grasp records are sorted within each world/tool;
dead payload is zeroed because capture overwrites it before use. Duplicate
records retain their multiplicity. Consumed clip nodes survive canonicalization.
Hook entry and exit retain their distinct roles. No tolerance floor or factor
changed. Old archives and the earlier failed exploratory recordings remain
separate evidence; fresh complete v2 records and replays live in `gate-recheck/`.

The following are binding requirements for Stage C, not a Newton elasticity
migration in this gate repair:

1. Tet consumers bind connectivity, active masks, part mapping, and support
   counts through `scene.attr(domain, key)`. Surface, TetContainment, Thermal,
   Cutting, StretchBreaking, Grasp, VolumetricGrasp, Hook, and Centreline must
   not obtain that data through a Tet instance. Refs retain ordering only.
2. Reserve `i4h:tet/native_tet` as an int32 column initialized to -1, with no
   reference metadata and no populated mappings.
3. Author exact variant-specific rest geometry for each world in `add_world`.
4. Keep the five hooks and ordered loop in `I4hSolver.step`. Document predict,
   elasticity iterations, velocity finalization, deletion, and surface rebuild
   there, with a shipped-case ordering test; introduce no phase classifier or
   scheduler.
5. Define and test tool wrench frame, moment origin, and the applied correction
   used for accounting. Current source products accumulate `sum(m * dx)` and
   `sum((q_before_apply - tool_root) cross (m * dx))` in the component mechanics
   frame, after correction averaging. The reaction sign is negative; frame
   averaging divides by `substep_dt * step_dt`. Existing `ToolForce` applies
   source gains and an optional force/jaw cap; those conditioned outputs must
   be distinguished from raw coupling reactions. Preserve source impulse,
   moment, force, and jaw-load baselines so later truncation is visible. Geometry
   remains in existing case units, not implicitly SI.

Native tet/triangle/bending-edge arrays, Newton iteration callbacks, and
replacement of i4h elasticity remain deferred. These choices do not authorize
that later project or bypass the unchanged-stack replay gate.

## Gate recheck results

`gate-recheck/exits.json` records successful freezes and same-stack replays of
all six cases on both Newton 1.6.1 and RC1. Each freeze still uses three fresh
processes, two environments, and 60 steps. The v2 array counts are 255
(`two_worlds`), 390 (`chole_tet_med`), 63 (`aorta_static`), 87 (`aorta_shell`),
75 (`tissue_contact`), and 111 (`aorta_centreline`). Original archives remain
untouched. The older exploratory v2 runs aborted when the recorder source hash
changed during scope review; they are not the completed recheck evidence.

Cross-stack comparison authenticates each archive and retains these physical
failures against the original envelope. All slot-set comparisons pass:

| Case / RC1 repetition | Array | Maximum absolute difference | Allowed original envelope |
| --- | --- | ---: | ---: |
| `aorta_shell` / 0 | `masked_reset/body/aorta/q` | 5.364418e-6 | 0 |
| `aorta_shell` / 0 | `masked_reset/body/aorta/q_prev` | 5.006790e-6 | 0 |
| `aorta_shell` / 0 | `masked_reset/body/aorta/qd` | 5.149841e-4 | 0 |
| `aorta_centreline` / 1 | `masked_reset/system/lumen_containment/contact_depth` | 9.728596e-6 | 7.951632e-6 |
| `chole_tet_med` / 2 | `stepped/body/tissue/qd` | 0.7161712 | 0.7010650 |

The masked-reset differences are confined to unselected world 1; restored world
0 matches. The chole maximum occurs in world 0 before reset. These remain
physical drift candidates, with exact details in `physical-drift-detail.json`;
they are not waived as scratch or slot-order differences. Both stacks passing
one independent replay does not prove that all future samples will fit their
three-run envelopes. No numerical tolerance or original archive changed.
These residual differences must remain visible when Stage B parity is assessed.

The chole isolation test now records all three environments' `own / others`
ratios as JUnit properties before applying its existing assertions. The 0.2
threshold and integer equality checks are unchanged. Ten fresh-process runs
per stack were alternated serially, without other qualification GPU jobs.
The original stack passed 9/10 (one ratio 0.22326); RC1 passed 10/10. All integer
comparisons matched in all twenty runs.

| Stack | Mean of all 30 ratios | Median | Range | Ratios >= 0.2 |
| --- | ---: | ---: | --- | ---: |
| Newton 1.6.1 | 0.052919 | 0.041988 | 0.008402–0.223260 | 1 |
| Newton RC1 | 0.059968 | 0.037667 | 0.003792–0.199837 | 0 |

| Environment | Original min / median / max | RC1 min / median / max |
| --- | --- | --- |
| 0 | 0.00840 / 0.03201 / 0.08977 | 0.00379 / 0.01932 / 0.03870 |
| 1 | 0.01645 / 0.07496 / 0.22326 | 0.02498 / 0.06989 / 0.19984 |
| 2 | 0.01374 / 0.03489 / 0.11581 | 0.01101 / 0.04171 / 0.18648 |

Using the mean of each process's three environments as the independent sample
(10 per stack), an exact two-sided Mann–Whitney test gives U=50, p=1.0.
No distribution shift is detected; ten runs do not establish equivalence. The
previous isolated threshold miss is not evidence of a Newton regression.
`isolation-samples.json`, individual JUnit XML/logs, and
`isolation-distribution.json` retain all raw observations and integer outcomes.

The shared component-layout fixture now passes the full gravity vector. All
four layout tests pass on both stacks. The baseline and qualification-runner
contract tests pass (11 tests), and focused Ruff checks and `git diff --check`
pass. No full runtime-suite rerun is claimed for this follow-up.

**Exit:** main probes were already passing; patched 3.0 CPU/CUDA probes now
pass; all six rescoped original-stack baselines replay cleanly. Stage B planning
can proceed under the user's stated gate. The next bounded implementation is
the shared authoring/construction/binding split, followed by main adapter
lifecycle qualification and the patched 3.0 bridge. The Stage C constraints
above carry into that plan; no elasticity replacement is part of it.

## Stage C private helper dependencies

The storage schema uses three private helpers verified against Newton
`1.8.0rc1`, revision `91fb115d685a96311b2f1bac5527754ee4e9c21b`:

| Private helper | Purpose | Verification |
| --- | --- | --- |
| `Model._normalize_attribute_reference` | Normalize enum/string/custom-frequency references before builder mutation | Reference validation and two-builder merge/remapping tests |
| `ModelBuilder._add_custom_attribute_model_finalizer` | Install padding finalizer for empty native domains | Empty native PARTICLE tests, without inventing entities |
| `Model._attribute_frequency_count` | Obtain native row count in that finalizer | Empty-domain padded backing and logical-empty view tests |

These are explicit pinned-dependency assumptions, not stable public APIs.
Requalify them on a Newton update. Installed helper source paths and source-text
SHA-256 hashes are retained in
`.tmp_tests/stage-c-shared-domain-20261105/private-helpers.json`.
