# B1 follow-up: paired replay, refresh and Lab cloner

The 2026-11-04 follow-up preserves the frozen RC1 archive, zero tolerance floors,
factor three, slot canonicalization, dependency pins and qualified checkouts.
Raw results, source copies, commands and manifests are under
`.tmp_tests/stage-b-followup-20261104/`. No adapter replacement has started.

## Discriminating experiment

Three fresh repetitions of all six cases ran on each preserved source, alternating
source order by round, with the same RC1 core environment, GPU and recorder.
Each case ran in a new process. B1 was copied before the refresh correction so
that implementation work could not change the paired experiment midway.

Maximum pairwise absolute spread across the three fresh runs:

| Product | Pre-change | B1 before refresh fix | B1 / pre-change |
| --- | ---: | ---: | ---: |
| Chole tool force | 21.894409 | 18.871948 | 0.862 |
| Chole jaw load | 13.040649 | 11.097839 | 0.851 |
| Centreline aorta positions | 0.00554293 | 0.01163506 | 2.099 |
| Centreline reference samples | 0.00588121 | 0.01368256 | 2.326 |

The earlier apparent threefold chole spread increase did not reproduce. Both
sources exhibit a large chole outlier absent from the frozen three repeats.
Maximum absolute differences of sample means were 1.314 (force), 0.810 (jaw),
0.00864 (aorta positions), and 0.00998 (reference samples). With three samples and
rare outliers, these are descriptive values, not an equivalence test or evidence
of a statistically established bias.

Failures against the **unchanged frozen gate**, by repetition:

| Case | Pre-change | B1 before refresh fix |
| --- | --- | --- |
| two_worlds | 0, 0, 0 | 0, 0, 0 |
| chole_tet_med | 19, 0, 0 | 0, 3, 17 |
| aorta_static | 0, 0, 0 | 0, 0, 0 |
| aorta_shell | 11, 0, 0 | 0, 0, 0 |
| tissue_contact | 0, 0, 0 | 0, 0, 0 |
| aorta_centreline | 0, 0, 0 | 5, 0, 5 |

All 887 products classified as zero noise still match byte for byte in these
three B1 runs. **That assertion no longer holds for the preserved pre-change
source:** one chole run changed `instrument_contact/impulse_count` and
`tool_force/count`; one shell run changed the masked-reset catheter q, qd,
q_prev, rod frame and omega products. A finite sample's zero spread did not prove
determinism. These failures remain in the evidence. They are not silently
reclassified as noisy products or given a tolerance floor.

The original extracted pre-change snapshot omitted the unchanged standalone
package. All numerical package files match the frozen manifest; all 64 omitted
standalone source files match the B1 snapshot byte for byte. Baseline replay does
not use standalone rendering; suite-prefix tests resolved that unchanged package
from the installed editable checkout. A complete, separately retained pre-change
copy includes those verified files for the subsequent holdout. The original
snapshot and archive remain untouched.

## Four isolated mechanism reversions

Each experiment used its own copy of the pre-refresh B1 snapshot, changed one
mechanism only, and ran both noisy cases three times. All initial recorded
products remained byte-identical to the unmodified B1 initial products.

| Experimental reversion | Chole failure counts | Centreline failure counts | Aorta spread | Reference-sample spread |
| --- | --- | --- | ---: | ---: |
| Seed q_prev from model instead of state | 3, 0, 0 | 0, 0, 0 | 0.00651747 | 0.01313499 |
| Store exact authored inverse mass in model array | 0, 15, 2 | 5, 5, 0 | 0.01157077 | 0.01359048 |
| Template plus replicate construction | 1, 0, 16 | 0, 0, 5 | 0.01146148 | 0.01346066 |
| Old episode reset dispatch without extra cache refresh | 6, 6, 0 | 6, 5, 2 | 0.01450167 | 0.01647413 |

The inverse-mass experiment preserves exact numerical values; it does not
reintroduce reciprocal-rounding changes. The template experiment rejects unequal
world variants and is restricted to these two identical-variant workloads. The
reset experiment changes episode dispatch only, not the five-hook numerical loop.
These experimental changes are not production fixes and are not promoted.

No reversion establishes a cause or consistently removes failures in both cases.
Three passing centreline samples with model-seeded history do not justify using
model defaults when binding an externally moved state. The tested standalone
states start at the same positions, and their initial history bytes match.

## Refresh and clipping

The [reset audit](stage-b-refresh-audit.md) records every system reset override
and the next read/write of its scratch or history. It found a missing Centreline
refresh: clipping can read stale positions/counts after external host edits.
The new regression fails without the fix and passes with it, checking empty,
partial and repeated selections, removed rings and preserved history. A separate
binding test verifies tracking of current host positions. Graph candidate
invalidation has added preservation coverage.

The clipping experiment includes the runtime suite's entire file prefix through
`test_clipping.py`, in the same process. Both sources passed **3/3**, each run
reporting 98 passed and 1 skipped. Thus the observed target-test failure rate is
0/3 per source in this context. The previous isolated 0/3 per source and previous
full-suite failure remain separate evidence; a zero observed rate from three
runs does not resolve the flake.

## Lab 3.0 compatibility delta

The new isolated branch `stage-b-rc1-cloner-20261104` is based on unchanged
`e53cd288727a5ca3dae7187d1d415b4d32b67e11`. Its additional change is saved as
[`newton18-lab30-cloner.patch`](newton18-lab30-cloner.patch), including the required
upstream changelog fragment. It enumerates surface/volume, uses public family
labels and retains the corresponding private particle-start arrays.

Using the existing Lab 3.0 environment with imports verified to resolve from the
new worktree, all three focused tests pass:

- `test_spaces_steps_and_time[newton]` (60 action steps, observations and clock).
- `test_hard_reset_rebinds[newton]` (actual model/solver rebuild and continued stepping).
- `test_close_and_reopen[newton]` (stale-handle rejection and a new environment).

These qualify the narrow legacy-host compatibility correction, not the unbuilt
Stage B adapters or their full lifecycle matrix. The patch also passes Ruff and
`git apply --check` against the preserved e53 checkout. Neither supplied checkout
nor environment was changed, and no additional dependency pin move was made.
The existing pin change remains in place; final stack qualification is pending.

## Full-suite follow-up

After the Centreline fix, the complete package/runtime suite reports **727
passed, 46 skipped, 1 failed**. The clipping tests, new refresh/binding tests,
construction/ordering, graph-reset preservation and coupling tests passed.
The remaining failure is `test_chole_batched_matches_single`: world 1 differed
from its independent run by 0.00275970 versus 0.01191652 from the other worlds,
a ratio of approximately 0.232 against the unchanged requirement below 0.2.
This is retained as a failure, not relabeled a pass because other runs pass.
The Stage A isolation test itself is unchanged.

Another simulation process from the separate `i4h-sim-rod-physical` worktree was
resident on the GPU during the follow-up. Its start time precedes these runs;
residency and utilization observations are saved in `gpu-residency.json`. No
claim of exclusive-GPU performance or attribution of numerical failures to that
process is made. It was not stopped or modified.

## Independent holdout and acceptance decision

Three additional fresh-process runs per source and case were kept separate from
calibration. The complete pre-change source copy is verified byte for byte
against **every** source entry in the frozen manifest. The final candidate source
includes the Centreline refresh correction. Against the unchanged frozen gate:

| Case | Pre-change holdout failures | Final B1 holdout failures |
| --- | --- | --- |
| two_worlds | 0, 0, 0 | 0, 0, 0 |
| chole_tet_med | 19, 0, 0 | 0, 0, 0 |
| aorta_static | 0, 0, 0 | 0, 0, 0 |
| aorta_shell | 0, 0, 0 | 0, 0, 0 |
| tissue_contact | 0, 0, 0 | 0, 0, 0 |
| aorta_centreline | 0, 5, 5 | 1, 5, 0 |

All 887 frozen zero-noise products remain byte-identical in the three final B1
holdouts. The old chole holdout again violates the two supposedly zero-noise
contact-count products. The centreline failures now reproduce on the old source
as well, weakening the earlier suggestion that only B1 enlarged the spread.

A **diagnostic**, separately retained six-repeat archive uses the original three
pre-change repeats plus the first three fresh pre-change repeats. It is under
`.tmp_tests/stage-b-followup-20261104/diagnostic-prechange-six/`, with per-product
hashes, repeat provenance and explicit unaccepted status. No holdout sample or
B1 sample enters its calibration. All 36 independent holdout case runs (six
cases × two sources × three repeats) pass that diagnostic envelope, with zero
floors and factor three unchanged.

This supports the conclusion that the original three-repeat envelope
underestimated existing variability. It does **not** make recalibration a small
administrative change: the chole force allowance would increase from about
0.886 to 65.683, jaw load from 0.504 to 39.122, and two contact-count products would
change from exact byte equality to allowing 30% differing entries. Five shell
products would also acquire nonzero measured noise. These changes are explicitly
listed in `holdout-summary.json` and have not been promoted to the accepted gate.

The isolation follow-up ran the unchanged target test three times each on the
pre-change, pre-refresh B1 and final B1 sources. Observed failures were **1/3,
0/3 and 0/3**, respectively. The pre-change failure had an own/other-world ratio
of 0.401 against the unchanged 0.2 limit; integer topology comparisons matched
in all nine runs. The full-suite B1 failure remains recorded. This demonstrates
an existing intermittent test failure, not a consistently passing suite.

**Decision: hold B1 at the acceptance boundary.** The missing refresh is fixed,
the narrow Lab 3.0 test gate is green, the clipping rate and reset audit are
recorded, and the noise experiment has an answer: the original envelope misses
pre-existing variability. The current candidate still fails the selected frozen
centreline gate, and the full suite has an unwaived intermittent failure. Do not
promote the diagnostic thresholds or start host replacement as if those gates
passed. A numerical acceptance change must explicitly account for the enlarged
force/count allowances and the isolation-test policy. No baseline, test limit,
dependency pin, five-hook order or Stage A cross-stack finding was silently
changed. Stage C remains separate.

Final supporting checks pass: CPU synthetic example, focused Ruff, whitespace
checks, all six package wheel/source builds and artifact-boundary checks. The
preservation audit verifies the original four Stage A working files, dependency
manifests/lockfile, all frozen reference/repetition files and the clean qualified
Newton, Lab main and e53 Lab 3.0 checkouts. No commit was made. Artifact hashes
and final source hashes are saved with the private evidence index.
