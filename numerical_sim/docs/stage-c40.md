# C4.0 — Formatted baseline, C3.3 follow-up qualification, timing and inventory

**The frozen campaign is clean and the timing gate failed.** All 36 reference groups and all three suites passed on the formatted source, which fully qualifies the C3.3 follow-up. The exclusive timing campaign is valid, and `tissue_contact` at 64 worlds exceeds the +5% limit on both hosts. Nothing was tuned or rerun. **Stop for the user's decision before C4.1.**

C4.0 changed no ownership, kernel, hook order, capacity, comparator, tolerance, reference or dependency pin. The three retained environments are unchanged. Nothing was pushed.

## Commits

| Commit | Content | Hooks |
| --- | --- | --- |
| `c9339e9` | Hook exclusions for dependency patches and the stage A to C3 reports; C3.3 disposition; C4 plan | pass |
| `7a5291b` | Import ordering in 67 files; allocation inventory re-anchored | pass |
| `02ea7f0` | Timing harness options; rod construction script promoted to `scripts/` | pass |

The campaign and the timing ran on `02ea7f0`. The commit that adds this report contains documents only.

## Preservation and formatting

- The tree matched the latest C3.3 review manifest before any edit: 89,072 files, no mismatch ([preflight][preflight]).
- `trailing-whitespace`, `end-of-file-fixer` and `markdownlint` now skip `docs/*.patch` and the historical reports by name. Reports from C4.0 on are linted. After the change the hooks modified no file.
- isort ran on tracked files only, never on a directory, so nothing under `.tmp_tests` was rewritten.
- [Format audit][format]: for each of the 67 files the AST without imports is identical, and every scope imports the same names. 36 files change the order of modules.
- Six production files reorder imports across packages: `i4h_physics/__init__.py`, `coupling.py` and `sim.py` (order of `newton` and `i4h_physics` modules), and the Lab adapters `newton_solver.py`, `lab30.py`, `newton_manager.py`, which now import `i4h_*` before `isaaclab`. No order pin was needed: every package imports in all three environments, and the development selection and frozen suites pass.
- The allocation inventory was re-anchored: 511 sites, 18 line moves, 26 files rehashed, no site added, removed or reclassified. Its stale `site_count` (587) is corrected to 511. `scripts/check_stage_c_inventory.py` passes.

## Frozen campaign

The [certificate][cert] covers **89,087 files** at commit `02ea7f0`, manifest SHA256 `9e64fd1f9d060f7d6404646d1404e661afe79041ae359fbc5de264c613a8c2a1`. The independent auditor verified 85 startup and exit pairs and 104,521 file-backed import observations, and recomputed every comparison from the retained arrays. One campaign, no retries, no reference promotion.

| Suite | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Full core | 1,096 | 49 | 0/0 |
| Lab main six-module selection | 121 | 5 | 0/0 |
| Patched Lab 3.0 six-module selection | 118 | 8 | 0/0 |

The development selection on the same commit gave the same counts. Neither known intermittent (clipping count, `two_worlds` fluid count) recurred; that does not explain them.

| Environment / mode | two_worlds | chole_tet_med | aorta_static | aorta_shell | tissue_contact | aorta_centreline |
| --- | --- | --- | --- | --- | --- | --- |
| standalone-captured | pass | pass (2/3) | pass | pass | pass | pass |
| standalone-eager | pass | pass (3/3) | pass | pass | pass | pass |
| main-captured | pass | pass (3/3) | pass | pass | pass | pass |
| main-eager | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-captured | pass | pass (3/3) | pass | pass | pass | pass |
| lab30-eager | pass | pass (2/3) | pass | pass | pass | pass |

**36/36 groups, 46/48 replays, 16/18 chole trials.** The two failed trials are standalone-captured trial 1 and lab30-eager trial 0, each with 19 failed products in the usual families (grasp 6, instrument_contact 6, tool_force 4, volumetric_grasp 3). Both groups pass the unchanged 2-of-3 policy. Their reports and arrays are retained under the certificate directory.

All six `aorta_shell` groups passed. The largest stepped catheter-impulse difference is 1.05e-05 against the 4.63e-05 threshold. The certificate stores the exact stepped and masked-reset values for C3.1, C3.2, C3.3 and C4.0. Passing here does not establish a cause for the earlier failures.

### Construction (one sample each)

| Workload | Worlds | Seconds | C3.3 follow-up seconds | Retained Warp bytes | Peak Warp bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| chole_tet_med | 1 | 1.129 | 0.924 | 27,823,129 | 28,029,373 |
| chole_tet_med | 64 | 15.000 | 16.028 | 1,219,312,846 | 1,232,518,132 |
| aorta_shell (rods) | 1 | 0.613 | 0.089 | 3,354,144 | 18,032,112 |
| aorta_shell (rods) | 64 | 0.652 | 0.708 | 172,785,410 | 205,359,656 |

Every byte count, shape and layout in all four reports equals the C3.3 follow-up sample; only timings and device pointers differ. The six residue arrays remain 2,020,320 and 129,300,480 bytes with no template or checkpoint copy. The slower one-world samples differ in one phase only, runtime binding: 0.006 to 0.530 s for rods and 0.201 to 0.411 s for chole. The rod log shows a single cached kernel module load of 524 ms in that phase; all other phases match the C3.3 follow-up within 3 ms. The 64-world samples are not slower. The chole script's `variant` label still reads C3.3 because the script is unchanged.

**This campaign is the full qualification of the C3.3 follow-up source** (public storage registration, `Control.clear` rejection): full suites, references and construction all ran on it.

## Exclusive timing campaign

Run once on the frozen commit after the campaign, with the manifest verified before and after. 16 workers, all valid; 359 process observations with no unrelated compute process; zero replay allocation; cadence test passed. The worker and its measured loop are unchanged (20 warmup, 200 measured steps, captured, zero controls). [Decision][decision]: **failed**.

Gate: median step time against the references Stage B was gated on, limit +5.0%.

| Workload | Worlds | Lab 3.0 reference ms | Lab 3.0 ms | Change | Main reference ms | Main ms | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tissue_contact | 1 | 0.362472 | 0.376695 | +3.92% | 0.361239 | 0.378598 | +4.81% |
| tissue_contact | 64 | 0.529243 | 0.569134 | **+7.54%** | 0.528342 | 0.569247 | **+7.74%** |
| aorta_static | 1 | 7.789783 | 7.405409 | -4.93% | 7.963070 | 7.409723 | -6.95% |
| aorta_static | 64 | 10.560836 | 9.819630 | -7.02% | 10.270203 | 9.821954 | -4.37% |

Facts that bound the failure, without attributing it:

- The excess is an absolute cost of about 0.040 ms per step at 64 worlds and 0.014 to 0.017 ms at one world, on a case with eight particles per world. The catheter case is 0.38 to 0.74 ms **faster** per step.
- Against Stage B's own results the picture is the same (tissue_contact 64 worlds: +7.30% and +7.74%).
- Export overhead is at most +0.55%.
- Standalone, reported and not gated. It ran directly after the hosted campaign, bracketed by two GPU process snapshots that both show no compute process, but it was not continuously monitored, and another session on this machine runs short GPU jobs:

| Workload | Worlds | Stage A ms | B1 ms | C4.0 ms |
| --- | ---: | ---: | ---: | ---: |
| tissue_contact | 1 | 0.351 | 0.358 | 0.369 |
| tissue_contact | 64 | 0.526 | 0.535 | 0.537 |
| aorta_static | 1 | 7.378 | 7.386 | 7.395 |
| aorta_static | 64 | 10.183 | 10.153 | 9.907 |

At 64 worlds standalone `tissue_contact` is 2.1% over Stage A while hosted is 7.5% over its reference, so most of that excess appears only when hosted. At one world standalone is 5.1% over Stage A, about the same absolute cost as hosted (0.014 to 0.018 ms). This describes where the cost shows; it does not attribute it.

- Retained Warp bytes per hosted run rose with the migrated storage: `tissue_contact` 6,064 to 8,248 (1 world) and 310,669 to 405,589 (64 worlds); `aorta_static` 5,855,154 to 7,142,766 and 18,163,527 to 21,149,367. Construction time is between 0.62 and 1.13 s, close to Stage B.
- Stage B was measured on Newton 1.8rc1 and Stage A Lab pins. This campaign is on the C3.0 pins. The measurement cannot separate the dependency change from C1 to C3.
- The two hosts agree within 0.02% at 64 worlds (0.569134 and 0.569247 ms), so this is not a noisy draw.

**What the decision covers.** `tissue_contact` holds 128 bytes of remaining State per world and `aorta_static` holds none (see the inventory). C4.1 to C4.3 are therefore unlikely to move these timings, and the closure gate in C4.4 will probably show the same result. Deciding on this failure now is in effect deciding the closure outcome of the performance gate.

Before the freeze, the unchanged worker was started once per host with `tissue_contact`, one world, exports on, to check that it still runs on the current pins. Those two outputs are startup checks, not measurements, and are kept in `.tmp_tests/stage-c40-20261110/worker-dry-run/`.

## Remaining-State inventory

[stage-c4-inventory.md](stage-c4-inventory.md) and its [JSON companion](stage-c4-inventory.json) confirm the 65 State sites of the C4 plan with file and line, row domain, reset behaviour, aliases and measured bytes per case. Five findings change the plan:

1. TipEvidence is created after binding, not by case authoring, so it cannot be declared before finalize as it stands.
2. `Centreline.positions` is history only for fully removed rings; `active_count` is derived.
3. GraphContainment forces a candidate rebuild on every `prepare`, so a fresh binding already differs from an uninterrupted run in its cache.
4. Thermal `heat` and `burn` exist twice; C4.2 removes memory there.
5. Hook anchors two alias chains (contact suppression and Thermal entry views) and must migrate with InstrumentContact.

`aorta_static` and `aorta_shell` hold none of the remaining State. `chole_tet_med` holds 2,482,368 bytes per world, of which the six residue arrays are 2,020,320.

## Open and unverified

- **Timing gate: failed**, pending the user's decision.
- HDF5 recording and replay: unverified, deferred by user choice.
- The `aorta_shell` catheter-impulse failures of earlier campaigns, the clipping intermittent and the fluid count intermittent remain unexplained and retained.
- Absolute home-directory paths remain in four committed files; cleaning them is a closure item.

[preflight]: ../.tmp_tests/stage-c40-20261110/c33-preflight.json
[format]: ../.tmp_tests/stage-c40-20261110/format-audit.json
[cert]: ../.tmp_tests/stage-c40-20261110/certificate/frozen-campaign-certificate.json
[decision]: ../.tmp_tests/stage-c40-20261110/timing/decision.json
