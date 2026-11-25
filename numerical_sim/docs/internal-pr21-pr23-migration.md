# Internal PRs 21 and 23: public migration audit

Public physics base: `ca1c8702ad994c090eebc21f5531ca44b4c4da8a` (PR 17).
Standalone base: `14d7f3e` (main), repository `isaac-for-healthcare/i4h-standalone-simulator`.
Internal source heads: #21 `55869707f8ebd64948793fe1f1e58c266d0ff7b2`,
and PR #23 `6cf9eae60e16091397539262ee0a074aa95c827d`.
Changes are ported individually; neither internal branch is merged. Workflows PR 292,
production assets, APIs, mechanics, versions, dependency pins and HashGrid opt-in
behavior remain unchanged. Raw evidence: `/home/pkorzeniowsk/Projects/i4h/qualification-evidence/migrate-internal-21-23-20261125/`
on the validation host (working logs also under `/tmp/i4h-migration-21-23/`).

## Historical evidence and superseding decisions

The original reports remain available in
[internal PR 21](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/21)
and [internal PR 23](https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal/pull/23).
Existing tracked historical reports in both public repositories are preserved byte for byte.
Their patched-Warp qualification helper, exact artifact hashes, host restrictions and
artifact-specific helper tests are superseded by the current locked stock Warp 1.17.0 setup.
No private wheel requirement is restored. Opt-in HashGrid regressions retain their
assertions and are not enabled for this ordinary suite.

Historical results include clipping/haptics 44 passed and 30/30 fresh-process clipping
runs; these are not fresh results. The historical full campaign was interrupted by
SIGTERM after 906 results, then continued in batches: 1,330 passed, two retained failed
observations and eight skipped across 1,340 nodes. The failures were
`test_gl_sequence_omits_markers_and_keeps_legacy` and `test_contact_events_capped_and_reset`;
a subsequent focused pass did not erase either failure. Catheter crossing and liver
encoded-normal exceptions accepted for local Workflows progression remain exceptions.
NVBug 6940614 stays deferred; GB300/aarch64 and physical haptics remain unqualified.

The old private-checkout fetch instructions are replaced by the public repository's
existing `origin tag v0.9.0` instructions. Both identify the same public reference tag.
The standalone extraction moved neutral camera and composition imports into `i4h_host`;
this supersedes old `i4h_isaaclab` imports, without changing camera behavior.
The removed SERVO_LOCK explanatory comment is superseded by the retained lock scopes
and their inline comments; all three locking changes remain present.

## Complete file and hunk mapping

Hunk coordinates refer to the internal PR diff, not current destination line numbers.
`physics:` paths are relative to public `numerical_sim/`; `standalone:` paths are
relative to the standalone repository. Every internal hunk is listed below.

| PR | Internal file | Internal hunk | Public destination or explicit decision |
| --- | --- | --- | --- |
| #21 | `numerical_sim/docs/catheter-parity.md` | `-9,7 +9,7` | `physics: docs/catheter-parity.md` retains v0.9.0 and output-directory guidance; internal-only fetch instruction superseded by public origin instructions. |
| #21 | `numerical_sim/docs/catheter-parity.md` | `-38,8 +38,13` | `physics: docs/catheter-parity.md` retains v0.9.0 and output-directory guidance; internal-only fetch instruction superseded by public origin instructions. |
| #21 | `numerical_sim/docs/nvbug-7011011.md` | `-0,0 +1,54` | Historical internal report retained at source; prior outcomes and limitations preserved above, fresh results below. |
| #21 | `numerical_sim/docs/standalone-qualification.md` | `-0,0 +1,20` | Historical internal report retained at source; superseded as current guidance by this audit and fresh results below. Private-wheel requirement is retired. |
| #21 | `numerical_sim/docs/v010-qa-review.md` | `-0,0 +1,68` | Historical internal report retained at source; prior outcomes and limitations preserved above, fresh results below. |
| #21 | `numerical_sim/endoluminal/src/i4h_endoluminal/cases/aorta_centreline.yaml` | `-70,5 +70,5` | Already present in `physics: endoluminal/src/i4h_endoluminal/cases/aorta_centreline.yaml`. |
| #21 | `numerical_sim/endoluminal/src/i4h_endoluminal/cases/aorta_shell.yaml` | `-55,5 +55,6` | Already present in `physics: endoluminal/src/i4h_endoluminal/cases/aorta_shell.yaml`. |
| #21 | `numerical_sim/physics/src/i4h_physics/mesh_utils.py` | `-3,6 +3,7` | Already present in `physics: physics/src/i4h_physics/mesh_utils.py`. |
| #21 | `numerical_sim/physics/src/i4h_physics/mesh_utils.py` | `-96,3 +97,318` | Already present in `physics: physics/src/i4h_physics/mesh_utils.py`. |
| #21 | `numerical_sim/physics/src/i4h_physics/systems/contact/rod_containment.py` | `-16,14 +16,34` | Already present in `physics: physics/src/i4h_physics/systems/contact/rod_containment.py`. |
| #21 | `numerical_sim/physics/src/i4h_physics/systems/contact/rod_containment.py` | `-36,8 +56,10` | Already present in `physics: physics/src/i4h_physics/systems/contact/rod_containment.py`. |
| #21 | `numerical_sim/physics/src/i4h_physics/systems/contact/rod_containment.py` | `-50,11 +72,53` | Already present in `physics: physics/src/i4h_physics/systems/contact/rod_containment.py`. |
| #21 | `numerical_sim/scripts/isaaclab_teleop.py` | `-16,9 +16,10` | Already present in `standalone: integration/src/i4h_isaaclab_standalone/cli.py`. |
| #21 | `numerical_sim/scripts/parity/catheter.py` | `-3,8 +3,8` | Already present in `physics: scripts/parity/catheter.py`. |
| #21 | `numerical_sim/scripts/parity/setup_reference.sh` | `-3,13 +3,23` | `physics: scripts/parity/setup_reference.sh`: reference-source preflight ported; public v0.9.0/origin fetch instruction preserved. |
| #21 | `numerical_sim/scripts/setup_qualification.py` | `-0,0 +1,70` | Superseded: artifact-specific patched-wheel installer/tests are retired in favor of locked stock Warp setup; no mandatory private wheel or native hash gate. |
| #21 | `numerical_sim/standalone/pyproject.toml` | `-14,7 +14,7` | Already present in `standalone: pyproject.toml`. |
| #21 | `numerical_sim/standalone/src/i4h_standalone/render/gl.py` | `-18,6 +18,7` | `standalone: src/i4h_standalone/render/gl.py`: Camera import relocated to neutral `i4h_host.camera.camera` by extraction. |
| #21 | `numerical_sim/standalone/src/i4h_standalone/render/gl.py` | `-41,6 +42,19` | Already present in `standalone: src/i4h_standalone/render/gl.py`. |
| #21 | `numerical_sim/standalone/src/i4h_standalone/render/gl.py` | `-66,8 +80,7` | Already present in `standalone: src/i4h_standalone/render/gl.py`. |
| #21 | `numerical_sim/tests/packages/test_qualification_setup.py` | `-0,0 +1,46` | Superseded: artifact-specific patched-wheel installer/tests are retired in favor of locked stock Warp setup; no mandatory private wheel or native hash gate. |
| #21 | `numerical_sim/tests/runtime/test_app_loop.py` | `-201,18 +201,53` | `standalone: tests/runtime/test_app_loop.py`: ten-frame cadence and blocked-render tests ported; step count and intervals sampled before shutdown to exclude cleanup. |
| #21 | `numerical_sim/tests/runtime/test_clipping.py` | `-24,11 +24,12` | `physics: tests/runtime/test_clipping.py`: first-fire ring reference and float64 radius already present; second-target reacquisition ported from #23, preserving physical assertions. |
| #21 | `numerical_sim/tests/runtime/test_clipping.py` | `-102,7 +103,9` | `physics: tests/runtime/test_clipping.py`: first-fire ring reference and float64 radius already present; second-target reacquisition ported from #23, preserving physical assertions. |
| #21 | `numerical_sim/tests/runtime/test_clipping.py` | `-113,7 +116,7` | `physics: tests/runtime/test_clipping.py`: first-fire ring reference and float64 radius already present; second-target reacquisition ported from #23, preserving physical assertions. |
| #21 | `numerical_sim/tests/runtime/test_clipping.py` | `-121,17 +124,35` | `physics: tests/runtime/test_clipping.py`: first-fire ring reference and float64 radius already present; second-target reacquisition ported from #23, preserving physical assertions. |
| #21 | `numerical_sim/tests/runtime/test_haptics.py` | `-188,6 +188,10` | Standalone retains all three servo-lock scopes and inline explanations; original standalone-only comment superseded by those retained comments. |
| #21 | `numerical_sim/tests/runtime/test_haptics.py` | `-423,12 +427,13` | Already present in `standalone: tests/runtime/test_haptics.py`. |
| #21 | `numerical_sim/tests/runtime/test_haptics.py` | `-816,17 +821,18` | Already present in `standalone: tests/runtime/test_haptics.py`. |
| #21 | `numerical_sim/tests/runtime/test_haptics.py` | `-935,26 +941,28` | Already present in `standalone: tests/runtime/test_haptics.py`. |
| #21 | `numerical_sim/tests/runtime/test_render.py` | `-63,6 +63,26` | Already present in `standalone: tests/runtime/test_render.py`. |
| #21 | `numerical_sim/tests/runtime/test_render.py` | `-1086,7 +1106,8` | Already present in `standalone: tests/runtime/test_render.py`. |
| #21 | `numerical_sim/tests/runtime/test_rods.py` | `-13,6 +13,8` | Already present in `physics: tests/runtime/test_rods.py`; section-heading wording differs only. |
| #21 | `numerical_sim/tests/runtime/test_rods.py` | `-80,6 +82,171` | Already present in `physics: tests/runtime/test_rods.py`; section-heading wording differs only. |
| #21 | `numerical_sim/tests/runtime/test_rods.py` | `-297,6 +464,56` | Already present in `physics: tests/runtime/test_rods.py`; section-heading wording differs only. |
| #21 | `numerical_sim/tests/runtime/test_rods.py` | `-346,6 +563,33` | `physics: tests/runtime/test_rods.py`: missing eager/graph capped-query continuation regression ported, including immutable render mesh and query identity assertions. |
| #21 | `numerical_sim/uv.lock` | `-394,6 +394,7` | PyOpenGL dependency and GL extra metadata already present in `standalone: uv.lock`; removed public standalone workspace entry is superseded by standalone ownership. |
| #21 | `numerical_sim/uv.lock` | `-423,6 +424,7` | PyOpenGL dependency and GL extra metadata already present in `standalone: uv.lock`; removed public standalone workspace entry is superseded by standalone ownership. |
| #23 | `numerical_sim/docs/nvbug-7011011.md` | `-0,0 +1,72` | Historical internal report retained at source; prior outcomes and limitations preserved above, fresh results below. |
| #23 | `numerical_sim/tests/runtime/test_clipping.py` | `-127,11 +127,29` | `physics: tests/runtime/test_clipping.py`: first-fire ring reference and float64 radius already present; second-target reacquisition ported from #23, preserving physical assertions. |

## Publication checkpoints

The clipping correction is commit `752f96f`; the source preflight and missing
capped-vessel continuation regression are commit `94b76b2`. The standalone worker correction is commit `fee9a88`, published in
[standalone PR 1](https://github.com/isaac-for-healthcare/i4h-standalone-simulator/pull/1).
The standalone repository has internal visibility; its PR was verified with
authenticated GitHub access. Anonymous Markdown link checking excludes that exact
PR URL because GitHub returns 404 without repository access.
This audit records the final validation outcomes.
Internal PR closure follows publication and passing replacement behavior; broader
failures remain recorded and do not establish release qualification.

## Fresh validation (2026-11-25)

All 87 explicit production assets verified against both manifests before testing.
Fresh environments use Python 3.12.12, stock Warp 1.17.0 and pinned Newton
`7f0c13af6e4dd75253709c31dc65c0edcaf22b44`. Installed package roots and Git origins
were inspected before tests. Standalone retains its upstream physics pin `4532a216`;
the bridge run uses isolated installed wheels and Lab source `4cfdfcfa`.
The suites ran sequentially on the GPU, with the two unrelated GPU jobs left running.

| Check | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| physics-focused | 12 | 0 | 0 |
| continuation | 2 | 0 | 0 |
| physics-full | 1128 | 1 | 23 |
| standalone-focused | 56 | 0 | 24 |
| standalone-full | 205 | 1 | 36 |
| standalone-render | 178 | 1 | 4 |
| bridge-full | 26 | 0 | 1 |

Rows overlap by design: the full suites are the required final runs, and standalone
rendering is the separately required `I4H_RENDER_TESTS=1` configuration. Focused passes
do not replace failures from either full configuration. No failed numerical check
was repeated merely to obtain a pass. This campaign is not release qualification.

The forced clipping diagnostic used the same released-fire action perturbation in
two saved test copies. The original failed at the second-attachment assertion (count
stayed one; nearest eligible gap 1.057634830 versus reach 0.25). The corrected copy
passed all unchanged physical/reset checks from nearest gap 0.909163773. Both copies
received identical perturbation code; their measured gaps differ and both satisfy
the same forced out-of-reach precondition. This constructs the reported condition on this host;
it is not a spontaneous GB300 reproduction. Both scripts and complete logs are saved.

Parity setup checks passed: missing tag, present tag with missing source, valid
reference archive reaching environment creation, and shell syntax. The public
v0.9.0 reference and origin-fetch guidance remain unchanged.

Both repositories passed build, wheel/sdist artifact and private-access boundary
checks. All six physics wheels passed installed-root/version checks outside the
checkout. Minimal standalone wheels passed independent import checks with no
editable siblings, Torch, MuJoCo, Lab adapter or bridge; its CPU headless smoke
passed. Standalone CPU package checks passed 39 tests. The physics core-only CPU
example passed. Production sources, assets, pins and historical reports are unchanged.

Physics pre-commit hooks passed. Standalone's required Ruff F checks passed, as did
its changed-test Ruff check, Markdown with an explicit configuration, and whitespace
checks. Borrowing the physics hook configuration for standalone produced retained
tooling failures: baseline extracted import formatting fails isort; the standalone
checkout has no `.markdownlint.yaml`; an initial cross-repository large-file check
used the wrong Git context. The correct-context large-file check and explicit-config
Markdown check passed. No unrelated import reformatting was introduced. All failed
attempt logs remain in the evidence directory. New Markdown links were checked.

### Retained fresh failures

- `physics-full`: `tests.runtime.test_env_isolation.test_chole_batched_matches_single`.
- `standalone-full`: `tests.runtime.test_instruments.test_hook_acquires_from_record`.
- `standalone-render`: `tests.runtime.test_instruments.test_hook_acquires_from_record`.

The physics scene-excitation failure measured 0.0030006766319274902 against
the unchanged greater-than-0.005 requirement. It was also recorded in the historical
standalone extraction report. The standalone hook failure observed `[1, 0]`
against the required paired anchors `[1, 1]`; that failure is also historical. Fresh failures above remain failed observations regardless of any
older accepted progression. See the raw XML/logs for exact values and failure arrays.
Skipped rendering/XR/hardware coverage is not qualified by other passing checks.

Preservation checks compare all 82 original worktrees, their staged diffs and 509
pre-existing dirty/untracked file hashes. All pre-existing file hashes and staged
diffs remain unchanged. A new untracked `FollouDeviceLogs/` directory appeared in the
original standalone checkout during validation and was left untouched; its status
change is recorded rather than described as an entirely unchanged checkout.
