# PR 12 coupling fixes on the public sync branch

## Scope and source

Implemented on local branch `fix/pr12-coupling-v010` in the separate
`pr12-coupling-v010` worktree. The recorded base is
`37f03006e68b6e6a1a1ac4d07c952d82ab468f56` (`Sync catheter and guidewire validation`).
Implementation commit: `06a9e642b76fc4bd604c487b7d425677207ee99a`.

The sync worktree was clean and that validation commit existed before this
worktree was created. No uncommitted validation work was staged, copied, stashed or committed.

No push, merge, PR update, Workflows repin, asset edit, case edit, dependency pin
change, reference replacement or tolerance change was made.

## Implementation and findings mapped to regressions

| Finding | Implementation and maintained evidence |
| --- | --- |
| Mounted episode reset samples stale holder poses | Optional `PhysicsBinding.hosted(..., prepare_episode_reset=...)` runs after selection and before numerical restoration. Shared Lab construction code owns the articulation selection buffer and refreshes selected FK and mount poses. `test_hosted_mount_reset_uses_current_joints` covers instruments/catheters, eager/captured reset, either selected world, a rotated holder, unchanged peer poses/mounts/particles, pending reactions, cache reset and continuation. |
| Catheter reset retains the previous root | `CatheterControl.restore_episode` places restored geometry, frames and particle history in the current mount frame. Generic system reset and the stepping kernels retain their prior behavior. `test_manual_mount_defaults_and_reset_persistence` and the six `test_defaults_declarations_aliases_and_unchanged_binding` modes check the new episode behavior and peer preservation. |
| Modern adapter updates every mount during cache reset | Removed that unconditional refresh. `test_physics_reset_reads_the_mount_pose_the_host_just_set` checks the current binding API with eager/captured host stepping. |
| Unsupported rigid host | Existing rejection retained. `test_configure_rejects_an_unsupported_rigid_backend` checks PhysX rejection and unchanged rejected host configuration. |
| Disabled catheter publishes reactions | Existing clearing/skipping retained. `test_a_disabled_catheter_driver_stops_its_reaction` now also checks disabled graph capture/replay and re-enable with fresh capture. |
| Rotated declared gravity | Existing world-frame validation retained. `test_declared_case_gravity_is_the_world_gravity_of_a_rotated_component` covers valid and invalid declarations. |
| Physical rod inertia uses environment zero | Existing model-owned per-environment calculation retained. `test_physical_inertia_follows_each_envs_parameters` compares both environments with independent rods after mass/radius edits and preparation, eager and captured. |
| Stale hosted API assumptions | Removed the obsolete positive `reset_graph` assertion; use `prepared`, `runtime` and `state`. Host-settings coverage now checks supported settings preservation and explicit unsupported collision rejection separately. |

The episode hook is shared by both Lab adapters. Cache-only `CoupledSolver.reset`
and continuation do not invoke it. Selected episode reset clears only the selected
worlds' pending reactions; cache reset and continuation keep their ownership.
No rod integration kernel or storage declaration changed, so frozen rod mechanics
benchmarks were not rerun.

## Qualification environment

Python 3.12.12, locked Newton `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`,
Warp 1.17.0, MuJoCo/MuJoCo-Warp 3.17.0, NVIDIA RTX PRO 6000 Blackwell.
Other GPU jobs were observed during the final suite; elapsed times are not isolated
performance measurements. No other jobs were stopped.
The standalone environment was created with `uv sync --locked --group dev`.
It uses the upstream Warp binary, SHA256
`a05f6fa483b43ea4c7d6bba17e75d9ee600b0d22a2a162ae962b51598bd674cb`;
it is not the historical privately patched Warp binary.

The final qualification uses production assets explicitly from
`/home/pkorzeniowsk/Projects/i4h/i4h-physics-simulation-worktrees/sync-internal-v0.10/numerical_sim/data`.
The first full run used the older `seven-qa-sync` asset directory, which lacked
`instruments/dvrk/dvrk_420049.usdc`. A read-only comparison found the final sync
assets identical to the earlier PR 12 review assets apart from `README.md`.
That earlier, pre-extraction environment also included its locked render extra,
SlangPy 0.43.1.
Lab probes use source overlays and the installed Lab 3.0 dependency environment;
no external checkout was edited. An additional isolated ovstorage 0.3.0 installation
satisfied the current Lab main import dependency without changing project pins.

## Results

The final full runtime/package suite completed after the last test/code change:
**1,297 passed, 3 failed, 61 skipped** in 1,373.31 seconds. See
`full-validated-final.log` and `full-validated-final.xml` in the retained evidence.
The continuation allocation counter check and all new/reset-related regressions
pass in this full run.

- Base mounted-reset reproduction: **4 failed**, each showing the 40 mm stale
  pose, across instrument/catheter and eager/captured reset.
- Final focused subsystem and boundary checks: **109 passed**.
- Hosted-reset checks followed by the continuation allocation check after explicit
  test binding cleanup: **9 passed**. The cleanup releases the test callback's
  reference cycle instead of deferring model destruction to another test.
- Lab main at retained C3 pin `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1`:
  **12 passed**, covering contact sensors, host settings, unsupported host rejection,
  selected reset, pending reactions and rebuild, including eager/captured stepping.
- Patched Lab 3.0 at `e53cd288727a5ca3dae7187d1d415b4d32b67e11`:
  required runtime checks are **blocked** by the host cloner reading the absent
  Newton builder `_cloth_label` metadata. Four contact-sensor variants reproduce
  that same error using the exact archived base. Three configuration checks pass.
- CPU synthetic example: **passed**.
- Six packages built; **all 12 wheels/source archives passed inspection**.
- Applicable pre-commit hooks: **passed**.

Retained earlier full run: **1,144 passed, 155 failed, 62 skipped**. Of those
failures, 145 directly report the missing da Vinci asset, one subprocess fails
while loading the same case, one lacks optional SlangPy, six assert the old
mounted-catheter reset behavior, one is the new alternate-constructor field's
missing private-reach inventory entry, and one reports a device memory counter
decrease. The stale assertions and inventory are updated, the final asset path
and locked render extra correct setup, and the test callback now has explicit
lifetime cleanup. The original allocation failure remains retained; the isolated
base allocation check passed, so it is not treated as proof of a production leak.

An intermediate corrected-setup run was interrupted to fix that test callback
lifetime: **141 passed, 1 failed, 13 skipped** before interruption. Its clipping
failure (`test_tracker_and_clip`, tracker count 1 instead of 2) also fails on the
archived base with the final assets. Base clipping endpoint arrays are retained.
No clipping assertion or mechanics was changed.

The three remaining full-suite failures are:

| Check | Final observation | Base comparison |
| --- | --- | --- |
| `test_tracker_and_clip` | The second press leaves one clip instead of two. | Same failure on the exact archived base. |
| `test_chole_batched_matches_single` | Phase separation `0.00301623` is below the unchanged `0.005` minimum. | Same failure on the exact archived base. |
| `test_hook_acquires_from_record` | Entry/exit anchor activity is `[1, 0]` instead of `[1, 1]`. | Same failure on the exact archived base. |

The 61 skips include rendering opt-ins, the explicitly opt-in upstream Warp
HashGrid regressions, unavailable Lab/GL/XR dependencies and global-WORLD-row
coverage unsupported by this Newton revision. They do not establish hardware,
renderer or alternative-Warp-build qualification.

Overall qualification remains **incomplete**: the full suite is not green, and
the real Lab 3.0 reset/contact path cannot construct on the recorded host stack.
These gates are not waived by the focused passes. The local implementation is
available for independent review; no merge readiness is asserted.

## Retained evidence and reproduction

Private raw logs, XML results, dependency identity and the exact archived base
are retained under `numerical_sim/.tmp_tests/pr12-coupling/`. This includes failed
initial probes, missing-data runs and incompatible host-stack attempts; none are
reported as passing qualification.

Run from `numerical_sim/` with the production data directory exported as
`I4H_DATA`:

```bash
uv sync --locked --group dev
uv run --locked pytest tests/runtime tests/packages -q
uv run python examples/core_synthetic.py --device cpu
uv build --all-packages
uv run python scripts/check_artifacts.py dist
```

The extracted standalone repository owns the `render` extra. Its rendering command
is `I4H_RENDER_TESTS=1 uv run --extra usd --extra render --extra gl pytest tests/runtime -q`
from that repository with `I4H_DATA` exported; upstream physics has no `render` extra.

Run the repository pre-commit hooks on changed files. Hosted tests additionally
require a compatible Lab source and dependency overlay; see the retained
`lab_probe.py` and per-stack logs for the exact invocations and failures.

## PR 17 local integration on `8812708`

This section records a separate qualification campaign; the earlier PR 12 results
above remain historical. The integration lives on `fix/pr17-coupling-local` in
the sibling `pr17-coupling-local` worktree, based on
`8812708`. Both `06a9e64` and `76cfb2a` were applied without creating commits.
No push, PR update or downstream repin is part of this integration.

### Reviewed resolutions

- Retained PR 17's six-package list, neutral `i4h_host` imports and standalone
  extraction. The private-access inventory retains PR 17's allowances and adds
  only `_prepare_episode_reset`; removed standalone allowances stay removed.
- Retained per-environment mass and radius inertia coverage, including eager
  stepping and graph recapture after parameter edits and preparation.
- Hosted reset assertions use public preparation, runtime and state access.
  Only the captured host variant requires a captured step graph.
- Supported host settings configure successfully. Unsupported collision settings
  raise while preserving the rejected configuration object and its field values.
- Episode reset retains selected-world FK/mount preparation before restoration,
  mounted-catheter geometry/history restoration and pending-reaction clearing.
  Cache reset and continuation remain separate and never invoke the episode hook.

The public addition is still
`PhysicsBinding.hosted(..., prepare_episode_reset=None)`. The callback receives
the selected device world mask before numerical episode restoration.

### Integration environment and evidence

Python 3.12.12; locked Newton
`7f0c13af6e4dd75253709c31dc65c0edcaf22b44`; Warp 1.17.0;
MuJoCo/MuJoCo-Warp 3.17.0; NVIDIA RTX PRO 6000 Blackwell; driver 615.71.09.
Production assets are explicitly selected from the same
`sync-internal-v0.10/numerical_sim/data` path recorded above. GPU checks run
sequentially; unrelated jobs remain running. No performance or physical-device
qualification is claimed.

Private evidence is retained under `numerical_sim/.tmp_tests/pr17-coupling/`:
raw logs/XML, commands, isolated environments, package origins and wheel hashes,
the exact archived `8812708` baseline and a disposable standalone source copy.
The standalone copy's tracked `pyproject.toml` and `uv.lock` are preserved.
Its environments install the locally built integration wheels without repinning
or regenerating the standalone lockfile. Tests use those environments' Python
directly, so `uv run` cannot silently restore the tracked upstream Git pin.
The root wheel installation is verified outside source import paths and contains
neither Torch, MuJoCo, Lab nor the bridge. The bridge uses a separate environment.

The retained Lab probe now loads `host`, replacing the removed `standalone`
workspace member. Its archived Lab source is verified byte-for-byte against
main `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1` (3,019 source files) and patched
3.0 `e53cd288727a5ca3dae7187d1d415b4d32b67e11` (3,137 source files).
No dependency changes bypass the Lab 3.0 incompatibility.

### Integration results

| Check | Passed | Failed / errors | Skipped |
| --- | ---: | ---: | ---: |
| Focused mount, catheter, coupling, inertia and ownership checks | 79 | 0 | 0 |
| Full upstream runtime/package suite after the final code/test edit | 1,124 | 3 | 22 |
| Unmodified `8812708` full upstream baseline | 1,113 | 2 | 22 |
| Affected checks on pinned Lab main | 25 | 0 | 2 |
| Affected checks on patched Lab 3.0 | 6 | 18 | 3 |
| Standalone configured runtime/package/Versius suite, integration wheels | 203 | 2 | 36 |
| Standalone rendering-enabled runtime suite, integration wheels | 176 | 2 | 4 |
| Standalone bridge configured suites on pinned Lab main, integration wheels | 26 | 0 | 1 |
| Standalone bridge configured suites on patched Lab 3.0, integration wheels | 15 | 9 failures, 2 setup errors | 1 |

The CPU synthetic example, private-access/import checks, six-package builds and
all 12 upstream archive inspections pass. Both standalone packages also build;
their four archives pass inspection and the standalone private-access check
passes. Applicable source and final documentation hooks pass.
Installed wheel contents match the built artifacts byte-for-byte in both the
standalone and bridge environments. The host, surgical and assessment wheel
payloads are unchanged from `8812708`; all production payload differences are
confined to the six binding, coupling, adapter and catheter-reset source files.
Package origins and wheel hashes are retained.

The full upstream suite fails `test_tracker_and_clip`,
`test_chole_batched_matches_single` and `test_scissors_cut_the_artery`. The targeted
unmodified `8812708` comparison reports two passes (clipping and scissors) and
one failure (chole phase separation). The full unmodified baseline suite then
reports **1,113 passed, 2 failed, 22 skipped**: clipping again leaves one clip
instead of two, and chole phase separation is `0.00301594` against the unchanged
`0.005` minimum (integration: `0.00300914`). Clipping therefore reproduces on the
exact baseline under full-suite ordering. Scissors passes that baseline suite;
the integration's full-suite scissors failure remains unresolved. The original
integration failures remain recorded.

A separate two-case diagnostic captures endpoint arrays without changing any
numerical assertion: the isolated integration reports two passes; baseline
reports clipping failed and scissors passed. This establishes run-context
sensitivity, not a successful replacement qualification. The original integration
full-suite scissors failure is retained as an unresolved gate. Both cases' endpoint
arrays, system lists and comparisons are retained privately. Endpoint states can
come from different stopping steps and are not a parity measurement. No further
unchanged qualification runs were used to seek a pass.

Both standalone runs fail `test_physics_rate_independent_of_render` and
`test_hook_acquires_from_record`. Both tests also fail against baseline wheels.
The rate observations (physics steps, drawn frames) are `(79, 17)` in the configured
integration run and `(83, 17)` in its targeted baseline; with rendering enabled,
they are `(83, 17)` and `(85, 17)` respectively. The unchanged assertion requires
more than five physics steps per frame. These shared-host observations do not
constitute performance qualification and the failed gate is not waived.
The integration hook failure reports anchors `[1, 0]` instead of `[1, 1]` in both
runs. The first baseline reaches a later displacement assertion and fails there;
the rendering-enabled baseline reproduces `[1, 0]`. All observations are retained.

All 18 Lab 3.0 affected-check failures and all nine bridge failures plus two setup
errors report the host cloner's missing Newton `ModelBuilder._cloth_label`.
The unmodified baseline adapter checks reproduce that error in 14 tests
(three passed, three skipped). The matching baseline bridge selection reproduces
all nine failures and two setup errors with the same missing `_cloth_label`.
Lab 3.0 runtime qualification is blocked. Its six passing configuration checks do not
establish reset, contact or recording/replay readiness on that stack.

Lab main's bridge checks pass reset, recording/replay, action rows and fake-device
contracts using installed integration wheels. The four rendering skips require
the separate XR packages/SlangPy fork; the bridge collection skip has the same XR
requirement. The upstream skips include unavailable Lab modules, nine opt-in Warp
HashGrid regressions, two unsupported global WORLD-row checks and one optional
Slang rendering check. Lab checks run separately as reported above. No skip is
counted as successful XR, hardware or alternative-Warp qualification.

Qualification remains **incomplete** and no merge readiness is asserted. Clipping,
chole phase separation and hook failures remain open, together with the scissors
and standalone rate observations. No assertions, numerical tolerances, production
assets or dependency pins have been changed to clear these gates.

### Local handoff

All integration changes remain uncommitted on `fix/pr17-coupling-local` at
`8812708`. Existing worktrees and the original standalone repository were not
modified; unrelated rod-axial validation files in `sync-internal-v0.10` were left
alone. The original and disposable standalone `pyproject.toml` and `uv.lock`
checksums still match the recorded originals. Upstream dependency pins are
unchanged. Nothing was pushed, published, repinned or sent to a PR.

The final patch and source status, logs/XML, failed-baseline comparisons,
commands, environment freezes, archive inspections, wheel byte checks and
endpoint arrays are retained in the private integration evidence directory.
Outstanding failures require resolution or an explicit later disposition before
merge readiness can be claimed.
