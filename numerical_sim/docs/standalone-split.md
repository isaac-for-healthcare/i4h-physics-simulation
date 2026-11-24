# Standalone extraction migration

Source baseline: `37f03006e68b6e6a1a1ac4d07c952d82ab468f56`. Version metadata remains
0.9.0; this extraction is not a package release. Solvers, mechanics, calibration,
production assets and numerical qualification evidence are unchanged.

| Previous symbol or command | Replacement |
| --- | --- |
| `i4h_isaaclab.composition` | `i4h_host.composition` |
| `i4h_isaaclab.appearance` | `i4h_host.appearance` |
| `i4h_isaaclab.camera.camera` | `i4h_host.camera.camera` |
| `i4h_isaaclab.camera.rig.resolve_tip`, `TIP_DEFAULTS` | `i4h_host.camera.tip` |
| `i4h_isaaclab.camera.rig.TipCamera` | `i4h_standalone.render.tip.TipCamera` |
| `i4h_isaaclab.views.controls` | `i4h_host.controls.controls` |
| `i4h_standalone.devices` | `i4h_isaaclab_standalone.devices` |
| `i4h_standalone.teleop` | `i4h_isaaclab_standalone.teleop` |
| `i4h_standalone.cartesian` | `i4h_isaaclab_standalone.cartesian` |
| `i4h_standalone.render.versius.Viewer` | `i4h_isaaclab_standalone.viewer.Viewer` |
| `i4h_isaaclab.versius.status.StatusPanel`, `guidance` | `i4h_isaaclab_standalone.status` |
| `scripts/isaaclab_run.py --render gl ...` | `i4h-lab-run --render gl ...` |
| `scripts/isaaclab_teleop.py ...` | `i4h-lab-teleop ...` |

Headless upstream `scripts/isaaclab_run.py` remains available. Its installed implementation
is `i4h_isaaclab.run`; `main(..., viewer_factory=..., view_loop_factory=...)` explicitly
accepts optional presentation. Old viewer invocations name the replacement command.
Task measurements and `failure_status` remain in `i4h_isaaclab.versius.status`.
No forwarding modules recreate dependencies between hosts. Ordinary app APIs and
`i4h-sim` remain available; case/YAML/action/recording contracts retain their formats.

Only source-branch ancestors were filtered into the standalone repository. Transferred
app tests and scripts retain their available history. The extracted repository records
rewritten commit mappings and test-level relocation mappings in `docs/history/`.
Frozen qualification reports remain upstream and are not rewritten as current evidence.

Final publication requires separately authorized operations:

1. Review and commit the coordinated upstream changes, then make that exact commit remotely reachable.
2. In the standalone repository run `python scripts/pin_upstream.py <40-character-commit>`.
   It sets every upstream distribution to the same immutable Git subdirectory source,
   then resolves `uv.lock`. On failure it restores prior metadata and lock.
3. Run the locked standalone installation, build and installed-package checks outside
   both checkouts. Validate the optional bridge in the qualified Lab environment.
4. Review and commit the standalone changes. Push/tag operations remain separate.

The pin script performs no commit, push or tag. Until step 2 succeeds, no final lockfile
is claimed. The root project does not depend on the bridge; `--all-packages` opts into
building both distributions, while ordinary `uv sync` installs standalone only.
