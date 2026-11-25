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

## Repository independence

`i4h-standalone-simulator` owns the application and the optional bridge described above.
Physics builds, installed-package checks and CI require no access to that repository.
Negative import checks retain the application module names to reject reverse dependencies;
they neither import nor install the application. Missing-asset errors describe `I4H_DATA`
and the local `data=` case-loader argument.

Frozen qualification reports retain their original bytes and test paths. The Markdown
link checker resolves the two moved test links in `stage-c25.md` at the source baseline
recorded above, rather than requiring those application tests in the current workspace.

The application pins all six upstream distributions to published split commit
`4532a216c25bd7e5fee49678b1c1359727fe523e` through Git subdirectory sources.
The application owns its lockfile, setup instructions and validation evidence.
