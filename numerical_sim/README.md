# i4h numerical simulation

Six distributions share the Warp/Newton mechanics and one numerical workspace lockfile. Every distribution uses a `src/` layout and version `0.9.0`. The workspace root is not a distributable package.

| Distribution | Import | Owns |
| --- | --- | --- |
| i4h-physics | i4h_physics | Scene, Body, staged systems, solver, Stepper, physical binding/reset |
| i4h-endoluminal | i4h_endoluminal | Anatomy metadata, catheter control/binding, reach measurements |
| i4h-surgical | i4h_surgical | Instruments, appearance, surgical systems, grasp measurements |
| i4h-assessment | i4h_assessment | Shared executor, history, evidence consumption, reward products |
| i4h-isaaclab | i4h_isaaclab | Isaac Lab hosts, gym tasks, tiled camera |
| i4h-standalone | i4h_standalone | i4h-sim app, Slang/GL, recording, XR, devices, panel adapters |

From the repository root:

```bash
cd numerical_sim
uv sync --locked --extra render --extra gl --group dev
uv run i4h-sim chole_tet_med --data ./data
uv run i4h-sim two_worlds --data ./data --render noop --steps 200
I4H_DATA="$PWD/data" uv run pytest tests/runtime -q
```

The window uses Tab to select a tool, IJKL/UO to move it, Space to grip, Shift for energy, F to fire, P to pause, N to step, Backspace to reset and F1 for panels. XR uses `--extra xr` instead of `--extra render`; these extras are mutually exclusive. Device and haptic support is optional and physical devices require separate qualification.

Production assets are included in `data/` and licensed under Apache-2.0; `docs/asset-manifest.json` records each file's SHA-256, size and license. See `data/README.md`. They are excluded from wheels and source distributions. `scripts/fetch_assets.py --source PATH_OR_URL` can copy a matching asset tree into `data/` (or `$I4H_DATA`) and verifies every file. `${DATA}` resolves in this order: explicit `--data PATH` (absolute and propagated to child processes), then `I4H_DATA`; installed packages do not search for a checkout or download assets. Headless runs do not load unused textures.

Use each domain directly:

```python
from i4h_surgical import load_case
sim = load_case("two_worlds", data="/absolute/path/to/data", envs=2)
sim.step()
sim.reset([1])  # physical reset only
```

There is no `i4h_sim` compatibility package. Assessment is constructed and advanced by the host after each completed physics sample, before rewards or terminations. Episode reset explicitly resets both physical state and the assessment object. Constructor maps are passed explicitly; systems may receive ordinary sidecars from factories without registration or a sidecar lifecycle framework.

Build members with `uv build --all-packages`. Use Python `>=3.12,<3.13`; development
continues to pin Python 3.12.12 in the environment. Keep Warp 1.17.0 and Newton 1.6.1.

Camera observations are owned by Isaac Lab and do not require standalone. Teleoperation
adapters now import from `i4h_standalone.devices` and `i4h_standalone.teleop`.
Follou support requires the separately installed `follou-sdk` Python wrappers and
`FOLLOU_LIB_PATH` pointing to the vendor native library file. Local development uses
`uv run --with /path/to/follou-sdk i4h-sim CASE --input devices --data /path/to/data`.
Fake devices, keyboard and headless runs remain independent of the SDK.

Missing textures raise an explicit asset error when rendering consumes them. Physical
surfaces without appearance metadata render with host defaults. Haptic force sliders
can only reduce and restore each device's configured pose and handle ceilings.

See [architecture](docs/package-migration.md), [contribution](CONTRIBUTING.md)
and [license notices](../NOTICE).
[Performance measurements](docs/performance.md)
and the [ReachTarget training example](docs/reach-target-example.md) include reproducible commands and observed results.

Physical surfaces without appearance sidecars use the host renderer's default material.
A requested texture that is missing raises an error naming its path; supply the original
external data tree rather than substitute textures. Headless stepping does not require
unused presentation assets.
