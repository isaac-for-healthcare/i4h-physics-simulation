# i4h numerical simulation

Start with the [documentation index](DOCUMENTATION.md) for architecture, guides and history.

Six distributions share the Warp/Newton mechanics and one numerical workspace lockfile. Every distribution uses a `src/` layout and version `0.9.0`. The workspace root is not a distributable package.

| Distribution | Import | Owns |
| --- | --- | --- |
| i4h-physics | i4h_physics | Scene, Body, staged systems, solver, Stepper, physical binding/reset |
| i4h-endoluminal | i4h_endoluminal | Anatomy metadata, catheter control/binding, reach measurements |
| i4h-surgical | i4h_surgical | Instruments, appearance, surgical systems, grasp measurements |
| i4h-assessment | i4h_assessment | Shared executor, history, evidence consumption, reward products |
| i4h-isaaclab | i4h_isaaclab | Isaac Lab hosts, gym tasks, tiled camera |
| i4h-host | i4h_host | Case composition, procedures, camera geometry, appearance, controls |

The standalone viewer, device backends, capture and XR moved to a separate application.
See [package ownership and migration](DOCUMENTATION.md).
The optional `i4h-isaaclab-standalone` bridge supplies `i4h-lab-run` and `i4h-lab-teleop`.
The numerical repository retains headless Lab training/evaluation and all task measurements.

```bash
cd numerical_sim
uv sync --locked --group dev
I4H_DATA="$PWD/data" uv run pytest tests/runtime tests/packages -q
uv run python examples/core_synthetic.py --device cpu
```

Production assets are included in `data/` and licensed under Apache-2.0, except for the MIT-licensed dVRK instruments; `data/asset-manifest.json` records each file's SHA-256, size and license. See `data/README.md`. They are excluded from wheels and source distributions. `scripts/fetch_assets.py --source PATH_OR_URL` can copy a matching asset tree into `data/` (or `$I4H_DATA`) and verifies every file. `${DATA}` resolves in this order: explicit `--data PATH` (absolute and propagated to child processes), then `I4H_DATA`; installed packages do not search for a checkout or download assets. Headless runs do not load unused textures.

Use each domain directly:

```python
from i4h_surgical import load_case
sim = load_case("two_worlds", data="/absolute/path/to/data", envs=2)
sim.step()
sim.reset([1])  # physical reset only
```

Newton owns the simulation data: particles, topology, history and commands are fields of the Newton Model, State and
Control, declared before finalize. Systems hold views of them and keep only scratch, outputs and tables. A State and
Control pair can be saved and restored with `sim.physics.save(state, control)` and `sim.physics.restore(state, control)`;
see `physics/component-contracts.md` for what that covers.

There is no `i4h_sim` compatibility package. Assessment is constructed and advanced by the host after each completed physics sample, before rewards or terminations. Episode reset explicitly resets both physical state and the assessment object. Constructor maps are passed explicitly; systems may receive ordinary sidecars from factories without registration or a sidecar lifecycle framework.

Build members with `uv build --all-packages`. Use Python `>=3.12,<3.13`; development
continues to pin Python 3.12.12 in the environment. Keep Warp 1.17.0 and Newton at revision `b70ad54` (1.9.0.dev0, pinned in `physics/pyproject.toml`); coupling uses MuJoCo Core and MuJoCo-Warp 4.0.1. See [Isaac Lab compatibility](isaaclab/README.md#isaac-lab-compatibility) for the host revision and required patches.

Camera observations are owned by Isaac Lab and require neither standalone nor the bridge.
Shared helpers live in `i4h_host`; neither host depends on the other. See
[split migration](DOCUMENTATION.md) for moved Python symbols and launchers.

See [architecture](DOCUMENTATION.md), [contribution](CONTRIBUTING.md)
and [license notices](../NOTICE).
[Performance measurements](physics/performance.md)
and the [ReachTarget training example](isaaclab/reach-target-example.md) include reproducible commands and observed results.

Physical surfaces without appearance sidecars use the host renderer's default material.
A requested texture that is missing raises an error naming its path; supply the original
external data tree rather than substitute textures. Headless stepping does not require
unused presentation assets.

Versius robot tasks use the [shared Isaac Lab launchers](isaaclab/versius.md).
See [experiment results](examples/versius/README.md) for results and retained limitations.
