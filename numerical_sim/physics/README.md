# i4h-physics

Version 0.9.0, import `i4h_physics`, Python `>=3.12,<3.13`.
Dependencies: Warp 1.17.0, Newton 1.8.0rc1, NumPy and PyYAML. Install locally with `uv pip install ./numerical_sim/physics`
or use the workspace lockfile for development. The `coupled` extra pins MuJoCo and MuJoCo-Warp 3.17.0.

The qualified Stage B CUDA stack uses the [Warp 1.17.0 capture correction](../docs/warp117-hashgrid-capture.md).
The local patched wheel fixes metadata ownership across captured grids; the version pin stays unchanged.

Shared construction uses `author_case(cfg, loaders, systems)`, `add_world(builder, world, xform, label_template)`,
then `bind(model, state, control)`. Binding returns a fresh scene and ordered systems without a Stepper.
`I4hSolver.reset(flags=0)` preserves episode history; `restore_episode(state, control, world_mask)` restores it.
Standalone `load_case()` uses the same authoring and binding path, then creates its Stepper.

```python
from i4h_physics import load_case
# Pass a generated physics mapping; see examples/core_synthetic.py.
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
