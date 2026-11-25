# i4h-physics

Version 0.9.0, import `i4h_physics`, Python `>=3.12,<3.13`.
Dependencies: Warp 1.17.0, Newton at revision `7f0c13af` (1.9.0.dev0, pinned in `physics/pyproject.toml`), NumPy and PyYAML. Install locally with `uv pip install ./numerical_sim/physics`
or use the workspace lockfile for development. The `coupled` extra pins MuJoCo and MuJoCo-Warp 3.17.0.

The historical Stage B CUDA stack used the [Warp 1.17.0 capture correction](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/warp117-hashgrid-capture.md).
Normal setup uses stock Warp 1.17.0. The historical patched wheel was local to that
campaign; its results do not establish stock-Warp HashGrid capture correctness.
The known HashGrid capture failures remain visible; do not weaken their tests.

Shared construction uses `author_case(cfg, loaders, systems)`, `add_world(builder, world, xform, label_template)`,
then `bind(model, state, control)`. Binding returns a fresh scene and ordered systems without a Stepper.
`I4hSolver.reset(flags=0)` preserves episode history; `restore_episode(state, control, world_mask)` restores it.
Standalone `load_case()` uses the same authoring and binding path, then creates its Stepper.

Systems declare their State and Control fields in `declare_storage` before the Model is finalized and bind views of
them afterwards; scratch, outputs and tables stay on the system. `I4hSolver.save_i4h(state, control, mask=None)` and
`restore_i4h(state, control, mask=None)` copy every declared State and Control field and the scene's own particle rows
between containers of one Model. To continue an episode in a new runtime: bind, prepare, then restore.

```python
from i4h_physics import load_case
# Pass a generated physics mapping; see examples/core_synthetic.py.
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
