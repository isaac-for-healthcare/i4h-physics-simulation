# i4h-physics

Version 0.9.0, import `i4h_physics`, Python `>=3.12,<3.13`.
Dependencies: Warp 1.17.0, Newton 1.6.1, NumPy and PyYAML. Install locally with `uv pip install ./numerical_sim/physics`
or use the workspace lockfile for development.

```python
from i4h_physics import load_case
# Pass a generated physics mapping; see examples/core_synthetic.py.
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
