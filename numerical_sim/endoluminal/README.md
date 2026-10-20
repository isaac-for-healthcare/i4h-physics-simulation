# i4h-endoluminal

Version 0.9.0, import `i4h_endoluminal`, Python `>=3.12,<3.13`.
Dependencies: i4h-physics; procedure cases and their measurements need i4h-assessment (the `assessment` extra); USD assets need `pxr`: the `usd` extra (`usd-core`) outside Isaac Lab, Lab's own USD inside it. Install locally with `uv pip install ./numerical_sim/endoluminal`
or use the workspace lockfile for development.

```python
from i4h_endoluminal import load_case
sim = load_case("aorta_static", data="/path/to/data", device="cpu")
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
