# i4h-assessment

Version 0.9.0, import `i4h_assessment`, Python `>=3.12,<3.13`.
Dependencies: i4h-physics. Install locally with `uv pip install ./numerical_sim/assessment`
or use the workspace lockfile for development.

```python
from i4h_assessment.executor import Procedure
# Supply explicit measurement constructors and optional host horizon.
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
