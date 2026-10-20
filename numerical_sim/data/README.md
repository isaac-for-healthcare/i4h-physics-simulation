# Production assets

The meshes, textures, centrelines and USD files in this directory were created by
Przemyslaw Korzeniowski and are licensed under the Apache License, Version 2.0.
They may be used, modified and redistributed under the terms of the repository's
root `LICENSE`. The asset manifest records each file's SHA-256, size, creator and
license.

The assets are included in this repository and can be passed directly to cases:

```bash
uv run i4h-sim chole_tet_med --data ./data
uv run python scripts/fetch_assets.py --check
```

`scripts/fetch_assets.py --source PATH_OR_URL` can also copy a verified asset tree
to `data/` or `$I4H_DATA` for installations that keep assets separately.
