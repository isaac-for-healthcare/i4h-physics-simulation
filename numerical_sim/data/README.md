# Production assets

Except for the dVRK instruments below, the meshes, textures, centrelines and USD
files in this directory were created by Przemyslaw Korzeniowski and are licensed
under the Apache License, Version 2.0.
They may be used, modified and redistributed under the terms of the repository's
root `LICENSE`. The asset manifest records each file's SHA-256, size, creator and
license.

`instruments/dvrk/` contains da Vinci Si instrument USDs copied from
`i4h-mps-cmr/gtc/assets/dvrk`, using the matching definitions in
`gtc/cases/instruments/dvrk_si_all.yaml`. These derive from the dVRK
`dvrk_model` catalog and use its MIT license, retained in
`instruments/dvrk/LICENSE`. The selected models are Cadiere forceps (420049),
permanent cautery hook (420183), curved scissors (420179), and clip applier
(420327). Their coordinates are in metres; chole cases retain the source's
7.5 mesh scale, divided by seven for the `1x` cases. Collision and cutting
chains use the same case scale. The chole grasper retains its original jaw-sphere
radius, spacing and offsets for alignment review with the da Vinci visual mesh;
the spheres still follow the da Vinci jaw bind transforms. Press X in Slang to
toggle the collision overlay. Existing input mappings and tool poses are
preserved. The source set has no suction instrument, so chole retains the
existing suction mesh.

The assets are included in this repository and can be passed directly to cases:

```bash
uv run i4h-sim chole_tet_med --data ./data
uv run python scripts/fetch_assets.py --check
```

`scripts/fetch_assets.py --source PATH_OR_URL` can also copy a verified asset tree
to `data/` or `$I4H_DATA` for installations that keep assets separately.
