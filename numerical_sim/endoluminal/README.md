# i4h-endoluminal

Version 0.9.0, import `i4h_endoluminal`, Python `>=3.12,<3.13`.
Dependencies: i4h-physics; procedure cases and their measurements need i4h-assessment (the `assessment` extra); USD assets need `pxr`: the `usd` extra (`usd-core`) outside Isaac Lab, Lab's own USD inside it. Install locally with `uv pip install './numerical_sim/endoluminal[usd]'`
(inside Isaac Lab, `uv pip install ./numerical_sim/endoluminal`; every bundled case loads a USD vessel)
or use the workspace lockfile for development.

```python
from i4h_endoluminal import load_case
sim = load_case("aorta_static", data="/path/to/data", device="cpu")
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.

## Vessel centrelines from i4h-digital-twin

A `vessel_mesh` asset's `centreline` reads the arrays that an
[i4h-digital-twin](https://github.com/isaac-for-healthcare/i4h-digital-twin)
patient bundle writes; there is no other centreline format. Export the vessels
to simulate from the twin:

```python
body.export_patient_twin("patient_bundle", vessel_names=["aorta", "iliac_artery_left", "iliac_artery_right"])
```

| File | Content |
| --- | --- |
| `centerline_points.npy` | (N, 3) points in the bundle's frame and units (see `patient_twin.yaml`) |
| `centerline_edges.npy` | (E, 2) point-index pairs |
| `centerline_radii.npy` | (N,) lumen radius per point |
| `centerline_regions.npy` | Optional (E,) region ID per edge; every edge is region 0 without it |
| `vessel_regions.yaml` | Optional region names and wall moduli ([vessel regions](../docs/vessel-regions.md)) |

Start from the `aorta_centreline` case and point its vessel at the bundle:

```yaml
  aorta:
    type: vessel_mesh
    path: /data/patient_bundle/patient_anatomy.usdc
    prim: /HumanBody/Anatomy/aorta   # one anatomy prim; the centreline covers every exported vessel
    prim_transform: true             # the twin places each prim with a transform; apply it to match the arrays
    scale: 0.01                      # bundle units (mm for CT) to scene units
    translation: [0.0, 0.0, 0.0]
    rotation_euler_degrees: [0.0, 0.0, 0.0]
    centreline:
      path: /data/patient_bundle     # or its patient_twin.yaml
      root_position: [0.0, 0.0, 0.0] # scene point at the access site; the nearest node becomes the root
      radius_scale: 1.0
```

The mesh and the centreline share the vessel transform. Weld and stitch
tolerances (`weld_epsilon`, `stitch_threshold`) are in bundle units. To start each
region's bend and twist from the bundle's wall moduli, add
`reference_young_modulus_kpa` to the `rod_pbd_material` system. The shipped
`data/vessels/aorta_centreline/` is the `aorta_centreline` centreline in this layout.

## Catheter and guidewire validation

[Bench and phantom results](CATH_GDWR_VALIDATION.md) · [Reproduce the measurements](validation/README.md)
