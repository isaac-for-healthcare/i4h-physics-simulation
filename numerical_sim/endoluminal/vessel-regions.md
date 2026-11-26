# Vessel regions and material properties

`aorta_centreline` ships four coarse groups: right iliac/femoral (branch 0),
aortic trunk (1), left iliac/femoral (2), and arch outlets (3–11). The configured
right-femoral entry identifies branch 0. Arch outlets deliberately carry no more
specific arterial labels. These groups cover all 254 original edges.

The centreline asset `data/vessels/aorta_centreline/` uses the i4h-digital-twin
array layout (see the [endoluminal README](../endoluminal/README.md#vessel-centrelines-from-i4h-digital-twin)):
`centerline_regions.npy` holds each edge's region ID (here its VMTK branch) and
`vessel_regions.yaml` names the regions. The centreline's `vessel_regions.yaml` is
the default; `assets.aorta.centreline.regions` may replace it with an inline mapping
or a YAML file path. File paths are relative to the process working directory, not
the case; `${DATA}` expands from explicit `--data` or `I4H_DATA`. The shipped sidecar is:

```yaml
schema_version: 1
default: unassigned
branches:
  0: right_iliac_femoral
  1: aortic_trunk
  2: left_iliac_femoral
  3: arch_outlets
  4: arch_outlets
  5: arch_outlets
  6: arch_outlets
  7: arch_outlets
  8: arch_outlets
  9: arch_outlets
  10: arch_outlets
  11: arch_outlets
```

Version 1 allows only `schema_version`, `default`, `branches` and the optional
`young_modulus_kpa` (below). Region names must be nonempty strings. `default` is
optional and defaults to `unassigned`; `branches` defaults to an empty mapping.
Keys are the integer region IDs of `centerline_regions.npy` (the string "0"
differs from 0); a bundle's own sidecar may also list vessels without centreline
edges. Other unknown branch IDs fail. Unlisted branches use the default region.
Welding preserves source provenance; conflicting assignments on a deduplicated
edge fail. Artificial stitching edges use the default region.

A `rod_pbd_material` system must precede the matching `rod_pbd` solver.
Each regional stretch multiplier multiplies the solver's stretch setting and
layout scale. Bend and twist use the harmonic average of the adjoining regional
values, multiplied by their solver settings (and the bend layout scale).
Radius weighting is disabled in this case. All regional values start at 1.
These are numerical PBD multipliers, not material moduli. Calibration is deferred.

A sidecar may also give each region a wall Young's modulus in kPa. Patient bundles
from i4h-digital-twin write one (`vessel_regions.yaml`) with literature defaults per
segmented vessel; their region IDs are the centreline's branch IDs:

```yaml
schema_version: 1
default: unassigned
branches: {0: aorta, 1: iliac_artery_right, 2: iliac_artery_left}
young_modulus_kpa: {aorta: 400.0, iliac_artery_right: 400.0, iliac_artery_left: 400.0}
```

Names must be regions of the same sidecar, and values positive and finite. The
moduli only take effect when the `rod_pbd_material` sets
`reference_young_modulus_kpa`: each region with a modulus then starts with bend and
twist multipliers of its modulus over the reference, and stretch stays 1. Regions
without a modulus, such as `unassigned`, keep 1. A ratio above the multiplier limit
of 10 fails at declaration. Set the reference to the modulus the solver settings
were tuned for, so that region keeps them:

```yaml
- type: rod_pbd_material
  id: vessel_material
  body: aorta
  reference_young_modulus_kpa: 400
```

This linear mapping is a first-order default, not a calibration: it ignores wall
thickness and radius (see `radius_weighting`).

The existing physics API edits regional tables between steps:

```python
body = sim.scene.body("aorta")
body.region_properties.edit({"bend": 0.5}, "aortic_trunk", envs=[0])
sim.reset([0])  # episode reset preserves regional material edits
body.region_properties.restore(envs=[0])  # restore every region's initial values
```

Omitting `envs` applies a material edit or restoration to every environment.
Restoration uses each region's initial value: 1, or the Young's-modulus ratio
for bend and twist. The physics API accepts multipliers in [0, 10].
`tests/runtime/test_regions.py` covers eager and captured edits, environment
isolation, masked episode reset and restoration of modulus-derived defaults.
