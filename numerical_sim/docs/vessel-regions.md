# Vessel regions and stiffness editing

`aorta_centreline` ships four coarse groups: right iliac/femoral (branch 0),
aortic trunk (1), left iliac/femoral (2), and arch outlets (3–11). The configured
right-femoral entry identifies branch 0. Arch outlets deliberately carry no more
specific arterial labels. These groups cover all 254 original edges.

The `assets.aorta.centreline.regions` value accepts an inline mapping or a YAML
file path. File paths are relative to the process working directory, not the case;
`${DATA}` expands from explicit `--data` or `I4H_DATA`. An equivalent sidecar is:

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

Version 1 allows only `schema_version`, `default`, and `branches`. Region names
must be nonempty strings. `default` is optional and defaults to `unassigned`;
`branches` defaults to an empty mapping. Keys match the source JSON's `branch_id`
(or its list index when omitted), including their type: integer 0 differs from
string "0". Unknown branch IDs fail. Unlisted branches use the default region.
Welding preserves source provenance; conflicting assignments on a deduplicated
edge fail. Artificial stitching edges use the default region.

A `rod_pbd_material` system must precede the matching `rod_pbd` solver.
Each regional stretch multiplier multiplies the solver's stretch setting and
layout scale. Bend and twist use the harmonic average of the adjoining regional
values, multiplied by their solver settings (and the bend layout scale).
Radius weighting is disabled in this case. All regional values start at 1.
These are numerical PBD multipliers, not material moduli. Calibration is deferred.

The standalone Slang workspace selects and highlights one region. Edits apply
to all environments by default; disable that option to edit the displayed one.
Episode reset preserves material edits. Restore all region defaults resets every
region of the selected body to 1 in the selected environment scope.
Save/Load and `--overrides FILE` use the existing `regions` override section:

```yaml
regions:
  aorta:
    vessel_material:
      aortic_trunk:
        bend: 0.5
```

Scalar values apply to all environments; per-environment lists are also supported.
The standalone workspace's group `limits` maps a declared knob label to a finite
`[minimum, maximum]` within the knob's underlying bounds. The aorta case uses
stretch 0–1 and bend/twist 0–10. Its base solver coefficients are 1, 0.1 and 0.1,
respectively. Panel edits and standalone override files enforce these limits;
the lower-level physics API retains its existing bounds. Limits do not establish
physiological realism or guarantee stability. Vascular templates remain neutral,
uncalibrated UI presets and do not rename region membership.
