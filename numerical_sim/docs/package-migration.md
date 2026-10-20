# Architecture and package boundaries

Physics and assessment are neutral packages. Domains depend on physics, and import
neither hosts nor each other. Standalone owns device adapters, teleoperation, input
calibration, UI, rendering, XR and capture. Isaac Lab owns environments, tasks and
cameras; its cameras operate without standalone. Environment configurations declare no
teleop device by default; `i4h_standalone.devices.configure_teleop(cfg)` installs the `i4h`
device, whose case then follows the environment case.

`PhysicsBinding` owns physical lifetime and masked reset. Host teardown and replacement
invalidate bindings. Assessment evaluates every completed physics sample, before rewards
and termination checks. Empty reset selections leave assessment and host pacing alone.
`Stepper.step()` advances one physics sample; `NewtonSim.step()` invokes the manager,
including its configured decimation. Per-sample reward pulses and action totals retain
their existing meanings.

Loader and constructor maps are explicit. Ordinary owning sidecars carry presentation;
physical surfaces can render with host defaults without appearance metadata. Structural
edits require preparation before capture; staged kernel order is preserved.

Catheter material keys follow the backend: direct rods expose `young`, `torsion`, `bend`
and `twist`; PBD rods expose `stretch`, `bend` and `twist` multipliers. These are backend
fields, with no conversion between physical stiffness and PBD multipliers.

Mesh Euler angles use Rx @ Ry @ Rz; input orientation offsets use Rz @ Ry @ Rx. Explicit
helper names state these orders; existing YAML interpretation and compatibility aliases
are retained. There is no `i4h_sim` compatibility package.

Host adapters pass their episode horizon (`max_episode_length * step_dt`) through
`horizon_s` when constructing assessment. The earlier host horizon or procedure
deadline wins; completion takes precedence at that boundary. Terminal exports identify
`timeout_source` and the effective `timeout_s`. External hosts must supply their
horizon explicitly; assessment has no host dependency.

## Renamed and removed cases

| Before | Now |
| --- | --- |
| `aorta`, `aorta_reach` | `aorta_static`, `aorta_static_reach` |
| `aorta_def`, `aorta_def_reach` | `aorta_shell`, `aorta_shell_reach` |
| `I4h-AortaDefReachTarget-Direct` | `I4h-ReachTarget-Direct` with `env_cfg.case = "aorta_shell_reach"` and `observe = "catheter"` |
| `i4h_physics/data/vessels/{vascular_templates,instrument_regions}.yaml` | `i4h_endoluminal/data/` |

The airway, airrc, colon, porcine CTA, S0065 and Syndaver cases are no longer shipped. Gym task ids
`I4h-Aorta-Direct` and `I4h-ReachTarget-Direct` are unchanged and now load `aorta_static` and `aorta_static_reach`.
