# Pinned-liver numerical handoff

The implementation is a numerical qualification fixture and observer. Arena will
own objective predicates and continuous hold timers; Workflow Tasks will execute
controllers. `GraspTeacher` is the standalone reference used to qualify the fixture.
No Arena imports, training, Workflows recorder integration or production workflow
engine are added. Numerical traces are not Workflows-compatible demonstrations.

The approved acceptance specification is
[liver-grasp-workflows-integration-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/liver-grasp-workflows-integration-plan.md).
The numerical worktree starts at immutable baseline
`d1fc95c99d59be793da1f938e8c58458a538b168`. The sibling upgrade checkout, its
runtime, dependency pins, staged Versius boundary and baseline remain separate.

## Construction and coordinates

`i4h_surgical.config.load_case('liver_pinned_grasp_1x', data=ASSET_ROOT)` builds one
liver body (1,986 particles), one grasper and no other bodies, tools, inter-organ
connectors, cavity, support surface or bounds solver. There are 391 soft attachments
and 1,595 unattached vertices. The selected particles still have positive inverse
mass: attachment stiffness is `0.0005`, and they can drift. Source geometry is scaled
by `0.142857143` and translated by `(0,-0.428571,0)`; the existing per-particle mass,
materials, collision radii, gravity and instrument parameters are retained.
Inversions and attachment drift are measured; mechanics remain uncalibrated.

All positions, normals, sphere geometry, displacements and commands exposed here
are in the **component frame**, in metres. Component +Y is up and gravity is -Y.
Quaternions are **XYZW**. A future host applies its component placement and environment
origin once when converting to world coordinates; it must invert that placement
before writing physical commands. Probe arrays contain no environment origins.

For one tool the physical action is float32 `(E,10)`:

`[px,py,pz,qx,qy,qz,qw,grip,energy,fire]`.

The instrument system's backing view is `(E,1,10)`. This reference holds authored
orientation, uses normal trocar control, writes grip 0=open / 1=closed, and leaves
energy and fire at zero. It never writes tissue positions or grasp slots. Physics
is 60 Hz, eight substeps, eight iterations. Teacher commands are issued at 30 Hz
and applied unchanged for two physics steps. Acquisition and retention are sampled
after **each** physics step, including the first step between teacher commands.

## Observer, streams and lifetime

Standalone:

```python
from i4h_surgical.grasp_probe import GraspTargetProbe
probe = GraspTargetProbe(sim.physics, tissue='tissue', grasper='grasper')
# At a completed physics boundary, outside capture:
probe.select(vertex_id, env=0)
probe.arm(env=0)  # before writing the closure command
# Host/Stepper advances ONE physics step:
probe.sample()
views = probe.torch_views()
```

Hosted: `i4h_isaaclab.grasp.bind_grasp_probe(component)` binds through the public
`component.physics` facade. The host calls `sample()` in its after-physics callback
**before decimation**, then consumes `torch_views()`. This adapter does not step,
issue commands, create host state, register another clock or automatically attach a
callback. Integration against the final upgraded runtime remains pending.

`sample()` launches kernels on the current Warp stream without reading device data
or allocating arrays. All tables and outputs exist before capture. Changing the
selected target only updates a preallocated ID array; physics recapture is unnecessary.
A host may capture observer launches with its after-step work. Its command writes
must already be ordered before physics through the existing component action
boundary. `torch_views()` makes Torch's current stream wait for Warp's producer
stream, then returns zero-copy views. `read()` synchronizes to host and is for the
numerical harness and diagnostics. A consumer on another stream must order its own
work and keep the binding alive.

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `target`, `nearest` | `(E,)` int32 | body-local target vertex and global instrument-sphere ID; -1 unset |
| `geometry` | `(E,4,3)` float32 | live target, area-weighted unit normal (zero invalid), active fixed-chain jaw-sphere mean, nearest active jaw-sphere position |
| `nearest_radius`, `gap` | `(E,)` float32 | nearest sphere radius and signed radial clearance: distance minus sphere radius minus particle radius |
| `jaw_state` | `(E,2)` float32 | actual jaw angle in radians and held flag |
| `mechanism_counts` | `(E,4)` int32 | surface count, volumetric count, surface active slots, volumetric active slots |
| `armed`, `latched` | `(E,)` int32 | acquisition arm and permanent first-nonempty latch |
| `original_particle`, `original_sphere` | `(E,G)` int32 | originally acquired slot identities; particle -1 means unused |
| `original_rest` | `(E,G,3)` float32 | original particle-minus-sphere offset at capture |
| `membership`, `lost` | `(E,G)` int32 | original patch membership and sticky original-binding loss |
| `displacement` | `(E,G,3)` float32 | current particle minus its original sphere |
| `residual` | `(E,G)` float32 | absolute current radius minus captured radius, metres |
| `counts` | `(E,6)` int32 | acquisition size, original patch size, valid original patch particles, retained particles, invalid original patch bindings, wrong-patch flag |
| `stats` | `(E,2)` float32 | maximum and mean radial residual over the original acquired patch set |

`G` is the existing volumetric capacity (64). Neither count mechanism inflates the
other. The first nonempty volumetric capture freezes all acquired identities,
rest offsets and membership, including off-patch particles. Any off-patch capture
fails this reference. Empty captures may retry for one second with a stationary
root. Opening, later captures or replacement slots never replace the original
acquisition; invalid originals remain invalid for that attempt. All asset vertices
are surface vertices; “volumetric” identifies the mechanism, not interior tissue.

Retention requires the original slot's particle, sphere and rest identity to remain
valid and radial residual <=2 mm. At least `ceil(original_patch_size/2)` must remain
retained at every sample in qualification, retraction and hold. Tangential motion
is allowed and retained as displacement vectors. A one-particle acquisition is
explicitly counted as one, not described as a robust multi-particle grasp. The
material target vertex must independently reach its frozen +10 mm Y destination.

Views last only as long as the binding. On hard rebuild/close, discard all observer
and controller objects and bind a fresh probe; old access raises. Ordinary masked
reset does not require rebinding. Selected-world reset/restoration order is:

1. Perform physical episode reset, or restore through continuation (never both).
2. Clear the selected probe and controller state, preserving peers.
3. Select the frozen manifest target and recompute geometry.
4. Check eligibility; reject disagreement without reselection.

## Continuation and qualification artifacts

`HarnessSnapshot` uses `PhysicsBinding.save/restore`, hence `save_i4h/restore_i4h`,
to copy every continuation-managed State and Control field. Native particle state,
particle history, instrument state and grasp state follow the existing continuation
contract. A private checkpoint uses separately allocated State and cloned Control;
there is no reset or history initialization after restoring it.

The NPZ bundle records all copied arrays, attachment rest/index data, harness time
and counters, settling evidence, schema/layout hash, asset hashes, package source
hashes, dependency versions/source hashes, Model-array identity and snapshot hash.
Loading validates identity, every layout and byte hash, and attachment rest before
assigning checkpoint buffers. Restoring rechecks constructed attachment rest before
calling continuation, restores harness counters and clears only selected observers.
Full standalone counters rewind on full restore; selected restore rewinds selected
world counters without rewinding the host's global clock.

Each scored attempt must first restore copied bytes exactly. Three repeated
restore/open-step trials in each execution mode must differ by <=10 micrometres
maximum pointwise distance, including the eager/captured first-step comparison.
Save/restore rebuilds shared contact caches. Selected reset/restore tests require
exact peer State/Control and latch preservation at the operation boundary, not
bitwise equality of later trajectories.

Settling uses completed-step position differences, <5 mm/s maximum speed for 60
consecutive steps, at least three seconds observed and a 30-second deadline.
`not_settled` blocks scoring. One canonical settled bundle is used for all starts,
targets and execution modes of that world-count configuration. A replay command can
load that bundle instead of settling again; mismatched source/assets/configuration
are rejected.

The frozen target/workspace manifest includes mesh-derived patch membership, target
IDs, settled normals/facing, target-selection measurements and snapshot hash. The
axis-aligned workspace contains all three starts and computed pregrasp, closure and
retraction root goals plus 20 mm per axis. It is a **controller sanity bound, not an
independently validated safety or reachability constraint**. Root/command exit fails;
commands are never silently clipped. It is not a bounds constraint on the tissue.

`physical_trace.npz` stores a physical `(steps,E,10)` action at every physics step,
particle positions, probe geometry, original-particle displacements/residuals and
invalid-binding masks. `trace.jsonl` stores phases, actual jaw angle and root,
mechanism counts, radial statistics, destination error, existing tool force and jaw
load in simulation units, attachment drift and signed-volume diagnostics. These
force channels are diagnostics, not calibrated force validation.

`result.json` identifies closure geometry/clearance, acquisition, failure, frozen
destination and release-command/first-clear steps separately for both mechanisms.
`report.json` publishes every attempt, capture-size histogram and the supported
set (all three starts in both modes). Four supported targets are required for
reusable-set support. No targets or tolerances are changed after scoring.
The video script also emits explicit release latencies and eager/captured trace
comparisons. `review.mp4` is a labelled X/Y mesh projection at 1x speed; it does not
claim production renderer or recorder integration.
