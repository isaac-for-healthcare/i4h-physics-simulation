# Catheter runtime capabilities

This candidate is stacked on public physics PR #17. PR #8 is the source
proposal for external root commands, curved guides, startup/material separation
and linear interior pull; the existing workflow mechanics candidate supplies the
opt-in colored wall solve and portable continuation. Defaults remain unchanged.
Dependency versions, case assets and numerical acceptance limits are retained.

## Commands and ownership

`catheter_control.root_mode` defaults to `track`. Its normalized `actions (E,3)`
integrate insertion, roll and bend rates once per step. With
`target_mode: absolute`, the declared Newton Control field `absolute_targets`
holds those three absolute values instead. Targets are clamped to the same
bounds. This is intended for hosts whose action terms already integrate rates.

`root_mode: external_pose` takes complete component-frame position `(E,3)` and
xyzw quaternion `(E,4)` batches through `write_root_poses(positions, rotations)`.
NumPy, Warp and Torch float32 inputs copy into persistent `root_positions` and
`root_rotations` Control buffers. Torch producers are ordered onto the current
Warp stream. Warp callers must order any other producer stream first. Device
inputs must contain finite positions and finite nonzero quaternions; the step
normalizes the quaternion and preserves axial roll. External ownership ignores
insertion/roll integration; bend remains controlled by the selected target mode.
It excludes a rigid mount, both at declaration and through `mount()`.

Authored reset commands and guide geometry belong to Newton Model. Selected
reset restores only those worlds' commands; track mode retains its existing
published-root reset behavior. State and Control are independently bindable.
Continuation includes pending commands; reconstruction must acquire new views.
`I4hComponent.capture_continuation`, `validate_continuation` and
`restore_continuation` export/import selected episode rows between batch sizes.
The host validates source/configuration identity and owns its task, renderer,
rigid-body and recorder state.

## Geometry and material

`add_rod` accepts independent `startup_positions (P,3)`,
`startup_frames (P,4)`, `rest_lengths (P-1,)` and
`rest_curvature (P-1,3)`. Curved placement alone retains straight material.
Curvature stores the vector part of the short relative quaternion, with norm
at most one. Direct rods consume it directly; chain PBD reconstructs its scalar
part from that material value. Centreline-tree rest construction is unchanged;
`precise_rest_frames=True` optionally computes tree rest strain before rounding
the uploaded frames, matching the workflow wall reference.

`track_path` optionally supplies an immutable component-frame polyline. Guidance
uses its closest clamped segment point. `distal_free_nodes` explicitly controls
the excluded distal nodes; when absent, `guide_tip` and live structural
`tip_edges` retain their original behavior.

Linear `graph_containment` supports `interior_deadband` and
`interior_stiffness`. Defaults disable interior pull. Interior pull corrects only
the wire and reports no wall contact or vessel reaction. Penetrating samples
retain two-way contact. Pinned nodes, spline containment and exact candidate
overflow fallback retain their existing contracts.

## Workflow wall support

Body-specific `particles` systems can independently configure gravity and
prediction versus final-velocity damping. They must cover every main body once;
the default single system still integrates the entire main block.

`rod_pbd` defaults to Jacobi. Opt-in `scheme: colored` preserves authored
Gauss-Seidel ordering using Model color tables and batched launches.
`pre_sweeps`, `limit_motion` and live `max_delta` reproduce the existing wall
prediction/projection sequence without changing other cases.

## Validation

Run from `numerical_sim/`, with production assets supplied explicitly:

```bash
uv sync --locked --group dev
I4H_DATA="$PRODUCTION_DATA" uv run pytest tests/runtime tests/packages -q -ra
uv run python examples/core_synthetic.py --device cpu
uv build --all-packages
uv run python scripts/check_artifacts.py dist
```

The added navigation fixtures cover both rod backends at 1/2/16 worlds, eager
and captured execution, complete-batch commands, roll, Torch stream ordering,
guide exclusion, material separation, buffer identity, selected reset,
independent State/Control binding, portable continuation, interior/wall contact,
pinned nodes and overflow equivalence. Colored-tree fixtures cover capture,
isolation and selected reset. The frozen navigation fixture JSON and acceptance
bounds are copied unchanged from the prior mechanics campaign.

Workflow migration and default promotion require separate native-host,
production-patient, recording/replay and policy-update qualification. Historical
physics and workflow failures remain evidence; these APIs do not waive them.
