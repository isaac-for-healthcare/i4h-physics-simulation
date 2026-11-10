# Stage C field contracts

This is the semantic inventory accompanying the source allocation census. These
are migration contracts, not claims that the fields have migrated. Source-review
coverage is recorded per section; runtime scratch-poison and continuation tests
remain separate gates. The static source inventory is reconciled in [stage-c-inventory.json](stage-c-inventory.json),
with a separate [scratch lifecycle audit](stage-c-scratch-audit.md). This certifies
the recorded source contracts, not runtime continuation or completed migration.

`M`, `S`, `C`, `X`, `O` mean Model, State, Control, solver scratch and completed-step
outputs respectively. `E` is worlds, `V` mesh variants, and all other capacities
are the exact local constructor quantities. `f`, `i`, `b` mean float32, int32,
bool; vector/quaternion/matrix names denote Warp element types, not array axes.
Fields grouped in one row have the same ownership/reset policy; their separate
names, dtypes and shapes remain explicit. Custom fields on zero-capacity domains
retain logical empty views with a single padded backing row; empty native aliases
have no backing bytes. No body-local topology integer
receives Newton reference-offset metadata.

For Model fields, episode reset leaves live edits intact. Authored defaults are
retained separately when a State/Control reset needs them. For X fields, the
listed source overwrite or self-clearing path is the required next-use proof;
fresh binding must initialize self-clearing accumulators before the first solve.
Poison tests must distinguish that initialization from a full overwrite on every
step. `O` never means that a physics consumer can be moved to a later phase.

C2.1 implementation: main q0 aliases native Model.particle_q without a custom
rest copy. inv_mass/flags0 alias Model.i4h particle attributes, and q_prev/flags
alias State.i4h particle attributes. Native
Model particle flags remain untouched. Four Scene clones and their post-finalize
readback/assignment path are removed. Newton retains native defaults and custom
State initialization templates. C2.2 additionally binds all eight pool fields,
body variant tables, component gravity and static collision geometry to Newton.
Pool Model fields cost 24 bytes/logical row; each State and its initialization
template cost 40 bytes/row each, plus identity columns and empty-domain padding.
Fluid settings/cursors/overflow and the six legacy residue arrays retain their owners.
The accepted C1 inventory is preserved in [stage-c-inventory-c1-accepted.json](stage-c-inventory-c1-accepted.json).

## Scene, native storage and views

Source: [scene.py](../physics/src/i4h_physics/scene.py),
[newton_model.py](../physics/src/i4h_physics/newton_model.py).

| Field | Owner / dtype / layout | Authored value; reset | First subsequent reader/writer; aliases |
| --- | --- | --- | --- |
| main.q, main.qd | S, vec3, full native particle arrays | Exact authored rows / zero velocity; selected owned rows reset | Particles.predict/finalize; aliases State.particle_q/qd, Scene.q/qd and cached body/part views. Foreign rows excluded from reset/copy. |
| main.q_prev | S, vec3, full native rows | Exact authored rest at initial model construction; selected rest reset; explicit post-write initialization copies selected live q only | Particles.predict writes before integration, but control/drag/contact consumers also use history; binding and cache-only refresh preserve restored history. Startup/reset events and fresh component rebuild callbacks complete their position writes before the history copy. |
| main.q0 | M, vec3, native particle frequency | Exact ParticleRest.q0 authored through native add_particles positions; subsequent in-place native rest edits become the reset source | Scene.reset reads native Model.particle_q through checked cached world-major/body/part aliases. No i4h rest allocation or column; replacement of the native buffer invalidates the binding. Empty native aliases have zero backing bytes. |
| main.inv_mass | M, f, native particle frequency | Exact ParticleRest.inv_mass, not model.particle_inv_mass | Prediction and mechanics read; selected body/part views alias same storage. |
| main.flags0 | M, i, native particle frequency | Exact authored flags | Scene.reset reads. |
| main.flags | S, i, native particle frequency | flags0; selected-world flags0 reset | Prediction/contact/deletion read/write; must cease aliasing Model.particle_flags so states stay independent. |
| main.radius | M, f, native particle frequency | Authored radius; retain native Model.particle_radius alias | Contact kernels read; no separate allocation. |
| pool.q0 | Newton M.i4h, vec3, body-major sum(E*body.count) | Concatenated exact rest rows; one pool domain includes non-fluid gaps, world identities identify actual worlds, native owner is -1 | Pool reset and fluid q0_span read. |
| pool.q, pool.q_prev, pool.qd | Newton S.i4h, vec3, same body-major layout | q0, q0, zero; selected body/world rows reset | Emission then Fluid.predict/finalize; all span and q_b/qd_b/q_prev_b aliases refer to these owners. |
| pool.inv_mass, pool.radius | Newton M.i4h, f, same body-major layout | Exact authored pool rest fields | Fluid/contact/suction read; no native particle rows added for pools. |
| pool.flags0 | Newton M.i4h, i, same body-major layout | Exact authored flags | Pool reset reads. |
| pool.flags | Newton S.i4h, i, same body-major layout | flags0; selected-world restore | Emitters activate, suction removes, fluid reads; flags_b/flags_span alias. |
| Body.variant | Newton M.i4h, i, (E,) WORLD | variant_of_env, including padding-world variants | Stable body declaration order independent of system instances; every consumer aliases one Model table. |
| Scene.gravity | Newton M.i4h, vec3, (E,) WORLD | Scene-specific finalizer uses completed transforms and finalized builder gravity; unrotated aliases native gravity, rotated allocates once | Particle/fluid predict read; explicit refresh_gravity(model) after host edits before binding, or model-property notification after binding; in-place refresh retains captured pointers. Binding performs no refresh writes. |
| cached static mesh points / indices | Newton M.i4h, vec3 (vertices,) / i (3*faces,) | StaticMesh.points / flattened indices; mesh-local indices with no Newton reference offsets | StaticContact and RodContainment share Scene.static_wp_mesh. |
| cached wp.Mesh acceleration structure | X | Build from declared static geometry before use | Mesh queries; cache ownership is rebuildable, geometry arrays are not scratch. |

All eight pool columns are authored together per logical row before finalization.
Direct builder callers complete with `newton_model.complete_scene(builder, scene)`
after all worlds; CaseAuthoring uses the same helper. Repeated completion is
idempotent; changed capacities, mappings, geometry layouts or transforms require
reconstruction. Binding preserves every selected State byte, including independent
position/history, and pool reset remains full even when main position/velocity
reset is disabled. Acceleration structures remain runtime scratch.

Scene._fields, Block.rows, Scene.field/part_view, fluid span/body views and renderer
readers are aliases, not additional owners. Native particle_f is host-owned
accumulation, outside the i4h particle-force-input contract. Its current episode
reset behavior must remain distinct from continuation copying.

## Tet mechanics and physical surface

C2.3 implementation: all fifteen Tet mechanics fields below bind Newton aliases
through `tet_storage.tet_fields(scene, system_id)`. The body's C2.2 variant table
remains shared. Direct Scene builders call `declare_tet` before allocation/schema
registration. CaseAuthoring declares each configured Tet separately, even for
repeated systems on the same body. Edge variant, world-edge, shared edge-part
and world-material domains extend the existing tet domains. All topology and
native_tet columns are local, without Newton reference offsets; identity columns
resolve actual worlds/native owners after world creation. Missing declarations or
changed topology/layouts require reconstruction. Every migrated field and cached
compatibility alias participates in stale-buffer checks.

New State masks clone authored defaults. Binding/preparation does not reset them;
explicit selected-world Tet reset gathers the live Model defaults through the
current body variant table. All four live Model material aliases survive reset.
Scratch delta/count and XPBD/PBD multipliers stay local with unchanged clearing.
C2.4 migrates all physical Surface fields below to Newton through
`surface_storage.surface_fields(scene, system_id)`. Direct callers predeclare
`declare_surface(scene, tet_id=..., system_id=...)` after the matching Tet and
before allocation/schema registration. CaseAuthoring publishes the same records.
Repeated Surfaces retain distinct fields while sharing their referenced Tet's
variant-tet, edge and material domains. Explicit Surface face, particle-support
and shared-part domains use authored native-particle identities; padding, unused
variants and initially unused cut slots keep owner -1 permanently. Shared rows
use world -1; shared face/part rows use the first world as representative. The
three world counters use native WORLD frequency. Local topology has no offsets.

New State templates initialize authored variant triangles, owners and supports,
and zero cut_count/overflow/deleted/dirty. Binding and preparation preserve every
selected byte, including pending dirty work and latched overflow. Explicit reset
reads live Model defaults through the current body variant table and clears only
selected bookkeeping; it does not reset Tet masks. Rebuild order and atomic slot
semantics are unchanged. All physical compatibility aliases are checked. Authored
topology, mappings, cut ranges and dimensions are frozen; changed authoring
requires reconstruction. Valid live Model edits survive binding and reset.

Surface's three additive/runtime duplicates are removed. Thermal heat/burn
additive and legacy copies remain. Surgical Appearance still allocates zeroed
history after reference validation and has explicit appearance reset; Thermal
history still follows legacy construction. Commands, rods, coupling and the six
residue arrays retain their owners. Physical Tet/Surface rebinding is supported;
this is not full fresh-runtime continuation.

Logical Surface bytes per system, excluding identities and inaccessible backing:
Model `56VT + 16VF + 4F + 4VN + 4VS + 16P`; each State and initialization
templates separately `16EF + 4EN + 4ES + 4EP + 12E`. The three existing per-tet
Model declarations already contributed `56VT` before C2.4. New Surface identities
cost `8(VF+EF+F+VN+EN+P)` logical bytes; reused Tet domains are counted once
under Tet. Empty custom domains keep one inaccessible backing row, separately
reported by owner and with an extra State-sized template allocation.

Logical Tet bytes per system (identity columns and empty backing excluded):
Model `24VT + 4T + 16VS + 4S + 4ET + 16EP`; each State and its initialization
templates separately `4E(T+S)`. C2.2 already declared Model `24VT+4T+4ET`,
State `4ET` and templates `4ET` additively. C2.3 adds edge/material Model
`16VS+4S+16EP`, State `4ES`, templates `4ES` and the edge/material identities.
It removes legacy topology/default/mask/material allocations rather than adding
second runtime copies. Zero-count domains retain the existing one-row backing rule.

Sources: [Tet](../physics/src/i4h_physics/systems/mechanics/tet.py),
[Surface](../physics/src/i4h_physics/systems/mechanics/surface.py).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| Tet.tets | M, vec4i, (V,T) | m.tets; solve_tet and surface/deletion consumers read local vertex indices. |
| Tet.active0 | M, i, (V,T) | m.active0; selected reset gathers variant rows. |
| Tet.active | S, i, (E,T) | active0[variant]; solve_tet reads, surface deleters atomically clear. |
| Tet.rest_volume | M, f, (V,T) | m.rest_volume; solve_tet reads. |
| Tet.tet_part | M, i, (T,) | m.part; solve_tet/deleters read shared local part ids (not (V,T)). |
| Tet.edges | M, vec2i, (V,S) | m.edges; solve_tet and stretch-breaking read. |
| Tet.edge_active0 | M, i, (V,S) | m.edge_active0; variant reset reads. |
| Tet.edge_active | S, i, (E,S) | edge_active0[variant]; solve_tet reads, deletion/breaking clear. |
| Tet.edge_rest | M, f, (V,S) | m.edge_rest; solve_tet and stretch-breaking read. |
| Tet.edge_part | M, i, (S,) | m.edge_part; local part indexing. |
| Tet.spring, spring_damping, volume, volume_damping | M, f, (E,parts) each | Tile each matching m material table; edits persist across reset; solve_tet reads. |
| Tet.delta / delta_count | X, vec3 / i, (E*body.count,) | Zero; solve_tet accumulates, apply_average consumes and clears all rows. |
| Tet.edge_lambda / tet_lambda | X, f, (E,S)/(E,T), or (1,1) PBD fallback | Zero; XPBD substep_start clears before solve; PBD does not consume. |
| reserved native_tet | M, i, (E,T), per i4h tet row | Always -1; no references metadata and no native Newton tets authored. |
| Surface.tet_edges | M, i, (V,T,6) | m.tet_edges; deactivate_tet reads local edge indices; six scalar columns of each variant-tet row. |
| Surface.face_neighbor, face_tri | M, i, (V,T,4) each | Matching m fields; emit_cut_faces, thermal and cutting read local neighbors/face slots. |
| Surface.tris0 / tri_tet0 | M, vec3i / i, (V,F) | m.tris/m.tri_tet, including degenerate capacity padding; reset/rebuild_authored read. |
| Surface.tri_part | M, i, (F,) | m.tri_part; local face-to-part map. |
| Surface.support0 | M, i, (V,body.count) | m.support0; selected reset reads. |
| Surface.edge_support0 | M, i, (V,S) | m.edge_support0; selected reset reads. |
| Surface.heat_enabled, deletable, cut_start, cut_capacity | M, i, (parts,) each | m.heat/deletable/cut_start/cut_capacity; thermal/cutting/face emission read; local slots, no references. |
| Surface.tris / tri_tet | S, vec3i / i, (E,F) | Variant defaults; contact systems read before step_end rebuild, so retain exact slots. |
| Surface.support | S, i, (E*body.count,) | support0[variant]; contacts read and deactivate_tet decrements. |
| Surface.edge_support | S, i, (E,S) | edge_support0[variant]; deactivate_tet decrements. |
| Surface.cut_count | S, i, (E,parts) | Zero; rebuild_authored resets only dirty worlds, emit_cut_faces accumulates. Preserve even when no rebuild occurs. |
| Surface.overflow, deleted | S, i, (E,) each | Zero; latched/cumulative deletion history, also exposed as measurements. Never exclude merely because state_products lists them. |
| Surface.dirty | S, i, (E,) | Zero; deactivate_tet sets, step_end consumes then clears. Preserve declaration even though ordinary completed steps leave zero. |

Surface.variant, active, edge_active, tets, tet_part, edges, edge_rest and edge_part
alias the corresponding Body/Tet domains. TetContainment, Thermal, Cutting,
StretchBreaking, Grasp and Hook read the referenced cached Tet fields without
changing their earlier-system ordering references. VolumetricGrasp and Centreline
consume no migrated Tet topology/mask field and have no Tet reference: they keep
their existing Scene particle flags/positions and Body.variant paths, preserving
the effects of deletion without inventing a new dependency.

## Rod material containers and mechanics

Sources: [regions.py](../physics/src/i4h_physics/regions.py),
[material.py](../physics/src/i4h_physics/systems/rods/material.py),
[RodDirect](../physics/src/i4h_physics/systems/rods/rod_direct.py),
[RodPbd](../physics/src/i4h_physics/systems/rods/rod_pbd.py).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| RegionProperties.edge | M, i, (segments,) | membership.edge; regional material kernels read local region ids. |
| RegionProperties.tables[stretch,bend,twist] (PBD) | M, f, (E,regions) each | One; pbd_regions at first substep reads; episode reset preserves edits. |
| RegionProperties.tables[young_multiplier,torsion_multiplier,bend_multiplier,twist_multiplier] (direct) | M, f, (E,regions) each | One; direct_regions at first substep reads; edits persist. |
| RegionProperties.tables[curvature_x,curvature_y] (direct) | M, f, (E,regions) each | Zero; direct_regions reads; edits persist. |
| RodControls.stretch_scale / bend_scale | C, f, (E,segments)/(E,pairs) | One; PBD reset restores one; control layouts may write in step_start before first-substep material updates. |
| RodControls.bend_offset | C, vec3, (E,pairs) | Zero; rod reset restores zero; catheter/layout writes then rod constraints read. Solver aliases must not allocate a second copy. |
| PBD material.pair | M, vec2i, (1,pairs) | Shared local pair topology; aliases RodPbd.pair. |
| PBD material.radius_baseline | M, f, (segments,) | (mean endpoint radius / maximum)^3; pbd_regions reads when radius weighting enabled. |
| PBD material.stretch / bend / twist | X, f, (E,segments)/(E,pairs)/(E,pairs) | Allocated empty; pbd_regions writes each at s==0 before any enabled regional solver reads. Inactive provider routes solver to control/default arrays. |
| Direct material.rest_len / rest_darboux | M, f / vec3, (1,edges) | Exact rod records; aliases RodDirect rest arrays. |
| Direct material.young_multiplier, torsion_multiplier, bend_multiplier, twist_multiplier | X, f, (E,edges) each | Allocated empty; direct_regions overwrites at s==0 before prepare_regional_compliance. |
| Direct material.preferred_rest | X, vec3, (E,edges) | Allocated empty; direct_regions overwrites before edge_constraints. |
| RodDirect.frames0 / inv_mass_q | M, quat / f, (1,nodes) | Exact rest frames/angular inverse masses; reset/predict read. |
| RodDirect.rest_len / rest_darboux | M, f / vec3, (1,edges) | Alias material provider if present, otherwise own exact record arrays. |
| RodDirect.inertia_local | M, vec3, (nodes,) | cylinder_inv_inertia result; world_inv_inertia reads only physical mode. |
| RodDirect.frame / omega | S, quat / vec3, (E*nodes,) | Variant frames / zero; selected reset restores; predict_rot reads previous frame/omega and integrate_rot writes. |
| RodDirect.frame_pred | X, quat, (E*nodes,) | Initial frames; predict_rot writes all rows before constraints; root writer order must be preserved. |
| RodDirect.compliance | X, f, (E*edges*6,) | Zero; prepare_compliance or prepare_regional_compliance writes at s==0 before constraint read. |
| RodDirect.lambda_sum | X, f, (E*edges*6,) | Zero; edge_constraints ignores old lambda when first==1, compute_corrections overwrites first iteration, accumulates later. |
| RodDirect.rhs / delta_lambda | X, f, (E*edges*6,) each | Zero; edge_constraints / thomas overwrite before thomas / compute_corrections reads. |
| RodDirect.jacobian_rot | X, f, (E*edges*36,) | Zero; edge_constraints overwrites before assembly/correction. |
| RodDirect.diag_blocks / offdiag_blocks / c_blocks | X, f, (E*edges*36,) each | Zero; assemble_jmjt overwrites first two, thomas overwrites recurrence rows before reuse. |
| RodDirect.edge_w | X, vec3, (E*edges,) | Zero; edge_constraints overwrites before assemble/correction. |
| RodDirect.pos_corr / rot_corr | X, vec3, (E*nodes,) each | Zero; compute_corrections accumulates and apply_corrections clears. |
| RodDirect.inv_inertia | X, mat33, (E*nodes,) physical or (1,) unit | Zero; physical world_inv_inertia writes before assembly; unit mode does not consume fallback. |
| RodPbd.seg / pair | M, vec2i, (1,segments)/(1,pairs) | Chain/tree local topology; pair aliases provider when present; cosserat_jacobi reads. |
| RodPbd.frames0 / rest_darboux | M, quat, (1,segments)/(1,pairs) | Authored frames / rest_darboux_quats; reset/constraints read. |
| RodPbd.rest_len / inv_mass_q | M, f, (1,segments) each | Exact records; prediction/constraints read. |
| RodPbd.unit_twist | M, f, (E,pairs) | One; nonregional effective_twist aliases it. |
| RodPbd.frame / omega | S, quat / vec3, (E*segments,) | Rest / zero; selected reset restores; predict/finalize read/write. |
| RodPbd.frame_pred | X, quat, (E*segments,) | Initial rest; predict_orientations overwrites before constraints. |
| RodPbd.pos_corr / pos_count | X, vec3 / i, (E*nodes,) | Zero; cosserat_jacobi accumulates, apply_corrections consumes/clears. |
| RodPbd.ori_corr / ori_count | X, vec4 / i, (E*segments,) | Zero; same self-clearing contract. |

RodDirect.bend_offset and RodPbd.bend_offset/stretch_scale/bend_scale are
RodControls aliases. preferred_rest/effective_stretch/effective_bend/effective_twist
route to declared material/control/default owners, never independent history.
RegionProperties.edit allocates a temporary CPU staging array; it is not retained
physics. Its explicit restore API remains different from episode reset.

## Commands, tool history and mounts

Sources: [InstrumentControl](../surgical/src/i4h_surgical/systems/instruments/instrument_control.py),
[CatheterControl](../endoluminal/src/i4h_endoluminal/systems/rods/catheter_control.py),
[coupling.py](../physics/src/i4h_physics/coupling.py),
[ControlLayout](../endoluminal/src/i4h_endoluminal/systems/rods/control_layout.py),
[Drag](../endoluminal/src/i4h_endoluminal/systems/rods/drag.py).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| InstrumentControl.actions | C, f, (E,n,10) | Initial position, quaternion xyzw, zero grip/energy/fire; selected reset restores valid pose; update_pose reads packed columns in existing order. |
| InstrumentControl.root | S, vec3, (E,n) | Initial position; reset/step_start pose update. |
| InstrumentControl.shaft_q, wrist_q, jaw_q | S, quat, (E,n) each | Existing reset-pose kernel results, not zero quaternions; update_pose/geometry read/write. |
| InstrumentControl.jaw_angle, jaw_delta | S, f, (E,n) each | Existing reset result / zero; update_pose reads previous jaw and writes delta; cutting consumes delta. |
| InstrumentControl.held, grasp_pending, released, energized, fire_on, fired | S, i, (E,n) each | Existing reset zeros; trigger edge/held history updated at step_start before grasps/clips/cutting. |
| InstrumentControl.sphere_q, sphere_q_prev | S, vec3, (E,spheres) each | Valid authored sphere geometry in both; place_spheres advances history before contact/hook. |
| InstrumentControl.sphere_active | S, i, (E,spheres) | Reset-pose geometry result; place_spheres writes before consumers. |
| InstrumentControl.mounted | M, i, (max(n,1),) | Zero then pre-finalization mount mapping; episode reset preserves mounting. |
| InstrumentControl.mount_pose | C, transform, (E,n) | Initial authored root/orientation; mounted drive writes before i4h, external writers may write between steps. |
| InstrumentControl._all | X, b, (E,) | Constant true initialization mask; no history. |
| InstrumentControl.trocar, initial_position, wrist_pivot, jaw_anchor, wrist_end | M, vec3, (n,) each | Matching instrument records, zero fallback trocar/wrist_end; pose/reset/geometry kernels read. |
| InstrumentControl.initial_rotation | M, quat, (n,) | Authored rotation; reset reads. |
| InstrumentControl.jaw_open, jaw_closed, jaw_response, grasp_threshold, shaft_spacing | M, f, (n,) each | Matching authored values; update_pose/place_spheres read. |
| InstrumentControl.has_trocar, has_wrist_end, prim_start, prim_counts, jaw_root_a, jaw_root_b | M, i, (n,) each | Authored presence and local primitive/sphere ranges; geometry kernels read. |
| InstrumentControl.wrist_end_bind | M, mat44, (n,) | Authored bind matrix; geometry reads. |
| InstrumentControl.sphere_instrument, sphere_role, sphere_chain, sphere_chain_index, sphere_chain_count | M, i, (spheres,) each | InstrumentLayout columns; place_spheres reads local instrument/chain ids. |
| InstrumentControl.sphere_jaw_sign, sphere_radius | M, f, (spheres,) each | Layout values; geometry/contact reads. |
| InstrumentControl.sphere_local / sphere_bind | M, vec3 / mat44, (spheres,) | Layout values; place_spheres reads. |
| InstrumentControl.prim_a, prim_b, prim_instrument, prim_kind | M, i, (primitives,) each | Layout columns; contact and tool selection read local indices. |
| CatheterControl.actions | C, f, (E,3) | Zero insertion/rotation/bend rates; selected reset zero; advance_targets reads. |
| CatheterControl.targets | S, vec3, (E,) | Zero; advance_targets integrates and track/root guidance reads. |
| CatheterControl.track_start / track_direction | M, vec3, (V,) | Exact rod start/direction; guidance reads. |
| CatheterControl.base_orientation | M, quat, (V,) | m.frames[:,0]; guidance reads. |
| CatheterControl.mounted | M, i, (1,) | Pre-finalization mapping; reset preserves. |
| CatheterControl.mount_pose | C, transform, (E,1) | Authored variant start/root frame; mounted drive writes before controls. |
| CatheterControl.root | S, vec3, (E,1) | Variant start; guide updates; coupling reads root for moment conversion. |
| CatheterControl.impulse, moment | O, vec3, (E,1) each | Zero; step_start clears, guidance accumulates, coupling copies into pending State reactions after step. |
| MountGroup.target | M, i, (mounts,) | Target local tool index; retain current indexing, owner reference belongs in domain metadata. |
| MountGroup.offset | M, transform, (mounts,) | Normalized authored offset; drive_mounts reads. |
| MountGroup.body | M, i, (E,mounts) | Builder-resolved host body indices; genuine BODY reference; apply_reaction and drive_mounts read. |
| MountGroup.force, torque | S, vec3, (E,mounts) each | Zero; apply_reaction reads previous step before rigid stepping, fill_reaction writes next pending reactions. Selected episode reset clears. |
| Mounts.env_xform | M, transform, (E,) | Authored world/component transforms; drive and reaction conversion read. |
| ControlLayout.hops / arc_offset | M, i / f, (L,C,P) | Authored graph hops/root-relative arc; material/layout projection reads. |
| ControlLayout.edges / pairs | M, vec2i, (S,)/(K,) | Local tree topology; materials/materials_pairs reads. |
| ControlLayout.origin / direction | M, vec3, (L,C) | Authored live layout configuration, shared across worlds using a layout; projection reads. These are layout parameters, not per-episode action channels. |
| ControlLayout.active | M, i, (L,C) | Authored live layout enabled flags; materials and projection read. |
| ControlLayout.layout_of_env | M, i, (E,) | arange(E)%L; authored world-to-layout indexing; material/projection reads. |
| ControlLayout.stiffness, falloff, bend_mult, stretch_mult | M, f, (L,C) each | Authored/clamped values; materials/projection read; preserve material edits. |
| ControlLayout.rest_bend | M, vec2, (L,C) | Authored rest bend; materials_pairs reads; material edits persist. |
| Drag.active / node | C, i, (E,) | Zero; begin/move/end write, apply_drag reads; episode reset clears active only. |
| Drag.target | C, vec3, (E,) | Zero; begin/move writes; apply_drag reads. |
| Drag.weights / baseline | S, f / vec3, (E*P,) | Zero initially; begin computes weights and snapshots current positions; persists until next begin. Episode reset leaves buffers but disables active. |
| Drag.ref | X, vec3, (E*P,) | Zero; substep_start gather_rows writes before apply_drag reads. |

ControlLayout.prepare writes RodControls defaults, and CatheterControl.prepare
clears bend_offset (including slots outside its active tip). Those writes must
move to authoring, leaving prepare history-preserving. Layout configuration stays
Model-owned and shared, matching the existing round-robin lookup. Continuation
requires unchanged Model parameters; it checkpoints the per-world RodControls
channels rather than silently modifying a shared layout for unselected worlds.

## Graph containment and accumulated evidence

Sources: [GraphContainment](../physics/src/i4h_physics/systems/contact/graph_containment.py),
[TipEvidence](../endoluminal/src/i4h_endoluminal/systems/tip_evidence.py).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| GraphContainment.edges / controls / neighbors | M, vec2i / vec4i / vec2i, (tree_edges,) each | Exact first-variant tree topology; candidate construction/projection reads. |
| cand_idx | S, i, (E*samples*capacity,) | -1; _refresh_candidates may retain old candidates, projection consumes. |
| cand_count, cand_overflow | S, i, (E*samples,) each | Zero; retained with candidates; projection consumes count/overflow. |
| ref_samples / ref_tree | S, vec3, (E*samples,)/(E*tree_nodes,) | Zero; displacement checks read before deciding whether to rebuild. |
| rebuild / built | S, i / vec2, (E,) | One / zero; request_rebuild consumes thresholds/history; candidate build updates together. |
| max_disp | X, f, (2*E,) | Zero; displacement accumulation then request_rebuild consumes and clears; initialize fresh scratch to zero. |
| cath_corr / cath_count | X, vec3 / f, (E*cath_nodes,) | Zero; project accumulates, apply_corrections consumes and clears all rows. |
| tree_corr / tree_count | X, vec3 / f, (E*tree_nodes,) | Zero; same when two_way; one-way projection does not accumulate tree corrections. |
| contact_depth / contact_count | O, f / i, (E,) | Zero; substep_start clears then projection accumulates. |
| overflow | O, i, (E,) | Zero; step_start clears, projection counts current-step overflows. Different from retained cand_overflow. |
| TipEvidence.positions / active | S, vec3 / i, (E,capacity) | Zero; record_tip appends one slot per step; reset selected rows zero. |
| TipEvidence.count / overflow | S, i, (E,) | Zero; record_tip advances/latches across steps; never classify as disposable measurements. |

TipEvidence.q/flags are aliases of the declared catheter particle domain. Body,
distal node and capacity must be authored in the systems list. All seven
GraphContainment candidate/reference/rebuild fields are copied together; restore
must not invoke refresh or change candidate-rebuild timing. Existing GraphContainment.prepare sets rebuild=1; C4 must remove that history write from fresh binding/capture preparation without replacing it with an unconditional refresh.

## Other mechanics and contact

Sources are the named modules under [physics systems](../physics/src/i4h_physics/systems)
and [surgical contact](../surgical/src/i4h_surgical/systems/contact).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| Attachments.index / rest | M, i / vec3, (V,pins) | b.pin.index/rest, including -1 padded indices; attachment projection reads; body.variant alias. |
| Connectors.particle / tri / bar / rest | M, i / vec3i / vec3 / f, (V,C) | c.particle/tri/bar/rest; solve_connectors reads local topology and rest geometry. |
| Connectors.group | M, i, (C,) | c.group; connector material lookup reads. |
| Connectors.stiffness, damping, weight_ratio | M, f, (E,groups) | Tile c material values; projection reads; preserve edits. |
| Connectors.delta | X, vec3, (E*body.count,) | Zero; solve_connectors accumulates, apply_direct consumes and clears. |
| Shell.edges / edge_rest | M, vec2i / f, (V,S) | Matching shell records; solve_shell_xpbd reads. |
| Shell.dihedrals / dihedral_rest / dihedral_edge_len | M, vec4i / f / f, (V,D) | Matching shell records; solve_shell_xpbd reads. |
| Shell.lambda_s / lambda_d | X, f, (E,S)/(E,D) | Zero; substep_start clears before XPBD accumulation. |
| Shell.delta / delta_count | X, vec3 / i, (E*body.count,) | Zero; solve accumulates, apply_average consumes/clears. |
| Physical StretchBreaking.target | M, i, (parts,) | Authored selected parts; measure_and_break/break_edges read. |
| Physical StretchBreaking.stretch | X, f, (E*body.count,) | Zero; measure_and_break accumulates, finish_stretch reads and unconditionally clears. Surgical subclass reads same scratch when painting. |
| Physical StretchBreaking.max_stretch | O, f, (E,) | Zero; step_start clears, finish_stretch writes; reset zero. |
| StaticContact.particle_enabled | M, i, (body.count,) | Selected authored part ranges, including internal gaps; project_static reads. Static wp_mesh aliases Scene cache. |
| FluidContact.lowers / uppers | X, vec3, (E*F,) | Zero; bounds kernel writes before BVH refit/query. |
| FluidContact.tri_group | M, i, (E*F,) | repeat(arange(E), F); BVH grouping. |
| FluidContact.bvh | X, Bvh | Rebuildable from bounds/groups; no physics history. |
| PartContact.targets | M, i, (N,) | Selected local particle ids; query/projection read. |
| PartContact.lowers / uppers | X, vec3, (E*F,) | Zero; _bounds overwrites before bvh.refit and query. |
| PartContact.groups | M, i, (E*F,) | repeat(arange(E),F); BVH grouping. |
| PartContact.candidates / candidate_count | X, i, (E,N*max_candidates)/(E,N) | Zero; query_candidates overwrites valid rows/count before accumulate_candidates; unused slots are masked by rewritten counts. |
| PartContact.delta / delta_count | X, vec3 / i, (E,body.count) | Zero; candidate/fallback kernels accumulate, apply_deltas consumes/clears every row. |
| PartContact.contact_count / overflow_count | O, i, (E,) | Zero; step_start clears, query/accumulation writes; reset zero. |
| PartContact.bvh | X, Bvh | Rebuildable from current bounds; body q/inv_mass/radius/flags are Scene aliases. |
| ShellContainment.tris | M, vec3i, (V,F) | sh.shell.tris; classification/projection reads; variant aliases wall Body.variant. |
| TetContainment.variant | M, i, (E,) | arange(E); mutable surface rows already world-major. tris aliases Surface.tris, not an independent immutable topology. |
| ShellContainment.lowers, uppers; TetContainment.lowers, uppers | X, vec3, (E*F) each | Zero; _refit writes before queries. |
| ShellContainment.tri_group; TetContainment.tri_group | M, i, (E*F,) each | repeat(arange(E),F); BVH grouping. |
| ShellContainment.sides; TetContainment.sides | X, f, (E*rod_nodes,) each | -1; two-sided substep_start writes classification before projection; one-sided mode does not read. Reset -1. |
| ShellContainment.wall_delta; TetContainment.wall_delta | X, vec3, (E*wall_nodes,) each | Zero; two-way projection accumulates, apply_direct consumes/clears; one-way does not accumulate. |
| ShellContainment.bvh; TetContainment.bvh | X, Bvh | Rebuild/refit before query from declared topology/current positions. |
| RodContainment.contact_depth; ShellContainment.contact_depth; TetContainment.contact_depth | O, f, (E,) each | Zero; substep_start clears, projection accumulates; reset zero. |
| RodContainment.contact_count; ShellContainment.contact_count; TetContainment.contact_count | O, i, (E,) each | Same current-substep lifecycle. |
| InstrumentContact.tri_front_only | M, i, (F,) | front[body.tet.tri_part]; project_contacts reads authored front policy per slot. |
| InstrumentContact.prim_records | M, i, (primitives,) | Authored tip-record flags; project_contacts reads. |
| InstrumentContact.delta / delta_count | X, vec3 / i, (E*body.count,) | Zero; project_contacts accumulates; apply_average_tools consumes/clears. |
| InstrumentContact.delta_k / delta_jaw | S, vec3, (E*body.count,max(tools,1)) each | Zero initially; projection accumulates, apply_average_tools conditionally consumes/clears. Nonzero underflow residues survive. Episode reset/disabled prepare leave them intact; preserve exact bytes in continuation. |
| InstrumentContact.impulse, jaw_impulse, moment | O, vec3, (E,max(tools,1)) each | Zero; step_start clears; apply_average_tools accumulates; ToolForce/exports/coupling consume completed step. |
| InstrumentContact.impulse_count | O, i, (E,max(tools,1)) | Zero; step_start clears, project_contacts accumulates. |
| InstrumentContact.record_count | O, i, (E,primitives) | Zero; substep_start clears before records and physics consumers. |
| InstrumentContact.rec_tri / rec_tet | O, i, (E,primitives,records) each | Zero; final contact event writes valid slots; record_count masks unused slots. |
| InstrumentContact.rec_verts / rec_bary / rec_normal | O, vec3i / vec3 / vec3, (E,primitives,records) | Same valid-slot contract; grasp/thermal/hook read after generation. |
| InstrumentContact.rec_pen / rec_t | O, f, (E,primitives,records) each | Same valid-slot contract. |
| InstrumentContact.record_overflow | O, i, (E,) | Zero; step_start clears; count_record_overflow accumulates substep overflow. |
| InstrumentContact.suppress_active / suppress_verts fallback | M, i / vec3i, (E,) | Constant zeros when no hook; Hook replaces aliases with its retained entry State. Remove superseded fallback allocation when binding the alias. |

Particles and Bounds have no private history beyond Scene arrays. Bounds uses
host scalar configuration; Particles' live arrays are listed below. The surgical
StretchBreaking subclass adds live parameters only, sharing physical state and
Appearance.visual. Contact measurements can be invalid after restore until the
next completed step; physics records must still be generated before consumers.

## Fluid, emission and smoke

Sources: [Fluid](../physics/src/i4h_physics/systems/fluids/fluid.py) and
[surgical fluids](../surgical/src/i4h_surgical/systems/fluids).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| Fluid.const[prefix] | M, i, (nf+1,) | cumsum([0]+counts); pool_slot in prediction/neighbors/etc. reads. |
| Fluid.const[counts,base,nbr_base,max_neighbors] | M, i, (nf,) each | Exact counts/span-relative bases/neighbour offsets/capacities; span-wide kernels read. |
| Fluid.const[radius,rest_density,rest_distance,density_epsilon] | M, f, (nf,) each | Authored pool constants; neighbors/lambdas/deltas read. |
| Fluid.group | X, i, (span,) | -1; predict_fluid rewrites owned pool slots before grouped grid build. Gaps retain -1; fresh scratch must initialize them. |
| Fluid.lam_span / density_span | X, f, (span,) | Zero; lambdas writes selected live slots before deltas/consumers; density per-pool views alias density_span. |
| Fluid.delta / qd_scratch | X, vec3, (span,) each | Zero; deltas overwrites before apply; finalize_fluid overwrites before xsph. Inactive/gap rows are excluded by flags/pool layout. |
| Fluid.nbr / nbr_count | X, i, (nbr_base[-1],)/(span,) | Zero; neighbors writes valid slots/counts before lambdas/xsph; unused slots count-masked. |
| Fluid.grid | X, HashGrid | Reserved grouped capacity before capture; rebuilt after predict_fluid, before neighbor queries. Corrected Warp binary required. |
| Fluid.cursor | S, i, (E,nf) | Zero; emitters advance ring slots before fluid; selected reset zero. |
| Fluid.overflow | S, i, (E,nf) | Zero; neighbors accumulates overflow since reset; retain even though exposed as a readout. |
| Fluid.alive_count | O, i, (E,nf) | Zero; current live occupancy recomputed; reset zero. Not cumulative blood loss. |
| Fluid.relaxation, viscosity, cohesion | M, f, (E,nf) | Tile authored pool values; apply/xsph/deltas read; edits persist. |
| Bleeding.cut / source | S, i, (E,K) each | Zero; mark_cuts latches at step_end; find_leaks reads cut and writes sources at next step_start before bleed. Keep both explicitly. |
| Bleeding.acc | S, f, (E,K) | Zero; bleed carries fractional spawn remainder. |
| Bleeding.serial | S, i, (E,) | Zero; find_leaks advances, bleed uses deterministic spawn sequence. |
| Bleeding.clipped | Alias to Clipping.clipped S, or M zero fallback i (E,K) | No clipping: constant zero; otherwise same retained clipped buffer, not a copy. |
| Bleeding.sealed / sources | O, i, (E,) | Zero; find_leaks rewrites; reset zero. Phase is step_start, before current-step cuts. |
| Bleeding.gate | Alias | Cutting.cut_on or InstrumentControl.fire_on; consume at mark_cuts after current cut signal. |
| Emitter.position / direction | C, f, (E,3) each | Authored position/direction, or zero unused position in tool mode; emit_point/tool reads. Episode reset preserves source commands. |
| Emitter.enabled_env | C, f, (E,) | Authored enable; emit_point/tool reads; reset preserves. |
| Emitter.acc / serial | S, f / i, (E,) | Zero; emit_point/tool carries fractional remainder and serial; selected reset zero. |
| Emitter.spawned | O, i, (E,) | Zero; emitter writes current-step count; reset zero. |
| Smoke.source_prim | M, i, (primitives,) | Authored tool-owned non-wrist/non-shaft selection; spawn_smoke reads. |
| Smoke.pos_age / vel_life | S, vec4, (E*count,) each | Zero/dead lifetime; spawn_smoke and update_smoke retain particle age/lifetime; selected reset zero. |
| Smoke.cursor / serial | S, i, (E,) each | Zero; spawn_smoke advances ring and sequence; selected reset zero. |
| Smoke.acc | S, f, (E,) | Zero; spawn_smoke carries fractional remainder; reset zero. |
| Smoke.hot / alive_count | O, i, (E,) each | Zero; spawn/update write current values; reset zero. |
| Smoke.centre | O, vec3, (E,) | Zero initially; spawn_smoke writes only when a source exists. Early returns retain the previous value, which is invalid when hot=0. No physics consumer reads it; reset does not clear it. It is not proven-overwritten solver scratch. |
| Suction.removed | S, i, (E,nf) | Zero; suck only atomic-adds removed particles, never clears per step; selected reset zero. It is cumulative history, not disposable output. |

Fluid.q_span/qd_span/q_prev_span/inv_mass_span/flags_span/radius_span/q0_span and
q_b/qd_b/q_prev_b/flags_b are zero-copy aliases of Scene.pool, preserving its
body-major layout and non-fluid gaps. Native and pool storage are not conflated.

## Surgical attachment, thermal and appearance history

Sources: [instruments](../surgical/src/i4h_surgical/systems/instruments) and
[mechanics](../surgical/src/i4h_surgical/systems/mechanics).

| Field | Owner / dtype / shape | Authored/reset value; first subsequent use |
| --- | --- | --- |
| Grasp.grasps | M, i, (tools,) | Instrument-kind flags; capture_grasps reads. |
| Grasp.stiffness; VolumetricGrasp.stiffness | M, f, (E,tools) each | Authored per-tool grasp stiffness; hold kernels read; edits persist. |
| Grasp.prim / tet | S, i, (E,tools,slots) each | -1 initially; reset prim=-1 and active=0, leaves tet history inert; release/hold/revalidate/capture read/write in existing phases. |
| Grasp.t / gap | S, f, (E,tools,slots) each | Zero initially; capture writes before active hold; episode reset leaves inactive slot payload. |
| Grasp.radial / prev_axis / bary | S, vec3, (E,tools,slots) each | Zero initially; capture/hold write; retained across steps and inactive reset slots. |
| Grasp.verts | S, vec3i, (E,tools,slots) | Zero initially; capture writes anchor; hold/revalidate read; reset leaves inactive payload. |
| Grasp.active | S, i, (E,tools,slots) | Zero; release/hold/revalidate/capture update; selected reset zero. |
| Grasp.count / pending | S, i, (E,tools) each | Zero; release/capture update; selected reset zero. |
| Grasp.delta / delta_count | X, vec3 / i, (E*body.count,) | Zero; hold_grasps accumulates, apply_average_tools consumes/clears. |
| Grasp.delta_k / delta_jaw | S, vec3, (E*body.count,max(tools,1)) each | Zero initially; hold_grasps accumulates, apply_average_tools conditionally clears. Retain underflow residues; episode reset/disabled prepare leave them intact. |
| Grasp.impulse / jaw_impulse / moment | O, vec3, (E,max(tools,1)) each | Zero; step_start clears, apply_average_tools accumulates, coupling consumes after step. |
| VolumetricGrasp.grasps / jaw_start / jaw_count | M, i, (tools,) each | Authored tool/geometry selection; arm_capture/capture_particles read. |
| VolumetricGrasp.particle | S, i, (E,tools,capacity) | -1; release/arm/capture update; reset -1. Local particle indices, no Newton references. |
| VolumetricGrasp.sphere | S, i, (E,tools,capacity) | Zero; capture writes, hold reads; reset zero. |
| VolumetricGrasp.rest | S, vec3, (E,tools,capacity) | Zero initially; capture writes, hold reads; reset leaves inactive payload. |
| VolumetricGrasp.active | S, i, (E,tools,capacity) | Zero; release/arm/capture update; reset zero. |
| VolumetricGrasp.count / pending / claim | S, i, (E,tools) each | Zero; release/arm/capture/finish update, reset zero. Retain claim with attachment history. |
| VolumetricGrasp.delta / delta_count | X, vec3 / i, (E*body.count,) | Zero; hold_particles accumulates, apply_average_tools clears. |
| VolumetricGrasp.delta_k / delta_jaw | S, vec3, (E*body.count,max(tools,1)) each | Zero initially; hold_particles accumulates, apply_average_tools conditionally clears. Retain underflow residues; episode reset/disabled prepare leave them intact. |
| VolumetricGrasp.impulse / jaw_impulse / moment | O, vec3, (E,max(tools,1)) each | Zero; current-step reaction accumulation, reset zero. |
| Hook.verts / tet | S, vec3i / i, (E,2) | -1; acquire writes, retain reads; selected reset -1. |
| Hook.bary / radial | S, vec3, (E,2) | Zero initially; acquire/retain updates; reset leaves inactive payload. |
| Hook.friction_coord | S, f, (E,2) | Zero initially; retain updates axial history, acquire seeds; reset leaves inactive payload. |
| Hook.active | S, i, (E,2) | Zero; retain/acquire update, reset zero. entry_active/entry_tet/entry_verts are first-column aliases, also used by contact suppression/thermal. |
| Hook.impulse / moment | O, vec3, (E,tools) | Zero; step_start clears, retain accumulates, coupling reads. |
| Clipping.eligible | M, i, (K,) | Authored centreline selection; attach_clip reads. |
| Clipping.node / active | S, i, (E,slots) each | -1 / zero; attach_clip and clamp projection read/write; selected reset defaults. |
| Clipping.count | S, i, (E,) | Zero; attached-slot history remains occupied after tear; reset zero. |
| Clipping.clipped | S, i, (E,K) | Zero; latched clipping history also read by Bleeding; reset zero. |
| Thermal.source_prim | M, i, (primitives,) | Tool tip selection; deposit_heat reads. |
| Thermal.heat / burn | S, f, (E,T) each | Zero; deposit/diffuse/burn retain thermal history; selected reset zero. |
| Thermal.coag | S, f, (E*body.count,) | Zero; burn_and_delete atomic-maxes; paint_coag reads, remains accumulated; reset zero. |
| Thermal.heat_next | X, f, (E,T) | Zero; diffuse_heat writes before burn; even sweep count explicitly copies heat to heat_next. |
| Thermal._no_hook | M, i, (E,) | Constant zero fallback; deposit_heat reads only when hook absent. |
| Cutting.cut_on | O, i, (E,tools) | Zero; arm_cuts overwrites step_start, Bleeding reads step_end; disabled prepare clears stale signal. |
| Cutting.cut_tets | X, i, (E,T) | Zero; cut_tets writes each row before neighbour_blood reads in last-substep path; reset zero. |
| Cutting.blood_acc | X, f, (E*body.count,) | Zero; neighbour_blood accumulates, paint application consumes/clears. |
| Appearance.visual | S, vec4, (E*body.count,) | Zero; damage/coag/blood writers accumulate, renderer reads; selected reset zero. Surgical Surface owns Appearance; physical Surface does not. |
| ContactPaint.paint | M, f, (E,3) | Authored damage/coag/blood color; paint_contacts reads; material/config edits persist. |
| ContactPaint.painted | X, i, (E*body.count,) | Zero; paint_contacts increments, clamp_paint clears touched rows; zero untouched rows remain zero. |
| Centreline.clamp_start / clamp_count | M, i, (V,K) | Authored clamp ranges/counts; update_centreline/reset reads. |
| Centreline.clamp | M, i, (V,clamp_entries) | Exact c.clamp; update_centreline reads local particle ids. |
| Centreline.rest | M, vec3, (V,K) | c.rest; selected reset reads. |
| Centreline.positions / active_count | S, vec3 / i, (E,K) | Rest / full variant clamp counts; step_end refresh updates, fully removed ring retains last position; Clipping reads before next refresh. |
| ToolForce._zero | M, vec3, (E,max(tools,1)) | Constant zero fallback for absent grasp/hook sources. |
| ToolForce.force / count / jaw_load | O, vec3 / i / f, (E,max(tools,1)) | Zero; tool_force overwrites completed step; reset zero. Conditioned outputs distinct from raw wrench exports. |

## Expanded live parameter arrays

Every scalar `_live` below is M/f/(E,), initialized from the YAML/dataclass value
and read at the named system's ordinary kernel phase. Vector exceptions are
listed explicitly. These arrays are not reconstructed or reset during binding;
episode reset preserves material/configuration edits. Externally commanded
Emitter fields were assigned C above and are excluded from the Model list.

| System | Model live fields | First subsequent reader |
| --- | --- | --- |
| Particles | damping, max_velocity | predict/finalize |
| Attachments | stiffness | attachment projection |
| Shell | stretch, bend, relaxation | solve_shell_xpbd |
| Physical StretchBreaking | threshold | measure_and_break / break_edges |
| Surgical StretchBreaking | damage_start, damage_full, blood | surgical finish/appearance update |
| RodDirect | young, torsion, bend, twist, angular_damping | compliance preparation / predict_rot |
| RodPbd | stretch, bend, twist, angular_damping | cosserat_jacobi / finalize_orientations |
| StaticContact, FluidContact, PartContact (separate instances) | margin, stiffness, max_correction | each contact query/project |
| RodContainment | sign | project_static |
| ShellContainment, TetContainment (separate instances) | sign, target_response, max_distance | classify_sides / project_coupled |
| GraphContainment | max_distance, skin, target_response, source_max_delta, target_max_delta | candidate displacement/rebuild and projection |
| Fluid | damping, max_velocity | predict_fluid / finalize_fluid / xsph |
| InstrumentContact | margin, stiffness, static_friction, kinetic_friction, max_correction | project_contacts |
| InstrumentControl | max_velocity, energy_threshold, fire_threshold | update_pose |
| CatheterControl | track_length, track_stiffness, max_delta; rates, lower, upper are f (E,3) | advance_targets / guide |
| Drag | max_delta | apply_drag |
| Bleeding | direction is f (E,3); margin, rate, speed, spawn_radius | mark_cuts / bleed |
| Emitter | rate, speed, spawn_radius, threshold | emit_point / emit_tool |
| Smoke | rate and spawn_radius are f (E,); lifetime, speed, upward are f (E,2) ranges | spawn_smoke |
| Suction | radius, removal_radius, power | suck |
| Clipping | reach, lumen, stiffness | attach_clip / clamp projection |
| ContactPaint | rate | paint_contacts |
| Cutting | radius_scale, margin, closure_delta, blood | arm_cuts / cut_tets / neighbour_blood |
| Hook | engage_overlap, engage_inward_speed, retention_slack, axial_friction, puncture_distance | retain_hook / acquire_hook |
| Thermal | power, diffusion, cooling, burn_rate, coag_heat, delete_heat, delete_burn | deposit / diffuse / burn |
| ToolForce | contact_gain, grasp_gain, hook_gain, max_force | tool_force |

## Boundary allocations, temporary storage and output containers

These allocations are inventoried but excluded from i4h continuation by the
agreed host/assessment/renderer boundary. Their exact dtype/size expressions are
retained at the source sites in the census. Exclusion does not authorize their
removal or changes to their runtime sequencing.

| Source / fields | Ownership and initialization | First subsequent use / aliases / reset |
| --- | --- | --- |
| I4hSolver._all | X, b (E+1), all true | Solver restore/reset launches; constant full-world mask. |
| Stepper.world_mask; hosted PhysicsBinding.world_mask | X, b (E+1), zero | select_envs writes before reset; _mask aliases first E rows. Rebuild on new binding. |
| _int32_ids temporary arrays | X, i (selected ids,), empty or authored ids | Selection kernel consumes; not retained across operations. |
| Constructor wp.ones masks in rods, CatheterControl, Centreline | Temporary b (E,), true | Existing constructor reset/refresh; these writes must not recur at fresh binding. |
| Storage custom-domain world / owner metadata | M, i, (max(1,domain rows),) each | Authored world/entity reference indices, -1 for unused/shared-world identity. Newton builder merge remaps genuine references; local topology stays local. These additional metadata buffers are counted separately from physics fields. |
| Storage initialization templates and allocated State/Control copies | Declared field dtype/backing row layout; independent allocations except intentional shared controls | Newton finalization retains State/Control defaults; model.state/control allocate copies. Bind creates aliases only. Accounting distinguishes templates, each container and shared control overlap; additive C1 declarations coexist with old runtime allocations. |
| Storage empty-native attribute.build_array | Newton-owned padded backing, declared dtype and max(1,native count) | The finalizer allocates a single element for an empty native field without inventing entities. Model ownership or State/Control initialization templates follow the field assignment; bind exposes only the logical empty view. Allocated bytes include this backing. |
| Storage Views._views | Alias, declared dtype/shape and native placement stride | Cached Warp views retain backing references; storage.attr validates identity/layout. No new device owner; Model/State/Control backing is allocated by Newton. |
| System.live / _live | Allocation helpers, not separate owners | Each expanded live field above owns its array; helper temporaries do not create extra fields. |
| RegionProperties.edit.source | Temporary CPU f (E,regions) | Validated edit copies selected entries; released after operation. |
| Centreline loader bind_skin: verts, positions, frames, edges_d, bound_edge, bound_t, rest_center, rest_orientation | Temporary geometry-processing vec3/vec3/quat/vec2i/i/f/vec3/quat arrays | Bind kernels produce host SkinMesh attributes; runtime Skinner owns renderer copies. Not physics history. |
| Observables.i4h_wrench | O, spatial_vector, output_tool rows | Publisher sums raw completed-step sources; selection aliases same output buffer. |
| Observables.i4h_contact_depth | O, ContactDepth struct, output_depth rows | Publisher copies completed-step depth/count; clear selected outputs after restore. |
| Observables.i4h_surgical_contact | O, SurgicalContact struct, output_record rows | Optional publisher fills final-event valid slots and completed-step geometry. |
| Observables.i4h_surgical_count | O, SurgicalCount struct, output_capsule rows | Publisher writes counts/capacity; invalidation retains capacity. |
| ComponentTorch action/position/product views; observation concatenation; map_actions clamp temporary | Warp-to-Torch aliases retain source owner; normalized-action clamp is a temporary tensor | `_flat` checks pointer identity and reshapes without copies. Observation torch.cat allocates a host-facing (E,observation_width) tensor and promotes state columns to float32. map_actions clamp feeds in-place multiply/add into caller output; neither stores integrated physics history. |
| I4hEnv._actions; I4hActionTerm._raw/_processed | Host action staging Torch tensors | Policy/action processing writes then scatters to packed i4h commands; these host tensors are not authoritative i4h Control storage. |
| I4hEnv._zero_reward / _never_terminated | Host return tensors, Torch zero/falses (E,) | Environment API reads; excluded clocks/rewards/dones. |
| ProcedureTorch.terminal_task, terminal_history, terminal_metrics, new_terminal, recorded; save_terminal.selection | Host terminal records; zeros/-1 and temporary masks | Assessment completion snapshots; reset uses existing terminal policy, independent of i4h restore. |
| TargetMeasurement.rows / mask | Assessment f (E,12) / b (E,), zero | Target measurement computation; excludes assessment state. |
| Procedure.samples, holds, radii | Assessment f (steps,E,12)/(steps,)/(steps,) | Samples zero, authored thresholds; measurement/executor update reads/writes. |
| Procedure.active_step, result, consumed | Assessment i (E,), zero | Assessment state machine; excluded history. |
| Procedure.elapsed / active_hold | Assessment float64 (E,), zero | Assessment clocks; excluded from i4h continuation. |
| Procedure.history / current | Assessment float64 (E,steps,2), -1 / f (E,9), zero | Per-step task history and measurement; excluded. |
| Procedure.path_length / previous_tip | Assessment f (E,) / vec3 (E,), zero | Accumulated path measurement; excluded assessment history, not physics. |
| Procedure.completion_pulses / action_completions | Assessment f (E,steps), zero | Physics/action-cadence aggregation; preserve existing cadence. |
| Procedure.rows / progress / reward / action_reward / mask | Assessment f (E,12)/(E,4)/(E,)/(E,), b (E,), zero | Published assessment observations and masks. |
| TipCamera.host_q / host_f | Camera CPU pinned vec3 (2,) / quat (1,), zero | Camera pose readback; not physics history. |
| TiledCamera.cap_start/cap_stride/cap_node, piece owner/index columns, piece_jaw_sign, piece_bind_q/piece_bind_t | Camera i/f/quat/vec3 authored geometry tables | Camera placement kernels consume; body_q is camera-owned transform staging, not host rigid State. |
| TiledCamera.walls and auxiliary meshes: points/indices/groups | Camera-owned static/triangle geometry and rebuildable Mesh structures | Ray-tracing reads; scene.q references remain aliases. |
| TiledCamera.base_radius/particle_color/row_color, points/radii/tris/tint/normal_accum/normals | Camera authored f/vec3 colors/radii and zeroed render workspaces | Camera update/draw overwrites geometry; tint reads Appearance.visual but does not own accumulated appearance. |
| TiledCamera.model.particle_world, row_env, mesh, camera_transforms | Camera's separate finalized model/index groups and transform arrays | Rendering uses world grouping; this is not a mutation of the physics Model. |
| DragDriver.host; InstrumentDriver.node_frac | Input/app readback vec3 CPU pinned / f (E,) scripted target workspace | Drag picking or scripted source writes commands; source timing remains excluded. |
| ForceMailbox.slots (per slot force/position/rotation/jaw) | Input/output bridge CPU pinned vec3/vec3/quat/f buffers | Servo/render readback ring; no authoritative pending coupling history (MountGroup State owns that). |
| FrameCapture.host | Render capture CPU pinned uint32 pixels | Readback/encoder consumes; excluded renderer state. |
| ClipDraw.centre/tube/line | Renderer vec3/vec3/i authored templates | Clip placement reads current Clipping State; no duplicate clip history. |
| PieceSkin.local/local_normals | Renderer vec3 geometry | Instrument placement uses snapshot/control poses. |
| Skinner.edges/rest_points/bound_edge/bound_t/rest_center/rest_orientation/adj_offsets/adj_vertices | Renderer vec2i/vec3/i/f/vec3/quat/i/i authored SkinMesh copies | vertex_dual_quaternions and smoothing read. |
| Skinner.real[0:2], dual[0:2] | Renderer quat (skin vertices,), zero, two buffers each | Vertex DQ writes first; smoothing ping-pongs; transform reads final. No physics history. |
| render.kernels.Scratch.normal/tangent/bitangent | Renderer vec3 per rendered vertex, zero | Surface shading accumulation and normalization; not persistent tissue appearance. |
| PnDraw.bary | Renderer vec3 tessellation template | PN draw geometry consumes. |
| RegionHighlight.template/edges/membership | Renderer vec3/vec2i/i geometry/region lookup | Highlight placement reads current snapshot; excluded display state. |
| GlRenderer static points/indices/normals/uvs; body/skin/clip/rod draw arrays; piece_points/piece_normals/piece_indices | Renderer geometry copies and zeroed output buffers | Renderer updates/draws; OpenGL buffers/textures remain renderer-owned. |
| GlRenderer fluid/smoke position/radius/color arrays; sphere_radii/sphere_colors; _GlMarkers.q/radii/colors | Renderer copies/staging of live physics output | Refresh from snapshot/marker inputs; accumulated smoke remains Smoke State, not these draw buffers. |
| BodyDraw.uv_array/index_array/color_array; SkinDraw.uv_array/index_array | Renderer authored UV/index buffers and render colors | index_array may alias shared graphics memory; color_array is generated from Appearance State. |
| Markers.q/radius/q_dev/radius_dev/ones/show/template; SlangRenderer.sphere_template/show_regular/show_cut | Renderer staging/display-selection/template arrays | Marker upload/draw; no physics history. |
| SharedBuffer.array | Warp pointer view of graphics-owned storage | Slang/Vulkan buffer allocates; CUDA import/map owns mapping; Warp does not own another copy. Graphics buffers, textures and synchronization resources are excluded. |
| Snapshot and ControlState copied arrays, Snapshots ring slots | Renderer-owned copies (or zero-copy aliases for synchronous Snapshot) of declared source arrays | publish copies completed physics state; render acquire reads. Not an alternative checkpoint format. |
| Snapshot host/task_host_array/task_progress_host_array/task_history_host_array | CPU pinned readback arrays matching source dtype/shape | Snapshot readouts; assessment history remains excluded. |

The ten `I4hComponent._live()` calls in the original census are **not allocation
helpers**: they obtain a live binding. The updated census excludes these false
positives; `System._live("field", ...)` remains counted and expanded above.

## Allocation-site reconciliation follow-up

The source checklist now also records Torch `tensor`, `as_tensor`, `arange` and
`from_numpy` sites, and Warp/Torch interop views. A conversion may alias or
allocate depending on its source device; these entries are not counted as proven
new device buffers. Tensor method calls such as `.to(device)` and external
graphics allocations still need manual review rather than guessed ownership.

Additional reviewed sites:

| Site / expansion | Contract / boundary |
| --- | --- |
| `centreline.bind_skin`: verts, positions, frames, edges_d | Temporary authored vec3/vec3/quat/vec2i device inputs. `bind_vertices` reads vertices/positions/edges; `capture_rest_transforms` reads positions/frames/edges. Construction-only, no episode reset or continuation owner. |
| `centreline.bind_skin`: bound_edge, bound_t, rest_center, rest_orientation | Temporary i/f/vec3/quat outputs per skin vertex, zero before binding kernels; read back into authored host SkinMesh. Renderer later allocates its own copies, already listed above. |
| `ProcedureTorch.step_weights` | Host assessment float32 (steps,) tensor from authored weights; terminal reward aggregation reads it. Excluded assessment configuration; no physics checkpoint owner. |
| `I4hEnv.low/high`, action-manager `_low/_high` | Authored action-bound float32 tensors; normalization reads them. Host action adapter configuration, not packed physics command history. |
| `I4hEnv.reset` as_tensor and manager reset arange | Temporary selected-world ids; reset selection consumes them, no retained physics history. |
| Recording device position/rotation/grip/buttons tensors | CPU episode payload copies, consumed by recorder; preserve recording format, excluded continuation. |
| Device/teleop from_numpy and `.to(device)` | Input-source staging copied into packed command storage. Input-source timing/history remains outside continuation. |
| Stepper from_torch/to-device selection | Zero-copy mask or temporary int32 selected-world ids; selected reset consumes them. Neither is independent authoritative physics history. |

## Explicit boundary-site expansion

| Allocation family | Exact expansion and disposition |
| --- | --- |
| TiledCamera.rays / color / depth4 / shape4 | Newton sensor utility allocations: pinhole rays and per-world color/depth/shape outputs. Renderer-owned; refreshed by camera render, not i4h history. color is uint32 (E,1,H,W); depth/shape have matching image axes. rgb is a zero-copy uint8 RGBA view; depth and shape_index reshape outputs; rgb_t/depth_t alias via to_torch. |
| I4hEnv image `.clone()` | Per-observation RGB/depth snapshots; newly allocated copies, excluded renderer/observation boundary. |
| ProcedureTorch clones | `new_terminal`, rows[:,:8], terminal_task/history/metrics copies published into obs/extras. These are independent assessment payloads, not new physics history. |
| Recording/replay `.clone()` / `.to()` | Initial episode-length copy and action/reference device staging. Host recording/replay owns lifetimes; packed controls remain the destination. |
| Agent `.to()` | Policy observation/reward/done transfer and action upload; potentially alias on same device, otherwise external training buffers. |
| Device `_mid/_half.to()` | Selected authored normalization bounds on simulation device; preserved input-source configuration, not integrated physics targets. |
| Teleop `cat` of action-view clones | Independent physical command staging assembled from command views, then copied into packed commands. Not a second authoritative Control. |
| Renderer SharedBuffer / mesh vertex/index buffers | Graphics-owned shared buffers plus static vertex/index/UV uploads. SharedBuffer Warp view aliases the imported resource; graphics resource is not counted twice as physics allocation. |
| Fluid render textures | Sized render targets allocated by the local texture helper; refreshed from fluid draw passes. No fluid emission/lifetime state lives here. |
| GBuffer normal_depth / label / depth / pool_label | Renderer attachments per viewport; formats and dimensions from framebuffer construction. Written by geometry passes and read by postprocessing. |
| Slang `_textures`, scene_color, depth, ldr, zero_color | Cached image/bitmap textures, resize-time framebuffer targets and constant fallback vertex colors. Renderer-owned, excluded from physics continuation. |
| PN index_buffer / postprocessing textures | Tessellation topology and postprocessing render targets; authored indices or pass output, excluded renderer storage. |

The census includes explicit graphics factories, sensor utility allocations,
Tensor `.clone()`/`.to()` sites and `cat`/`stack` factories. These broaden the
boundary checklist; generic tensor operators and third-party allocator internals
are not inferred to be new persistent i4h fields.

## Source coverage and remaining runtime gates

All 627 recorded allocation/helper sites are linked to these contracts in the
source-hashed inventory. Helper expansions and conditional branches are explicit;
a textual name match alone was not treated as ownership proof. The certificate
applies only to the recorded source and documented external-allocation boundary.

- [x] Separate native aliases, persistent history, outputs and excluded host/renderer storage.
- [x] Expand nested material tables and `_live` system-field families.
- [x] Review construction-only skin binding buffers and additional Torch constructors.
- [x] Reconcile every census site to an explicit field-table entry, including helper/loop expansions and conditional allocations.
- [x] Review explicit tensor method conversions and external graphics factories against the stated boundary.
- [x] Check every scratch first-use claim against reset, capture preparation and fresh-binding paths; dynamic poison qualification remains a later gate.
- [x] Certify the static inventory against preserved source hashes; checker rejects missing sites/contracts and stale source/review documents.
- [x] Connect existing per-tet declarations to CaseAuthoring completion and both adapter hooks.
- [ ] During subsequent migration milestones, extend declarations to remaining families and complete mount metadata before finalization.

Run `python3 scripts/check_stage_c_inventory.py` from `numerical_sim` to verify
coverage and provenance. The checker cannot prove numerical overwrite semantics;
that source review is recorded in the scratch audit, including the CPU/CUDA
counterexample that promoted six per-tool correction arrays to State. Dynamic
poison and continuation tests remain separate C2–C4 acceptance gates.

## C3.1 Instrument ownership (supersedes legacy InstrumentControl allocation rows)

`actions` is Newton Control float32 `(E,N,10)`. Newton State owns `root`,
`shaft_q`, `wrist_q`, `jaw_q`, `jaw_angle`, `jaw_delta`, `held`, `grasp_pending`,
`released`, `energized`, `fire_on`, `fired`, `sphere_q`, `sphere_q_prev`,
`sphere_active`. Nonempty State bytes are `92*E*N + 28*E*S`; Control bytes are
`40*E*N`. Each has a Model initialization template and each independent live or
checkpoint container has independent backing. Explicit `clone_variables=False`
aliases the Control template and changes later clones; ordinary shared live
Control does not.

Model owns the existing default poses, trocar flags/positions, jaw parameters,
wrist/jaw geometry, sphere tables and primitive tables. `max_velocity`,
`energy_threshold`, `fire_threshold` are WORLD-frequency Model arrays.
Instance-qualified shared instrument/sphere/primitive rows have world `-1`;
root instrument owner `-1` is explicitly absent. Sphere/primitive identity
owners refer to their instrument domain. Runtime instrument/sphere rows have
actual world ids and refer to their shared rows. Kernel geometry indices remain
local integer inputs. Empty custom domains retain one inaccessible backing row.

A trailing ONCE Model marker invokes package initializers in declaration order.
Instrument initialization resolves that finalized Model's own templates and
runs unchanged reset then sphere-placement launches. Temporary finalizer-only
unmounted flags, default mount poses and mask use `4*max(N,1)+28*E*N+E` bytes.
Binding and cache preparation validate and alias without resetting history.
Explicit episode reset reads the current Model defaults.

`mounted` and `mount_pose` remain legacy; `_all` remains rebuildable mask scratch.
Host geometry records remain host-owned. Six existing contact/grasp residue
arrays remain legacy State with no new template/checkpoint copies.

`System.command_view_spec` declares half-open final-axis slices. PhysicsBinding
and component mappings are cached and read-only; their Warp/Torch values remain
writable zero-copy views. They retain backing references and validate ownership,
pointers, dtype, device, shape, strides and nested aliases. Replacing storage
requires rebinding, including at `prepare()`. Packed commands and recording
formats remain unchanged. Mounted-tool episode restore remains partial until C3.3.

## C3.2 Rod and endoluminal ownership (supersedes legacy allocation/preparation rows)

Physics declares rod storage in `i4h_physics.rod_storage`; endoluminal declares
catheter, layout and drag storage in `i4h_endoluminal.rod_storage`. Physics has no
endoluminal import. Every constructor requires a compatible pre-allocation
declaration. Direct callers use `declare_rod_direct`, `declare_rod_pbd`,
`declare_rod_direct_material`, `declare_rod_pbd_material`, `declare_catheter_control`,
`declare_control_layout` and `declare_drag`. Case authoring calls each class's
`declare_storage` hook. Provider declarations require exactly one matching solver;
providers precede solvers, and references resolve before dependent publication.

Rest/topology fields and the three RodControls channels have body-qualified
`body_<index>_rod` domains. Other fields have system-instance-qualified domains.
Custom identity columns carry Newton reference semantics; topology and geometry
fields retain local integers. Shared rows have world -1; runtime rows have the
actual world. Owners resolve after host particle authoring, including foreign
prefixes; rows without a particle owner and padding use -1. No pool ownership or
native entity topology is invented for rods.

| Owner | Migrated fields |
| --- | --- |
| Model | Exact rod rest frames, angular inverse masses, rest lengths and Darboux data; PBD segment/pair topology and unit_twist; Direct local inertia; all existing live solver parameters |
| State | Rod frame and omega, with unchanged flat kernel shapes |
| Control | Exactly one stretch_scale, bend_scale and bend_offset backing per body per live Control |
| Model | Regional membership, tables, radius baseline; shared topology/rest aliases |
| Model | Catheter track_start, track_direction, base_orientation and live parameters |
| State / Control | Catheter targets and root / actions `(E,3)` |
| Model | Every ControlLayout configuration/topology/hop/arc/world-map array |
| State / Control / Model | Drag weights and baseline / active, node and target / max_delta |

Binding and ordinary preparation only validate and alias these buffers. Both
Newton containers and nested body, provider, table and solver aliases are checked.
Replaced buffers require a fresh binding. Topology, capacities, declaration layout,
backend and inertia allocation mode are fixed. PBD sweeps, provider routing and
catheter tip settings retain supported recapture behavior.

Initial templates contain rod frames/history and shared channel defaults, catheter
commands/history/root and Drag defaults. Direct inertia initialization reads the
Model being finalized: its exact i4h inverse masses, native radii and declared rest
lengths, through the unchanged float64 cylinder calculation. The trailing generic
finalizer finishes before template clones. Builder records retain host data and
weak scene references; no finalized device arrays are cached in them.

ControlLayout preparation no longer writes shared commands. Its existing per-step
writes remain. Catheter preparation first validates tip_edges; on shrink old to new
it clears only `[B-old, B-new)` across worlds. Retained slots and all other commands
and history survive. Increase, unchanged ranges and repeated preparation write no
commands. Existing step kernels write the active range. Explicit resets preserve
the previous behavior, including no additional catheter-root reset and Drag
clearing only active. Live Model edits persist.

Predicted rod frames (initial scratch copies of State frames), derived regional
outputs, solver workspaces and Drag.ref remain scratch with their prior overwrite
or zero-accumulator contracts. Catheter impulse/moment remain completed-step
outputs. Mounted flags, mount poses and pending reactions remain under their prior
ownership. **Mounted-tool episode restore remains partial until C3.3.**

CatheterControl supplies the generic command specification: actions `0:3`, insertion
`0:1`, rotation `1:2`, bend `2:3`. Warp and Torch mappings are cached/read-only;
the contained views are writable zero-copy slices of packed normalized rates.
Named shape is `(E,1)`, Warp strides `(12,4)`, Torch strides `(3,1)`, offset `4*c`.
C3.1 validation, backing retention, invalidation and stream ordering are reused;
recording format and rate integration are unchanged. Drag has no packed action API.

For nonempty domains each State container uses `28*E*F` rod bytes and `24*E`
catheter bytes; each Control uses `4*E*S+16*E*K` shared rod bytes and `12*E`
catheter bytes. Drag uses `16*E*P` State bytes and `20*E` Control bytes. Matching
initialization templates, Model fields, identity columns, padding, live/checkpoint
containers and intentional sharing are separately counted. Aliases count once by
maximum backing extent. Direct finalization's host readbacks/arithmetic and upload
are temporary; no inertia scratch/template duplicate persists. The six residue
arrays are unchanged and receive no new template or checkpoint copies.

## C3.3 Mount ownership (supersedes earlier mounted-restore boundaries)

`InstrumentControl.mounted` and `CatheterControl.mounted` now alias declared
Newton Model fields. Their defaults identify configured mounts or remain zero;
`control.mount(k)` explicitly edits the existing flag in place without recapture,
creating no host mapping or pending-reaction group. The corresponding mount poses
alias Control fields initialized at authored roots/orientations. Instrument
initialization uses those finalized Model flags and pose templates directly;
only its E-byte mask remains a device temporary. There is no initial host drive.

Physics owns `Mountable` descriptors and coupling declarations, without importing
surgical/endoluminal implementations. Direct callers declare target controls then
`declare_mounts(scene, mounts, two_way=...)` before schema registration. Case
construction accepts `mounts`/`coupling_mode`, publishes coupling after systems,
and passes the same configuration through host rebuilds. `Mounts(...)` only
validates and aliases matching declarations. It never allocates these buffers or
calls `control.mount()`.

Shared mount rows use instance-qualified control IDs, world -1 and an owner
reference to the declared target-control domain. `target` is a local kernel index;
`offset` is the normalized authored transform. Runtime rows use actual worlds and
native BODY owners; `body` is a genuine BODY-reference field. Deferred custom host
values and identities resolve after all foreign/host entities and worlds exist,
before any custom rows are emitted. Exact-label/suffix matching requires exactly
one body per world. Nonentity and padded owners are explicitly -1.

Binding/preparation/cache resets preserve poses and queued reactions. Before
coupling writes, checks cover selected Model/State/Control containers, Newton
buffers, shaped aliases, groups, sources, root aliases and configuration. The
step order remains apply previous reactions, rigid substeps, drive mounts, i4h
step, accumulate next reactions. The force conversion, root-to-COM moment shift
and shared-body atomic addition kernels are unchanged. Episode reset clears only
selected pending force/torque; flags and pose commands persist. Instrument reset
reads the current pose; catheter reset introduces no root reset.

For E worlds, N instrument targets per controller, C catheter controllers and M
configured mounts, mounted flags use `4*max(N,1)` per instrument controller and
`4*C` bytes for catheters. Pose Control storage and its matching template each use
`28*E*(N+C)` bytes; pending-reaction State storage and its matching template each
use `24*E*M`. Target/offset Model data uses `32*M`, native body mappings `4*E*M`,
and environment transforms `28*E` per aggregate. Every custom identity domain
adds `8*max(rows,1)` bytes. Empty instrument poses retain a 28-byte inaccessible
backing row per container/template; empty instrument flags retain 4 bytes with
absent owners. Coupling adds no scratch arrays. Allocation accounting deduplicates
aliases by maximum backing extent and is separate from the established chole
construction samples.

Mount poses remain outside packed/named actions and recording formats. Completed
source impulse/moment outputs keep their previous ownership. Continuation tests
use a deterministic rigid solver whose complete integration history is in native
State; they make no claim to restore MuJoCo-Warp private history or unmigrated
grasp/hook/contact history. Required campaign gates and disposition are recorded
in [stage-c33.md](stage-c33.md); C4 remains outside this change.

## C3.3 review follow-up: registration and Control clearing

Package integration uses one public `register_package` entrypoint on storage.
During authoring, `Declarations.register_package(records=...)` publishes
`(scene, attribute, {id: host_record})` entries, rejects duplicates atomically and
freezes publication at schema registration. Tet, surface, instrument, shared rod,
rod solver/material, endoluminal and mount declarations all use it. The existing
Scene record dictionaries remain readable by package code and accounting.

After binding, `Views.register_package(identities=..., checks=...)` registers
`(container, attribute, expected_object)` identities and callable layout checks.
Array identities also freeze pointer, shape, stride, dtype and device metadata.
Repeated identical identities are accepted; conflicting identities are rejected.
Explicit `replace=True` validates the old binding before routing aliases, as used
by rod material selection. `Views.container(owner)` exposes selected Newton
containers without access to private dictionaries. Packages no longer mutate
storage's private identity, array or check lists. No device allocation is added.

For a Model declaring mountable controls, **`Control.clear()` and
`Control.clear(model)` raise before changing any buffer**. Pinned Newton otherwise
zeros all namespaced arrays, producing zero mount-pose quaternions. Because that
version has no custom-clear hook, finalization wraps only that Model's `control`
factory and installs the rejecting `clear` on returned Controls. This applies
before binding, to independent Controls and explicit template-alias Controls, and
to each Model finalized from a retained builder. Newton classes/dependencies are
unchanged; Models without mountable controls retain ordinary clear behavior.
Calling Newton's base method directly bypasses this instance policy and is not a
supported i4h clear operation. Packed/named action views remain directly editable
and clearable. Their clearing and explicit episode resets preserve mount poses.
This host-only policy adds no device buffers, synchronization or capture kernels.

The approved post-freeze construction pass and the separate follow-up source
qualification are documented in [stage-c33.md](stage-c33.md). HDF5 is deferred by
user choice. No C4 implementation or progression disposition is implied.
