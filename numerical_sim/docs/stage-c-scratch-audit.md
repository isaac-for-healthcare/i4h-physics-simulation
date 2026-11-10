# Stage C scratch lifecycle source audit

This audit qualifies the **source inventory**, not checkpoint continuation. The
field contracts identify the individual dtypes, dimensions, reset behavior and
first consumers. Scratch below requires either overwrite-before-read or explicit
fresh-solver initialization. A poison replay must test those conditions during
C4; self-clearing does not mean an uninitialized fresh allocation is safe.

| Source / fields | First-use and reset path | Fresh binding requirement |
| --- | --- | --- |
| Tet.delta/count; Shell.delta/count; Connectors.delta | Solve kernels atomically accumulate; `apply_average`/`apply_direct` consume and clear. Ordinary completed steps leave zero. | Allocate zero; preserve clear-before-return behavior, including inactive particles. Never restore from an old runtime. |
| Tet XPBD edge/tet multipliers; Shell lambda_s/d | Substep-start resets before first constraint accumulation. PBD fallback arrays are not read by PBD solve. | Allocate correct branch layout; overwrite in substep-start before consuming. |
| Rod material outputs | `pbd_regions`/`direct_regions` run at substep zero, before their solver's material consumers in the ordered loop. Disabled providers route solver aliases to defaults/controls. | Allocate scratch without changing Model tables or Control channels; retain provider-before-solver ordering and enabled routing. |
| RodDirect.frame_pred, compliance | `predict_rot` writes predicted frames; substep-zero preparation writes compliance before `edge_constraints`. | Preserve root writer order and regional/nonregional branch; initialize scratch only. |
| RodDirect.lambda_sum/rhs/delta_lambda/jacobian/blocks/edge_w | `edge_constraints` ignores old lambda on first iteration; assembly writes blocks; Thomas recurrence writes factors/delta; corrections overwrite lambda on first iteration. | Preserve first-iteration flag and recurrence order. No general refresh or constructor reset is required to restore history. |
| RodDirect.inv_inertia | Physical branch writes world inertia immediately before assembly/corrections. Unit branch does not consume its one-element fallback. | Match structural branch and initialize fallback safely. |
| RodDirect.pos_corr/rot_corr; RodPbd.pos_corr/count/ori_corr/count | Constraints accumulate; apply kernels consume and clear. | Initialize all accumulators to zero, including inactive and padded rows. |
| RodPbd.frame_pred | `predict_orientations` writes before rod constraints. Frame/omega remain State. | Do not copy constructor rest poses over restored frame/omega. |
| Drag.ref | Substep-start gathers current particle rows before drag projection. Weights/baseline remain State captured by begin-drag. | Reallocate ref; preserve weights/baseline and commands. |
| GraphContainment.max_disp | Displacement kernels atomic-max into zero; `request_rebuild` reads and clears. | Initialize zero. The seven candidate/reference/rebuild arrays are State, not acceleration scratch. |
| GraphContainment cath/tree correction/count | Projection accumulates; apply clears; one-way projection does not accumulate tree corrections. | Initialize zero; preserve one-/two-way behavior. |
| PartContact candidates/count | Bounds are written/refitted; query overwrites counts and valid slots before accumulation. Slots beyond count are never read. | Rebuild BVH from current geometry. `prepare` may clear these scratch/output arrays. |
| PartContact delta/count | Candidate and overflow passes accumulate; `apply_deltas` clears each row. | Initialize zero. |
| FluidContact, PartContact, ShellContainment, TetContainment bounds/BVH | Bounds kernels precede refit/query. Static Mesh geometry is Model data; its acceleration structure is rebuildable. | Build acceleration structures from declared geometry/current positions, then preserve ordinary refit timing. |
| Shell/TetContainment.sides | Two-sided substep-start classifies before projection; one-sided projection ignores sides. | Initialize -1 for fallback/unused rows; follow branch classification timing. |
| Shell/TetContainment.wall_delta | Two-way projection accumulates; apply_direct clears; one-way does not accumulate. | Initialize zero. |
| Fluid.group | Prediction writes owned pool slots before grouped HashGrid build. Non-fluid gaps remain -1. | Initialize the whole span to -1 before prediction. Poisoning only owned slots would miss this requirement. |
| Fluid neighbors/count | Neighbor build writes valid slots/counts before lambda/XSPH readers. Unused slots are count-masked. | Rebuild grouped grid and neighbors before use with the qualified Warp binary. |
| Fluid lambda/density/delta | Due-mask lambda pass writes a pool before its corresponding delta/apply passes; inactive slots are explicitly zeroed or excluded. No physics consumer reads the per-pool density view outside Fluid. | Preserve due-mask order and private pool layout; do not read gaps or skipped pools through a new bulk path. |
| Fluid.qd_scratch | Finalization writes before XSPH; inactive rows are zeroed. | Run normal finalization before XSPH; do not replace particle history. |
| InstrumentContact/Grasp/VolumetricGrasp delta/count | Projection/hold kernels accumulate; `apply_average_tools` unconditionally clears delta/count. Reaction outputs are distinct from pending coupling State. | Initialize zero. Per-tool delta_k/delta_jaw are State, not scratch; see the reproduced exception below. |
| StretchBreaking.stretch | Measurement accumulates; finish reads and clears unconditionally. Surgical finish uses the same scratch to paint. max_stretch is a measurement. | Initialize zero, retain subclass finish phase. |
| Thermal.heat_next | Diffusion writes destination each sweep. Even sweep counts copy heat to heat_next before burn reads it. | Preserve odd/even handling. heat/burn/coag remain State. |
| Cutting.cut_tets / blood_acc | On the final substep, cut kernel writes before neighbor-blood reads; apply_blood consumes and clears accumulator before particle-flag branching. When cut launch is skipped, dependent launches are skipped too. | Initialize blood_acc zero; preserve branch/last-substep sequence. |
| ContactPaint.painted | Paint increments; clamp skips only zero entries and clears every touched entry. | Initialize zero; Appearance.visual remains State. |
| Full-world masks / selected-world buffers | Constant masks are initialized true. Selection buffers are written before reset; temporary index arrays are operation-local. | Recreate constants/selections without triggering an episode reset. Constructor masks themselves are temporary, not physics state. |

Source review also identified lifecycle writes that **must change during
migration**, and therefore are not claimed safe for fresh-runtime continuation:

- GraphContainment.prepare writes `rebuild=1`. Preserve the restored candidate
  cache as a unit; do not replace this with unconditional refresh.
- ControlLayout.prepare fills RodControls defaults; CatheterControl.prepare
  clears bend offsets. Move initial values to authoring and preserve restored
  command storage during preparation.
- Current constructors/reset helpers still initialize particle, topology, rod,
  tool and surgical history. C2–C4 must remove those persistent writes when
  binding Newton-owned fields. Current additive declarations do not fix them.
- Smoke.centre is an output written only when `hot=1`; it has no physics reader.
  It is intentionally excluded as a measurement, not justified as overwritten
  scratch. Smoke particle lifetimes, emission fractions and cursors remain State.

Disabled systems, padded variants, empty domains and partial/empty resets remain
required runtime poison/continuation cases. This source audit does not substitute
for those tests or imply that unselected-world bytes may be changed.

## Reproduced exception: per-tool correction residues

`apply_average_tools` clears delta_k and delta_jaw only inside the active,
positive-count, positive-inverse-mass branch, and only when `length_sq(value)>0`.
A finite nonzero float32 vector `(1e-30,0,0)` has a squared length that underflows
to zero. The kernel retains its bytes while clearing delta_count. This was
reproduced on both CPU and CUDA using the qualified patched Warp environment;
`tool-residue.json` and `tool-residue.log` are retained under
`.tmp_tests/stage-c-inventory-20261105/`.

Consequently **InstrumentContact, Grasp and VolumetricGrasp delta_k/delta_jaw are
State** in the inventory. Each is zero at authored construction, but existing
episode reset and disabled preparation do not clear it. The migration must keep
that behavior. This classification preserves the existing kernel; changing it
to clear unconditionally would be a separate numerical change. The direct probe
establishes failed overwrite/clear proof, not a measured error magnitude in a
clinical workload. Additional State and checkpoint memory must include these six
arrays when they migrate; no memory-reduction claim is made.
