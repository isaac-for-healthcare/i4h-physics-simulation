# Stage C4 inventory: remaining system-owned State

Verified against source at commit `02ea7f0`. The machine-readable companion is [stage-c4-inventory.json](stage-c4-inventory.json).
It lists the 65 allocation sites that [the reviewed inventory](stage-c-inventory.json) codes `S` (State) and that systems still allocate themselves.
Bytes are measured at one world on CPU by constructing each reference case; they scale linearly with worlds.

## Remaining State by family

| Order | Family | Checkpoint | Arrays (file:line) | Row domain | Reset today | Aliases and readers |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `surgical/src/i4h_surgical/systems/mechanics/centreline.py` | C4.1 | `positions`:45, `active_count`:46 | centreline node `(E,K)` | masked `restore_variant_rows` from Model tables; the constructor also runs a full reset and `refresh`; `step_end` refreshes | Clipping (`positions`, `active_count`), Bleeding |
| 2 | `surgical/src/i4h_surgical/systems/contact/instrument_contact.py` | C4.1 | `delta_k`:105, `delta_jaw`:106 | tissue particle x tool `(E*Nb, max(n,1))` | never cleared by reset; `apply_average_tools` skips values that underflow to zero (residue) | none |
| 3 | `surgical/src/i4h_surgical/systems/instruments/hook.py` | C4.1 | `verts`:75, `tet`:76, `bary`:77, `friction_coord`:78, `radial`:79, `active`:80 | world x 2 anchors `(E,2)` | masked: `active`, `verts`, `tet` only; `bary`, `friction_coord`, `radial` keep stale bytes. Disabled `prepare` zeroes `active` unmasked | `active[:,0]`, `verts[:,0]` are InstrumentContact `suppress_active/verts`; `entry_active`, `entry_tet` views read by Thermal |
| 4 | `surgical/src/i4h_surgical/systems/instruments/grasp.py` | C4.1 | `delta_k`:59, `delta_jaw`:60, `prim`:66, `tet`:67, `t`:68, `radial`:69, `prev_axis`:70, `verts`:71, `bary`:72, `gap`:73, `active`:74, `count`:75, `pending`:76 | slot `(E,n,G)`; instrument `(E,n)`; residue `(E*Nb, max(n,1))` | masked: `active`, `prim`, `count`, `pending` only; other slot fields keep stale bytes; residue never cleared | reads InstrumentContact records and tet `active` |
| 5 | `surgical/src/i4h_surgical/systems/instruments/volumetric_grasp.py` | C4.1 | `particle`:74, `sphere`:75, `rest`:76, `active`:77, `count`:78, `pending`:79, `claim`:80, `delta_k`:83, `delta_jaw`:84 | slot `(E,n,G)`; instrument `(E,n)`; residue `(E*Nb, max(n,1))` | masked: `particle`, `active`, `sphere`, `count`, `pending`, `claim`; `rest` keeps stale bytes; residue never cleared | `count` is a state product |
| 6 | `surgical/src/i4h_surgical/systems/instruments/clipping.py` | C4.1 | `node`:67, `active`:68, `count`:69, `clipped`:70 | slot `(E,G)`; world `(E,)`; centreline node `(E,K)` | masked, all four fields | `clipped` is aliased as `Bleeding.clipped` |
| 7 | `surgical/src/i4h_surgical/systems/instruments/thermal.py` | C4.2 | `heat`:85, `burn`:87, `coag`:88 | tet `(E,T)`; particle `(E*Nb,)` | masked zero of `heat`, `burn`, `coag` | Smoke reads `heat`; Newton State copies of `heat`/`burn` already exist from `declare_thermal_tet_fields` but kernels still use these arrays |
| 8 | `surgical/src/i4h_surgical/systems/mechanics/surface.py` | C4.2 | `visual`:13 | particle `(E*Nb,)` | masked zero through `Surface.reset` | written by Thermal, Cutting, StretchBreaking, ContactPaint; read by `isaaclab/appearance.py`, `isaaclab/camera/tiled.py`, standalone `gl.py`, `slang.py`, `snapshots.py` |
| 9 | `physics/src/i4h_physics/systems/fluids/fluid.py` | C4.2 | `cursor`:137, `overflow`:139 | pool `(E,nf)` | masked zero (`cursor`, `overflow`; `alive_count` is output) | Emitter and Bleeding advance `cursor` |
| 10 | `surgical/src/i4h_surgical/systems/fluids/bleeding.py` | C4.2 | `cut`:102, `source`:103, `acc`:104, `serial`:107 | centreline node `(E,K)`; world `(E,)` | masked zero | reads `Cutting.cut_on`, `Clipping.clipped` (alias), Centreline |
| 11 | `surgical/src/i4h_surgical/systems/fluids/emitter.py` | C4.2 | `acc`:71, `serial`:72 | world `(E,)` | masked zero | writes Fluid `cursor` |
| 12 | `surgical/src/i4h_surgical/systems/fluids/smoke.py` | C4.2 | `pos_age`:81, `vel_life`:82, `cursor`:83, `acc`:84, `serial`:85 | smoke particle `(E*512,)`; world `(E,)` | masked zero | reads Thermal `heat` |
| 13 | `surgical/src/i4h_surgical/systems/fluids/suction.py` | C4.2 | `removed`:52 | pool `(E,nf)` | masked zero | state product |
| 14 | `physics/src/i4h_physics/systems/contact/graph_containment.py` | C4.2 | `cand_idx`:111, `cand_count`:112, `cand_overflow`:113, `ref_samples`:114, `ref_tree`:115, `rebuild`:117, `built`:118 | sample `(E*ns*cap)`, `(E*ns)`; tree particle `(E*Pt)`; world `(E,)` | masked reset of all; `refresh` sets `rebuild=1` masked; `prepare` fills `rebuild=1` unmasked | none |
| 15 | `endoluminal/src/i4h_endoluminal/systems/tip_evidence.py` | C4.2 | `positions`:34, `active`:35, `count`:36, `overflow`:37 | world ring `(E,cap)`; world `(E,)` | masked zero; `begin_action` zeroes `count`, `overflow` unmasked | created after binding by `endoluminal/bindings.py::configure_tip_evidence`, not by case authoring |

## Measured bytes per world

| Family | `aorta_centreline` | `aorta_shell` | `aorta_static` | `chole_tet_med` | `tissue_contact` | `two_worlds` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| centreline | 0 | 0 | 0 | 400 | 0 | 0 |
| instrument_contact | 0 | 0 | 0 | 673,440 | 0 | 3,000 |
| hook | 0 | 0 | 0 | 96 | 0 | 0 |
| grasp | 0 | 0 | 0 | 847,560 | 0 | 0 |
| volumetric_grasp | 0 | 0 | 0 | 681,180 | 0 | 0 |
| clipping | 0 | 0 | 0 | 136 | 0 | 0 |
| thermal | 0 | 0 | 0 | 173,032 | 0 | 3,060 |
| surface | 0 | 0 | 0 | 89,792 | 128 | 2,000 |
| fluid | 0 | 0 | 0 | 16 | 0 | 8 |
| bleeding | 0 | 0 | 0 | 304 | 0 | 0 |
| emitter | 0 | 0 | 0 | 8 | 0 | 8 |
| smoke | 0 | 0 | 0 | 16,396 | 0 | 16,396 |
| suction | 0 | 0 | 0 | 8 | 0 | 4 |
| graph_containment | 214,212 | 0 | 0 | 0 | 0 | 0 |
| tip_evidence | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **214,212** | **0** | **0** | **2,482,368** | **128** | **24,476** |

`aorta_static` and `aorta_shell`, the timing and rod construction workloads, hold none of the remaining State, so C4.1 to C4.3 cannot change their storage. `chole_tet_med` holds 54 of the 65 arrays; the six residue arrays are 2,020,320 of its 2,482,368 bytes per world.

## Findings that change the C4 plan

1. **TipEvidence is not created by case authoring.** `endoluminal/bindings.py::configure_tip_evidence` constructs it after binding, with a capacity chosen by the caller, and no reference case contains it. Newton State must be declared before finalize, so C4.2 needs a decision: declare tip evidence in the case with a fixed capacity, or leave it host-owned and exclude it from the continuation claim.
2. **`Centreline.positions` is history only for removed rings.** `refresh` recomputes `positions` and `active_count` from particle positions and flags at every `step_end`, but keeps the last position of a ring whose clamp particles are all gone. `active_count` is fully derived. C4.1 can migrate both as State (simplest, exact) or migrate `positions` only and rebuild `active_count` at bind.
3. **GraphContainment candidates are a skin cache with an unmasked rebuild on every `prepare`.** A fresh binding today therefore rebuilds candidates from current positions, while an uninterrupted run keeps the older candidate lists. C4.2 must first test whether a forced rebuild is bitwise-neutral for the projection. If it is not, the candidate arrays migrate as State and `prepare` stops forcing the rebuild for preserved history.
4. **Thermal `heat` and `burn` exist twice.** `surgical/tet_storage.py::declare_thermal_tet_fields` already declares Newton State copies, but the kernels read and write the legacy arrays at `thermal.py:85,87`. C4.2 repoints the kernels and deletes the legacy arrays, which removes memory instead of adding it.
5. **Hook is the anchor of two alias chains.** `InstrumentContact.suppress_active/verts` and Thermal's `entry_active/entry_tet` are row views of Hook's `(E,2)` arrays. Hook must migrate with layouts that keep column 0 addressable as a contiguous view, in the same checkpoint as the InstrumentContact residue.

## Lifecycle writes that change history outside episode reset

| System | Write | Where | Effect on a fresh binding |
| --- | --- | --- | --- |
| Hook | disabled `prepare` zeroes `active` for all worlds | `hook.py` `prepare` | clears a set anchor and the contact suppression that aliases it |
| GraphContainment | `prepare` fills `rebuild` with 1 for all worlds | `graph_containment.py` `prepare` | discards the candidate cache |
| Centreline | constructor runs a full `reset`, then `refresh` | `centreline.py` `__post_init__` | overwrites retained positions of removed rings |
| TipEvidence | `begin_action` zeroes `count` and `overflow` for all worlds | `tip_evidence.py` `begin_action` | by design per action; not a binding effect |
| Grasp, VolumetricGrasp, Hook | reset clears only some slot fields | each `reset` | stale bytes in the other fields are part of State and must be copied exactly |

Each needs a pre-migration test that records today's behaviour, as the C4 plan requires.

## Caches derived from history

| System | Cache | Rebuilt today |
| --- | --- | --- |
| Centreline | `active_count`, and `positions` of rings with live particles | every `step_end` and in the constructor |
| GraphContainment | `cand_idx`, `cand_count`, `cand_overflow`, `ref_samples`, `ref_tree`, `built` | when displacement exceeds the skin, and on every `prepare` |
| Fluid | neighbour lists, density, lambda | every substep (scratch, not migrated) |
| Hook | `entry_active`, `entry_tet`, contact `suppress_*` | views, never rebuilt |

## Confirmed scratch and outputs (not migrated)

- `ShellContainment.sides` and `TetContainment.sides` carry a "two-sided history" comment, but `classify_sides` rewrites every row at `substep_start` before the projection reads it, and one-sided mode never reads it.
- `ToolForce.force`, `count` and `jaw_load` are completed-step products.
- `Cutting` owns no history: the cut itself lives in the migrated tet `active` flags.

## Private storage reaches still present

- `authoring.py:149` reads `storage._authored`.
- `tet_storage.py:193,206` and `surface_storage.py:116,131` keep pending declarations and field caches in `scene.__dict__`; `storage.py:555` does the same for deferred declarations.
- `mount_storage.py:58-65,157`, `scene_storage.py:24` and `coupling.py:129` use `scene._mount_declaration`.
- No package writes `storage._identities` or `storage._checks` any more.

## Outcome after C4.2

C4.1 migrated families 1 to 6 (36 arrays) and C4.2 families 7 to 14 (25 arrays). Family 15, TipEvidence, stays host-owned assessment state by user decision (2026-11-10) and is excluded from the continuation claim; the allocation inventory codes its four sites `A`. One correction to the table above: `isaaclab/appearance.py` reads authored appearance metadata, not the `visual` State array.
