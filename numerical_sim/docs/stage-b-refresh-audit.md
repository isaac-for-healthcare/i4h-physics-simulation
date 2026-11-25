# B1 cache-refresh audit

Audited on 2026-11-04 against the preserved pre-change source and B1 source.
`I4hSolver.reset(flags=0)` is a boundary between completed physics steps. It
must preserve history, commands, topology, measurements and pending reactions.
It is not recovery from an interrupted kernel or an arbitrary edit to private
solver scratch. Episode restoration continues through each system's existing
`reset` implementation, followed by cache refresh.

## Findings and changes

`Centreline` was missing a refresh. Clipping reads its positions and live clamp
counts in `step_start`, before the tracker's ordinary `step_end` update. After a
host position edit, the default no-op refresh exposed stale geometry. A focused
CPU regression failed with the old implementation (a translated ring remained
at its old position) and passed after adding a masked update from current
particles. Binding also initializes the tracker from current host state.
Unselected worlds remain untouched; an entirely removed ring retains its last
position and gets zero live count. Empty and repeated refreshes preserve history.

`GraphContainment` already invalidates selected worlds through `rebuild=1`.
Its candidate query rebuilds before consuming the candidate list. Clearing its
references, counts or measurements at cache refresh is unnecessary. The existing
episode-reset test now also checks empty and repeated cache-only reset, selected
invalidation, and preservation of positions, frames, targets and measurements.

## Old reset writes and next-use analysis

| Systems | What old reset writes | Cache-refresh decision and next use |
| --- | --- | --- |
| Centreline | Rest positions and clamp counts | Recompute selected live geometry immediately; clipping and bleeding can read it before the next tracker update. |
| GraphContainment | Candidate lists, reference geometry, rebuild state, displacement scratch, contact measurements | Invalidate selected candidates. `request_rebuild` consumes/clears displacement scratch; candidate construction precedes projection. Preserve measurements. |
| PartContact | Delta/count accumulators, candidate arrays/counts, contact/overflow measurements | No additional refresh. `apply_deltas` unconditionally clears every delta/count row after use, including rows receiving no correction. Query rewrites candidate counts before reading candidates; bounds refit before queries. Measurements clear at step start. |
| ShellContainment, TetContainment | Side classification and contact measurements | No additional refresh. Two-sided classification is written in substep start; one-sided projection does not consume it. Bounds refit before queries. Measurements clear at substep start. |
| RodContainment | Contact measurements | Preserve until next substep start clears them; static geometry has no state-derived cache. |
| InstrumentContact | Reaction products, record counts | Preserve products. Step start clears reactions; substep start clears record counts before contact generation and dependent systems. Delta accumulators clear during application; BVH refits before contact queries. |
| Fluid | Ring cursor, alive count, overflow | Preserve episode state and measurements. Predict/grid build/neighbour search precede density and projection; density, lambda, velocity scratch and deltas are rewritten before use. |
| StretchBreaking (physical and surgical) | Stretch scratch and maximum measurement | No additional refresh. `finish_stretch` consumes and unconditionally zeros scratch at step end; step start clears the maximum. |
| Tet | Active tet/edge masks | Topology, not a cache; preserve. |
| Surface (physical and surgical), Appearance | Authored triangles, owners, support counts, dirty/deletion/overflow counters and visual state | Preserve topology and appearance. Normal deletion/re-emission remains in its authored phase; particle-position edits do not change these topology tables. |
| RodDirect | Frames, predicted frames, angular velocity, bend commands | Preserve history and commands. Prediction replaces predicted frames; first iteration starts XPBD multipliers; assembly/solve rewrites matrices and right-hand sides; correction application clears accumulators. |
| RodPbd | Frames, angular velocity, bend commands and per-element scales | Preserve at cache refresh. Prediction and correction application initialize/consume scratch. Episode restoration still resets the existing scales to one. |
| InstrumentControl | Packed actions, roots, jaw state, trigger history, current/previous spheres | Preserve commands and history. Step start updates pose and sphere geometry before consumers; particle edits do not move instrument geometry. |
| Grasp, VolumetricGrasp, Hook | Slots, anchors, active/pending state and reaction products | Preserve. Step start clears reactions; projection accumulators clear during apply. Retention/revalidation remains in existing phases. |
| Clipping | Attached nodes, slots, clip count and clipped flags | Preserve surgical history. Its geometry dependency is fixed by Centreline refresh. |
| Thermal | Heat, burn and coagulation | Preserve history. Diffusion writes `heat_next` before burn reads it. |
| Cutting | Cut signal, deleted-tet count and blood accumulator | Preserve history/measurements. Existing phase updates remain unchanged. |
| Bleeding | Cut/source state, fractional emission, serial and sealing measurements | Preserve history and pending emission; `find_leaks` updates source selection before spawning. |
| Emitter | Fractional emission, serial and spawned count | Preserve episode state/measurements. |
| Smoke | Particle ages/lifetimes, cursor, serial, emission remainder, hot/alive measurements | Preserve smoke history. Source search and update write their derived scratch before use. |
| Suction, ContactPaint | Removed/painted measurements | Preserve until the next ordinary measurement update. |
| ToolForce | Force, contact count, jaw load | Preserve the last completed step's measurements until step end recomputes them. |
| CatheterControl | Packed actions, targets, reaction products | Preserve commands/reactions. Step start clears products, substep start writes root controls before guidance. |
| Drag | Active drag command | Preserve. Substep start gathers reference positions before drag projection. |
| TipEvidence | Tip positions, active/count/overflow measurements | Preserve the previous observation until the next step end. |

Systems with no reset override (particles, bounds, attachments, connectors,
shell, static/fluid contact, rod materials and control layout) introduce no old
reset clearing that was lost in this split. Their existing prediction, refit,
material preparation and self-clearing correction paths remain in place.

The previous masked-reset baseline products exercised **episode restoration**,
not cache-only reset. Their passing status was not evidence for the missing
Centreline refresh. This audit adds no new scheduling or persistent-state schema.
