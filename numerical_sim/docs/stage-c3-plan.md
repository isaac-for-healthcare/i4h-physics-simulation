**C3.2 disposition: accepted for progression with retained failures.** The frozen
campaign passed 34/36 groups, 43/48 replays and 15/18 chole trials; every chole
group passed the existing 2-of-3 policy. Standalone captured `aorta_shell` failed
stepped catheter impulse; patched Lab 3.0 captured failed stepped and masked-reset
catheter impulse. Causes remain unresolved. Frozen suites passed core 1,017,
main 117 and Lab 3.0 114, with their recorded skips. The subsequent production
alias-validation correction passed focused checks only (core 103, each host 104);
full qualification was not rerun. HDF5 recording/replay remains unverified because
the retained hosts lack `h5py`. Earlier clipping/fluid observations remain retained.
This disposition leaves all historical reports and certificates unchanged and
authorizes C3.3 only; its required gates are not waived. Stop for review before C4.

---

**C3.3 implementation:** mounted Model flags, Control mount poses and State pending
reactions now use declarations and checked aliases. Focused mount checks pass on
CPU eager and CUDA eager/captured in all retained environments; the frozen
campaign and disposition are recorded in [stage-c33.md](stage-c33.md). Rigid
private history and unmigrated interaction history remain outside this slice.
Stop for review before C4. The historical plan below is retained.

**2026-11-10 review follow-up:** the user approved construction only and deferred
HDF5. Four post-freeze samples (chole and rod-bearing aorta, each at 1/64 worlds)
passed once each under source/import guards. The separate follow-up adds a
Control.clear rejection contract and public package registration; its focused
evidence and qualification boundaries are in [stage-c33.md](stage-c33.md).
No full-suite/reference reruns or C4 implementation. Stop for review.

# Preserved C3 plans

Source: retained planning session `01a1b4f9-301d-7af2-aef0-29de43b2f1f0`.
The two entries below are preserved verbatim. The second supersedes the first entry conditions and clarifies the binding contracts; C3.0 must be reviewed before C3.1.

## Preceding plan (verbatim)

<proposed_plan>
# C3 — Commands, rods and coupling ownership

## Summary and checkpoint boundaries

Continue Stage C from the accepted C2.5 closeout, using three separately qualified checkpoints:

| Checkpoint | Migration | Stop condition |
|---|---|---|
| **C3.1** | Instrument commands, geometry, live parameters and pose/trigger history | Qualify and report; stop before C3.2 |
| **C3.2** | Rod mechanics/materials, shared rod controls, catheter controls, ControlLayout and Drag | Qualify and report; stop before C3.3 |
| **C3.3** | Mount metadata, mounted pose commands and pending coupling reactions | Qualify and report; stop before C4 |

**The next implementation covers C3.1 only.** Later checkpoints establish the roadmap and ownership boundaries; they require subsequent progression decisions.

C4 history, general save/restore, full fresh-runtime continuation, robot-grasper examples and final steady-step performance qualification remain outside this work. No commit or push.

## Entry and common contracts

- Preserve the C2.4 and C2.5 certificates and every failed observation. The C2.5 manifest currently matches all **10,130 files**, with digest `2932e8e9a7fc006d90a810bf5ed17aaa4bf320550205a7d6edc138263151daca`. Recheck before editing and preserve a separate pre-C3 source snapshot.
- Use the three preserved qualification environments under `.tmp_tests/stage-b-hashgrid-fix-20261105/`, including their patched Warp binaries and recorded Lab sources. Verify imported paths and hashes; do not sync or upgrade these environments.
- Follow the explicit ownership contracts in `numerical_sim/docs/stage-c-field-contracts.md`: package-owned declarations, `i4h:{system.id}_{name}` fields, truthful domain identities, unchanged shapes and local indexing.
- Declare persistent storage before finalization. Bind cached aliases without modifying selected State, Control or Model values. Structural changes require reconstruction.
- Preserve numerical kernels, five-hook ordering, episode-reset behavior, action ordering, recording formats, contact capacity 256, dependencies, approved references and comparison tolerances.
- Retain the six residue arrays under their existing owners. Thermal, Appearance and other C4 history remain unmigrated.

## C3.1 — Instrument commands and history

**Ownership migration**

- Add surgical-owned instrument declarations and checked binding helpers. Integrate both `CaseAuthoring` and direct Scene construction; update direct callers to predeclare their instrument systems.
- Move packed `actions (E, instruments, 10)` to Newton **Control**.
- Move root/orientation/jaw state, held/released/pending flags, energy/fire history, current/previous sphere positions and sphere activity to Newton **State**.
- Move authored instrument/sphere/primitive geometry, reset defaults and live instrument parameters to Newton **Model**. Include arrays currently created by `_live`, not just explicit constructor allocations.
- Keep `mounted` and `mount_pose` under their current ownership until C3.3. Preserve their existing mounted-tool behavior and account for them explicitly; C3.1 does not claim complete mounted-state preservation.
- Remove the migrated runtime allocations. Keep host geometry records and genuinely rebuildable scratch separately identified.

**Initialization and reset**

- Preserve valid initial quaternions, jaw poses and collision-sphere geometry.
- Use a surgical-owned model-finalization initializer that runs the existing reset/geometry kernels against Newton’s initialization templates after the required fields exist. Do not duplicate pose mathematics or initialize a selected live State during binding.
- Keep declaration/completion host-only; device initialization happens during finalization. Initialize each finalized Model independently, including repeated finalization of a retained builder.
- Keep explicit masked reset using the existing kernels and current Model defaults. Binding, preparation and cache refresh must preserve all migrated bytes.

**Command interface**

- Add `PhysicsBinding.command_views(system_id)` for migrated instrument controls, returning cached writable Warp views named `actions`, `position`, `quaternion`, `grip`, `energy` and `fire`.
- Preserve packed columns: position `0:3`, quaternion xyzw `3:7`, grip `7`, energy `8`, fire `9`. Named scalar channels retain a singleton final dimension.
- Add the matching component method returning cached zero-copy Torch views. Preserve existing `action_views()`, `write_actions()`, bounds and system/instrument ordering.
- Reject unknown or not-yet-supported systems explicitly. C3.2 extends this interface to catheter commands.
- Validate owning arrays and cached views before execution. Invalidate caches on hard rebuild; replacing nested storage requires rebinding.

## C3.2 and C3.3 — Subsequent checkpoints

**C3.2: rods and catheter controls**

- Migrate both RodDirect and RodPbd persistent fields: topology, rest data, exact angular masses, inertia and live material tables to Model; frames and angular history to State.
- Put shared `RodControls` in Control, preserving the single shared allocation used by solvers, material providers and catheter/layout writers.
- Migrate catheter actions and mount-independent history/defaults; extend named command views with insertion, rotation and bend channels.
- Migrate ControlLayout configuration to Model and Drag commands/history according to the field contracts.
- Move `ControlLayout.prepare()` and `CatheterControl.prepare()` command initialization to construction defaults. Preserve legitimate per-step writers and their ordering.
- Keep derived material and solver workspaces as scratch only with demonstrated initialization/overwrite behavior. Preserve chain/tree and existing variant restrictions.

**C3.3: mounts and reactions**

- Declare mount configuration before finalization and resolve host body identities after all worlds and bodies exist.
- Move target/offset/body/environment-transform metadata and mounted flags to Model; mount poses to Control; pending force/torque to State.
- Make mount binding validate and alias storage without mounting, clearing reactions or rewriting restored commands.
- Preserve MuJoCo-Warp support, one-way/two-way behavior, rigid substeps, frame transformations and root-to-COM moment conversion.
- Preserve the exact sequence: apply previous reactions → rigid step → drive mounts → i4h step → accumulate next reactions.
- Test pending-reaction preservation without claiming restoration of the rigid solver’s private history.

## Verification, accounting and acceptance

**Focused tests at each checkpoint**

- Exercise CPU eager, CUDA eager and CUDA captured execution in all three qualified environments.
- Test multiple systems/worlds, empty domains, padding, foreign host entities, independent States/Controls and intentional shared Control/template aliases.
- For C3.1, verify packed↔named↔Torch writes, initial geometry, valid reset poses, trigger edges, jaw relaxation, recording replay and partial/empty/repeated resets.
- Rebind a selected allocation containing nondefault commands/history; verify exact preservation before stepping and compare subsequent behavior with an uninterrupted runtime. Preserve untouched allocations, unselected worlds and initialization templates.
- Cover stale arrays, nested aliases, rebuild invalidation and rejection of undeclared structural changes. Retain existing mounted, contact, grasp, hook, cutting, teleoperation, renderer-snapshot and host-lifecycle coverage.

**One frozen qualification campaign per checkpoint**

- Freeze production code, tests, harnesses, host sources and dependency binaries under a new manifest. Require startup/exit guards, imported-source records and successful process receipts, including nested workers.
- Run the existing full core suite and established Lab 3.0/main host selections.
- Run all six approved references in standalone and both hosts, eager and captured. Keep exactly three fresh-process chole trials per host/mode under the existing majority policy; retain every failed trial and endpoint.
- Take exactly one fresh construction sample at one world and one at 64 worlds. Report Model buffers, State/Control templates, live/checkpoint containers, intentional sharing, identities, inaccessible padding, construction peaks and maximum backing extent for aliased pointers.
- Reconcile removed runtime owners and remaining duplicates. Preserve residue totals of **2,020,320 bytes** and **129,300,480 bytes**, with no new residue template/checkpoint copies.
- Update the current allocation inventory while preserving historical inventories and removal records. Do not run attribution experiments or the final performance campaign.

**Closeout**

Write `numerical_sim/docs/stage-c31.md` with the ownership matrix, new versus inherited evidence, accounting reconciliation, failures and certificate; prepend the current status in the Stage C roadmap.

C3.1 closes only when its provenance, ownership/accounting checks, focused tests and required qualification gates pass. C2’s progression acceptance does not automatically accept a new failed reference group. Preserve failures and leave the checkpoint open with a concrete blocker report rather than changing tolerances or expanding into numerical fixes.

After reporting C3.1, stop for review. C3 remains incomplete until C3.2 and C3.3 are separately accepted.
</proposed_plan>

<oai-mem-citation>
<citation_entries>
MEMORY.md:104-112|note=[Preserved checkpoint boundaries explicit declarations failed evidence and residue ownership]
MEMORY.md:120-125|note=[Applied initialization template shared Control and qualified environment constraints]
</citation_entries>
<rollout_ids>
01a1b0eb-28ed-7972-8c7a-0c6b7787a9b6
01a1b09e-1bd4-7fb1-b4cf-045c9264a0f7
</rollout_ids>
</oai-mem-citation>

## Revision (verbatim; binding)

<proposed_plan>
# Stage C revision — C3.0 pin checkpoint, then separately gated ownership migrations

## Decisions and boundaries

- **Add C3.0 before C3.1.** Upgrade and qualify the host/dependency combination separately from storage migration. Stop for review after C3.0.
- **Keep patched Isaac Lab 3.0 support.** Retain `lab30.py`, the host selector and the cloner patch. Continue qualifying standalone, Lab main and patched Lab 3.0.
- **Packed commands with named views are the final design.** No subsequent typed-command migration or recording-format-v2 change is added to C4.
- Preserve the C3 sequence: C3.1 instruments; C3.2 rods/catheter controls; C3.3 mounts/coupling. Each checkpoint has its own qualification and review stop.
- No numerical kernel, execution-order, capacity, comparator, tolerance or approved-reference changes. No commit or push.

Correction to the handoff: the three preserved qualification environments already contain **MuJoCo and MuJoCo-Warp 3.17.0**. C3.0 changes Lab and Newton; it does not perform a 3.15→3.17 migration.

## C3.0 implementation

**Preserve and pin**

1. Recheck the C2.5 manifest: **10,130 files**, SHA256 `2932e8e9a7fc006d90a810bf5ed17aaa4bf320550205a7d6edc138263151daca`. It currently matches.
2. Before edits, preserve the complete pre-C3 worktree source, including untracked implementation files, in a new C3.0 evidence directory. Preserve existing certificates, environments, patches, references and failed observations.
3. Save the preceding C3 plan verbatim in `numerical_sim/docs/stage-c3-plan.md`. Append this revision separately, clearly identifying the superseded entry conditions and clarified contracts.
4. Use these exact targets; do not track moving branches:

   | Component | C3.0 target |
   |---|---|
   | Lab main | `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1` |
   | Newton, all three environments | `7f0c13af6e4dd75253709c31dc65c0edcaf22b44` — `1.9.0.dev0` |
   | Lab 3.0 | Existing `e53cd288727a5ca3dae7187d1d415b4d32b67e11` plus the preserved cloner patch |
   | Warp | Existing patched 1.17.0 wheel, verified by hash |
   | MuJoCo / MuJoCo-Warp | 3.17.0 |
   | Python | Existing Python 3.12 qualification version |

5. Update dependency declarations and lock records. Use exact git revisions for Newton and Lab packages, with host-specific qualification manifests so main and patched 3.0 do not conflict. Keep standalone installation independent of Lab.
6. Build fresh `core-env`, `main-env` and `lab30-env` under the C3.0 evidence directory. Resolve and record each dependency set, including required transitive changes. Do not modify the preserved environments or shared source checkouts.

**Port the host adapter**

- Convert the main solver adapters to the pinned Lab instance-method contract, using `self.cfg` and `self.native`. Retain pre-model hooks as classmethods where the upstream contract requires them.
- Preserve child-solver delegation, single-state stepping, observables, reset behavior, rigid substeps and coupling order.
- Verify manager configuration access, solver-init callbacks, backend graph invalidation, stop/rebuild cleanup and stream synchronization.
- Recheck contact-sensor delegation through solver observables: coupled rigid `CONTACT_F` remains supported; pure i4h rejection remains explicit.
- Retain the Lab 3.0 adapter and selector. Restrict any necessary compatibility edits to the host boundary.
- Add a physics-package capability guard for `_add_custom_attribute_model_finalizer`, naming the required Newton commit and expected callable contract. This guard is an explicitly permitted non-adapter production change.
- Update the target revision information in `QUALIFIED`, but distinguish the candidate from completed qualification in status documentation.

Allowed C3.0 edits are dependency/configuration changes, adapter compatibility, the capability guard, verification harnesses/tests and documentation. If compatibility requires mechanics, ownership or numerical changes, retain the failure and stop with a blocker report.

## C3.0 verification and acceptance

**Preflight tests**

- Verify actual imports, git revisions, package versions and patched Warp native hashes.
- Exercise the private finalizer hook on the target Newton version, including callback ordering, template availability and retained-builder finalization. Do not implement instrument migration during this probe.
- Run adapter construction, stepping, eager/captured execution, decimation, partial reset, hard rebuild, callback cleanup, observables and contact-sensor tests on both hosts.
- Run the existing C2.5 nonzero-capacity post-rebind Surface regression: PBD/XPBD × CPU eager/CUDA eager/CUDA captured in each environment.
- Re-run the existing HashGrid capture regression against the retained patched Warp build.

**Frozen qualification campaign**

After development checks pass, freeze the candidate source, tests, harnesses, host code, dependencies and native binaries. Require matching startup/exit guards and imported-source records for every worker and nested process.

Run once on the frozen candidate:

- Full core suite and the established host-suite selections for main and Lab 3.0.
- Six approved references across all three environments, eager and captured.
- Exactly three fresh-process chole trials per host/mode: **36 case groups, 48 individual replays, including 18 chole trials**.
- Exactly one construction sample at one world and one at 64 worlds.

Retain the existing zero floors, factor-three comparator, contact capacity 256 and 2-of-3 chole policy. Preserve all failures and numerical values; do not retry until passing, promote references or infer acceptance from C2’s progression decision.

Reconcile ownership/accounting against C2.5. C3.0 should perform no i4h ownership migration; explain any Newton-induced allocation differences. Retain templates, live/checkpoint States and Controls, alias overlap, padding, identities and maximum backing extents. Keep the six residue arrays explicit.

Write `numerical_sim/docs/stage-c30.md` with exact pins, adapter changes, environment records, suite results, reference matrix, accounting differences and the new certificate. Prepend the current status to the Stage C roadmap without rewriting historical results.

C3.0 closes only after required gates pass or a subsequent explicit user disposition accepts reported limitations. Otherwise it remains open. **Stop before C3.1 either way.** C3.1 adopts the new frozen baseline only after C3.0 is accepted.

## Binding amendments for C3.1–C3.3

**Command ownership and discovery**

- Each control package supplies its command-view specification. `PhysicsBinding.command_views(system_id)` and the component implement generic validation, caching and Warp/Torch exposure; they must not switch on concrete control classes.
- Preserve the packed action layout and recording format. Named views alias packed storage without copies.
- Document the actual instrument-domain frequency registration, world identities, owner references, view shapes, strides and pointer relationships. Preserve instance-qualified domains; do not collapse multiple control systems into a colliding global frequency.
- Record the typed-attribute/format-v2 proposal as superseded by the accepted packed-plus-named-view contract. Masterplan item 10 was not found in this worktree; record the decision in the roadmap without inventing or editing an unavailable source.

**Initialization and sharing**

- The instrument initializer must run after every required Model field and State/Control template exists, and finish before runtime containers can clone those templates.
- Test two Models finalized from one retained builder for independent backing. Subsequent State/Control allocations must receive valid initial commands and geometry; finalizing the second Model must not modify the first Model or its live containers.
- Ordinary Controls are cloned allocations. Two runtimes may deliberately share one such live Control; writes and resets must still leave initialization templates unchanged.
- Preserve the existing explicit `model.control(clone_variables=False)` semantics as a separately tested opt-in template alias. Its writes intentionally affect templates and later clones; never present this as ordinary live-Control sharing.

**Bounded preservation claims**

- C3.1 comparisons with uninterrupted execution use fixtures whose required instrument history is fully preserved.
- Keep integrated interaction tests, but make no fresh-runtime equivalence claim for unmigrated grasp, hook, Thermal or mounted state.
- State explicitly in the ownership matrix: **mounted-tool episode restore remains partial until C3.3**.

**Rod preparation and numerical history**

- Before C3.2 edits, extend `test_rods.py::test_structural_and_live_settings`, which already covers a structural `tip_edges` change followed by `prepare()`.
- Distinguish cache-only preparation from supported structural recapture and changes requiring reconstruction. Test preserved commands before stepping and the treatment of tip slots that cease to be controlled; do not silently clear restored commands or retain unintended stale bend contributions.
- Record that the accepted `aorta_shell` failures concern catheter-control impulse products, and C3.2 migrates that system’s ownership. Keep those exact failed endpoints visible in its comparison report.

The remaining C3 ownership boundaries, per-checkpoint qualification, accounting and review stops remain as previously planned. C4 and final performance qualification remain later work; no additional release-date-driven bump is authorized by this plan.
</proposed_plan>
