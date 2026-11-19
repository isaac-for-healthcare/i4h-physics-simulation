# Pinned-liver grasping: numerical fixture, procedural qualification and Workflows handoff

## 1. Scope and fixture

Deliver a liver-only numerical case, reusable grasp measurements, a procedural reference, focused tests and a documented integration contract. Arena integration and policy training remain separate follow-up work.

Create `liver_pinned_grasp_1x` with:

- The existing life-size liver geometry, scale, translation, mass, material settings, particle radius and gravity.
- The existing soft attachment selector and stiffness: 391 attached and 1,595 unattached vertices. These attachments permit drift; they are not fixed particles.
- **Exactly one surgical grasper**, preserving its geometry, orientation, trocar and control parameters.
- Particle/tet mechanics, surface updates, attachments, instrument contact, both existing grasp mechanisms and tool-force reporting.
- No other instruments, organs, connectors, cavity, support surface or bounds constraint.

Assert the physical action layout is `(E, 10)`:

`[px, py, pz, qx, qy, qz, qw, grip, energy, fire]`

Coordinates are component-frame metres, quaternions are XYZW, grip is `0` open and `1` closed, and energy/fire remain zero.

Run 60 Hz physics with eight substeps and eight iterations. Teacher commands run at 30 Hz and remain applied for two physics steps. Acquisition and retention sampling run after **every physics step**.

Preserve inversions as a reported limitation. Do not tune mechanics, strengthen attachments, change mass or repair the solver.

## 2. Settled snapshot, target manifest and sanity workspace

### Settle once and save

Keep the grasper open at its authored starting pose. Settling requires maximum particle speed below **5 mm/s for 60 consecutive physics steps**, with at least three seconds observed and a 30-second deadline. Compute speed from completed-step position differences.

If the criterion is not met, report `not_settled` and stop scored qualification without changing mechanics.

At a completed step boundary, save one canonical settled baseline per world-count configuration. Use that same snapshot for every attempt and for eager/captured comparisons of that configuration; do not settle independently for each attempt.

Reuse the existing `save_i4h`/`restore_i4h` continuation path. Add a harness snapshot bundle and disk round-trip support covering:

- All continuation-managed State and Control fields, including positions, velocities, particle history, instrument and grasp state.
- Attachment rest data and required Model/configuration identity.
- Harness time and counters; probe/controller state starts empty.
- Asset, schema, source, dependency and snapshot hashes.

Store attachment rest data and verify it matches the constructed fixture before restoration. Reject mismatched assets, layouts or configuration instead of loading partially compatible state.

Restore through the continuation API, without subsequent episode reset or history initialization. Then restore harness counters, clear the selected probe/controller state and select the manifest target. Recompute derived measurements from the restored state.

Before scoring, require exact restoration of copied State/Control bytes and attachment-rest identity. Repeated restore-plus-one-open-step trials must reproduce positions within **10 µm maximum pointwise distance**. Report measured differences; failure blocks reproducibility qualification. Save/restore rebuilds contact caches, so references must use this same continuation path.

### Mesh-aware target patches

At bind time:

- Compute median unique rest-surface edge length `L`.
- For every surface vertex, construct the union of its one-ring neighbourhood and vertices within rest-surface geodesic distance `1.5 × L`.
- Include the centre vertex and exclude attached vertices.
- Precompute padded patch-ID tables and counts before graph capture.

Eligible targets are unattached surface vertices at least 10 mm from attached vertices in rest-space Euclidean distance, with at least five patch vertices.

After restoring the settled snapshot, compute area-weighted live normals from authored triangle winding. Require a valid normal and:

`dot(normal, normalize(trocar − target)) ≥ 0.25`

Define the grasp centre as the mean of active grasper jaw spheres on `CHAIN_FIXED` chains.

Select the first target nearest that settled grasp centre. Select seven more by deterministic farthest-point sampling using **rest-space Euclidean distance**, restricted to eligible candidates within **30 mm rest-space Euclidean distance of the first target**. Break ties by vertex ID.

Freeze target IDs, patch membership, snapshot hash and selection measurements before scored runs. If fewer than eight candidates exist, publish the available set without substituting another selection rule.

### Sanity workspace

Construct a component-frame axis-aligned box containing the starting poses, computed teacher root goals and retraction commands, expanded by 20 mm per axis.

Freeze it in the manifest and label it **a controller sanity bound, not an independently validated safety or reachability constraint**. Exiting it fails the episode; commands are not silently clipped.

## 3. Grasp probe and procedural reference

### Probe interface and acquisition

Add a surgical-domain `GraspTargetProbe` with a thin Isaac Lab adapter. It consumes a physics binding and named tissue/grasper, exposes preallocated Warp arrays and synchronized Torch views, and never steps or commands physics.

Expose target position/normal, grasp centre, nearest jaw-sphere geometry, jaw state, acquisition identities, capture counts and geometric retention measurements.

The controller arms acquisition before issuing closure. At the **first non-empty volumetric capture**, latch particle IDs, sphere IDs, captured `rest` values and target-patch membership. Freeze that acquisition for the attempt.

- A first capture outside the intended patch fails.
- Empty capture may use the existing retry mechanism while the root remains stationary for at most one second.
- Release, binding loss or later capture never replaces the original acquisition.

All vertices in this asset are surface vertices. “Volumetric” names the capture mechanism, not interior particles. Its bindings are authoritative for v1; surface-grasp counts remain separate diagnostics.

### Geometric retention

For each originally acquired patch particle, compute:

`residual = abs(norm(particle − current_sphere) − norm(captured_rest))`

It is retained only if its original binding remains valid and residual is at most **2 mm**.

Require at least half the originally acquired patch particles, rounded up, throughout acquisition qualification, retraction and holding. Record acquisition size explicitly; one-particle acquisitions are expected and are not described as robust multi-particle grasps.

At every physics step, record:

- Maximum and mean residual across the original acquired patch set.
- Number of valid and geometrically retained particles.
- Invalid original bindings.
- Particle-to-sphere displacement vectors.

Tangential motion is permitted by this radial constraint and is reported rather than incorrectly classified as a radial failure. The original target vertex is tracked independently for destination success.

### Teacher sequence

Use authored orientation and trocar behaviour. Apply normal instrument commands only.

1. Restore the canonical snapshot, clear the probe/controller and select the manifest target.
2. Verify restored target eligibility; fail explicitly on disagreement without reselection.
3. Approach along the live outward normal. Place the nearest active jaw sphere at a signed gap of **+5 mm outside the capture envelope**.
4. Reduce the gap toward **−0.25 mm**, targeting the closure window **[−0.5 mm, 0 mm]**.
5. Once inside that window, arm acquisition and issue closure. Record pre-command jaw angle, sphere geometry and clearance.
6. Hold the root stationary while capture retries, then require intended-patch capture and retention for 0.5 seconds.
7. Retract toward a destination **+10 mm along component Y** from the target vertex’s first-acquisition position.
8. Hold the target vertex within 2 mm of that frozen destination for one continuous second while retaining the required original particles.
9. Open, require both grasp mechanisms to clear within 0.5 seconds, then observe one second of relaxation.

Root command increments are capped at **1 mm/control tick**. When the signed gap is below +2 mm, cap them at **0.1 mm/tick** and at the remaining distance to the −0.25 mm setpoint. Recompute live geometry each control tick. Overshoot triggers open-jaw repositioning within the same approach deadline, not immediate closure or an automatic regrasp.

Allow 20 seconds from approach through release, excluding snapshot restoration. Non-finite state, workspace exit, timeout, wrong-patch capture or retention loss fails the attempt.

Record release-command step and the first subsequent physics step at which each mechanism separately has zero counts and no active bindings.

## 4. Tests and qualification evidence

Test the authored starting root and ±5 mm X variants for each candidate, in eager and captured execution, restoring the same baseline each time. Starting-pose offsets affect only the open tool commands; tissue state comes from the snapshot.

A supported target must pass all three starts in both modes. Require at least four supported targets to claim reusable target-set support. Publish all candidate results and exclusions without adjusting targets or tolerances after scoring.

Focused tests cover:

- Liver-only storage, one instrument and the ten-column action layout.
- Snapshot disk round trip, exact restored state, first-step tolerance and incompatible-bundle rejection.
- Patch padding, size/attachment filters, normal-facing filters and target changes without recapture.
- Closure clearance and adaptive small steps near the capture envelope.
- First capture on either physics step between teacher commands; empty retry and wrong-patch capture.
- Injected geometric slip with slots still active, radial versus tangential motion, and synthetic binding removal/replacement.
- Interrupted hold streaks and independent target-vertex destination checks.
- Release timing for surface and volumetric mechanisms.
- Rebuild invalidation and fresh probe binding.

Use an explicit **two-world override** for isolation tests. Exercise both ordinary masked episode reset and selected snapshot restoration. Clear the selected probe after physical reset/restoration and verify the peer’s State/Control and acquisition latch remain unchanged. Do not require bitwise-identical later trajectories after continuation, because contact-cache rebuilding spans worlds.

Deliver:

- Reproduction command, snapshot bundle and target/workspace manifest.
- Per-physics-step traces and physical action arrays.
- Per-attempt results, a capture-count histogram, release latencies and maximum radial residual timelines.
- Settling, attachment drift, bounds and inversion diagnostics, kept separate from functional pass/fail.
- A visibly reviewed video with target/destination markers, phase labels and playback speed.
- Focused test results, failures, skips and hosted-compatibility status.

## 5. Parallel-work boundary and handoff

Preserve the Workflows upgrade agent’s checkout, immutable numerical baseline, pins, environments and qualification evidence. Perform numerical implementation in a separate worktree; preserve the staged Versius boundary.

The integration contract specifies frames, action layout, timing, probe shapes, first-acquisition semantics, stream ordering, reset/restoration order and view lifetime.

Arena will later own objective predicates and hold timers; Workflow Tasks will execute controllers. Do not add Arena imports or another production workflow engine to Numerical Sim.

Qualify hosted compatibility only against the approved upgrade’s exact final runtime. If unavailable, report it pending. Numerical traces are not advertised as Workflows-compatible demonstrations until recorder integration exists.

Before implementation, replace the earlier on-disk planning document with this specification. No training, staging, commits or pushes are included.
