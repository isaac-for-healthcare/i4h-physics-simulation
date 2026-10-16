# Catheter parity (F5)

**The question.** Which catheter rod should the workflows' catheter scenes use?

- the workflow's own (`catheter_vasculature_solver.CathRodSolver`);
- `i4h_endoluminal`'s (`rod_direct` + `catheter_control`).

The bounds are docs/component-contracts.md section 4. Numerical qualification is run by the owner; the numbers below are a development run, recorded to explain the decision.

## Reference

- **Solver.** `catheter_vasculature_solver` at this repo's `e4bed31`. That is what the i4h-workflows catheter-navigation reward branch installed, from `feature/newton-solverbase-isaaclab-integration`, now in `main`.
- **Stack.** The old stack's warp-lang 1.13.0. The solver needs only warp, torch and numpy.
- **Configuration.** As the workflow's `CatheterRodSpec` configures it:
  - a 0.4 m rod in 120 segments, radius 0.5 mm;
  - density 7800 kg/m³, E 1 GPa, bend 0.1 and twist 0.4 of EI and GJ;
  - damping 0.01 per substep;
  - dt 1/120 with one substep;
  - physical rotational inertia;
  - polyline tip-bend mapping;
  - 128 position-only distance sweeps.
- **Root drive.** The root rides a straight rail through `set_root_pose_gpu`, as the workflow's route-railed introducer drives it.

## Decision: ours, with two workflow features ported (opt-in)

| Feature | Ours before | Ported setting |
| --- | --- | --- |
| Rotational inertia | unit (omniendo) | `rod_direct.inertia: physical`: each node's cylinder, `12/(m(3r²+L²))` transverse and `2/(m r²)` about the tangent, applied as `R diag R^T` in the block assembly and the corrections |
| Tip bend | `angle / n` per tip edge (omniendo) | `catheter_control.tip_mapping: polyline`: `sin(angle / (2n − 1))` per edge, so the tip polyline turns by the commanded angle |

**Why inertia.** With unit inertia, a 2e-5 kg-per-node wire cannot turn its frames: it stays straight under gravity and barely answers a tip command. The defaults keep omniendo's numerics, so the shipped cases and PHYS-1..8 are unchanged. A workflow catheter opts in.

**Also opt-in, not from the workflow:** `catheter_control.guide_tip: true` holds the last `tip_edges` nodes on the track too while they are on it, so a tip bent there stays straight until it leaves the track, like a steerable tip inside its introducer. omniendo (the default) leaves them unguided. `aorta_centreline` opts in: its catheter rests on its track outside the deformable vessel, and a tip bent there could enter beside the lumen and drag the tree out with it. `aorta_shell` and `aorta_static` keep omniendo's behaviour.

**Not ported:**

- **Tip-bend taper** (`tip_bend_fraction`). It is off (1.0) in the shipped workflow scenes.
- **The 128 distance sweeps.** No measurable effect on the stiff stretch constraint here.
- **The routeless tangent push** (`apply_proximal_control`). The workflow's scenes drive a rail.

## How to run

```bash
scripts/parity/setup_reference.sh                       # .parity/reference: warp 1.13 + the e4bed31 solver
.parity/reference/venv/bin/python scripts/parity/catheter.py run --engine reference --out parity/reference.npz
$PY scripts/parity/catheter.py run --engine ours --out parity/ours.npz          # current stack
$PY scripts/parity/catheter.py compare parity/reference.npz parity/ours.npz --report parity/report.json
```

## Development run (2026-10-06, RTX PRO 6000; not a qualification)

- **Ours:** `inertia: physical` and `tip_mapping: polyline`. Every bounded row passed.

| Row | Value | Bound |
| --- | --- | --- |
| sag tip RMS / L | 1.5e-5 | 0.01 |
| sag tip max / L | 3.1e-5 | 0.02 |
| insertion tip RMS / L | 2.3e-5 | 0.02 |
| insertion tip max / L | 6.0e-5 | 0.04 |
| tip_bend curvature rel L2 | 0.0096 | 0.10 |
| twist tip angle RMS / peak | 0.0030 | 0.05 |
| root reaction rel error | 0.0053 | 0.10 |
| root moment rel error (added after the review) | 7.1e-5 | 0.10 |
| tube_straight tip RMS / L (added after the review) | 6.3e-7 | 0.02 |
| tube_bend tip RMS / L (added after the review) | 5.9e-5 | 0.02 |
| each refinement change | ≤ 0.016 | half its bound |

- **Ours at the defaults** (unit inertia, `per_edge`) failed sag, insertion, tip bend and twist by two orders of magnitude. `tube_bend` is also a retained legacy-default parity failure; the physical-inertia row above does not qualify those defaults.
- **Both engines move about 0.1 L on sag and insertion when the substep is halved.** This is informational. Both damp `v *= 1 − damping` per substep, so halving the substep doubles the damping per second. The change is the damping model's, the same on both sides; it is not a discretisation error of either solver.

## Tube and bend (added after the review)

`tube_straight` and `tube_bend` insert the wire 0.2 m into a rigid 2 mm lumen: straight, or turning 90° on a 5 cm radius. Both sides use centreline containment, matched as closely as the two models allow:

- **Reference:** `CenterlineVesselRuntime` with `two_way` off and no interior pull (`interior_deadband 1`, `interior_stiffness 0`).
- **Ours:** `graph_containment` on a pinned centreline tree, without the spline lumen or candidate search.
- **On both:** one-sided linear capsules, 0.005 motion clamps, a 0.05 activation shell, 2 containment passes, and 32 contact/solve alternations per substep.

The remaining differences are the solve order within an alternation and the wall model. The workflow's own scenes use a two-way wall with interior pull; that wall is not compared here.
