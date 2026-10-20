#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the catheter rod behind Newton's solver interface, as Isaac Lab does.

Isaac Lab's ``NewtonManager`` drives a Newton ``SolverBase`` and nothing else, so
:class:`~catheter_vasculature_solver.newton_rod_solver.NewtonCathRodSolver` wraps
the in-tree rod solver in that interface. This script exercises exactly that path
without Isaac Sim: build a Newton model holding the rod's particles, then step it
through ``SolverBase.step``.

Integration checklist for an actual Isaac Lab env:

1. ``XPBDRodSolverCfg`` (or ``CoupledMJWarpXPBDRodSolverCfg``) → ``NewtonCfg.solver_cfg``
2. During scene setup, before finalize: ``add_catheter_rod_to_builder(builder, cfg)``
3. Hand the returned range to ``NewtonXPBDRodManager.register_rod(...)``
4. Spawn robot / articulation assets as usual (MJWarp owns rigid DOFs)
5. Drive proximal insertion through the rod solver's ``apply_proximal_control``

Run::

    python examples/isaaclab_newton_manager_sketch.py
"""

from __future__ import annotations

import sys

from catheter_vasculature_solver import RodConfig, RodGeometryConfig, RodMaterialConfig
from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder, initial_rod_positions

LAB_WIRING = """
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg
from catheter_vasculature_solver.isaaclab_integration import (
    NewtonXPBDRodManager,
    XPBDRodSolverCfg,
    add_catheter_rod_to_builder,
)

# Scene setup, before ModelBuilder.finalize():
particle_range = add_catheter_rod_to_builder(builder, rod_config)
NewtonXPBDRodManager.register_rod(particle_range)

# Vessel mesh / centerline runtimes are GPU buffers, so pass a configured rod:
#   NewtonXPBDRodManager.register_rod(particle_range, rod=CathRodSolver(...))

sim_cfg = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=XPBDRodSolverCfg(num_segments=24, solver_backend="block_thomas"),
        num_substeps=4,
        use_cuda_graph=False,
    )
)
""".strip()


def build_rod_config(device: str) -> RodConfig:
    return RodConfig(
        material=RodMaterialConfig(bend_stiffness=0.1, twist_stiffness=0.4),
        geometry=RodGeometryConfig(num_segments=24, rest_length=0.48, radius=0.002),
        device=device,
    )


def run_under_newton(num_steps: int = 10, dt: float = 1.0 / 240.0) -> int:
    """Step the rod through Newton's ``SolverBase`` interface."""
    import newton
    from catheter_vasculature_solver import XPBDRodSolver
    from catheter_vasculature_solver.newton_rod_solver import NewtonCathRodSolver

    builder = newton.ModelBuilder()
    config = build_rod_config("cpu")

    particle_range = add_catheter_rod_to_builder(builder, config, lock_root=True)
    model = builder.finalize()
    print(f"Newton model finalized: {model.particle_count} particles")
    print(f"Rod owns [{particle_range.offset}, {particle_range.end})")

    # The rod must live on the model's device; finalize() decides that.
    config.device = str(model.device)
    rod = XPBDRodSolver(config, floor_z=None)
    solver = NewtonCathRodSolver(model, rod, particle_range)

    state_in, state_out = model.state(), model.state()
    control = model.control()
    for _ in range(num_steps):
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in

    tip = state_in.particle_q.numpy()[particle_range.end - 1]
    print(f"After {num_steps} steps, tip particle at {tip}")
    print("\nSolverBase path exercised successfully.")
    return 0


def main() -> int:
    print("Catheter rod under Newton's SolverBase (Isaac Lab's solver contract)")
    print("-" * 68)

    config = build_rod_config("cpu")
    positions = initial_rod_positions(config)
    print(f"RodConfig ready: segments={config.geometry.num_segments}, L={config.geometry.segment_length}")
    print(f"Authored {positions.shape[0]} rest particles from {positions[0]} to {positions[-1]}")

    try:
        status = run_under_newton()
    except ImportError as e:
        print(f"\n[skip] newton is not importable here: {e}")
        print("       The rod solver itself needs no newton; only this Newton/Isaac Lab path does.")
        status = 0

    print("\nIsaac Lab wiring:\n")
    print(LAB_WIRING)
    return status


if __name__ == "__main__":
    sys.exit(main())
