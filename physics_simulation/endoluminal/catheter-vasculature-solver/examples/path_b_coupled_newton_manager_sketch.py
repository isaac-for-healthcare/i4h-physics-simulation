#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path B wiring sketch: coupled MJWarp + XPBD catheter rod under Newton Manager.

This script documents the intended Isaac Lab integration. It does **not** launch
a full Lab app (that needs Isaac Sim + isaaclab_newton). Use it as a checklist
when building an env:

1. ``CoupledMJWarpXPBDRodSolverCfg`` → ``NewtonCfg.solver_cfg``
2. During scene / builder setup: ``register_xpbd_rod_builder_attributes`` +
   ``add_catheter_rod_to_builder``
3. Spawn robot/articulation assets as usual (MJWarp owns rigid DOFs)
4. Optional: ``NewtonCoupledMJWarpXPBDRodManager.register_vessel_track(...)``
5. Drive proximal insertion via root particle writers / action terms

Run a dry import check (fails clearly if Lab/Newton rod APIs are missing)::

    python examples/path_b_coupled_newton_manager_sketch.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print("Path B — coupled MJWarp + XPBD rod (Newton Manager)")
    print("-" * 60)

    # Builder helpers only need this package (+ Newton when spawning).
    from catheter_vasculature_solver import RodConfig, RodGeometryConfig, RodMaterialConfig
    from catheter_vasculature_solver.isaaclab_integration import (
        add_catheter_rod_to_builder,
        register_xpbd_rod_builder_attributes,
    )

    cfg = RodConfig(
        material=RodMaterialConfig(bend_stiffness=0.1, twist_stiffness=0.4),
        geometry=RodGeometryConfig(num_segments=24, rest_length=0.48, radius=0.002),
    )
    print(f"RodConfig ready: segments={cfg.geometry.num_segments}, L={cfg.geometry.segment_length}")

    try:
        from catheter_vasculature_solver.isaaclab_integration import (
            CoupledMJWarpXPBDRodSolverCfg,
            NewtonCoupledMJWarpXPBDRodManager,
            XPBDRodSolverCfg,
        )
    except ImportError as e:
        print("\n[skip] isaaclab_newton not installed in this environment.")
        print(f"       Import error: {e}")
        print("\nWhen Isaac Lab develop is available, wire sim as:\n")
        print(
            """
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from catheter_vasculature_solver.isaaclab_integration import (
    CoupledMJWarpXPBDRodSolverCfg,
    XPBDRodSolverCfg,
)

sim_cfg = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=CoupledMJWarpXPBDRodSolverCfg(
            coupling_mode="one_way",  # or "two_way"
            rigid_solver_cfg=MJWarpSolverCfg(),
            rod_solver_cfg=XPBDRodSolverCfg(
                num_segments=24,
                solver_backend="block_thomas",
            ),
            soft_contact_ke=5.0e3,
            soft_contact_mu=0.5,
        ),
        num_substeps=4,
    )
)
""".strip()
        )
        print("\nBuilder spawn (before finalize), inside your scene setup:\n")
        print(
            """
register_xpbd_rod_builder_attributes(builder)
add_catheter_rod_to_builder(builder, rod_config, lock_root=True)
""".strip()
        )
        return 0

    coupled = CoupledMJWarpXPBDRodSolverCfg(
        coupling_mode="two_way",
        rod_solver_cfg=XPBDRodSolverCfg(num_segments=24, solver_backend="block_thomas"),
    )
    print(f"Coupled cfg OK: class_type={coupled.class_type}")
    print(f"Manager class: {NewtonCoupledMJWarpXPBDRodManager.__name__}")
    print(f"Helpers: {register_xpbd_rod_builder_attributes.__name__}, {add_catheter_rod_to_builder.__name__}")
    print("\nPath B configs resolved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
