# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A volumetric grasp on a tet body without a surface system: its tool shares get their own particle rows."""
from i4h_physics import Scene, Stepper, declare_tet
from i4h_physics.loaders import add_tet_grid
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.instrument_storage import declare_instruments
from i4h_surgical.interaction_storage import declare_volumetric_grasp
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp


def block_and_grasper(declared, built, device="cpu"):
    scene = Scene(device, 2, gravity=(0.0, 0.0, 0.0))
    add_tet_grid(scene, "block", [3, 3, 2], 0.02, radius=0.005)
    add_instrument(scene, "ball", kind="sphere", radius=0.02)
    declare_tet(scene, "block", system_id="tet")
    declare_instruments(scene)
    declare_volumetric_grasp(scene, body="block", capacity=declared)
    scene.allocate()
    control = InstrumentControl(scene)
    grasp = VolumetricGrasp(scene, instruments=control, body="block", capacity=built)
    return Stepper.create(scene, [control, Tet(scene, body="block"), grasp], dt=1.0 / 60.0, substeps=1, iterations=1), grasp


def test_tool_shares_use_dedicated_particle_rows_without_a_surface():
    sim, grasp = block_and_grasper(8, 8)
    sim.prepare()
    sim.step()
    assert grasp.delta_k.ptr == sim.state.i4h.volumetric_grasp_delta_k.ptr
