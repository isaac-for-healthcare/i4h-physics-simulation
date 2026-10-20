# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Public binding parity with the standalone YAML assessor, on real Warp arrays."""

import numpy as np
import pytest
from assessment_host import AssessmentHost
from helpers import needs_external_data
from i4h_assessment.executor import Procedure
from i4h_endoluminal.assessment import ReachTarget, single_target_assessor
from i4h_endoluminal.bindings import CatheterBinding
from i4h_isaaclab.composition import load_case, read_case


@pytest.mark.parametrize("captured", [False, True])
@needs_external_data
def test_public_assessor_and_complete_selected_reset(captured):
    sim = load_case("aorta_static", envs=2)
    sim.cuda_graph = captured
    sim.prepare()
    binding = CatheterBinding(sim)
    assert binding.q.ptr == sim.scene.field("q", binding.body).ptr
    assert binding.distal_node == 191
    spec = read_case("aorta_static_reach")["procedure"]
    assessor = single_target_assessor(
        binding,
        target=spec["target"]["position"],
        radius=spec["radius"],
        hold=spec["hold"],
        timeout=spec["timeout"],
    )
    legacy = Procedure(spec, sim, measurements={"reach_target": ReachTarget})
    q0 = binding.q.numpy().copy()
    material = binding.materials["young"].numpy().copy()
    binding.commands.fill_(0.2)
    assessor.begin_action()
    legacy.begin_action()
    for _ in range(3):
        sim.step()
        assessor.update(sim.dt)
        legacy.update(sim.dt)
        assert np.array_equal(assessor.rows.numpy(), legacy.rows.numpy())
        assert np.array_equal(assessor.history.numpy(), legacy.history.numpy())
    stepped = binding.q.numpy().copy()
    sim.physics.reset([0])
    assert np.array_equal(binding.q.numpy()[0], q0[0])
    assert np.array_equal(binding.q.numpy()[1], stepped[1])
    assert np.array_equal(binding.commands.numpy()[0], np.zeros(3))
    assert np.array_equal(binding.materials["young"].numpy(), material)
    sim.physics.invalidate()
    with pytest.raises(RuntimeError, match="invalidated"):
        sim.physics.reset()
    with pytest.raises(RuntimeError, match="invalidated"):
        single_target_assessor(
            binding, target=[0, 0, 0], radius=0.1, hold=0.5, timeout=120
        )


@pytest.mark.parametrize("captured", [False, True])
def test_shared_owner_resets_every_body_and_preserves_assessment(captured):
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
    from i4h_physics.systems.rods.rod_direct import RodDirect

    cfg = read_case("two_worlds")
    for entry in cfg["systems"]:
        if entry["type"] == "rod_pbd":
            entry["type"] = "rod_direct"
    cfg["assets"]["rod2"] = {**cfg["assets"]["rod"], "start": [-0.2, 0.0, 0.2]}
    cfg["systems"].append(dict(type="rod_direct", id="second_rod", body="rod2"))
    sim = load_case(cfg, envs=2)
    sim.cuda_graph = captured
    rods = [s for s in sim.systems if isinstance(s, RodDirect)]
    rod = rods[0]
    for item in rods:
        sim.systems.append(CatheterControl(sim.scene, rod=item, track_length=0.0))
    first, second = CatheterBinding(sim, "rod"), CatheterBinding(sim, "rod2")
    assert first.owner is second.owner is sim.physics
    assert not hasattr(first, "reset")
    evidence = first.configure_tip_evidence(first, 8)
    sim = AssessmentHost(sim)
    sim.procedure = single_target_assessor(first, target=[100, 100, 100], radius=0.1, hold=0.5, timeout=120)
    sim.prepare()
    arrays = [sim.scene.q, sim.scene.qd, sim.scene.q_prev, sim.scene.flags,
              first.commands, first.targets, second.commands, second.targets,
              rod.frame, rod.frame_pred, rod.omega,
              rods[1].frame, rods[1].frame_pred, rods[1].omega,
              sim.scene.field("q", sim.scene.body("water")),
              sim.scene.field("qd", sim.scene.body("water")),
              sim.scene.field("q_prev", sim.scene.body("water")),
              sim.scene.field("flags", sim.scene.body("water"))]
    initial = [a.numpy().reshape(2, -1).copy() for a in arrays]
    first.commands.fill_(0.1)
    second.commands.fill_(0.2)
    for _ in range(4):
        sim.step()
    elapsed = sim.procedure.elapsed.numpy().copy()
    before = [a.numpy().reshape(2, -1).copy() for a in arrays]
    sim.physics.reset([0])
    for a, start, stepped in zip(arrays, initial, before):
        reset = a.numpy().reshape(2, -1)
        assert np.array_equal(reset[0], start[0])
        assert np.array_equal(reset[1], stepped[1])
    assert np.array_equal(sim.procedure.elapsed.numpy(), elapsed), "physical reset must not reset assessment"
    assert evidence.count.numpy().tolist() == [0, 4]
    sim.reset([0])
    assert sim.procedure.elapsed.numpy()[0] == 0
    assert sim.procedure.elapsed.numpy()[1] == elapsed[1]
    sim.prepare()
    assert first.valid and second.valid and evidence.valid
    sim.physics.invalidate()
    assert not first.valid and not second.valid and not evidence.valid
