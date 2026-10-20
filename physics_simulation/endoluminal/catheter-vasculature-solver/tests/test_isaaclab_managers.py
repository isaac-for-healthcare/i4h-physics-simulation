# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manager regressions, using Isaac Lab's real base class when installed."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from catheter_vasculature_solver import RodConfig, RodParticleBridge, RodParticleRange, XPBDRodSolver


def test_default_manager_constructs_a_bare_rod(rod_managers):
    cfg = rod_managers.cfg(num_segments=4)
    rod_managers.rod.register_rod(RodParticleRange(offset=3, count=10, num_envs=2))

    rod = rod_managers.rod._resolve_rod(SimpleNamespace(device="cpu"), cfg)

    assert type(rod) is XPBDRodSolver
    assert rod.num_envs == 2
    assert rod.num_points == 5


@pytest.mark.parametrize("feature", ["track_enabled", "collision_enabled"])
def test_geometry_features_require_an_explicit_rod(rod_managers, feature):
    cfg = rod_managers.cfg(**{feature: True})
    rod_managers.rod.register_rod(RodParticleRange(offset=0, count=25))

    with pytest.raises(ValueError, match="register_rod"):
        rod_managers.rod._resolve_rod(SimpleNamespace(device="cpu"), cfg)


@pytest.mark.parametrize("use_factory", [False, True])
def test_registered_rod_or_factory_supplies_geometry(rod_managers, use_factory):
    cfg = rod_managers.cfg(track_enabled=True, collision_enabled=True)
    model = SimpleNamespace(device="cpu")
    configured_rod = object()
    calls = []

    def factory(model_arg, cfg_arg):
        calls.append((model_arg, cfg_arg))
        return configured_rod

    kwargs = {"rod_factory": factory} if use_factory else {"rod": configured_rod}
    rod_managers.rod.register_rod(RodParticleRange(offset=0, count=25), **kwargs)

    assert rod_managers.rod._resolve_rod(model, cfg) is configured_rod
    assert calls == ([(model, cfg)] if use_factory else [])


@pytest.mark.parametrize(
    ("real_newton", "device"),
    [(False, "cpu"), (True, "cpu"), pytest.param(True, "cuda:0", marks=pytest.mark.gpu)],
    ids=["bridge", "newton-mujoco", "newton-mjwarp"],
)
def test_coupled_step_preserves_progress_and_rigid_output(rod_managers, monkeypatch, real_newton, device):
    """Swapping buffers must preserve every commanded rod substep."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = 4
    cfg.geometry.segment_length = 0.02
    cfg.solver.num_substeps = 1
    cfg.solver.gravity = (0.0, 0.0, 0.0)
    rod = XPBDRodSolver(cfg, num_envs=2, floor_z=None)
    bridge = RodParticleBridge(rod, RodParticleRange(0, rod.num_points * 2, 2))
    rigid_outputs = {}

    if real_newton:
        newton = pytest.importorskip("newton")
        from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder
        from catheter_vasculature_solver.newton_rod_solver import NewtonCathRodSolver
        from newton.solvers import SolverMuJoCo

        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        builder.add_shape_sphere(body, radius=0.1)
        particle_range = add_catheter_rod_to_builder(builder, cfg, num_envs=2)
        model = builder.finalize(device=device)
        rigid = SolverMuJoCo(model, use_mujoco_cpu=device == "cpu")
        solver = NewtonCathRodSolver(model, rod, particle_range)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        rigid_fields = ("body_q", "body_qd", "joint_q", "joint_qd")
    else:
        def advance(state_in, state_out, control, contacts, dt):
            bridge.read_from(state_in)
            bridge.advance(dt)
            bridge.write_to(state_out)

        solver = SimpleNamespace(rod=rod, step=advance)
        states = [SimpleNamespace(
            particle_q=wp.clone(rod.position_array),
            particle_qd=wp.clone(rod.velocity_array),
            body_q=wp.zeros(1, dtype=wp.vec3, device="cpu"),
        ) for _ in range(2)]
        state_0, state_1 = states
        control, rigid = None, None
        rigid_fields = ("body_q",)

    def rigid_step(state_in, state_out, control, contacts, dt):
        if rigid is not None:
            rigid.step(state_in, state_out, control, contacts, dt)
        else:
            state_out.body_q.assign(state_in.body_q.numpy() + 1.0)
        for field in rigid_fields:
            rigid_outputs[field] = getattr(state_out, field).numpy().copy()

    monkeypatch.setattr(rod_managers.coupled, "_rigid_solver", SimpleNamespace(step=rigid_step))
    monkeypatch.setattr(rod_managers.coupled, "_rod_solver", solver)
    monkeypatch.setattr(rod_managers.base, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(rod_managers.base, "_contacts", None, raising=False)
    dt = 1.0 / 240.0
    initial_roots = state_0.particle_q.numpy()[::rod.num_points].copy()
    for substep in range(1, 7):
        rod.apply_proximal_control_gpu([0.6, 0.6], [0.0, 0.0], dt)
        rod_managers.coupled._step_solver(state_0, state_1, control, None, dt)
        roots = state_1.particle_q.numpy()[::rod.num_points]
        np.testing.assert_allclose(roots[:, 0] - initial_roots[:, 0], substep * 0.6 * dt, atol=1e-6)
        for field, expected in rigid_outputs.items():
            np.testing.assert_array_equal(getattr(state_1, field).numpy(), expected)
        state_0, state_1 = state_1, state_0


@pytest.mark.parametrize("manager_name", ["rod", "coupled"])
def test_managers_reject_outer_cuda_graph_before_build(rod_managers, monkeypatch, manager_name):
    manager = getattr(rod_managers, manager_name)
    # The coupled manager must inspect its own outer config, even if the rod
    # manager has a different config or has never been initialized.
    monkeypatch.setattr(rod_managers.base, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)
    monkeypatch.setattr(manager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    cfg = rod_managers.cfg() if manager_name == "rod" else rod_managers.coupled_cfg()

    with pytest.raises(ValueError, match="use_cuda_graph=False"):
        manager._build_solver(SimpleNamespace(device="cpu"), cfg)


@pytest.mark.parametrize("manager_name", ["rod", "coupled"])
def test_managers_allow_eager_solver_construction(rod_managers, monkeypatch, manager_name):
    manager = getattr(rod_managers, manager_name)
    monkeypatch.setattr(manager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)
    solver = object()
    monkeypatch.setattr(rod_managers.rod, "build_nested_solver", lambda *args: solver)
    monkeypatch.setattr(rod_managers.coupled, "_build_rigid_solver", lambda *args: object())
    cfg = rod_managers.cfg() if manager_name == "rod" else rod_managers.coupled_cfg()

    manager._build_solver(SimpleNamespace(device="cpu"), cfg)

    assert rod_managers.base._solver is solver
    assert rod_managers.base._use_single_state is False


@pytest.mark.parametrize("manager_name", ["rod", "coupled"])
@pytest.mark.parametrize("entry_point", ["reset_rod", "_reset_solver_internals"])
def test_manager_reset_forwards_mask_and_current_state(rod_managers, monkeypatch, manager_name, entry_point):
    manager = getattr(rod_managers, manager_name)
    state = object()
    calls = []
    mask = [False, True, False]

    def reset_rod(state_arg, world_mask=None, flags=0):
        calls.append(("rod", state_arg, world_mask, flags))

    def reset_rigid(state_arg, world_mask=None, flags=0):
        calls.append(("rigid", state_arg, world_mask, flags))

    solver = SimpleNamespace(reset=reset_rod)
    monkeypatch.setattr(manager, "get_state_0", lambda: state, raising=False)
    monkeypatch.setattr(rod_managers.base, "_solver", solver, raising=False)
    monkeypatch.setattr(rod_managers.coupled, "_rod_solver", solver)
    monkeypatch.setattr(rod_managers.coupled, "_rigid_solver", SimpleNamespace(reset=reset_rigid))
    monkeypatch.setattr(rod_managers.coupled, "_coupler", SimpleNamespace(
        reset=lambda world_mask=None: calls.append(("coupler", world_mask)),
    ))

    getattr(manager, entry_point)(mask)

    assert ("rod", state, mask, 0) in calls
    if manager_name == "coupled":
        assert ("rigid", state, mask, 0) in calls
        assert ("coupler", mask) in calls
        assert len(calls) == 3
    else:
        assert len(calls) == 1


@pytest.mark.parametrize("manager_name", ["rod", "coupled"])
def test_missing_reset_hook_mask_is_not_a_full_reset(rod_managers, monkeypatch, manager_name):
    manager = getattr(rod_managers, manager_name)
    calls = []
    monkeypatch.setattr(manager, "reset_rod", lambda mask=None: calls.append(mask), raising=False)

    manager._reset_solver_internals(None)

    assert calls == []
