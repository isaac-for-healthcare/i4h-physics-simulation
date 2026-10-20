# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case and run-setting validation on CPU with generated assets (PR #10 review, track B)."""

import pytest
from i4h_physics.config import load_case

ROD = {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "count": 4, "segment_length": .02, "radius": .002}


def case(systems, sim=None):
    return {"sim": {"device": "cpu", "envs": 2, "dt": .01, **(sim or {})}, "assets": {"rod": dict(ROD)},
            "systems": systems}


def test_exactly_one_particles_system():
    load_case(case([{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]))
    load_case(case([]))  # no systems: a static scene
    with pytest.raises(ValueError, match="2 particles systems"):
        load_case(case([{"type": "particles"}, {"type": "particles", "id": "again"}, {"type": "rod_direct", "body": "rod"}]))
    with pytest.raises(ValueError, match="no particles system"):
        load_case(case([{"type": "rod_direct", "body": "rod"}]))


@pytest.mark.parametrize("key,value", [("cuda_graph", "false"), ("debug", 1), ("substeps", 2.5), ("substeps", 2.0),
                                       ("iterations", True), ("iterations", 0), ("dt", "0.01")])
def test_run_settings_are_validated_not_coerced(key, value):
    with pytest.raises(ValueError, match=f"sim.{key}"):
        load_case(case([{"type": "particles"}], sim={key: value}))


def test_stepper_arguments_are_validated():
    from i4h_physics.stepper import Stepper

    sim = load_case(case([{"type": "particles"}]))
    for kw, match in (({"cuda_graph": "false"}, "cuda_graph"), ({"substeps": 1.5}, "substeps"), ({"dt": 0.0}, "dt")):
        with pytest.raises(ValueError, match=match):
            Stepper(sim.solver, **kw)


def test_system_settings_are_validated_with_their_limits():
    from i4h_physics.checks import check_bool, check_float, check_int

    assert check_int("x", 0, 0) == 0 and check_bool("x", False) is False and check_float("x", 2) == 2.0
    systems = [{"type": "particles"}, {"type": "rod_pbd", "body": "rod"}]
    load_case(case(systems))
    for bad, match in (({"sweeps": 2.5}, "rod_pbd.sweeps"), ({"sweeps": True}, "rod_pbd.sweeps"),
                       ({"sweeps": 0}, "at least 1")):
        with pytest.raises(ValueError, match=match):
            load_case(case([systems[0], systems[1] | bad]))


def test_rebinding_a_live_parameter_is_detected():
    import warp as wp
    from i4h_physics.systems.mechanics.particles import Particles

    sim = load_case(case([{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]))
    sim.prepare()
    sim.step()
    particles = next(s for s in sim.systems if isinstance(s, Particles))
    particles.damping.fill_(0.9)  # in place: fine
    sim.step()
    particles.damping = wp.full(2, 0.5, dtype=wp.float32, device="cpu")  # a compatible rebind: the graphs are stale
    with pytest.raises(RuntimeError, match="rebound.*prepare"):
        sim.step()
    sim.prepare()
    sim.step()
    assert particles.damping.numpy().tolist() == [0.5, 0.5]
    for bad in (0.9, wp.full(3, 0.5, dtype=wp.float32, device="cpu")):
        particles.damping = bad
        with pytest.raises(TypeError, match="particles.damping is a live parameter.*fill_"):
            sim.step()
        with pytest.raises(TypeError, match="particles.damping is a live parameter"):
            sim.prepare()  # recapturing does not repair a scalar or a reshaped array


def test_every_live_array_is_registered():
    """The fields Particles once assigned with live() directly are in its registry."""
    from i4h_physics.systems.mechanics.particles import Particles

    sim = load_case(case([{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]))
    particles = next(s for s in sim.systems if isinstance(s, Particles))
    assert set(particles._live_layouts) == {"damping", "max_velocity"}



def test_a_strided_view_of_a_live_array_is_rejected():
    """Same buffer, same shape, another element order: the transpose of a square (E, 3) live array, E = 3."""
    from dataclasses import dataclass

    from i4h_physics.systems.base import System
    from i4h_physics.systems.mechanics.particles import Particles

    @dataclass(eq=False)
    class Gains(System):
        gain: tuple = (1.0, 2.0, 3.0)

        def __post_init__(self):
            self._live("gain")

    cfg = {"sim": {"device": "cpu", "envs": 3, "dt": .01}, "assets": {"rod": dict(ROD)},
           "systems": [{"type": "particles"}, {"type": "gains"}]}
    sim = load_case(cfg, systems={"particles": Particles, "gains": Gains})
    sim.prepare()
    gains = sim.systems[1]
    transposed = gains.gain.transpose()
    assert transposed.shape == gains.gain.shape and transposed.ptr == gains.gain.ptr
    gains.gain = transposed
    with pytest.raises(TypeError, match="gains.gain is a live parameter.*strides"):
        sim.step()


def test_env_count_and_enabled_are_validated():
    with pytest.raises(ValueError, match="sim.envs"):
        load_case(case([{"type": "particles"}], sim={"envs": 2.5}))
    with pytest.raises(ValueError, match="rod.enabled"):
        load_case(case([{"type": "particles"}, {"type": "rod_direct", "id": "rod", "body": "rod", "enabled": "false"}]))
    sim = load_case(case([{"type": "particles"}, {"type": "rod_direct", "id": "rod", "body": "rod"}]))
    sim.systems[1].enabled = "false"  # set through the API: rejected at the next prepare
    with pytest.raises(ValueError, match="rod.enabled"):
        sim.prepare()


def test_host_env_ids_are_checked_before_narrowing():
    import numpy as np

    sim = load_case(case([{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]))
    for bad in (np.int64([4294967296]), [0, 5], np.float32([0.0])):
        with pytest.raises(ValueError, match="env_ids"):
            sim.reset(bad)
    sim.reset(np.int64([1]))


@pytest.mark.parametrize("debug", [False, True])
def test_fractional_torch_env_ids_are_rejected(debug):
    import numpy as np
    import torch

    sim = load_case(case([{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]), debug=debug)
    sim.step()
    before = sim.scene.q.numpy().copy()
    for bad in (torch.tensor([0.5]), torch.tensor([1.0], dtype=torch.float64)):
        with pytest.raises(ValueError, match="env_ids must be integers"):
            sim.reset(bad)
    assert np.array_equal(before, sim.scene.q.numpy()), "a rejected selection resets nothing"
    sim.reset(torch.tensor([]))  # empty (float32 by default): selects nothing, accepted
    assert np.array_equal(before, sim.scene.q.numpy())
    sim.reset(torch.tensor([1]))


def test_env_rows_copy_one_env_only(monkeypatch):
    """add_env_particles (called once per env by Isaac Lab's per-world hook) copies W rows per env, not E * W."""
    import numpy as np
    from i4h_physics import newton_model
    from i4h_physics.config import load_scene

    cfg = case([])
    cfg["sim"]["envs"] = 3
    cfg["assets"]["solid"] = {"type": "tet_grid", "dim": [2, 2, 2], "cell": .02}
    scene = load_scene(cfg)
    W = scene.main.width
    for field in ("q0", "inv_mass", "radius", "flags"):
        for env in range(scene.envs):
            np.testing.assert_array_equal(newton_model.env_rows(scene, field, env), newton_model.main_rows(scene, field)[env])
    copied = []
    concatenate = np.concatenate

    def counting(arrays, axis=0, **kw):
        out = concatenate(arrays, axis=axis, **kw)
        copied.append(out.shape[0] if axis == 0 else out.shape[0] * out.shape[1])
        return out

    class Builder:
        def add_particles(self, *args):
            pass

    monkeypatch.setattr(newton_model.np, "concatenate", counting)
    for env in range(scene.envs):
        newton_model.add_env_particles(Builder(), scene, env)
    assert copied == [W] * 4 * scene.envs, copied
