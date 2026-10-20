# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched envs against independent single-env runs: env k of an ``E = 3`` scene must evolve exactly like an
``E = 1`` scene of the same mesh variant under the same actions.

This is the oracle for the particle layout: with one env every layout is the same, so a kernel that indexes
with the wrong stride only shows up here, as one env reading another env's or another body's particles.
The cases are made deterministic (no RNG-driven emission, no smoke) and get a ``spacer`` body between the
real ones, a short rod no system acts on, so the per-env width of a world-major layout never equals any single
body's count. Positions are compared in the same frame; no geometry is translated.
"""

import copy

import numpy as np
import pytest
import warp as wp
from helpers import deterministic, needs_chole, needs_vessels
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import load_case, read_case
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

# Measured run-to-run noise of these fixtures is below 1e-7. Burning is chaotic once tets start to die (two
# identical single-env runs drift ~3e-3 a few steps after the first deletion), so the burn check stops shortly
# after the first deletions and compares the topology masks exactly.
ATOL = 1e-5




def with_first_variant(case: dict, asset: str, variant: int) -> dict:
    """Rotate a tet_grid's variant list so env 0 of an ``E = 1`` scene loads ``variant``."""
    case = copy.deepcopy(case)
    dims = case["assets"][asset]["dim"]
    case["assets"][asset]["dim"] = dims[variant:] + dims[:variant]
    return case


def two_worlds_actions(sim, step, energy_from=None):
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    E = sim.scene.envs
    actions = np.zeros((E, control.count, 10), np.float32)
    t = step / 60.0
    actions[:, :, :3] = np.float32([0.004 * np.sin(3.0 * t), 0.0, 0.045 + 0.004 * np.sin(2.0 * t)])
    actions[:, :, 6] = 1.0
    if energy_from is not None and step >= energy_from:
        actions[:, :, 8] = 1.0  # energy: heat and burn the touched tets
    control.actions.assign(actions)


def catheter_actions(sim, step):
    control = next(s for s in sim.systems if isinstance(s, CatheterControl))
    E = sim.scene.envs
    rates = np.tile(np.float32([0.6, 0.3 * np.sin(step / 20.0), 0.4]), (E, 1))
    control.actions.assign(rates.reshape(control.actions.shape))


def run_case(case: dict, envs: int, steps: int, act) -> dict[str, np.ndarray]:
    sim = load_case(case, envs=envs)
    for k in range(steps):
        act(sim, k)
        sim.step()
    wp.synchronize()
    scene = sim.scene
    out = {b.name: scene.view(scene.q, b).numpy() for b in scene.bodies}
    for i, s in enumerate(sim.systems):
        for name in ("frame", "active", "edge_active", "deleted"):
            arr = getattr(s, name, None)
            if isinstance(arr, wp.array) and arr.size:
                out[f"{i}:{type(s).__name__}.{name}"] = arr.numpy().reshape(envs, -1)
    return out


def assert_env_matches(batched, single, env, what):
    for key, ref in single.items():
        got = batched[key][env]
        assert np.isfinite(got).all(), f"{what}: {key} env {env} not finite"
        if np.issubdtype(ref.dtype, np.integer):  # topology masks: exact
            assert np.array_equal(got, ref[0]), f"{what}: {key} env {env} differs from its single-env run"
            continue
        diff = float(np.abs(got - ref[0]).max()) if ref.size else 0.0
        assert diff < ATOL, f"{what}: {key} env {env} differs from its single-env run by {diff:.2e}"


@pytest.mark.parametrize("steps, energy_from", [(60, None), (12, 0)], ids=["contact", "burn"])
def test_two_worlds_batched_matches_single(steps, energy_from):
    case = deterministic(read_case("two_worlds"), spacer_before="rod")
    def act(sim, k):
        return two_worlds_actions(sim, k, energy_from)
    batched = run_case(case, 3, steps, act)  # envs cycle variants: 0, 1, 0
    for env, variant in enumerate((0, 1, 0)):
        single = run_case(with_first_variant(case, "block", variant), 1, steps, act)
        assert_env_matches(batched, single, env, f"two_worlds variant {variant}")
    if energy_from is not None:
        deleted = next(v for k, v in batched.items() if k.endswith("Surface.deleted"))
        assert (deleted > 0).all(), f"deleted {deleted.ravel().tolist()}: the burn check must see tets die in every env"


@needs_vessels
@pytest.mark.parametrize("name", ["aorta_static"])
def test_rod_case_batched_matches_single(name):
    case = read_case(name)
    first = next(iter(case["assets"]))
    case = deterministic(case, spacer_before=first)
    steps = 60
    batched = run_case(case, 3, steps, catheter_actions)
    single = run_case(case, 1, steps, catheter_actions)
    for env in range(3):
        assert_env_matches(batched, single, env, name)


def phased_tool_actions(phases):
    """Every tool sweeps through the tissue on its own per-env phase and closes its jaws from step 6; no energy or
    fire (burning and cutting are chaotic)."""
    from i4h_isaaclab.views import action_bounds, controls

    def act(sim, step):
        ctrl = controls(sim)
        low, high = action_bounds(sim, 0.5)
        mid, half = (low + high) / 2.0, (high - low) / 2.0
        t = step / 60.0
        rows = []
        for ph in phases:
            u = np.zeros(mid.shape[0], np.float32)
            for k in range(0, mid.shape[0] - 9, 10):  # instrument_control columns: position, quaternion, grip, energy, fire
                u[k : k + 3] = 0.8 * np.sin(4.0 * t + np.arange(3) + ph)
                u[k + 7] = 1.0 if step >= 6 else -1.0
                u[k + 8] = u[k + 9] = -1.0
            rows.append(mid + half * u)
        physical = np.asarray(rows, np.float32)
        offset = 0
        for s in ctrl:
            width = int(np.prod(s.actions.shape[1:]))
            s.actions.assign(physical[:, offset : offset + width].reshape(s.actions.shape))
            offset += width
    return act


@needs_chole
def test_chole_batched_matches_single():
    """Chole tissue is chaotic run to run even before contact (two identical single-env runs drift ~1e-4 by step 12,
    ~1e-3 by step 24), so exact comparison is impossible. Each env gets its own tool phase instead, so the envs differ
    by ~1e-2 or more: an env of the batched run must match its own single-env run far better than it matches any other
    env (a cross-env read makes the two comparable), and grasp and topology must agree exactly."""
    case = deterministic(read_case("chole_tet_med"), spacer_before='tissue')
    phases = (0.0, 1.7, 3.1)
    steps = 24
    batched = run_case(case, len(phases), steps, phased_tool_actions(phases))
    for env, ph in enumerate(phases):
        single = run_case(case, 1, steps, phased_tool_actions((ph,)))
        own = float(np.abs(batched['tissue'][env] - single['tissue'][0]).max())
        others = min(float(np.abs(batched['tissue'][env] - batched['tissue'][o]).max()) for o in range(len(phases)) if o != env)
        assert others > 5.0e-3, "the per-env phases must make the envs differ, or the check tests nothing"
        assert own < 0.2 * others, f"env {env}: {own:.2e} from its single run vs {others:.2e} from the other envs"
        for key, ref in single.items():
            if np.issubdtype(ref.dtype, np.integer):
                assert np.array_equal(batched[key][env], ref[0]), f"env {env}: {key} differs from its single-env run"
