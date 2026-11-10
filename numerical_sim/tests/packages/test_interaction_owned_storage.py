# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Interaction history (centreline, residue, hook, grasps, clips): lifecycle contracts that ownership must preserve."""
import numpy as np
import pytest
import warp as wp
from interaction_helpers import HISTORY, MODES, authored, history, poison, runtime, system

# What an episode reset clears; every other history array keeps its bytes (deliberately stale).
RESET = {
    ("Hook", "active"): 0, ("Hook", "verts"): -1, ("Hook", "tet"): -1,
    ("Grasp", "active"): 0, ("Grasp", "prim"): -1, ("Grasp", "count"): 0, ("Grasp", "pending"): 0,
    ("VolumetricGrasp", "particle"): -1, ("VolumetricGrasp", "active"): 0, ("VolumetricGrasp", "sphere"): 0,
    ("VolumetricGrasp", "count"): 0, ("VolumetricGrasp", "pending"): 0, ("VolumetricGrasp", "claim"): 0,
    ("Clipping", "node"): -1, ("Clipping", "active"): 0, ("Clipping", "count"): 0, ("Clipping", "clipped"): 0,
}
CENTRELINE = (("Centreline", "positions"), ("Centreline", "active_count"))


def rows(sim, key, env):
    """Every component of one world's rows (NumPy exposes vector dtypes as a trailing axis)."""
    return getattr(system(sim, key[0]), key[1]).numpy().reshape(sim.scene.envs, -1)[env]


@pytest.mark.parametrize("device,captured", MODES)
def test_episode_reset_clears_only_its_fields_in_selected_worlds(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim)
    before = history(sim)
    centre = system(sim, "Centreline")
    rest, full = centre.rest.numpy()[centre.variant.numpy()[1]], centre.clamp_count.numpy()[centre.variant.numpy()[1]]
    sim.reset_physics(mask=wp.array([False, True], dtype=wp.bool, device=device))
    after = history(sim)
    for key in before:
        old = np.frombuffer(before[key], np.uint8).reshape(2, -1)
        new = np.frombuffer(after[key], np.uint8).reshape(2, -1)
        np.testing.assert_array_equal(new[0], old[0], err_msg=f"unselected world changed: {key}")
        if key in RESET:
            np.testing.assert_array_equal(rows(sim, key, 1), RESET[key], err_msg=str(key))
        elif key not in CENTRELINE:
            np.testing.assert_array_equal(new[1], old[1], err_msg=f"reset cleared a field it leaves stale: {key}")
    np.testing.assert_array_equal(centre.positions.numpy()[1], rest)
    np.testing.assert_array_equal(centre.active_count.numpy()[1], full)


@pytest.mark.parametrize("device,captured", MODES)
def test_cache_only_preparation_preserves_history(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim)
    before = history(sim)
    sim.prepare()
    sim.prepare()
    assert history(sim) == before


@pytest.mark.parametrize("device,captured", MODES)
def test_disabled_hook_preparation_clears_its_anchor_flag_everywhere(device, captured):
    """A named exception: a disabled hook must release the contact suppression that aliases its entry anchor."""
    cfg, a, m = authored(device, disable=("hook",))
    sim = runtime(cfg, a, m, captured, prepare=False)
    poison(sim)
    before = history(sim)
    sim.prepare()
    after = history(sim)
    assert not system(sim, "Hook").active.numpy().any()
    assert not system(sim, "InstrumentContact").suppress_active.numpy().any()
    assert {k: v for k, v in after.items() if k != ("Hook", "active")} == {k: v for k, v in before.items() if k != ("Hook", "active")}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_hook_entry_views_alias_the_anchor_arrays(device):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m)
    hook, contact = system(sim, "Hook"), system(sim, "InstrumentContact")
    poison(sim)
    np.testing.assert_array_equal(contact.suppress_active.numpy(), hook.active.numpy()[:, 0])
    np.testing.assert_array_equal(contact.suppress_verts.numpy(), hook.verts.numpy()[:, 0])
    np.testing.assert_array_equal(hook.entry_active.numpy(), hook.active.numpy()[:, 0])
    np.testing.assert_array_equal(hook.entry_tet.numpy(), hook.tet.numpy()[:, 0])
    assert contact.suppress_active.ptr == hook.active.ptr and contact.suppress_verts.ptr == hook.verts.ptr
    assert system(sim, "Bleeding").clipped is system(sim, "Clipping").clipped


def test_history_set_is_the_inventoried_one():
    assert sum(len(v) for v in HISTORY.values()) == 36


def moved(sim, shift):
    """Translate every particle of the centreline's body in the bound State; returns the expected ring centres."""
    centre = system(sim, "Centreline")
    q = sim.scene.field("q", centre.body)
    host = q.numpy() + np.float32(shift)
    wp.copy(q, wp.array(host, dtype=wp.vec3, device=q.device))
    return centre.positions.numpy() + np.float32(shift)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cache_refresh_recomputes_centreline_from_a_moved_state(device):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m)
    expected = moved(sim, [0.01, 0.0, 0.0])
    sim.solver.reset(sim.state, flags=0)
    np.testing.assert_allclose(system(sim, "Centreline").positions.numpy(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_binding_a_moved_state_refreshes_centreline(device):
    """Pre-C4.1 behaviour: the constructor recomputes the tracker from the State it is bound to."""
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    expected = moved(sim, [0.01, 0.0, 0.0])
    fresh = runtime(cfg, a, m, state=state, control=control, prepare=False)
    np.testing.assert_allclose(system(fresh, "Centreline").positions.numpy(), expected, rtol=0, atol=1e-6)
