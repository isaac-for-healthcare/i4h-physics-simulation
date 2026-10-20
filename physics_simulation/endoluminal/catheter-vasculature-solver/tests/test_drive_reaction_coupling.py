# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-way coupling: feeding the catheter's proximal reaction to the rigid side.

These tests exist mostly to defend against a silent failure. The obvious ways to
harvest a reaction -- a body's momentum change, as Newton's upstream proxy
coupling does, or the proximal particle's momentum change -- both read exactly
zero here, because the rod solver never writes body state and ``lock_root``
gives the proximal particle zero mass. Either one would leave
``coupling_mode='two_way'`` running as one-way while looking like it worked,
which is worse than the ``NotImplementedError`` it replaced. So the vacuous
cases are pinned as tests alongside the arithmetic.

Everything runs on CPU with fabricated buffers; nothing here needs a simulator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="needs warp installed")

from catheter_vasculature_solver.isaaclab_integration.proxy_coupling import (  # noqa: E402
    DEBUG_ENV_VAR,
    DEFAULT_DEBUG_INTERVAL,
    DriveReactionCoupler,
    debug_log_interval,
    resolve_drive_body_index,
)

DEVICE = "cpu"


class FakeRod:
    """Stands in for the rod solver, reporting a fixed proximal reaction.

    Mirrors ``XPBDRodSolver.proximal_wrench``'s contract of filling a
    caller-supplied ``out`` array, which is what lets the coupler avoid
    allocating per substep.
    """

    def __init__(self, forces, moments=None) -> None:
        self.moments = np.zeros_like(np.asarray(forces)) if moments is None else np.asarray(moments)
        self.forces = np.asarray(forces, dtype=np.float32).reshape(-1, 3)
        self.calls = 0

    def proximal_wrench(self, out=None):
        self.calls += 1
        values = np.concatenate([self.forces, self.moments.reshape(-1, 3)], axis=1)
        out.assign(wp.array(values, dtype=wp.spatial_vector, device=DEVICE))
        return out


class FakeState:
    """Just the three buffers the coupler touches."""

    def __init__(self, body_count: int, particle_q, body_positions, body_rotations=None) -> None:
        self.body_f = wp.zeros(body_count, dtype=wp.spatial_vector, device=DEVICE)
        self.particle_q = wp.array(
            np.asarray(particle_q, dtype=np.float32).reshape(-1, 3),
            dtype=wp.vec3,
            device=DEVICE,
        )
        positions = np.asarray(body_positions, dtype=np.float32).reshape(-1, 3)
        rotations = (
            [wp.quat_identity()] * len(positions)
            if body_rotations is None
            else [wp.quat(*[float(v) for v in rotation]) for rotation in body_rotations]
        )
        self.body_q = wp.array(
            [
                wp.transform(wp.vec3(*position), rotation)
                for position, rotation in zip(positions, rotations)
            ],
            dtype=wp.transform,
            device=DEVICE,
        )


def make_coupler(
    *, body_count=2, num_envs=1, drive_bodies=(1,), relaxation=1.0, root_stride=4, mount_local=None
):
    return DriveReactionCoupler(
        body_count=body_count,
        num_envs=num_envs,
        root_offset=0,
        root_stride=root_stride,
        drive_bodies=drive_bodies,
        mount_local=mount_local,
        relaxation=relaxation,
        device=DEVICE,
    )


def wrench(coupler, body: int) -> np.ndarray:
    return coupler.coupling_forces.numpy()[body]


def origins(count: int):
    return [[0.0, 0.0, 0.0]] * count


# ------------------------------------------------- the opt-in diagnostic


def test_the_diagnostic_is_off_unless_asked_for() -> None:
    """Reading the wrench back costs a device sync, so silence is the default."""
    assert debug_log_interval({}) is None


@pytest.mark.parametrize("word", ["1", "true", "yes", "on", "TRUE", " On "])
def test_a_truthy_word_turns_it_on_at_the_default_interval(word) -> None:
    assert debug_log_interval({DEBUG_ENV_VAR: word}) == DEFAULT_DEBUG_INTERVAL


@pytest.mark.parametrize("word", ["0", "false", "no", "off", ""])
def test_a_falsey_word_leaves_it_off(word) -> None:
    assert debug_log_interval({DEBUG_ENV_VAR: word}) is None


def test_an_integer_sets_the_step_interval() -> None:
    """So a long run can report occasionally instead of every step."""
    assert debug_log_interval({DEBUG_ENV_VAR: "200"}) == 200


def test_nonsense_is_rejected_rather_than_ignored() -> None:
    """Silently ignoring it would look exactly like the flag not working."""
    with pytest.raises(ValueError, match="is not understood"):
        debug_log_interval({DEBUG_ENV_VAR: "loud"})


def test_a_negative_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        debug_log_interval({DEBUG_ENV_VAR: "-5"})


def test_the_report_names_the_body_and_splits_the_wrench() -> None:
    """What the debug line prints, so it has to match the applied wrench."""
    coupler = make_coupler(mount_local=[(2.0, 0.0, 0.0)])
    rod = FakeRod([[0.0, 5.0, 0.0]])
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))
    coupler.harvest(state, rod)

    body, force, torque = coupler.report()

    assert body == 1
    np.testing.assert_allclose(force, [0.0, 5.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(torque, [0.0, 0.0, 10.0], atol=1e-6)


def test_the_report_reads_the_requested_env() -> None:
    coupler = make_coupler(body_count=2, num_envs=2, drive_bodies=(0, 1), root_stride=4)
    rod = FakeRod([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = FakeState(2, particle_q=origins(8), body_positions=origins(2))
    coupler.harvest(state, rod)

    assert coupler.report(env=0)[0] == 0
    np.testing.assert_allclose(coupler.report(env=1)[1], [2.0, 0.0, 0.0], atol=1e-6)


def test_the_report_rejects_an_out_of_range_env() -> None:
    coupler = make_coupler()

    with pytest.raises(IndexError, match="out of range"):
        coupler.report(env=3)


# ------------------------------------------------- naming the drive body

#: Labels as IsaacLab leaves them on the builder: full USD paths, arm first.
PANDA_LABELS = [
    f"/World/envs/env_0/Robot/{leaf}"
    for leaf in (
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
        "TCP",
    )
]


def test_a_leaf_name_resolves_through_the_usd_path() -> None:
    """Scenes name the flange ``TCP``; the builder stores an absolute path."""
    assert resolve_drive_body_index(PANDA_LABELS, "TCP", 11) == 10


def test_a_full_path_also_resolves() -> None:
    assert resolve_drive_body_index(PANDA_LABELS, "/World/envs/env_0/Robot/TCP", 11) == 10


def test_only_the_first_env_is_searched() -> None:
    """The index is per-env, so env 1's copy must not be the one returned.

    Returning a later env's index would push the reaction onto the wrong
    robot, which for a single-arm scene means a body that is not holding
    the catheter at all.
    """
    two_envs = PANDA_LABELS + [label.replace("env_0", "env_1") for label in PANDA_LABELS]

    assert resolve_drive_body_index(two_envs, "TCP", 11) == 10


def test_an_unknown_name_lists_what_is_available() -> None:
    """A typo is the likely cause, so the message has to be diagnosable."""
    with pytest.raises(ValueError, match="no body named 'tcp'") as caught:
        resolve_drive_body_index(PANDA_LABELS, "tcp", 11)

    # Leaf names, not the paths, since that is what a scene would have written.
    assert "TCP" in str(caught.value)
    assert "/World" not in str(caught.value)


def test_an_ambiguous_leaf_name_is_refused() -> None:
    """Two links sharing a leaf name inside one env cannot be told apart."""
    labels = ["/World/envs/env_0/A/tool", "/World/envs/env_0/B/tool"]

    with pytest.raises(ValueError, match="matches 2 bodies"):
        resolve_drive_body_index(labels, "tool", 2)


def test_an_ambiguous_name_can_be_disambiguated_by_path() -> None:
    labels = ["/World/envs/env_0/A/tool", "/World/envs/env_0/B/tool"]

    assert resolve_drive_body_index(labels, "/World/envs/env_0/B/tool", 2) == 1


def test_a_body_outside_the_first_env_does_not_resolve() -> None:
    """Guards the slice: a name only present later must not be found."""
    labels = PANDA_LABELS + ["/World/envs/env_0/Cart/handle"]

    with pytest.raises(ValueError, match="no body named 'handle'"):
        resolve_drive_body_index(labels, "handle", 11)


@pytest.mark.parametrize("bodies_per_env", [0, -1])
def test_a_nonpositive_env_body_count_is_refused(bodies_per_env) -> None:
    with pytest.raises(ValueError, match="bodies_per_env must be positive"):
        resolve_drive_body_index(PANDA_LABELS, "TCP", bodies_per_env)


# ---------------------------------------------------------------- harvesting


def test_the_reaction_lands_on_the_driving_body() -> None:
    """And on no other body: an untouched body must stay unloaded."""
    coupler = make_coupler()
    rod = FakeRod([[3.0, 0.0, 0.0]])
    # Drive point sits at the driving body's origin, so there is no moment arm.
    state = FakeState(
        2,
        particle_q=[[1.0, 0.0, 0.0]] * 4,
        body_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    coupler.harvest(state, rod)

    np.testing.assert_allclose(wrench(coupler, 1), [3.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(wrench(coupler, 0), np.zeros(6), atol=1e-6)


def test_an_offset_mount_also_produces_a_moment() -> None:
    """A catheter gripped away from the body's origin twists it as well as pushes."""
    # Mount 2 m along +x in the body frame, force along +y.
    coupler = make_coupler(mount_local=[(2.0, 0.0, 0.0)])
    rod = FakeRod([[0.0, 5.0, 0.0]])
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, rod)

    # tau = r x f = (2,0,0) x (0,5,0) = (0,0,10)
    np.testing.assert_allclose(wrench(coupler, 1), [0.0, 5.0, 0.0, 0.0, 0.0, 10.0], atol=1e-6)


def test_the_default_mount_makes_the_reaction_a_pure_force() -> None:
    """Wire through the flange's own axis exerts no moment about it.

    The conservative default: under-state the moment rather than invent one.
    """
    coupler = make_coupler()
    rod = FakeRod([[0.0, 5.0, 0.0]])
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, rod)

    np.testing.assert_allclose(wrench(coupler, 1), [0.0, 5.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_the_moment_does_not_depend_on_how_much_wire_has_been_fed() -> None:
    """The bug this replaces, and the one worth a test of its own.

    The moment arm used to be the rod's proximal particle minus the body
    origin. Under a roller drive that particle travels away up the vessel as
    wire is fed, so the arm grew without bound and the torque scaled with
    insertion depth -- which drifted the flange 34 mm off the access site in two
    seconds on the s0011 twin. The wire is gripped by hardware bolted to the
    body, so feeding wire cannot change where the reaction is transmitted.
    """
    coupler = make_coupler(mount_local=[(0.01, 0.0, 0.0)])
    rod = FakeRod([[0.0, 5.0, 0.0]])

    coupler.harvest(FakeState(2, particle_q=origins(4), body_positions=origins(2)), rod)
    unfed = wrench(coupler, 1).copy()

    # Same grip, same force, but 300 mm of wire has since been fed up the vessel.
    coupler.harvest(FakeState(2, particle_q=[[0.3, 0.0, 0.0]] * 4, body_positions=origins(2)), rod)

    np.testing.assert_allclose(wrench(coupler, 1), unfed, atol=1e-6)


def test_the_mount_rotates_with_the_body() -> None:
    """The mount is fixed in the body's frame, so its world arm turns with it.

    A mount treated as a world offset would keep twisting about a stale axis
    once the arm reorients.
    """
    coupler = make_coupler(mount_local=[(2.0, 0.0, 0.0)])
    rod = FakeRod([[0.0, 5.0, 0.0]])
    # Body yawed a quarter turn about +z, so local +x points along world +y.
    quarter_turn = FakeState(
        2,
        particle_q=origins(4),
        body_positions=origins(2),
        body_rotations=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))],
    )

    coupler.harvest(quarter_turn, rod)

    # arm becomes (0,2,0), so tau = (0,2,0) x (0,5,0) = 0.
    np.testing.assert_allclose(wrench(coupler, 1)[3:], [0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    ("mount", "message"),
    [
        ([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], "mount_local has 2 entries"),
        ([(0.0, 0.0)], "must be a 3-vector"),
    ],
)
def test_nonsense_mount_geometry_is_rejected(mount, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_coupler(mount_local=mount)


def test_rods_sharing_a_body_accumulate() -> None:
    """Two envs held by the same body must sum, not overwrite each other."""
    coupler = make_coupler(body_count=1, num_envs=2, drive_bodies=(0, 0))
    rod = FakeRod([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = FakeState(1, particle_q=origins(8), body_positions=origins(1))

    coupler.harvest(state, rod)

    np.testing.assert_allclose(wrench(coupler, 0)[:3], [3.0, 0.0, 0.0], atol=1e-6)


def test_each_env_reads_its_own_mount() -> None:
    """Per-env mounts must not be shared, or env 1 twists about env 0's grip."""
    coupler = make_coupler(
        body_count=2,
        num_envs=2,
        drive_bodies=(0, 1),
        root_stride=4,
        mount_local=[(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    rod = FakeRod([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    state = FakeState(2, particle_q=origins(8), body_positions=origins(2))

    coupler.harvest(state, rod)

    # Env 0 grips on the body origin, so no moment; env 1 grips 3 m out.
    np.testing.assert_allclose(wrench(coupler, 0)[3:], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(wrench(coupler, 1)[3:], [0.0, 0.0, 3.0], atol=1e-6)


def test_a_negative_drive_body_disables_the_feedback() -> None:
    """The per-env opt-out upstream's proxy maps use, kept here."""
    coupler = make_coupler(drive_bodies=(-1,))
    rod = FakeRod([[9.0, 9.0, 9.0]])
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, rod)

    np.testing.assert_allclose(coupler.coupling_forces.numpy(), np.zeros((2, 6)), atol=1e-6)


def test_a_harvest_replaces_the_previous_one() -> None:
    """Feedback is a per-substep quantity, so it must not pile up across steps."""
    coupler = make_coupler()
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, FakeRod([[1.0, 0.0, 0.0]]))
    coupler.harvest(state, FakeRod([[1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(wrench(coupler, 1)[:3], [1.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------- relaxation blend


def test_under_relaxation_blends_toward_the_previous_iterate() -> None:
    """Halving the step is the standard cure for a ringing lagged fixed point."""
    coupler = make_coupler(relaxation=0.5)
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, FakeRod([[4.0, 0.0, 0.0]]))
    np.testing.assert_allclose(wrench(coupler, 1)[0], 2.0, atol=1e-6)

    # Second pass blends 4 against the stored 2.
    coupler.harvest(state, FakeRod([[4.0, 0.0, 0.0]]))
    np.testing.assert_allclose(wrench(coupler, 1)[0], 3.0, atol=1e-6)


def test_full_relaxation_takes_the_harvest_unchanged() -> None:
    coupler = make_coupler(relaxation=1.0)
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))

    coupler.harvest(state, FakeRod([[4.0, 0.0, 0.0]]))

    np.testing.assert_allclose(wrench(coupler, 1)[0], 4.0, atol=1e-6)


# -------------------------------------------------------------- application


def test_apply_adds_onto_body_f_rather_than_overwriting() -> None:
    """Other force sources write the same buffer; clobbering them would be a bug."""
    coupler = make_coupler()
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))
    coupler.harvest(state, FakeRod([[2.0, 0.0, 0.0]]))

    existing = np.zeros((2, 6), dtype=np.float32)
    existing[1, 0] = 10.0
    state.body_f.assign(wp.array(existing, dtype=wp.spatial_vector, device=DEVICE))

    coupler.apply(state)

    np.testing.assert_allclose(state.body_f.numpy()[1][0], 12.0, atol=1e-6)


def test_reset_drops_the_feedback_history() -> None:
    """A reset env restarts at rest, so last episode's load must not follow it."""
    coupler = make_coupler()
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))
    coupler.harvest(state, FakeRod([[7.0, 0.0, 0.0]]))

    coupler.reset()

    np.testing.assert_allclose(coupler.coupling_forces.numpy(), np.zeros((2, 6)), atol=1e-6)


# ------------------------------------------------------------- construction


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"relaxation": 0.0}, ValueError),
        ({"drive_bodies": (0, 0)}, ValueError),
        ({"drive_bodies": (5,)}, IndexError),
        ({"root_stride": 0}, ValueError),
        ({"body_count": 0}, ValueError),
    ],
)
def test_bad_configuration_is_rejected_at_construction(kwargs, exception) -> None:
    """Not on the first substep, where it reads as a coupled-solver failure."""
    with pytest.raises(exception):
        make_coupler(**kwargs)


# ------------------------------------------- the reaction against real physics

NUM_SEGMENTS = 24
SEGMENT_LENGTH = 0.01


@pytest.fixture(scope="module")
def hanging_rod():
    """A rod clamped at its proximal end and left to hang under gravity.

    Gravity is the one load whose transmitted force is known in closed form, so
    this is the fixture that checks the multiplier-to-force conversion rather
    than just its plumbing.
    """
    from catheter_vasculature_solver import CathRodSolver, RodConfig

    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    solver = CathRodSolver(
        cfg,
        num_envs=1,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.6,
        tip_num_edges=6,
        particle_radius=0.001,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=False,
        track_enabled=False,
    )
    return solver, cfg


@pytest.fixture(scope="module")
def settled_rod(hanging_rod):
    solver, cfg = hanging_rod
    for _ in range(200):
        solver.step(1.0 / 240.0)
    return solver, cfg


def test_the_accessor_refuses_before_a_step(hanging_rod) -> None:
    """The multipliers are zeroed per substep, so they mean nothing until one runs.

    Returning zeros instead would be indistinguishable from a catheter under no
    load, which is the failure this whole module is trying to avoid.
    """
    from catheter_vasculature_solver import CathRodSolver, RodConfig

    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    fresh = CathRodSolver(
        cfg,
        num_envs=1,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.6,
        tip_num_edges=6,
        particle_radius=0.001,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=False,
        track_enabled=False,
    )

    with pytest.raises(RuntimeError, match="needs a completed step"):
        fresh.proximal_reaction()


def test_a_hanging_rod_loads_its_holder_with_its_own_weight(settled_rod) -> None:
    """The closed-form check on the multiplier-to-force conversion.

    A rod clamped at one end and hanging in gravity transmits exactly its own
    weight to the clamp. The measured force comes out 1% light, which is the
    solver's ``linear_damping`` of 0.01 bleeding that fraction of velocity per
    substep, so the tolerance is set just wide enough to admit it.
    """
    solver, cfg = settled_rod

    inv_mass = solver._ws.inv_masses.numpy()
    moving = inv_mass[inv_mass > 0.0]
    rod_mass = float(np.sum(1.0 / moving))
    weight = rod_mass * abs(cfg.solver.gravity[1])

    measured = float(np.linalg.norm(solver.proximal_reaction().numpy()))

    assert measured == pytest.approx(weight, rel=0.02)
    # And specifically light rather than heavy, for the reason above.
    assert measured < weight


def test_the_reaction_pulls_in_the_direction_of_gravity(settled_rod) -> None:
    """A hanging catheter drags its holder down, it does not push it up.

    This pins the sign. Feeding back the negation would drive the holding arm
    away from the load instead of into it, which is the kind of error that only
    shows up as an unstable simulation much later.
    """
    solver, cfg = settled_rod

    force = solver.proximal_reaction().numpy()[0]
    gravity = np.asarray(cfg.solver.gravity, dtype=np.float64)

    assert float(np.dot(force, gravity)) > 0.0


def test_a_wrong_length_out_array_is_rejected(settled_rod) -> None:
    solver, _ = settled_rod

    with pytest.raises(ValueError, match="must hold 1 entries"):
        solver.proximal_reaction(out=wp.zeros(5, dtype=wp.vec3, device=DEVICE))


# ------------------------------------------------ why not a momentum harvest


def test_the_proximal_particle_has_no_mass_to_harvest(settled_rod) -> None:
    """The reason this module reads constraints instead of momentum.

    The clamped root is given zero inverse mass, i.e. infinite mass, so
    ``m * dv / dt`` -- the harvest Newton's upstream proxy coupling uses -- can
    never see anything there however hard the catheter pulls. If this ever
    becomes finite, a momentum harvest opens up and the specialisation in
    ``proxy_coupling`` is worth revisiting.
    """
    solver, _ = settled_rod

    assert solver._ws.inv_masses.numpy()[0] == 0.0


def test_the_rod_solver_never_writes_body_state() -> None:
    """The other half of why upstream's body-momentum harvest cannot work here.

    ``NewtonCathRodSolver.step`` discards contacts and touches no ``body_*``
    array, so ``body_qd`` is bit-identical either side of the rod solve.

    Read as text rather than imported, because importing it pulls in Newton and
    this check is about source content, not runtime behaviour.
    """
    import catheter_vasculature_solver

    source = (
        Path(catheter_vasculature_solver.__file__).parent / "newton_rod_solver.py"
    ).read_text()
    step = source.split("    def step(", 1)[1].split("    def ", 1)[0]

    assert "del control, contacts" in step
    assert "body_" not in step


def test_intrinsic_couple_reaches_the_arm_with_zero_force():
    coupler = make_coupler()
    rod = FakeRod([[0.0, 0.0, 0.0]], moments=[[0.001, -0.002, 0.003]])
    state = FakeState(2, particle_q=origins(4), body_positions=origins(2))
    coupler.harvest(state, rod)
    coupler.apply(state)
    np.testing.assert_allclose(state.body_f.numpy()[1], [0, 0, 0, 0.001, -0.002, 0.003], atol=1e-9)


def test_mount_moment_is_measured_from_body_center_of_mass():
    coupler = DriveReactionCoupler(
        body_count=1, num_envs=1, root_offset=0, root_stride=4, drive_bodies=[0],
        mount_local=[(0.0, 0.0, 0.10)],
        body_com=wp.array([[0.0, 0.0, 0.02]], dtype=wp.vec3, device=DEVICE), device=DEVICE,
    )
    rod = FakeRod([[1.0, 0.0, 0.0]], moments=[[0.0, 0.01, 0.0]])
    state = FakeState(1, particle_q=origins(4), body_positions=origins(1))
    coupler.harvest(state, rod)
    np.testing.assert_allclose(wrench(coupler, 0), [1, 0, 0, 0, 0.09, 0], atol=1e-8)
