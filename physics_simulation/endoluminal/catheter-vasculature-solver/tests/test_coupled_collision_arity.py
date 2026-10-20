# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the coupled manager calls Newton's collision pipeline.

This is pinned by a test because getting it wrong is both easy and expensive.
``CollisionPipeline`` is built from the model and keeps its own reference, so
passing the model in again looks harmless at the call site but shifts every
argument by one. The result is a ``TypeError`` raised on the first substep --
after scene composition, asset loading and solver construction have all
succeeded -- which reads as a coupled-solver failure rather than a typo.

The pipeline is reached through ``getattr`` on Isaac Lab's ``NewtonManager``
because the attribute is private and only exists once allocated, so these tests
substitute a recorder for it rather than standing up a simulator.
"""

from __future__ import annotations

import inspect

import pytest


class RecordingPipeline:
    """Stands in for ``newton.CollisionPipeline``, recording how it was called.

    Deliberately has no ``step``, so ``_run_collision_pipeline`` takes the
    ``collide`` branch. Lab revisions that expose ``step`` are preferred by the
    manager and are not what this file is about.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def collide(self, state, contacts, *, soft_contact_margin=None):
        self.calls.append((state, contacts, soft_contact_margin))
        return contacts


@pytest.fixture
def manager():
    pytest.importorskip("isaaclab_newton", reason="needs Isaac Lab's Newton bridge")
    from catheter_vasculature_solver.isaaclab_integration.coupled_manager import NewtonCoupledMJWarpXPBDRodManager

    return NewtonCoupledMJWarpXPBDRodManager


@pytest.fixture
def newton_manager(manager, monkeypatch):
    """Give the coupled manager a pipeline and contacts to find."""
    from isaaclab_newton.physics.newton_manager import NewtonManager

    pipeline = RecordingPipeline()
    contacts = object()
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", pipeline, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", contacts, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    return pipeline, contacts


def test_collide_receives_the_state_first(manager, newton_manager) -> None:
    """Not the model. This is the bug that cost a debugging session."""
    pipeline, contacts = newton_manager
    state_0 = object()

    manager._run_collision_pipeline(state_0, object(), object())

    assert len(pipeline.calls) == 1
    passed_state, passed_contacts, _ = pipeline.calls[0]
    assert passed_state is state_0
    assert passed_contacts is contacts


def test_the_recorder_matches_newtons_real_signature() -> None:
    """The recorder is the oracle for the tests above, so it has to be honest.

    Those tests catch a model-first call because ``RecordingPipeline.collide``
    takes two positional parameters and a third raises ``TypeError`` on the
    spot. That only holds while the recorder mirrors Newton, so compare the two
    directly -- this is what fails if Newton itself changes the signature.
    """
    newton = pytest.importorskip("newton", reason="needs newton installed")

    def positional(signature: inspect.Signature) -> list[str]:
        return [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]

    assert positional(inspect.signature(RecordingPipeline.collide)) == positional(
        inspect.signature(newton.CollisionPipeline.collide)
    )


def test_the_recorded_call_binds_to_newton(manager, newton_manager) -> None:
    """Belt and braces: what we passed must be bindable to the real pipeline."""
    newton = pytest.importorskip("newton", reason="needs newton installed")
    pipeline, _ = newton_manager

    manager._run_collision_pipeline(object(), object(), object())
    state, contacts, _ = pipeline.calls[0]

    signature = inspect.signature(newton.CollisionPipeline.collide)
    # ``self`` is supplied by the bound method, so bind the rest positionally.
    signature.bind(object(), state, contacts)


def test_a_pipeline_exposing_step_is_preferred(manager, monkeypatch) -> None:
    """``step`` carries state_1 and control too, so it is the richer path and
    must win when a Lab revision offers both."""
    from isaaclab_newton.physics.newton_manager import NewtonManager

    stepped: list[tuple] = []

    class SteppingPipeline(RecordingPipeline):
        def step(self, state_0, state_1, control, contacts):
            stepped.append((state_0, state_1, control, contacts))
            return contacts

    pipeline = SteppingPipeline()
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", pipeline, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", object(), raising=False)

    manager._run_collision_pipeline(object(), object(), object())

    assert len(stepped) == 1
    assert pipeline.calls == []


def test_no_pipeline_yields_the_existing_contacts(manager, monkeypatch) -> None:
    """Collision detection is optional; its absence is not an error."""
    from isaaclab_newton.physics.newton_manager import NewtonManager

    contacts = object()
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", contacts, raising=False)

    assert manager._run_collision_pipeline(object(), object(), object()) is contacts
