from types import SimpleNamespace as NS

import torch
from i4h_isaaclab.versius import status
from test_terms import fixture


def diagnostic_env():
    env = fixture()
    env.physical = NS(name="block")
    env.graph_seen = False
    env.i4h.state = lambda: {
        "tool_force.force": torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        "tool_force.count": torch.tensor([[2], [0]]),
    }
    return env


def test_snapshot_is_read_only_and_explains_no_contact_then_goal():
    env = diagnostic_env()
    env.i4h.state = lambda: {
        "tool_force.force": torch.zeros(2, 3),
        "tool_force.count": torch.zeros(2, 1),
    }
    before = env.hold_time.clone()
    state = status.snapshot(env)
    assert "No contact" in status.guidance(state)
    assert torch.equal(env.hold_time, before)
    env.i4h.state = lambda: {
        "tool_force.force": torch.ones(2, 3),
        "tool_force.count": torch.ones(2, 1),
    }
    assert "More measured indentation" in status.guidance(status.snapshot(env))
    env.points[:, :, 2] = -0.003
    assert "maintain contact" in status.guidance(status.snapshot(env))
    env.success[:] = True
    assert "achieved" in status.guidance(status.snapshot(env))
    assert torch.equal(env.hold_time, before)


def test_fault_diagnostics_identify_thresholds_and_nonfinite_state():
    env = diagnostic_env()
    env.points[0, :, 2] = -0.013
    env.tcp[1, 2] = -0.019
    state = status.snapshot(env)
    assert state["failed"]
    assert "env 0: tissue indentation limit exceeded" in state["reasons"]
    assert "env 1: tool tip below allowed depth" in state["reasons"]
    env.points[1, 0, 0] = float("nan")
    assert "env 1: nonfinite tissue/robot state" in status.snapshot(env)["reasons"]
    assert "Reset required" in status.guidance(status.snapshot(env))


def test_paused_physics_reports_reason_once_and_stops_stepping(
    tmp_path, monkeypatch, capsys
):
    from i4h_standalone.cartesian import teleop

    replay = tmp_path / "input.json"
    replay.write_text('[{"position":[0,0,0]}]')

    def unexpected_step(action):
        raise AssertionError("Invalid physics must stay frozen")

    env = NS(
        tcp_position=lambda: torch.zeros(1, 3),
        step_dt=1 / 30,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        step=unexpected_step,
    )
    args = NS(
        viewer=False,
        input="replay",
        replay=replay,
        follou_device_index=0,
        hd_device="Default Device",
        hd_library=None,
        axis_map=None,
        gain=1.0,
        steps=4,
        no_real_time=True,
    )
    monkeypatch.setattr(
        status,
        "failure_status",
        lambda env: {
            "failed": True,
            "reasons": ["env 0: tissue indentation limit exceeded"],
        },
    )
    result = teleop(env, args)
    output = capsys.readouterr().out
    assert output.count("Physics paused; reset required:") == 1
    assert "tissue indentation limit exceeded" in output
    assert result["paused_invalid_physics"]
    assert result["episode_steps"] == [0]


def test_invalid_control_configuration_closes_viewer(monkeypatch):
    import pytest
    from i4h_standalone import cartesian

    closed = []
    viewer = NS(status_panel=NS(), close=lambda: closed.append(True))
    monkeypatch.setattr(cartesian, 'Viewer', lambda *a: viewer)
    args = NS(viewer=True, input='replay', axis_map='x,x,z', gain=1.0)
    with pytest.raises(ValueError, match='axis map'):
        cartesian.teleop(NS(), args)
    assert closed == [True]
