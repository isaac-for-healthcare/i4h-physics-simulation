"""Exercise lost tracking, clutch transitions and HDAPI ownership without hardware."""

import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest
from i4h_versius.teleop_input import (
    FollouPositionInput,
    HDErrorInfo,
    OpenHapticsInput,
    PositionSample,
    RelativePositionControl,
    TeleopKeys,
    axis_mapping,
)


def test_relative_axes_rate_limit_and_workspace():
    controller = RelativePositionControl(gain=2, axes="-x,-z,y", speed=0.04)
    lower, upper = np.full(3, -0.01), np.full(3, 0.01)
    origin = np.zeros(3)
    np.testing.assert_array_equal(
        controller.target(PositionSample([4, 5, 6]), origin, 0.1, lower, upper), origin
    )
    target = controller.target(PositionSample([4.01, 5, 6]), origin, 0.1, lower, upper)
    np.testing.assert_allclose(target, [-0.004, 0, 0])
    for _ in range(20):
        target = controller.target(
            PositionSample([4.01, 5, 6]), target, 0.1, lower, upper
        )
    np.testing.assert_allclose(target, [-0.01, 0, 0])


@pytest.mark.parametrize(
    "interruption", ["clutch", "pause", "lost", "nan", "bad_shape", "recenter"]
)
def test_interruptions_discard_pending_motion_and_reanchor(interruption):
    controller = RelativePositionControl()
    lower, upper, target = np.full(3, -1), np.full(3, 1), np.zeros(3)
    controller.target(PositionSample([0, 0, 0]), target, 0.1, lower, upper)
    target = controller.target(PositionSample([1, 0, 0]), target, 0.1, lower, upper)
    # A large pending displacement must not continue through tracking loss or clutch.
    if interruption == "recenter":
        controller.recenter()
    else:
        position = {"lost": None, "nan": [np.nan, 0, 0], "bad_shape": [1]}.get(
            interruption, [5, 0, 0]
        )
        held = controller.target(
            PositionSample(position, clutch=interruption == "clutch"),
            target,
            0.1,
            lower,
            upper,
            clutch=interruption == "pause",
        )
        np.testing.assert_array_equal(held, target)
    recovered = controller.target(PositionSample([10, 0, 0]), target, 0.1, lower, upper)
    np.testing.assert_array_equal(recovered, target)
    moved = controller.target(
        PositionSample([10.001, 0, 0]), recovered, 0.1, lower, upper
    )
    np.testing.assert_allclose(moved, recovered + [0.001, 0, 0])


def test_keys_repeat_focus_and_clutch_release():
    keys = TeleopKeys()
    keys.on_key("Space", True)
    keys.on_key("Space", True)
    assert keys.paused
    keys.on_key("Space", False)
    keys.on_key("Space", True)
    assert not keys.paused
    keys.on_key("ControlLeft", True)
    assert keys.clutched
    keys.on_key("ControlLeft", False)
    assert not keys.clutched
    keys.focus_lost()
    assert keys.clutched and keys.recenter and not keys.held


@pytest.mark.parametrize("spec", ["x,x,z", "x,y", "--x,y,z", "x,y,w"])
def test_invalid_axis_map(spec):
    with pytest.raises(ValueError):
        axis_mapping(spec)


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


def fake_hd():
    log = []
    state = SimpleNamespace(calibration=0x5000)

    def get_position(code, values):
        assert code == OpenHapticsInput.CURRENT_POSITION
        values[:] = [10, 20, 30]

    def get_buttons(code, value):
        assert code == OpenHapticsInput.CURRENT_BUTTONS
        ct.cast(value, ct.POINTER(ct.c_int))[0] = 1

    library = SimpleNamespace()
    for symbol in (
        "hdDisableDevice",
        "hdMakeCurrentDevice",
        "hdDisable",
        "hdStartScheduler",
        "hdStopScheduler",
        "hdBeginFrame",
        "hdEndFrame",
    ):
        setattr(
            library, symbol, Function(lambda *args, s=symbol: log.append((s, args)))
        )
    library.hdInitDevice = Function(lambda name: 7)
    library.hdGetDoublev = Function(get_position)
    library.hdGetIntegerv = Function(get_buttons)
    library.hdGetError = Function(lambda: HDErrorInfo(0, 0, 7))
    library.hdCheckCalibration = Function(lambda: state.calibration)
    library.hdScheduleSynchronous = Function(
        lambda callback, data, priority: callback(data)
    )
    return library, log, state


def test_openhaptics_read_units_buttons_force_disabled_and_cleanup(monkeypatch):
    library, log, state = fake_hd()
    monkeypatch.setattr(ct, "CDLL", lambda name: library)
    device = OpenHapticsInput(library="fake")
    sample = device.sample(1 / 30)
    np.testing.assert_allclose(sample.position, [0.01, 0.02, 0.03])
    assert sample.clutch
    assert ("hdDisable", (OpenHapticsInput.FORCE_OUTPUT,)) in log
    assert log.index(("hdBeginFrame", (7,))) < log.index(("hdEndFrame", (7,)))
    state.calibration = 0
    assert device.sample(1 / 30).position is None
    device.close()
    device.close()
    assert [entry[0] for entry in log].count("hdStopScheduler") == 1
    assert [entry[0] for entry in log].count("hdDisableDevice") == 1
    assert device.sample(1 / 30).position is None


def test_openhaptics_failed_start_releases_device(monkeypatch):
    library, log, _ = fake_hd()
    calls = iter([0, 0, 0x0302])
    library.hdGetError = Function(lambda: HDErrorInfo(next(calls), 0, 7))
    monkeypatch.setattr(ct, "CDLL", lambda name: library)
    with pytest.raises(RuntimeError, match="start scheduler"):
        OpenHapticsInput(library="fake")
    assert ("hdStopScheduler", ()) in log
    assert ("hdDisableDevice", (7,)) in log


def test_openhaptics_missing_runtime_is_actionable(monkeypatch):
    def unavailable(name):
        raise OSError("not found")

    monkeypatch.setattr(ct, "CDLL", unavailable)
    with pytest.raises(RuntimeError, match="--hd-library"):
        OpenHapticsInput(library="missing")


def test_follou_adapter_keeps_raw_axes_for_common_mapping(monkeypatch):
    from i4h_versius.follou_input import FollouInput

    source = SimpleNamespace(
        raw_position=np.array([100, 200, 300]), failed=False, closed=False
    )

    def read():
        if source.failed:
            raise RuntimeError("disconnected")
        return np.array([-0.1, -0.3, 0.2])

    def close():
        source.closed = True

    source.read_position, source.close = read, close
    monkeypatch.setattr(FollouInput, "open", lambda **kwargs: source)
    device = FollouPositionInput()
    sample = device.sample(1 / 30)
    np.testing.assert_allclose(sample.position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        axis_mapping("-x,-z,y") @ sample.position, [-0.1, -0.3, 0.2]
    )
    source.failed = True
    assert device.sample(1 / 30).position is None
    device.close()
    assert source.closed
