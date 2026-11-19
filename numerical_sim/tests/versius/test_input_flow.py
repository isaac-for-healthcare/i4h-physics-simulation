import json

import numpy as np
from i4h_standalone.input.cartesian import (
    KeyboardInput,
    RelativePositionControl,
    ReplayInput,
    TeleopKeys,
)


def test_keyboard_clutch_focus_recovery_and_reset():
    keys = TeleopKeys()
    source = KeyboardInput(keys.held, 0.04)
    control = RelativePositionControl()
    target = np.zeros(3)
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)

    def tick():
        return control.target(
            source.sample(1 / 30), target, 1 / 30, lower, upper, clutch=keys.clutched
        )

    tick()
    keys.on_key("KeyW", True)
    target = tick()
    assert target[0] > 0
    keys.on_key("ControlLeft", True)
    np.testing.assert_array_equal(tick(), target)
    keys.focus_lost()
    np.testing.assert_array_equal(tick(), target)
    assert not keys.held
    keys.on_key("Space", True)
    np.testing.assert_array_equal(tick(), target)
    keys.on_key("KeyR", True)
    assert keys.reset
    target[:] = 0
    control.recenter()
    np.testing.assert_array_equal(tick(), target)


def test_replay_invalid_sample_reanchors_without_jump(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            [
                {"position": [0, 0, 0]},
                {"position": [0.01, 0, 0]},
                {"position": None},
                {"position": [10, 0, 0]},
                {"position": [10.01, 0, 0]},
            ]
        )
    )
    source = ReplayInput(path)
    control = RelativePositionControl()
    target = np.zeros(3)

    def tick():
        return control.target(source.sample(0.1), target, 0.1, [-1] * 3, [1] * 3)

    target = tick()
    target = tick()
    assert target[0] == 0.004
    np.testing.assert_array_equal(tick(), target)
    np.testing.assert_array_equal(tick(), target)
    assert tick()[0] > target[0]
