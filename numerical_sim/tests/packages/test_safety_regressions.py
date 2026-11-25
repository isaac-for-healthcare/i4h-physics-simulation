# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded safety regressions using generated assets and fake devices."""


import numpy as np
import pytest


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), -1, 0])
def test_nonfinite_timesteps_rejected(dt):
    from i4h_physics.config import check_dt, sim_settings
    with pytest.raises(ValueError):
        check_dt(dt, .01)
    with pytest.raises(ValueError):
        sim_settings({"sim": {"dt": dt}})


def test_euler_conventions_remain_distinct():
    from i4h_physics.mesh_utils import euler_rx_ry_rz_degrees_matrix
    from i4h_physics.quat import euler_rz_ry_rx_degrees_quat, quat_to_matrix
    angle = [20, 30, 40]
    # Reversing multiplication order preserves each historical YAML interpretation.
    x = euler_rx_ry_rz_degrees_matrix(angle)
    y = quat_to_matrix(euler_rz_ry_rx_degrees_quat(angle))
    assert not np.allclose(x, y)
    np.testing.assert_allclose(x @ x.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(y @ y.T, np.eye(3), atol=1e-6)
