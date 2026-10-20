# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M15 X5: XR in the Isaac Lab teleop loop. The real XrRenderer on a render thread beside the env stepping (XrView),
with a fake OpenXR runtime whose swapchain images are real textures (xr_fakes.install_fake_xr): the controller drives
env 0's instrument, both eyes get the scene. Isaac Lab venv with the xr packages (the slangpy fork, pyopenxr)."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
spy = pytest.importorskip("slangpy")
if not hasattr(spy.Device, "create_texture_from_native_handle"):
    pytest.skip("needs the xr packages (the slangpy fork) in the Isaac Lab venv", allow_module_level=True)

from helpers import haptic_case  # noqa: E402
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from i4h_standalone.teleop import Teleop, XrView  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from xr_fakes import install_fake_xr  # noqa: E402


@pytest.mark.parametrize("backend", ["newton", "i4h"])
def test_xr_teleop_render_thread_and_controller(monkeypatch, tmp_path, backend):
    from i4h_standalone.render.xr import XrRenderer
    from i4h_standalone.xr import hands

    xr, owned, holder = install_fake_xr(monkeypatch, tmp_path, spy)
    xr.auto_ready = True
    path = haptic_case(tmp_path, envs=2)
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(case=str(path), action_mode="physical", stagger_episodes=False, backend=backend)
    cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    env = I4hEnv(cfg)
    u = env.unwrapped
    render = {k: v for k, v in (read_case(str(path)).get("render") or {}).items() if k not in ("backend", "size")}
    renderer = XrRenderer(u.simulator, xr={"world_units_per_meter": 1.0, "runtime": {"mode": "external"}, "mirror": "off"}, **render)
    holder.session = renderer.session
    view = XrView(env, renderer)
    teleop = Teleop(env, {"pad": {"backend": "openxr", "settings": {"hand": "right"}}}, renderer=renderer)
    try:
        teleop.reset()
        view.start()
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline and renderer.frames < 5:
            teleop.step()
            view.check()
            view.publish()
            time.sleep(0.01)
        for _ in range(3):
            teleop.step()
            view.publish()
        view.check()
        assert renderer.frames >= 5 and view.running
        want, _ = renderer.anchor.transform((0.2, -0.3, -0.4), (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(teleop.rows[0, :3], want, atol=1e-5)  # the right controller holds the probe
        torch.cuda.synchronize()
        np.testing.assert_allclose(u.action_views[0][0, :3].cpu().numpy(), want, atol=1e-5)  # and env 0 was stepped with it
    finally:
        teleop.close()
        view.close()
        env.close()
        SimulationContext.clear_instance()
    assert not view.running and view.error is None
    assert not hands.read("right")["active"]
    assert xr.events.count(("end_frame", 1)) >= 5  # both eyes submitted every frame
