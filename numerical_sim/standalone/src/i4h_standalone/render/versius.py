"""Task labels and camera presets using the shared GL viewer."""
import numpy as np


class Viewer:
    """One Newton GL viewer for the native robot and explicitly published i4h mesh."""

    def __init__(self, env, args):
        import warp as wp
        from i4h_standalone.render.gl import GlRenderer

        self.env, self.wp, self.args, self.frame = env, wp, args, 0
        self.renderer = GlRenderer(
            env.i4h.sim, native_model=env.i4h.sim.model,
            native_state=lambda: env.i4h.sim.state,
            component_pose=env.physical.pose, headless=args.headless, show_instrument_spheres=False,
            camera={"position": (1.6, -1.8, 1.4), "pitch": -20, "yaw": 125})
        try:
            self.markers = self.renderer.add_points(
                "versius/targets", 3, [env.physical.case["instruments"]["probe"]["radius"], 0.002, 0.0015],
                [(0.2, 0.8, 1.0), (0.1, 1.0, 0.1), (1.0, 0.7, 0.1)])
            self.viewer = self.renderer.viewer
            if not args.headless and not self.viewer.gui.is_available:
                raise RuntimeError("Task UI requires imgui-bundle in the active environment")
            from i4h_isaaclab.versius.status import StatusPanel

            self.status_panel = StatusPanel()
            self.status_panel.focus_tip = self.focus_tip
            self.status_panel.overview = lambda: self.viewer.set_camera(
                wp.vec3(1.6, -1.8, 1.4), -20.0, 125.0
            )
            self.viewer.register_ui_callback(self.status_panel.draw, position="free")
            if args.record:
                # Draw once before requesting a framebuffer for the encoder dimensions.
                self.renderer.publish()
                self.renderer.draw()
                self.renderer.record_video(args.record)

        except BaseException:
            self.renderer.close()
            raise

    def focus_tip(self):
        import math

        target = (
            self.env.world_points()[0, self.env.patch_ids]
            .mean(0)
            .detach()
            .cpu()
            .numpy()
        )
        eye = target + np.array((0.22, -0.30, 0.13))
        delta = target - eye
        self.viewer.set_camera(
            self.wp.vec3(*map(float, eye)),
            math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))),
            math.degrees(math.atan2(delta[1], delta[0])),
        )

    def draw(self):
        import torch

        self.wp.to_torch(self.markers)[:] = torch.stack((
            self.env.tcp_position()[0], self.env.reference_patch[0],
            self.env.world_points()[0, self.env.patch_ids].mean(0)))
        # UI diagnostics are read at 6 Hz; physics failure checks remain per control step.
        if self.frame % 5 == 0:
            from i4h_isaaclab.versius.status import snapshot
            self.status_panel.state = snapshot(self.env)
        self.renderer.publish()
        self.renderer.draw()
        self.frame += 1

    def close(self):
        self.renderer.close()
