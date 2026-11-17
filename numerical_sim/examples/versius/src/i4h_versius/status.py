"""Read-only status snapshots and an always-visible Newton ImGui task panel."""

from . import terms


def snapshot(env):
    patch, tcp, depth, lateral, distance, failed, valid = terms.measurements(env)
    flags = terms.failure_conditions(env, env.world_points(), tcp, depth)
    products = env.i4h.state()
    force = products["tool_force.force"].reshape(env.num_envs, 3)
    contacts = products["tool_force.count"].reshape(env.num_envs, -1).sum(1)
    reasons = []
    names = {
        "nonfinite": "nonfinite tissue/robot state",
        "indentation_limit": "tissue indentation limit exceeded",
        "tip_depth_limit": "tool tip below allowed depth",
    }
    for name, mask in flags.items():
        for world in mask.nonzero().flatten().tolist():
            reasons.append(f"env {world}: {names[name]}")
    return {
        "profile": env.physical.name,
        "geometry_scale": getattr(env.physical, "geometry_scale", 1.0),
        "envs": env.num_envs,
        "depth": float(depth[0]),
        "depth_low": env.cfg.target_depth - env.cfg.depth_tolerance,
        "depth_high": env.cfg.target_depth + env.cfg.depth_tolerance,
        "depth_limit": env.cfg.max_depth,
        "tip_depth_limit": env.cfg.max_tool_depth,
        "tip_depth": float(env.reference_patch[0, 2] - tcp[0, 2]),
        "lateral": float(lateral[0]),
        "distance": float(distance[0]),
        "contacts": int(contacts[0]),
        "force": force[0].tolist(),
        "force_norm": float(force[0].norm()),
        "hold": float(env.hold_time[0]),
        "success": bool(env.success[0]),
        "valid": bool(valid[0]),
        "failed": bool(failed.any()),
        "reasons": reasons,
        "tcp": tcp[0].tolist(),
        "tracking_error": float((env.target_tcp[0] - tcp[0]).norm()),
        "steps": int(env.episode_length_buf[0]),
        "graph": bool(env.graph_seen),
        "patch": patch[0].tolist(),
    }


def guidance(state):
    if state["failed"]:
        return "Reset required: " + "; ".join(state["reasons"])
    if state["contacts"] == 0:
        return "No contact: move the cyan proxy toward the gold live patch."
    if state["depth"] < state["depth_low"]:
        return "Contact detected. More measured indentation is needed."
    if state["depth"] > state["depth_high"]:
        return "Too much indentation for the goal; withdraw slightly."
    if state["lateral"] >= 0.006:
        return "Indentation is in range; align laterally with the patch."
    if state["distance"] >= 0.012:
        return "Indentation is in range; the actual TCP is still too far away."
    if state["success"]:
        return "Poke-and-hold achieved. Teleoperation continues."
    return "All conditions met: maintain contact for 0.5 s."


class StatusPanel:
    def __init__(self):
        self.state = None
        self.keys = None
        self.input_status = "scripted control"
        self.physics_paused = False
        self.focus_tip = None
        self.overview = None
        self._first_frame = True

    def draw(self, imgui):
        state = self.state
        if state is None:
            return
        width = 420.0
        io = imgui.get_io()
        imgui.set_next_window_pos(
            imgui.ImVec2(max(10.0, io.display_size.x - width - 12.0), 12.0),
            imgui.Cond_.appearing,
        )
        imgui.set_next_window_size(imgui.ImVec2(width, 0.0), imgui.Cond_.appearing)
        if self._first_frame:
            imgui.set_next_window_focus()
            self._first_frame = False
        opened = imgui.begin("Versius | Poke status")
        try:
            if not opened:
                return
            imgui.text("Isaac Lab ManagerBasedRLEnv")
            imgui.text("MuJoCo-Warp arm + I4hSolver tissue")
            imgui.text(
                f"{state['profile']} | env 0 / {state['envs']} | step {state['steps']}"
            )
            mode = "CUDA graphs" if state["graph"] else "eager"
            imgui.text(f"Backend: {mode}")
            if state.get("geometry_scale", 1.0) != 1.0:
                imgui.text_disabled("1/7 reference geometry; mechanics uncalibrated")
            if self.physics_paused or state["failed"]:
                status = "PHYSICS PAUSED - RESET REQUIRED"
                color = (1.0, 0.35, 0.25, 1.0)
            elif self.keys is not None and self.keys.clutched:
                status = "INPUT PAUSED / CLUTCHED (target held)"
                color = (1.0, 0.8, 0.25, 1.0)
            elif state["success"]:
                status = "GOAL HELD - teleoperation continues"
                color = (0.3, 1.0, 0.5, 1.0)
            else:
                status = "CONTACT" if state["contacts"] else "APPROACH / NO CONTACT"
                color = (
                    (0.3, 1.0, 0.7, 1.0)
                    if state["contacts"]
                    else (0.65, 0.85, 1.0, 1.0)
                )
            imgui.text_colored(imgui.ImVec4(*color), status)
            imgui.text_wrapped(f"Input: {self.input_status}")
            imgui.separator()
            imgui.text(f"Contact count: {state['contacts']}")
            imgui.text(f"Reaction magnitude: {state['force_norm']:.5f}")
            imgui.text("Reaction XYZ: " + ", ".join(f"{x:.4f}" for x in state["force"]))
            imgui.text_disabled(
                "Reaction is in the component frame; no force feedback."
            )
            imgui.separator()
            rows = [
                (
                    "Indentation",
                    state["depth"],
                    f"{state['depth_low']:.4f} to {state['depth_high']:.4f}",
                    state["depth_low"] < state["depth"] < state["depth_high"],
                ),
                (
                    "Lateral error",
                    state["lateral"],
                    "< 0.0060",
                    state["lateral"] < 0.006,
                ),
                (
                    "Actual TCP / patch distance",
                    state["distance"],
                    "< 0.0120",
                    state["distance"] < 0.012,
                ),
            ]
            for label, value, goal, passed in rows:
                imgui.text(f"{'OK' if passed else '--'}  {label}: {value:.5f}")
                imgui.text_disabled(f"     goal: {goal}")
            imgui.progress_bar(
                min(1.0, max(0.0, state["hold"] / 0.5)),
                imgui.ImVec2(-1.0, 0.0),
                f"Hold: {state['hold']:.2f} / 0.50 s",
            )
            imgui.text(f"Target tracking error: {state['tracking_error']:.5f}")
            imgui.text("TCP XYZ: " + ", ".join(f"{x:.4f}" for x in state["tcp"]))
            imgui.text_disabled(
                "Distances use simulation units, not calibrated metres."
            )
            imgui.separator()
            imgui.text_wrapped(guidance(state))
            imgui.text_disabled(
                f"Failure limits: indentation > {state['depth_limit']:.3f}; tip depth > {state['tip_depth_limit']:.3f}"
            )
            if self.keys is not None:
                if imgui.button("Reset episode (R)"):
                    self.keys.reset = True
                imgui.same_line()
                if imgui.button("Resume input" if self.keys.paused else "Pause input"):
                    self.keys.paused = not self.keys.paused
                    self.keys.recenter = True
                imgui.text_disabled(
                    "Ctrl: clutch | C: recenter | Space: pause | Esc: quit"
                )
            if self.focus_tip is not None and imgui.button("Tip close-up"):
                self.focus_tip()
            imgui.same_line()
            if self.overview is not None and imgui.button("Arm overview"):
                self.overview()
            imgui.text_disabled("Cyan: proxy | gold: live patch | green: baseline")
        finally:
            imgui.end()
