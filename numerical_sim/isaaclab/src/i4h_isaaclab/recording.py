# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop demonstrations in Isaac Lab's HDF5 dataset format, and their replay (import only inside an Isaac Lab process).

``Recorder`` writes env 0 of a ``teleop.Teleop`` session with ``isaaclab.utils.datasets.HDF5DatasetFileHandler``: one
``data/demo_N`` group per episode (``seed``, ``success``, ``num_samples`` attributes), per step

- ``actions``: the physical row env 0 was stepped with; ``actions_normalized``: the same in [-1, 1] (clipped);
- ``obs/policy`` (and ``obs/rgb``, ``obs/depth`` with a camera and ``record_camera``): the observation after the step;
- ``states/<system id>.<product>``: the per-env state products (``views.state_names``);
- ``devices/<id>/{position, rotation, grip, buttons}``: each input device's state (buttons: grab, secondary, fire);
- ``haptics/{force, jaw_load}``: env 0's ``tool_force`` row, when the case has one.

The file's ``env_args`` (``format_version`` 1) carry everything a replay needs to rebuild the env: the case YAML (asset
paths under the data root written as ``${DATA}``, so a replay resolves them against its own ``I4H_DATA``), the backend,
env count, decimation, physics dt, workspace, observation and camera selection, episode horizon and staggering, step
weights, and the action layout (control systems, their column counts and the instruments or rods they drive, in
order). Each episode also stores ``initial_episode_length``, env 0's episode counter when the take began (staggering
gives a fresh env's first episode a random one), which the replay restores after its reset. ``env_cfg_from_recording`` rebuilds the env
config from them; explicit overrides that change the dynamics warn, and a different action layout is rejected. ``replay_episode`` steps a recorded episode's physical actions (``action_mode = "physical"``) from a
reset and returns, per step, the largest ``obs/policy`` difference of env 0 against the recording, relative to
``1 + |recorded|`` (positions and force products share the observation): deterministic cases reproduce within atomics
noise; chaotic ones (chole cutting and grasping) drift, which the curve shows.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

FORMAT_VERSION = 1
RECORDED = ("backend", "num_envs", "decimation", "physics_dt", "workspace", "observe", "camera", "camera_mode",
            "episode_length_s", "stagger_episodes", "step_weights")
DYNAMICS = ("case", "decimation", "physics_dt", "workspace", "episode_length_s")  # overrides that change the trajectory


def unresolve_data_path(value, root: str):
    """The inverse of ``i4h_physics.config.resolve_data_path`` for strings under ``root``: ``${DATA}/...``."""
    if isinstance(value, dict):
        return {k: unresolve_data_path(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [unresolve_data_path(v, root) for v in value]
    if isinstance(value, str) and (value == root or value.startswith(root.rstrip("/") + "/")):
        return "${DATA}" + value[len(root.rstrip("/")):]
    return value


def action_layout(sim) -> list:
    """``[[control system id, columns, [targets]], ...]`` in the order of the action row; the targets are what the
    columns drive, in order (an instrument control's instruments, a catheter control's rod body), so two layouts of the
    same width with the instruments reordered differ."""
    from i4h_isaaclab.views import action_views, controls

    def targets(s):
        if hasattr(s, "names"):
            return [str(n) for n in s.names]
        body = getattr(s, "body", None)
        return [str(body.name)] if body is not None else []

    return [[s.id, int(v.shape[1]), targets(s)] for s, v in zip(controls(sim), action_views(sim))]


class Recorder:
    def __init__(self, path, teleop, record_camera: bool = False):
        from i4h_isaaclab.views import state_names
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        try:  # optional: an endoluminal-only install has no tool force to record
            from i4h_surgical.systems.instruments.tool_force import ToolForce
        except ModuleNotFoundError as error:
            if error.name != "i4h_surgical":
                raise
            ToolForce = ()

        self.teleop, self.u = teleop, teleop.u
        u = self.u
        self.handler = HDF5DatasetFileHandler()
        self.handler.create(str(path), env_name=str(u.cfg.case))
        self.path = str(path) if str(path).endswith(".hdf5") else f"{path}.hdf5"
        from i4h_physics.config import data_dir

        root = str(data_dir())
        cfg = u.cfg
        self.handler.add_env_args({
            "format_version": FORMAT_VERSION, "case": str(cfg.case), "data_root": root,
            "case_yaml": yaml.safe_dump(unresolve_data_path(u._case, root), sort_keys=False),
            "backend": cfg.backend, "num_envs": int(u.num_envs), "decimation": int(cfg.decimation),
            "physics_dt": float(u.physics_dt), "workspace": float(cfg.workspace), "observe": str(cfg.observe),
            "camera": int(cfg.camera), "camera_mode": str(cfg.camera_mode), "episode_length_s": float(cfg.episode_length_s),
            "stagger_episodes": bool(cfg.stagger_episodes), "step_weights": [float(w) for w in cfg.step_weights],
            "action_layout": action_layout(u.simulator), "action_mode": "physical"})
        self.record_camera = bool(record_camera)
        self.state_names = state_names(u.simulator)
        self.tool_force = next((s for s in u.simulator.systems if isinstance(s, ToolForce)), None)
        self.episode = None
        self.written: list[str] = []

    @property
    def recording(self) -> bool:
        return self.episode is not None

    def start(self, reset: bool = True) -> None:
        """Begin an episode (the current one, if any, is discarded) from a reset, so it replays on its own:
        ``teleop.reset()`` resets the env and the teleop rows first. ``reset=False`` only where env 0 has just been
        reset anyway (right after ``teleop.reset()``, or after env 0's episode ended and Isaac Lab reset it)."""
        from isaaclab.utils.datasets import EpisodeData

        if reset:
            self.teleop.reset()
        self.episode = EpisodeData()
        self.episode.seed = int(self.u.cfg.seed)
        self.episode.add("initial_episode_length", self.u.episode_length_buf[0:1].clone())

    def record(self, obs: dict) -> None:
        """One step, after ``teleop.step()``: the action it applied and ``obs``, the observation that step returned."""
        ep = self.episode
        if ep is None:
            return
        u = self.u
        action = self.teleop.physical[0]
        ep.add("actions", action)
        ep.add("actions_normalized", torch.clamp((action - u._mid) / u._half, -1.0, 1.0), clone=False)
        ep.add("obs/policy", obs["policy"][0])
        if self.record_camera:
            for key in ("rgb", "depth"):
                if key in obs:
                    ep.add(f"obs/{key}", obs[key][0])
        for name, view in zip(self.state_names, u.state_views):
            ep.add(f"states/{name}", view[0])
        for dev, st in self.teleop.devices.states.items():
            ep.add(f"devices/{dev}", {"position": torch.tensor(np.asarray(st.position, np.float32)),
                                      "rotation": torch.tensor(np.asarray(st.rotation, np.float32)),
                                      "grip": torch.tensor([float(st.grip)]),
                                      "buttons": torch.tensor([float(st.buttons.get(b, False)) for b in ("grab", "secondary", "fire")])},
                   clone=False)
        if self.tool_force is not None:
            ep.add("haptics/force", wp.to_torch(self.tool_force.force)[0])
            ep.add("haptics/jaw_load", wp.to_torch(self.tool_force.jaw_load)[0])

    def end(self, success: bool) -> str | None:
        """Write the episode (``demo_N``, returned) with its ``success`` flag; an empty episode writes nothing."""
        ep, self.episode = self.episode, None
        if ep is None or ep.is_empty():
            return None
        ep.success = bool(success)
        ep.pre_export()
        name = f"demo_{self.handler.get_num_episodes()}"
        self.handler.write_episode(ep)
        self.handler.flush()
        self.written.append(name)
        return name

    def discard(self) -> None:
        self.episode = None

    def close(self) -> None:
        """An episode still open is written unsuccessful (not lost); then the file closes."""
        if self.episode is not None:
            self.end(False)
        self.handler.close()


def load_demo(path, demo: str | None = None, device: str = "cuda:0"):
    """``(episode, env_args)`` of ``demo`` (default: the first) in a Recorder file."""
    import json

    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    handler = HDF5DatasetFileHandler()
    handler.open(str(path))
    try:
        names = list(handler.get_episode_names())
        if not names:
            raise ValueError(f"{path}: no episodes")
        name = demo if demo is not None else names[0]
        episode = handler.load_episode(name, device)
        if episode is None:
            raise ValueError(f"{path}: no episode {name!r}; recorded: {names}")
        env_args = json.loads(handler._hdf5_data_group.attrs["env_args"])
    finally:
        handler.close()
    return episode, env_args


def env_cfg_from_recording(env_args: dict, overrides: dict | None = None, *, data=None, workdir=None, seed=None):
    """An ``I4hEnvCfg`` that rebuilds the recorded env: the recorded case YAML (``${DATA}`` resolved against ``data`` or
    the current ``I4H_DATA``; written to ``workdir`` or a temporary directory, so ``cfg.case`` stays a path) and every
    recorded setting, physical actions. ``overrides`` (None values ignored) replace recorded settings; one that changes
    the dynamics (``DYNAMICS``) warns. A recording without ``format_version`` keeps only its case name and decimation
    (with a warning)."""
    import tempfile

    from i4h_isaaclab.env import I4hEnvCfg
    from i4h_physics.config import resolve_data_path

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if "format_version" not in env_args:
        warnings.warn("a recording without format_version: only its case name and decimation are restored, the replay "
                      "may differ", DeprecationWarning, stacklevel=2)
        values = {"case": env_args["case"], "decimation": int(env_args["decimation"])} | overrides
    else:
        case = resolve_data_path(yaml.safe_load(env_args["case_yaml"]), data)
        folder = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="i4h-replay-"))
        path = folder / "recorded_case.yaml"
        path.write_text(yaml.safe_dump(case, sort_keys=False))
        values = {k: env_args[k] for k in RECORDED} | {"case": str(path)}
        for key, value in overrides.items():
            recorded = env_args["case"] if key == "case" else values.get(key)
            if key in DYNAMICS and value != recorded:
                warnings.warn(f"replay override {key}={value!r} differs from the recording ({recorded!r}): the "
                              "trajectory will differ", RuntimeWarning, stacklevel=2)
            values[key] = value
    cfg = I4hEnvCfg(case=str(values.pop("case")), action_mode="physical", gravity_from_case=True)
    if "physics_dt" in values:
        cfg.sim.dt = float(values.pop("physics_dt"))
    if "num_envs" in values:
        cfg.scene.num_envs = int(values.pop("num_envs"))
    if "step_weights" in values:
        cfg.step_weights = tuple(values.pop("step_weights"))
    if seed is not None:
        cfg.seed = int(seed)
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def check_replay_compatible(u, env_args: dict) -> None:
    """Raise when the env's action row is not laid out as recorded; warn when its dynamics settings differ."""
    if "action_layout" in env_args and action_layout(u.simulator) != env_args["action_layout"]:
        raise ValueError(f"the env's action layout {action_layout(u.simulator)} differs from the recording's "
                         f"{env_args['action_layout']}: the recorded rows would drive the wrong columns")
    live = {"decimation": int(u.cfg.decimation), "physics_dt": float(u.physics_dt), "workspace": float(u.cfg.workspace),
            "episode_length_s": float(u.cfg.episode_length_s)}
    for key, value in live.items():
        if key in env_args and abs(float(env_args[key]) - value) > 1.0e-9 * max(1.0, abs(value)):
            warnings.warn(f"the env's {key}={value} differs from the recording's {env_args[key]}: the replay will differ",
                          RuntimeWarning, stacklevel=2)


def replay_episode(env, episode, env_args: dict | None = None) -> np.ndarray:
    """Reset ``env`` (``action_mode = "physical"``) and step the episode's recorded actions (every env gets env 0's row);
    returns the largest ``|obs/policy|`` difference of env 0 against the recording, relative to ``1 + |recorded|``, per
    step. With the recording's ``env_args``, the env is checked against them first (``check_replay_compatible``)."""
    u = env.unwrapped
    if u.cfg.action_mode != "physical":
        raise ValueError("replay needs I4hEnvCfg.action_mode = 'physical'")
    if env_args is not None:
        check_replay_compatible(u, env_args)
    actions, recorded = episode.data["actions"], episode.data["obs"]["policy"]
    env.reset()
    restore_episode_length(u, episode)
    diffs = []
    for t in range(actions.shape[0]):
        obs, *_ = env.step(actions[t].to(u.device).expand(u.num_envs, -1))
        ref = recorded[t].to(u.device)
        diffs.append(((obs["policy"][0] - ref).abs() / (1.0 + ref.abs())).max())
    torch.cuda.synchronize()
    return torch.stack(diffs).cpu().numpy() if diffs else np.zeros(0, np.float32)


def restore_episode_length(u, episode) -> None:
    """After the replay's reset: every env's episode counter as env 0's was when the take began (a fresh env staggers
    its first episode; a later take began at 0), so time-outs fall on the recorded steps."""
    if "initial_episode_length" in episode.data:
        u.episode_length_buf.fill_(int(episode.data["initial_episode_length"].reshape(-1)[0]))


def replay_path(spec: str) -> tuple[Path, str | None]:
    """``FILE[:demo_N]`` -> (file, demo name or None)."""
    head, sep, tail = spec.rpartition(":")
    if sep and tail.startswith("demo_"):
        return Path(head), tail
    return Path(spec), None
