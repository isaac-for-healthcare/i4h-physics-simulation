# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab physics backend that steps an i4h_sim Stepper (an ``I4hSolver`` on the case's Newton model and state).

Isaac Lab 3.0 selects the backend from ``SimulationCfg.physics.class_type``; ``DirectRLEnv`` (kitless)
calls, in order, ``_prepare_stage_creation``, ``initialize``, ``get_scene_data_backend``,
``register_callback`` (the render context), then ``reset`` and ``play`` from ``sim.reset()`` after
``_setup_scene``, ``wait_for_playing`` + ``step`` once per physics substep (Isaac Lab keeps its own
decimation loop, ``handles_decimation()`` stays False), ``forward`` from ``env.reset()``, ``get_device``,
``get_physics_dt``, ``stop`` and ``close``. Everything not defined here is the base class's no-op. The
simulator is a class attribute bound by ``I4hEnv._setup_scene``, so one environment per process (Isaac
Lab's ``SimulationContext`` is a singleton anyway). The scene-data backend stays empty: ``I4hEnv``
exposes no Newton model to Isaac Lab; the camera observation renders from its own render model
(``i4h_isaaclab.camera.tiled``).
"""

from __future__ import annotations

from typing import ClassVar

from i4h_physics.stepper import Stepper
from isaaclab.physics.physics_manager import PhysicsEvent, PhysicsManager
from isaaclab.physics.physics_manager_cfg import PhysicsCfg
from isaaclab.scene_data.scene_data_backend import SceneDataBackend
from isaaclab.utils.configclass import configclass


class I4hPhysicsManager(PhysicsManager):
    sim: ClassVar[Stepper | None] = None  # set by I4hEnv._setup_scene
    procedure = None
    after_step = None  # I4hEnv: orders torch after Warp's step, before Isaac Lab's _get_dones/_get_rewards
    _ready: ClassVar[bool] = False  # PHYSICS_READY dispatched once, on the first reset with a simulator bound

    @classmethod
    def initialize(cls, sim_context) -> None:
        super().initialize(sim_context)  # records the context, cfg and device; selects the CUDA device for torch and Warp
        if cls.sim is not None:
            cls.sim.physics.invalidate()
        cls.sim = None
        cls.procedure = cls.after_step = None
        cls._ready = False

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """``sim.reset()`` from ``DirectRLEnv._init_sim`` (all envs; ``soft`` makes no difference here)."""
        if cls.sim is None:
            return
        cls.sim.reset()
        if cls.procedure is not None:
            cls.procedure.reset(mask=cls.sim._mask)
        if not cls._ready:
            cls._ready = True
            cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

    @classmethod
    def step(cls) -> None:
        if cls.sim is None:
            raise RuntimeError("I4hPhysicsManager.step before I4hEnv._setup_scene bound a simulator")
        cls.sim.step()
        if cls.procedure is not None:
            cls.procedure.update(cls.sim.dt)
        if cls.after_step is not None:
            cls.after_step()
        PhysicsManager._sim_time += cls.get_physics_dt()  # the base reads this attribute for get_simulation_time()

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        return SceneDataBackend()

    @classmethod
    def close(cls) -> None:
        if cls.sim is not None:
            cls.sim.physics.invalidate()
        cls.sim = None
        cls.procedure = cls.after_step = None
        cls._ready = False
        super().close()


@configclass
class I4hPhysicsCfg(PhysicsCfg):
    class_type: type = I4hPhysicsManager
