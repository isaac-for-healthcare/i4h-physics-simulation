# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only case records, exact world authoring, and fresh runtime binding."""

from copy import deepcopy
from uuid import uuid4

import newton
import numpy as np

from i4h_physics.checks import check_int
from i4h_physics.newton_model import add_env_particles


class CaseAuthoring:
    def __init__(self, cfg, loaders=None, systems=None, *, device=None, envs=None, mounts=None, coupling_mode="one_way"):
        from i4h_physics.config import load_scene, SYSTEMS

        self.cfg = deepcopy(cfg)
        self.constructors = dict(SYSTEMS if systems is None else systems)
        self.scene = load_scene(self.cfg, device, envs, loaders=loaders)
        seen = set()
        for spec in self.cfg.get("systems") or []:
            kind = spec["type"]
            sid = spec.get("id", kind)
            if kind not in self.constructors:
                raise ValueError(f"system {sid!r}: unknown type {kind!r}")
            if sid in seen:
                raise ValueError(f"duplicate system id {sid!r}")
            cls = self.constructors[kind]
            for ref in cls.refs:
                if ref in spec and (not isinstance(spec[ref], str) or spec[ref] not in seen):
                    raise ValueError(f"system {sid!r}: {ref!r} must name an earlier system")
            if hasattr(cls, "author"):
                cls.author(self.scene)
            seen.add(sid)
        from i4h_physics.coupling import COUPLING_MODES
        from i4h_physics.mount_storage import declare_mounts
        if coupling_mode not in COUPLING_MODES:
            raise ValueError(f"coupling_mode must be one of {COUPLING_MODES}")
        self.mounts, self.coupling_mode = deepcopy(mounts), coupling_mode
        if mounts is not None:
            declare_mounts(self.scene, mounts, two_way=coupling_mode == "two_way")
        self._mount_config = self._coupling_config()
        self.token = uuid4().hex
        self.builder = None
        self.worlds = {}
        self.storage = None
        self.storage_records = {}
        self._completed_specs = None

    def declare(self, builder):
        """Idempotent declarations; Newton rejects incompatible existing specifications."""
        previous = getattr(builder, "_i4h_authoring", None)
        if previous is not None and previous is not self:
            raise ValueError("builder already has a different i4h case authoring")
        for name, assignment in (("case_id", newton.Model.AttributeAssignment.MODEL),
                                 ("state_case_id", newton.Model.AttributeAssignment.STATE),
                                 ("control_case_id", newton.Model.AttributeAssignment.CONTROL)):
            existing = builder.custom_attributes.get(f"i4h:{name}")
            if existing is not None and existing.default != self.token:
                raise ValueError(f"conflicting i4h declaration: {name}")
            builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                name=name, namespace="i4h", dtype=str, frequency=newton.Model.AttributeFrequency.ONCE,
                assignment=assignment, default=self.token))
        if previous is None:
            from i4h_physics.storage import Declarations
            from i4h_physics.scene_storage import declare_scene, register_scene_finalizers
            self.storage = Declarations()
            declare_scene(self.storage, self.scene)
            self.particle_domain = self.scene.particle_domain
            for spec in self.cfg.get("systems") or []:
                hook = getattr(self.constructors[spec["type"]], "declare_storage", None)
                if hook is not None:
                    self.storage_records[spec.get("id", spec["type"])] = hook(self, self.storage, spec, self.storage_records)
            from i4h_physics.mount_storage import publish_mounts
            publish_mounts(self.scene, self.storage, builder)
            self.storage.register_schema(builder)
            register_scene_finalizers(builder, self.scene)
            self._completed_specs = deepcopy(self.cfg.get("systems") or [])
            self._declared_cfg = deepcopy(self.cfg)
            self._declared_layout = self._layout()
            from i4h_physics.observables import declare_layout
            self.output_layout = declare_layout(builder, self.scene, self.cfg.get("systems") or [], self.constructors)
        builder._i4h_authoring = self

    def add_world(self, builder, world, xform=None, label_template="i4h/{world}"):
        """Append exact component-frame rows inside the destination's active world."""
        world = check_int("world", world, 0)
        label = label_template.format(world, world=world)
        if not 0 <= world < self.scene.envs or builder.current_world != world:
            raise ValueError("authoring world must match the destination builder's active world")
        if self.builder is not None and self.builder is not builder:
            raise ValueError("a new builder requires fresh case authoring")
        if world in self.worlds:
            raise ValueError(f"duplicate i4h world authoring: {world}")
        xform = xform if xform is not None else ((0., 0., 0.), (0., 0., 0., 1.))
        position, rotation = (np.asarray(v, dtype=np.float64) for v in xform)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("world transform needs finite position (3,) and quaternion (4,)")
        if not np.isclose(np.linalg.norm(rotation), 1.0):
            raise ValueError("world quaternion must be normalized")
        self.declare(builder)
        self.builder = builder
        start = builder.particle_count
        add_env_particles(builder, self.scene, world, xform=(position, rotation),
                          custom_attributes=self.storage.entity_values(self.particle_domain, world * self.scene.main.width, self.scene.main.width))
        from i4h_physics.observables import add_world_layout
        add_world_layout(builder, self.output_layout, world)
        self.worlds[world] = (start, builder.particle_count, label)

    def complete(self, builder):
        """Complete package storage after all worlds/host entities, before finalize.

        Idempotent for a retained builder. No device arrays or systems are built.
        Packages explicitly declare fields; this loop only visits authored specs.
        """
        specs = self.cfg.get("systems") or []
        if builder is not self.builder or len(self.worlds) != self.scene.envs:
            raise ValueError("complete requires the authored builder and all worlds")
        if specs != self._completed_specs or self.cfg != self._declared_cfg or self._layout() != self._declared_layout:
            raise RuntimeError("persistent systems or capacities changed; reconstruct the model")
        from i4h_physics.scene_storage import complete_scene
        complete_scene(builder, self.scene)

    def _coupling_config(self):
        from i4h_physics.mount_storage import configuration
        return configuration(self.mounts, self.coupling_mode == "two_way"), self.coupling_mode

    def _layout(self):
        from i4h_physics.scene_storage import layout
        if self._coupling_config() != self._mount_config:
            raise RuntimeError("mount configuration changed; reconstruct the model")
        return layout(self.scene)

    def particle_starts(self, body):
        """Real native particle indices, including host prefixes."""
        return [self.worlds[e][0] + body.offset for e in range(self.scene.envs)]

    def bind(self, model, state, control):
        """Alias the chosen host allocation without changing its state or model parameters."""
        from i4h_physics.config import build_systems

        if len(self.worlds) != self.scene.envs:
            raise ValueError("all i4h worlds must be authored before binding")
        for obj, field in ((model, "case_id"), (state, "state_case_id"), (control, "control_case_id")):
            value = getattr(getattr(obj, "i4h", None), field, None)
            if value != [self.token] or (obj is not model and value is not getattr(model.i4h, field)):
                raise ValueError(f"foreign model/state/control: missing matching i4h {field}")
        if self.storage is None or not self.storage._authored:
            raise RuntimeError("call authoring.complete(builder) before model finalization")
        if self.cfg != self._declared_cfg or self._layout() != self._declared_layout:
            raise RuntimeError("persistent systems or capacities changed; reconstruct the model")
        scene = deepcopy(self.scene, {id(self.scene.device): self.scene.device, id(self.storage): self.storage})
        scene.particle_domain = self.particle_domain
        scene.declarations = self.storage
        scene.allocate(model, state, control=control)
        scene.control = control
        return scene, build_systems(scene, self.cfg, constructors=self.constructors)


def author_case(cfg, loaders=None, systems=None, *, device=None, envs=None, mounts=None, coupling_mode="one_way"):
    return CaseAuthoring(cfg, loaders, systems, device=device, envs=envs, mounts=mounts, coupling_mode=coupling_mode)
