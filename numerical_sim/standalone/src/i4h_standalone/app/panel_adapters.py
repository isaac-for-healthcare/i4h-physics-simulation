# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit presentation adapters. Simulation packages own no panel lifecycle."""

from dataclasses import dataclass

import warp as wp


@dataclass
class Knob:
    """One control of a system's panel: a live array (a slider per column) or a recapture-safe structural field."""

    label: str  # the attribute name; also the key in the overrides YAML
    lo: float = 0.0
    hi: float = 1.0
    array: wp.array | None = None  # live float32 (E,) or (E, n), edited in place with assign()
    field: str | None = None  # structural attribute: int -> integer slider lo..hi, bool -> checkbox; applied through prepare()
    labels: tuple = ()  # column names of an (E, n) array (parts, instruments, pools)
    format: str = "%.3f"
    log: bool = False  # logarithmic slider (lo > 0)
    scope: str | None = None  # body name for regional tables; labels are region names


def i4h_physics_systems_contact_fluid_contact_FluidContact(self):
    return [Knob("margin", 0.0, 0.02, array=self.margin, format="%.4f"), Knob("stiffness", 0.0, 1.0, array=self.stiffness),
            Knob("max_correction", 0.0, 0.05, array=self.max_correction, format="%.4f"), Knob("iterations", 1, 8, field="iterations")]


def i4h_physics_systems_contact_graph_containment_GraphContainment(self):
    return [Knob("max_distance", 0.0, 0.5, array=self.max_distance), Knob("skin", 0.0, 0.2, array=self.skin),
            Knob('target_response', 0.0, 1.0, array=self.target_response), Knob('source_max_delta', 0.0, 0.05, array=self.source_max_delta, format="%.4f"),
            Knob('target_max_delta', 0.0, 0.05, array=self.target_max_delta, format="%.4f"),
            Knob("iterations", 1, 8, field="iterations")]


def i4h_physics_systems_contact_part_contact_PartContact(self):
    return [Knob("margin", 0.0, 0.02, array=self.margin, format="%.4f"),
            Knob("stiffness", 0.0, 1.0, array=self.stiffness),
            Knob("max_correction", 0.0, 0.05, array=self.max_correction, format="%.4f"),
            Knob("iterations", 1, 8, field="iterations")]


def i4h_physics_systems_contact_rod_containment_RodContainment(self):
    return [Knob("iterations", 1, 4, field="iterations")]


def i4h_physics_systems_contact_shell_containment_ShellContainment(self):
    return [Knob('target_response', 0.0, 1.0, array=self.target_response), Knob("max_distance", 0.0, 0.5, array=self.max_distance),
            Knob("iterations", 1, 4, field="iterations")]


def i4h_physics_systems_contact_static_contact_StaticContact(self):
    return [Knob("margin", 0.0001, 0.05, array=self.margin, format="%.4f"), Knob("stiffness", 0.0, 1.0, array=self.stiffness),
            Knob("max_correction", 0.0, 0.05, array=self.max_correction, format="%.4f"), Knob("iterations", 1, 4, field="iterations")]


def i4h_physics_systems_contact_tet_containment_TetContainment(self):
    return [Knob('target_response', 0.0, 1.0, array=self.target_response), Knob("max_distance", 0.0, 0.5, array=self.max_distance),
            Knob("iterations", 1, 4, field="iterations")]


def i4h_physics_systems_fluids_fluid_Fluid(self):
    pools = tuple(b.name for b in self.bodies)
    return [Knob("relaxation", 0.0, 2.0, array=self.relaxation, labels=pools), Knob("viscosity", 0.0, 10.0, array=self.viscosity, labels=pools, format="%.2f"),
            Knob("cohesion", 0.0, 1.0, array=self.cohesion, labels=pools), Knob("damping", 0.0, 1.0, array=self.damping),
            Knob("max_velocity", 0.0, 1000.0, array=self.max_velocity, format="%.0f")]


def i4h_physics_systems_mechanics_attachments_Attachments(self):
    return [Knob("stiffness", 0.0, 1.0, array=self.stiffness)]


def i4h_physics_systems_mechanics_bounds_Bounds(self):
    return [Knob("iterations", 1, 4, field="iterations")]


def i4h_physics_systems_mechanics_connectors_Connectors(self):
    groups = tuple(f"group {g}" for g in range(self.stiffness.shape[1]))
    return [Knob("stiffness", 0.0, 5.0, array=self.stiffness, labels=groups),
            Knob("damping", 0.0, 500.0, array=self.damping, labels=groups, format="%.1f"),
            Knob("weight_ratio", 0.0, 1.0, array=self.weight_ratio, labels=groups)]


def i4h_physics_systems_mechanics_particles_Particles(self):
    return [Knob("damping", 0.0, 1.0, array=self.damping), Knob("max_velocity", 0.0, 1000.0, array=self.max_velocity, format="%.0f")]


def i4h_physics_systems_mechanics_shell_Shell(self):
    return [Knob("stretch", 1.0, 1.0e6, array=self.stretch, format="%.2e", log=True), Knob("bend", 1.0, 1.0e5, array=self.bend, format="%.2e", log=True),
            Knob("relaxation", 0.0, 1.0, array=self.relaxation), Knob("iterations", 1, 16, field="iterations")]


def i4h_physics_systems_mechanics_stretch_breaking_StretchBreaking(self):
    return [Knob("threshold", 1.0, 20.0, array=self.threshold, format="%.2f"), Knob("break_edges", field="break_edges")]


def i4h_physics_systems_mechanics_tet_Tet(self):
    parts = tuple(sorted(self.body.parts, key=lambda n: self.body.parts[n].index))  # (E, parts) column order
    return [Knob("iterations", 1, 16, field="iterations"),
            Knob("spring", 0.0, 5.0, array=self.spring, labels=parts),
            Knob("spring_damping", 0.0, 500.0, array=self.spring_damping, labels=parts, format="%.1f"),
            Knob("volume", 0.0, 5.0, array=self.volume, labels=parts),
            Knob("volume_damping", 0.0, 500.0, array=self.volume_damping, labels=parts, format="%.1f")]


def i4h_physics_systems_rods_material_RodPbdMaterial(self):
    return regional_knobs(self.body) + [Knob('radius_weighting', field='radius_weighting')]


def i4h_physics_systems_rods_material_RodDirectMaterial(self):
    return regional_knobs(self.body)


def i4h_physics_systems_rods_rod_direct_RodDirect(self):
    return [Knob("young", 1.0e3, 1.0e9, array=self.young, format="%.2e", log=True), Knob("torsion", 1.0e3, 1.0e9, array=self.torsion, format="%.2e", log=True),
            Knob("bend", 0.0, 10.0, array=self.bend), Knob("twist", 0.0, 10.0, array=self.twist), Knob("angular_damping", 0.0, 1.0, array=self.angular_damping)]


def i4h_physics_systems_rods_rod_pbd_RodPbd(self):
    return [Knob("stretch", 0.0, 10.0, array=self.stretch), Knob("bend", 0.0, 10.0, array=self.bend), Knob("twist", 0.0, 10.0, array=self.twist),
            Knob("angular_damping", 0.0, 1.0, array=self.angular_damping), Knob("sweeps", 1, 8, field="sweeps")]


def i4h_surgical_systems_contact_instrument_contact_InstrumentContact(self):
    return [Knob("margin", 0.0001, 0.02, array=self.margin, format="%.4f"), Knob("stiffness", 0.0, 1.0, array=self.stiffness),
            Knob("static_friction", 0.0, 2.0, array=self.static_friction), Knob("kinetic_friction", 0.0, 2.0, array=self.kinetic_friction),
            Knob("max_correction", 0.0, 0.05, array=self.max_correction, format="%.4f"),
            Knob("iterations", 1, 8, field="iterations"), Knob("sweep_samples", 1, 16, field="sweep_samples")]


def i4h_surgical_systems_fluids_bleeding_Bleeding(self):
    return [Knob("margin", 0.0, 0.02, array=self.margin, format="%.4f"), Knob("rate", 0.0, 2048.0, array=self.rate, format="%.0f"),
            Knob("speed", 0.0, 2.0, array=self.speed), Knob("spawn_radius", 0.0, 0.05, array=self.spawn_radius, format="%.4f")]


def i4h_surgical_systems_fluids_emitter_Emitter(self):
    return [Knob("rate", 0.0, 8192.0, array=self.rate, format="%.0f"), Knob("speed", 0.0, 10.0, array=self.speed),
            Knob("spawn_radius", 0.0, 0.05, array=self.spawn_radius, format="%.4f"), Knob("threshold", 0.0, 1.0, array=self.threshold),
            Knob("enabled_env", 0.0, 1.0, array=self.enabled_env, format="%.0f")]


def i4h_surgical_systems_fluids_smoke_Smoke(self):
    span = ("min", "max")
    return [Knob("rate", 0.0, 2000.0, array=self.rate, format="%.0f"), Knob("spawn_radius", 0.0, 0.1, array=self.spawn_radius, format="%.4f"),
            Knob("lifetime", 0.0, 2.0, array=self.lifetime, labels=span), Knob("speed", 0.0, 1.0, array=self.speed, labels=span),
            Knob("upward", 0.0, 1.0, array=self.upward, labels=span)]


def i4h_surgical_systems_fluids_suction_Suction(self):
    return [Knob("radius", 0.0, 0.5, array=self.radius), Knob("removal_radius", 0.0, 0.05, array=self.removal_radius, format="%.4f"),
            Knob("power", 0.0, 20.0, array=self.power)]


def i4h_surgical_systems_instruments_clipping_Clipping(self):
    return [Knob("reach", 0.0, 0.5, array=self.reach), Knob("lumen", 0.0, 0.02, array=self.lumen, format="%.4f"), Knob("stiffness", 0.0, 1.0, array=self.stiffness)]


def i4h_surgical_systems_instruments_contact_paint_ContactPaint(self):
    return [Knob("rate", 0.0, 10.0, array=self.rate)]  # paint is the keyboard's mask


def i4h_surgical_systems_instruments_cutting_Cutting(self):
    return [Knob("radius_scale", 0.0, 3.0, array=self.radius_scale), Knob("margin", 0.0, 0.02, array=self.margin, format="%.4f"),
            Knob("closure_delta", 1.0e-6, 0.1, array=self.closure_delta, format="%.1e", log=True), Knob("blood", 0.0, 1.0, array=self.blood)]


def i4h_surgical_systems_instruments_grasp_Grasp(self):
    return [Knob("stiffness", 0.0, 1.0, array=self.stiffness, labels=tuple(self.instruments.names))]


def i4h_surgical_systems_instruments_hook_Hook(self):
    return [Knob("engage_overlap", 0.0, 0.01, array=self.engage_overlap, format="%.4f"),
            Knob("engage_inward_speed", 0.0, 0.2, array=self.engage_inward_speed),
            Knob("retention_slack", 0.0, 0.02, array=self.retention_slack, format="%.4f"),
            Knob("axial_friction", 0.0, 2.0, array=self.axial_friction), Knob("puncture_distance", 0.0, 0.05, array=self.puncture_distance)]


def i4h_surgical_systems_instruments_instrument_control_InstrumentControl(self):
    return [Knob("max_velocity", 0.0, 10.0, array=self.max_velocity), Knob("energy_threshold", 0.0, 1.0, array=self.energy_threshold),
            Knob("fire_threshold", 0.0, 1.0, array=self.fire_threshold)]


def i4h_surgical_systems_instruments_thermal_Thermal(self):
    return [Knob("power", 0.0, 500.0, array=self.power, format="%.0f"), Knob("diffusion", 0.0, 1.0, array=self.diffusion),
            Knob("cooling", 0.0, 1.0, array=self.cooling), Knob("burn_rate", 0.0, 20.0, array=self.burn_rate, format="%.2f"),
            Knob("coag_heat", 0.0, 20.0, array=self.coag_heat, format="%.2f"), Knob("delete_heat", 0.0, 20.0, array=self.delete_heat, format="%.2f"),
            Knob("delete_burn", 0.0, 20.0, array=self.delete_burn, format="%.2f"), Knob("diffusion_substeps", 1, 4, field="diffusion_substeps")]


def i4h_surgical_systems_instruments_tool_force_ToolForce(self):
    return [Knob("contact_gain", 0.0, 10.0, array=self.contact_gain), Knob("grasp_gain", 0.0, 10.0, array=self.grasp_gain),
            Knob("hook_gain", 0.0, 10.0, array=self.hook_gain), Knob("max_force", 0.0, 100.0, array=self.max_force, format="%.2f")]


def i4h_surgical_systems_instruments_volumetric_grasp_VolumetricGrasp(self):
    return [Knob("stiffness", 0.0, 1.0, array=self.stiffness, labels=tuple(self.instruments.names))]


def i4h_endoluminal_systems_rods_catheter_control_CatheterControl(self):
    return [Knob("track_stiffness", 0.0, 1.0, array=self.track_stiffness), Knob("max_delta", 0.0, 0.05, array=self.max_delta, format="%.4f"),
            Knob("rates", 0.0, 5.0, array=self.rates, labels=("insert", "rotate", "tip"))]


def i4h_endoluminal_systems_rods_drag_Drag(self):
    return [Knob("max_delta", 0.0, 0.05, array=self.max_delta, format="%.4f")]


def regional_knobs(body):
    props = body.region_properties
    if props is None:
        return []
    return [Knob('region_' + name, -100.0 if name.startswith('curvature') else 0.0,
                 100.0 if name.startswith('curvature') else 10.0, array=table,
                 labels=props.membership.names, scope=body.name)
            for name, table in props.tables.items()]




def surgical_breaking(self):
    return [Knob("threshold", 1.0, 20.0, array=self.threshold, format="%.2f"), Knob("damage_start", 1.0, 20.0, array=self.damage_start, format="%.2f"),
            Knob("damage_full", 1.0, 20.0, array=self.damage_full, format="%.2f"), Knob("blood", 0.0, 1.0, array=self.blood),
            Knob("break_edges", field="break_edges")]


_ADAPTERS = {
    ('i4h_physics.systems.contact.fluid_contact', 'FluidContact'): i4h_physics_systems_contact_fluid_contact_FluidContact,
    ('i4h_physics.systems.contact.graph_containment', 'GraphContainment'): i4h_physics_systems_contact_graph_containment_GraphContainment,
    ('i4h_physics.systems.contact.part_contact', 'PartContact'): i4h_physics_systems_contact_part_contact_PartContact,
    ('i4h_physics.systems.contact.rod_containment', 'RodContainment'): i4h_physics_systems_contact_rod_containment_RodContainment,
    ('i4h_physics.systems.contact.shell_containment', 'ShellContainment'): i4h_physics_systems_contact_shell_containment_ShellContainment,
    ('i4h_physics.systems.contact.static_contact', 'StaticContact'): i4h_physics_systems_contact_static_contact_StaticContact,
    ('i4h_physics.systems.contact.tet_containment', 'TetContainment'): i4h_physics_systems_contact_tet_containment_TetContainment,
    ('i4h_physics.systems.fluids.fluid', 'Fluid'): i4h_physics_systems_fluids_fluid_Fluid,
    ('i4h_physics.systems.mechanics.attachments', 'Attachments'): i4h_physics_systems_mechanics_attachments_Attachments,
    ('i4h_physics.systems.mechanics.bounds', 'Bounds'): i4h_physics_systems_mechanics_bounds_Bounds,
    ('i4h_physics.systems.mechanics.connectors', 'Connectors'): i4h_physics_systems_mechanics_connectors_Connectors,
    ('i4h_physics.systems.mechanics.particles', 'Particles'): i4h_physics_systems_mechanics_particles_Particles,
    ('i4h_physics.systems.mechanics.shell', 'Shell'): i4h_physics_systems_mechanics_shell_Shell,
    ('i4h_physics.systems.mechanics.stretch_breaking', 'StretchBreaking'): i4h_physics_systems_mechanics_stretch_breaking_StretchBreaking,
    ('i4h_physics.systems.mechanics.tet', 'Tet'): i4h_physics_systems_mechanics_tet_Tet,
    ('i4h_physics.systems.rods.material', 'RodPbdMaterial'): i4h_physics_systems_rods_material_RodPbdMaterial,
    ('i4h_physics.systems.rods.material', 'RodDirectMaterial'): i4h_physics_systems_rods_material_RodDirectMaterial,
    ('i4h_physics.systems.rods.rod_direct', 'RodDirect'): i4h_physics_systems_rods_rod_direct_RodDirect,
    ('i4h_physics.systems.rods.rod_pbd', 'RodPbd'): i4h_physics_systems_rods_rod_pbd_RodPbd,
    ('i4h_surgical.systems.contact.instrument_contact', 'InstrumentContact'): i4h_surgical_systems_contact_instrument_contact_InstrumentContact,
    ('i4h_surgical.systems.fluids.bleeding', 'Bleeding'): i4h_surgical_systems_fluids_bleeding_Bleeding,
    ('i4h_surgical.systems.fluids.emitter', 'Emitter'): i4h_surgical_systems_fluids_emitter_Emitter,
    ('i4h_surgical.systems.fluids.smoke', 'Smoke'): i4h_surgical_systems_fluids_smoke_Smoke,
    ('i4h_surgical.systems.fluids.suction', 'Suction'): i4h_surgical_systems_fluids_suction_Suction,
    ('i4h_surgical.systems.instruments.clipping', 'Clipping'): i4h_surgical_systems_instruments_clipping_Clipping,
    ('i4h_surgical.systems.instruments.contact_paint', 'ContactPaint'): i4h_surgical_systems_instruments_contact_paint_ContactPaint,
    ('i4h_surgical.systems.instruments.cutting', 'Cutting'): i4h_surgical_systems_instruments_cutting_Cutting,
    ('i4h_surgical.systems.instruments.grasp', 'Grasp'): i4h_surgical_systems_instruments_grasp_Grasp,
    ('i4h_surgical.systems.instruments.hook', 'Hook'): i4h_surgical_systems_instruments_hook_Hook,
    ('i4h_surgical.systems.instruments.instrument_control', 'InstrumentControl'): i4h_surgical_systems_instruments_instrument_control_InstrumentControl,
    ('i4h_surgical.systems.instruments.thermal', 'Thermal'): i4h_surgical_systems_instruments_thermal_Thermal,
    ('i4h_surgical.systems.instruments.tool_force', 'ToolForce'): i4h_surgical_systems_instruments_tool_force_ToolForce,
    ('i4h_surgical.systems.instruments.volumetric_grasp', 'VolumetricGrasp'): i4h_surgical_systems_instruments_volumetric_grasp_VolumetricGrasp,
    ('i4h_endoluminal.systems.rods.catheter_control', 'CatheterControl'): i4h_endoluminal_systems_rods_catheter_control_CatheterControl,
    ('i4h_endoluminal.systems.rods.drag', 'Drag'): i4h_endoluminal_systems_rods_drag_Drag,
    ('i4h_surgical.systems.mechanics.stretch_breaking', 'StretchBreaking'): surgical_breaking,
}


def panel(system):
    for cls in type(system).__mro__:
        adapter = _ADAPTERS.get((cls.__module__, cls.__name__))
        if adapter is not None:
            return adapter(system)
    return []



def canonical_knob(sim, sid, name):
    from i4h_physics.regions import material_system
    name = {"vessel_response": "target_response", "catheter_max_delta": "source_max_delta",
            "vessel_max_delta": "target_max_delta"}.get(name, name)
    if name.startswith('region_') or name == 'radius_weighting':
        provider = material_system(sim, sid)
        if provider is not None:
            return provider.id, name
    return sid, name
