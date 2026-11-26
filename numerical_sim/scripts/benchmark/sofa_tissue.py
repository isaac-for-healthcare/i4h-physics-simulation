# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOFA replica of ``chole_tissue.yaml``: the cholecystectomy tissue, its pin, part connectors, cavity contact and the
five instruments' collision spheres replaying the motion i4h simulated.

One environment is one SOFA root node, built from SOFA's standard deformable-body components: one
``MechanicalObject`` for the 5,612 tissue particles with i4h's particle masses, a corotational
``TetrahedronFEMForceField`` per part (liver, fat, gallbladder; Young's moduli in i4h's 1.0 / 0.1 / 0.8 stiffness
ratio), an implicit Euler step solved by ``CGLinearSolver`` (25 iterations), a ``RestShapeSpringsForceField`` on the
pinned particles, and springs from every connector particle to the three corners of its triangle, weighted by the
barycentric coordinates. Contact uses ``MinProximityIntersection`` and SOFA's default penalty response: the
fat and gallbladder surface vertices against the cavity triangles, and the instrument spheres against the tissue
surface. Collision groups keep every other pair (tissue self-contact, tools against the cavity) out of the
detection, as i4h does.

Matching i4h: identical particles, tetrahedra, surface triangles, connectors, pin and cavity mesh (``export_scene.py
tissue``), gravity, 1/60 s frames of 8 substeps, and the instruments' sphere positions recorded from i4h under the
same seeded random actions, interpolated linearly across the substeps. Environment ``e`` replays recorded
trajectory ``e mod K``. Differences: SOFA's FEM and penalty contact replace i4h's XPBD tetrahedral constraints and
contact projections, the connector is three corner springs instead of a barycentric constraint, and the
instruments are kinematic spheres (i4h's capsule primitives are not replicated).
"""

from __future__ import annotations

import numpy as np

PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
]

YOUNG = 2.0e5  # Pa for stiffness 1.0 (liver): with the values below the tissue settles to i4h's sag (0.23 vs 0.22 m mean)
STIFFNESS = {"liver": 1.0, "fat": 0.1, "gallbladder": 0.8}
PIN_STIFFNESS = 1.0e3  # N/m per pinned particle
CONTACT_STIFFNESS = 1.0e3  # N/m penalty per contact
CONNECTOR_STIFFNESS = 1.0e5  # N/m per connector, split over the triangle corners by barycentric weight


class Env:
    """One tissue environment, stepped by ``frame()``."""

    def __init__(self, scene: dict, env: int, envs: int):
        import Sofa.Core
        import Sofa.Simulation

        self.dt = float(scene["dt"]) / int(scene["substeps"])
        self.substeps = int(scene["substeps"])
        trajectories = scene["sphere_trajectory"]
        self.trajectory = trajectories[env % trajectories.shape[0]]
        self.frame_index = 0
        points = scene["points"]
        parts = [str(p) for p in scene["parts"]]

        root = self.root = Sofa.Core.Node(f"env{env}")
        root.dt = self.dt
        root.gravity = scene["gravity"].tolist()
        root.addObject("RequiredPlugin", pluginName=PLUGINS)
        root.addObject("DefaultAnimationLoop")
        root.addObject("CollisionPipeline", depth=6)
        root.addObject("BruteForceBroadPhase")
        root.addObject("BVHNarrowPhase")
        root.addObject("MinProximityIntersection", alarmDistance=0.02, contactDistance=0.005)
        root.addObject("CollisionResponse", response="PenalityContactForceField")

        tissue = root.addChild("Tissue")
        tissue.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=2.0)
        tissue.addObject("CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9)
        self.dofs = tissue.addObject("MechanicalObject", name="DOFs", template="Vec3d", position=points.tolist())
        tissue.addObject("UniformMass", totalMass=float((1.0 / scene["inv_mass"]).sum()))  # i4h: equal masses
        for name in parts:
            part = tissue.addChild(name)
            part.addObject("TetrahedronSetTopologyContainer", tetrahedra=scene[f"tets_{name}"].tolist(),
                           position=points.tolist())
            part.addObject("TetrahedronFEMForceField", method="large", poissonRatio=0.45,
                           youngModulus=YOUNG * STIFFNESS[name])
        tissue.addObject("RestShapeSpringsForceField", points=scene["pin"].tolist(), stiffness=PIN_STIFFNESS)
        springs = []
        for p, tri, bary in zip(scene["connector_particle"], scene["connector_triangle"], scene["connector_bary"]):
            for corner, w in zip(tri, bary):
                if w > 1e-6:
                    rest = float(np.linalg.norm(points[p] - points[corner]))
                    springs.append([int(p), int(corner), CONNECTOR_STIFFNESS * float(w), 1.0, rest])
        tissue.addObject("SpringForceField", object1="@DOFs", object2="@DOFs", spring=springs)

        # Fat and gallbladder surface vertices against the cavity (i4h static_contact parts: [fat, gallbladder]).
        free = np.unique(np.concatenate([scene[f"tris_{p}"].reshape(-1) for p in scene["contact_parts"]]))
        cavity_contact = tissue.addChild("CavityContact")
        cavity_contact.addObject("MechanicalObject", template="Vec3d", position=points[free].tolist())
        cavity_contact.addObject("PointCollisionModel", group=[3, 4], contactStiffness=CONTACT_STIFFNESS)
        cavity_contact.addObject("SubsetMapping", indices=free.tolist(), handleTopologyChange=False)
        # The whole tissue surface against the instrument spheres (i4h instrument_contact).
        surface = tissue.addChild("Surface")
        surface.addObject("TriangleSetTopologyContainer",
                          triangles=np.concatenate([scene[f"tris_{p}"] for p in parts]).tolist())
        surface.addObject("TriangleCollisionModel", group=[1, 4], contactStiffness=CONTACT_STIFFNESS)

        cavity = root.addChild("Cavity")
        cavity.addObject("MeshTopology", position=scene["cavity_points"].tolist(),
                         triangles=scene["cavity_triangles"].tolist())
        cavity.addObject("MechanicalObject")
        cavity.addObject("TriangleCollisionModel", moving=False, simulated=False, group=[1, 2],
                         contactStiffness=CONTACT_STIFFNESS)

        tools = root.addChild("Instruments")
        self.tools = tools.addObject("MechanicalObject", template="Vec3d", position=self.trajectory[0].tolist())
        tools.addObject("SphereCollisionModel", listRadius=scene["sphere_radius"].tolist(), moving=True,
                        simulated=False, group=[2, 3], contactStiffness=CONTACT_STIFFNESS)
        Sofa.Simulation.initRoot(root)

    def frame(self) -> None:
        """One 1/60 s frame: 8 substeps with the spheres interpolated from the previous recorded frame."""
        import Sofa.Simulation

        frames = self.trajectory.shape[0]
        a = self.trajectory[self.frame_index % frames]
        b = self.trajectory[(self.frame_index + 1) % frames]
        for k in range(1, self.substeps + 1):
            self.tools.position.value = a + (b - a) * (k / self.substeps)
            Sofa.Simulation.animate(self.root, self.dt)
        self.frame_index += 1

    def finite(self) -> bool:
        return bool(np.isfinite(self.dofs.position.array()).all())
