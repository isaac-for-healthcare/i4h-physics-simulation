# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOFA replica of ``aorta_static`` with the scripted catheter motion of ``measure.py --navigation``.

One environment is one SOFA root node, built with BeamAdapter's standard catheter template
(``SingleBeamDeploymentCollision.py`` in the SOFA release): a ``RodStraightSection`` with the i4h rod's
length, radius, node spacing and mass, ``InterventionalRadiologyController`` to push and twist it, an implicit
Euler step solved by ``BTDLinearSolver`` (block tridiagonal: the beam counterpart of i4h's block-Thomas rod solve),
and frictional contact against the same vessel triangles (``LocalMinDistance`` + ``LCPConstraintSolver``).

Matching i4h: the same vessel mesh in world coordinates (``export_scene.py aorta``), the rod leaves the end of the
6.35 m guide at the same point and direction, 1/60 s frames of 8 substeps, no gravity, the per-environment
insertion speed ``0.25 (env + 1) / E`` m/s and the rotation rate ``0.45 sin(2 pi t / 4)`` rad/s of
``i4h_host.navigation.scripted_rates``. Differences: BeamAdapter simulates only the deployed part of the rod (the
part on the guide is outside the solve; i4h projects those nodes onto the guide every iteration); it has no
steerable-tip actuator, so the tip-bend action is not replicated; contact is a frictional complementarity solve
where i4h projects nodes back inside the vessel.
"""

from __future__ import annotations

import math

import numpy as np

PLUGINS = [
    "BeamAdapter",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
]

YOUNG = 1.0e8  # Pa: a stiff polymer catheter; the cost of the BTD solve does not depend on it


class Env:
    """One catheter-in-aorta environment, stepped by ``frame()``."""

    def __init__(self, scene: dict, env: int, envs: int):
        import Sofa.Core
        import Sofa.Simulation

        self.dt = float(scene["dt"]) / int(scene["substeps"])
        self.substeps = int(scene["substeps"])
        self.t = 0.0
        self.speed = float(scene["insertion_rate"]) * 0.5 * (env + 1.0) / max(envs, 1)
        self.rotation_rate = float(scene["rotation_rate"])
        nodes = scene["rod_nodes"]
        length = float(np.linalg.norm(nodes[-1] - nodes[0]))
        radius = float(scene["rod_radius"])
        segment = length / (len(nodes) - 1)
        density = float(scene["node_mass"]) / segment / (math.pi * radius * radius)
        exit_point = nodes[0] + (nodes[-1] - nodes[0]) / length * float(scene["track_length"])
        deployed = length - float(scene["track_length"])

        root = self.root = Sofa.Core.Node(f"env{env}")
        root.dt = self.dt
        root.gravity = [0.0, 0.0, 0.0]
        root.addObject("RequiredPlugin", pluginName=PLUGINS)
        root.addObject("FreeMotionAnimationLoop")
        root.addObject("LCPConstraintSolver", mu=0.1, tolerance=1e-6, maxIt=200, build_lcp=False)
        root.addObject("CollisionPipeline", depth=6)
        root.addObject("BruteForceBroadPhase")
        root.addObject("BVHNarrowPhase")
        # alarmDistance: the rod moves < 1 mm per substep, so 1 cm beyond contact suffices (i4h searches 9 cm; 9 cm here
        # costs SOFA 4x more).
        root.addObject("LocalMinDistance", contactDistance=radius, alarmDistance=radius + 0.01, angleCone=0.2)
        root.addObject("CollisionResponse", response="FrictionContactConstraint")

        shape = root.addChild("RestShape")
        shape.addObject("RodStraightSection", name="Straight", length=length, radius=radius,
                        nbBeams=len(nodes) - 1, nbEdgesCollis=len(nodes) - 1, nbEdgesVisu=len(nodes) - 1,
                        youngModulus=YOUNG, massDensity=density, poissonRatio=0.3)
        shape.addObject("WireRestShape", name="Shape", template="Rigid3d", wireMaterials="@Straight")
        shape.addObject("EdgeSetTopologyContainer", name="meshLines")
        shape.addObject("EdgeSetTopologyModifier")
        shape.addObject("EdgeSetGeometryAlgorithms", template="Rigid3d")
        shape.addObject("MechanicalObject", template="Rigid3d")

        beam = root.addChild("Catheter")
        beam.addObject("EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1)
        beam.addObject("BTDLinearSolver", verification=False, subpartSolve=False)
        beam.addObject("RegularGridTopology", name="MeshLines", nx=len(nodes), ny=1, nz=1,
                       xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, p0=[0, 0, 0])
        beam.addObject("MechanicalObject", name="DOFs", template="Rigid3d")
        beam.addObject("WireBeamInterpolation", name="Interpolation", WireRestShape="@../RestShape/Shape")
        beam.addObject("AdaptiveBeamForceFieldAndMass", massDensity=density, interpolation="@Interpolation")
        # Rigid3d starting pose: the guide exit, with the rod's local x along the world +x insertion direction.
        self.controller = beam.addObject(
            "InterventionalRadiologyController", name="Controller", template="Rigid3d", instruments="Interpolation",
            topology="@MeshLines", startingPos=[*exit_point.tolist(), 0, 0, 0, 1], xtip=[deployed],
            rotationInstrument=[0.0], step=0.0, speed=0.0, listening=False, controlledInstrument=0)
        beam.addObject("LinearSolverConstraintCorrection", wire_optimization=True)
        beam.addObject("FixedProjectiveConstraint", indices=0)
        beam.addObject("RestShapeSpringsForceField", points="@Controller.indexFirstNode", angularStiffness=1e8,
                       stiffness=1e8)
        collision = beam.addChild("Collision")
        collision.addObject("EdgeSetTopologyContainer")
        collision.addObject("EdgeSetTopologyModifier")
        collision.addObject("MechanicalObject")
        collision.addObject("MultiAdaptiveBeamMapping", controller="../Controller", useCurvAbs=True)
        collision.addObject("LineCollisionModel")
        collision.addObject("PointCollisionModel")

        vessel = root.addChild("Vessel")
        vessel.addObject("MeshTopology", position=scene["vessel_points"].tolist(),
                         triangles=scene["vessel_triangles"].tolist())
        vessel.addObject("MechanicalObject")
        vessel.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel.addObject("LineCollisionModel", moving=False, simulated=False)
        Sofa.Simulation.initRoot(root)
        self.xtip, self.rotation = deployed, 0.0

    def frame(self) -> None:
        """One 1/60 s frame: 8 substeps, each advancing the scripted insertion and rotation."""
        import Sofa.Simulation

        for _ in range(self.substeps):
            self.xtip += self.speed * self.dt
            self.rotation += self.rotation_rate * 0.3 * math.sin(2.0 * math.pi * self.t / 4.0) * self.dt
            self.controller.xtip.value = [self.xtip]
            self.controller.rotationInstrument.value = [self.rotation]
            Sofa.Simulation.animate(self.root, self.dt)
            self.t += self.dt

    def finite(self) -> bool:
        return bool(np.isfinite(self.root.Catheter.DOFs.position.array()).all())
