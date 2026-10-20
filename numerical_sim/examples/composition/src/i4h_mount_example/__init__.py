# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated rod/tet geometry and a one-way mount, without library edits or registration."""
from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_endoluminal.loaders import LOADERS as ENDO_LOADERS
from i4h_endoluminal.systems import SYSTEMS as ENDO_SYSTEMS
from i4h_physics import System
from i4h_physics import config as physical
from i4h_surgical.scene import Instrument, SphereSpec
from i4h_surgical.systems import SYSTEMS as SURGICAL_SYSTEMS
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


def generated_jaws(scene, name):
    identity = np.eye(4, dtype=np.float32)
    spheres = [SphereSpec(2, sign, np.array([0, .015 * sign, z], np.float32), identity,
                          .015, 0, i, 3)
               for sign in (1., -1.) for i, z in enumerate((.01, .03, .05))]
    scene.instruments = {name: Instrument(
        name=name, kind="grasper", input=None, trocar=None,
        initial_position=np.zeros(3, np.float32), initial_rotation=np.array([0, 0, 0, 1], np.float32),
        pieces=[], spheres=spheres, prims=[(0, 2, "jaw_left"), (3, 5, "jaw_right")],
        wrist_pivot=np.zeros(3, np.float32), jaw_anchor=np.zeros(3, np.float32), wrist_end=None,
        wrist_end_bind=identity, jaw_open=.5, jaw_closed=0., jaw_response=30.,
        grasp_threshold=.5, grasp_stiffness=1., shaft_spacing=.02)}


@dataclass
class MountState:
    grip: wp.array
    samples: wp.array
    last_tip: wp.array


@wp.kernel
def copy_tip(q: wp.array2d(dtype=wp.vec3), tip: int, grip: wp.array(dtype=float),
             commands: wp.array3d(dtype=float), samples: wp.array(dtype=int), last: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    p = q[e, tip]
    last[e] = p
    for axis in range(3):
        commands[e, 0, axis] = p[axis]
    commands[e, 0, 7] = grip[e]
    samples[e] += 1


@wp.kernel
def reset_mount(mask: wp.array(dtype=wp.bool), grip: wp.array(dtype=float),
                samples: wp.array(dtype=int), last: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    if mask[e]:
        grip[e] = 0.
        samples[e] = 0
        last[e] = wp.vec3(0.)


@dataclass(eq=False)
class OneWayMount(System):
    commands: wp.array
    sidecar: MountState
    body: str = "wire"
    tip: int = 5
    structural = System.structural + ("tip",)

    def __post_init__(self):
        self.body = self.scene.body(self.body)

    def step_start(self, scene, dt):
        wp.launch(copy_tip, dim=scene.envs,
                  inputs=[scene.field("q", self.body), self.tip, self.sidecar.grip],
                  outputs=[self.commands, self.sidecar.samples, self.sidecar.last_tip], device=scene.device)

    def reset(self, scene, mask):
        wp.launch(reset_mount, dim=scene.envs, inputs=[mask],
                  outputs=[self.sidecar.grip, self.sidecar.samples, self.sidecar.last_tip], device=scene.device)


def build(captured=True, device="cuda:0"):
    owned = {}
    def mount_factory(scene, **kwargs):
        control = InstrumentControl(scene, id="tools")
        state = MountState(wp.zeros(scene.envs, dtype=float, device=scene.device),
                           wp.zeros(scene.envs, dtype=int, device=scene.device),
                           wp.zeros(scene.envs, dtype=wp.vec3, device=scene.device))
        mount = OneWayMount(scene, commands=control.actions, sidecar=state, **kwargs)
        owned.update(control=control, state=state, mount=mount)
        return mount
    def control_factory(scene, **kwargs):
        return owned["control"]
    mount_factory.refs = ()
    control_factory.refs = ()
    cfg = {"sim": {"device": device, "envs": 2, "gravity": [0, 0, 0], "dt": .01,
                   "substeps": 2, "iterations": 4, "cuda_graph": captured},
           "assets": {"block": {"type": "tet_grid", "dim": [6, 6, 4], "cell": .02,
                                "origin": [-.06, -.06, 0], "fix_top": True, "radius": .005},
                      "wire": {"type": "rod", "start": [-.1, 0, 0], "direction": [1, 0, 0],
                               "count": 6, "segment_length": .02, "radius": .002},
                      "jaws": {"type": "generated_jaws"}},
           "systems": [{"type": "particles"}, {"type": "rod_direct", "id": "rod", "body": "wire"},
                       {"type": "catheter_control", "rod": "rod", "track_length": .2},
                       {"type": "mount"}, {"type": "instrument_control", "id": "tools"},
                       {"type": "tet", "body": "block"},
                       {"type": "volumetric_grasp", "id": "grasp", "instruments": "tools", "body": "block"}]}
    sim = physical.load_case(cfg, loaders={**physical.LOADERS, **ENDO_LOADERS, "generated_jaws": generated_jaws},
                             systems={**physical.SYSTEMS, **ENDO_SYSTEMS, **SURGICAL_SYSTEMS,
                                      "mount": mount_factory, "instrument_control": control_factory})
    return sim, owned


def qualify(captured):
    sim, owned = build(captured)
    sim.prepare()
    mount, state = owned["mount"], owned["state"]
    grasp = next(s for s in sim.systems if s.id == "grasp")
    control = next(s for s in sim.systems if s.id == "catheter_control")
    sim.step()
    previous = sim.scene.field("q", mount.body).numpy()[:, mount.tip].copy()
    state.grip.assign(np.array([1., 0.], np.float32))
    sim.step()
    np.testing.assert_array_equal(state.last_tip.numpy(), previous)
    assert grasp.count.numpy()[0, 0] > 0 and grasp.count.numpy()[1, 0] == 0
    q_before = sim.scene.field("q", sim.scene.body("block")).numpy().copy()
    control.actions.assign(np.array([[.5, 0, 0], [0, 0, 0]], np.float32))
    for _ in range(30):
        sim.step()
    q_after = sim.scene.field("q", sim.scene.body("block")).numpy()
    assert np.max(np.abs(q_after[0] - q_before[0])) > .001
    np.testing.assert_allclose(q_after[1], q_before[1], atol=1e-6)
    state.grip.zero_()
    sim.step()
    assert not grasp.count.numpy().any()
    samples = state.samples.numpy().copy()
    sim.reset([0])
    assert state.samples.numpy().tolist() == [0, int(samples[1])]
    assert state.grip.numpy().tolist() == [0., 0.]
    mount.tip = 4
    try:
        sim.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("a structural edit must be rejected before execution")
    print(f"one-way mount captured={captured}: grasp, displacement, release, isolation, reset and structural rejection passed")


def main():
    for captured in (False, True):
        qualify(captured)
