# SPDX-License-Identifier: Apache-2.0
"""Thin hosted adapter; the caller samples after every host physics step, before decimation."""

from i4h_surgical.grasp_probe import GraspTargetProbe


def bind_grasp_probe(component, tissue="tissue", grasper="grasper"):
    return GraspTargetProbe(component.physics, tissue, grasper)
