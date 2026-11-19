# SPDX-License-Identifier: Apache-2.0
"""Thin hosted adapter; the caller samples after every host physics step, before decimation."""

def bind_grasp_probe(component, tissue="tissue", grasper="grasper"):
    try:
        from i4h_surgical.grasp_probe import GraspTargetProbe
    except ModuleNotFoundError as error:
        if error.name != "i4h_surgical":
            raise
        raise ModuleNotFoundError("Grasp probes require the optional i4h-surgical package") from error

    return GraspTargetProbe(component.physics, tissue, grasper)
