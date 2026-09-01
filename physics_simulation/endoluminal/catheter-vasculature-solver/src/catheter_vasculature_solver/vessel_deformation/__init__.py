# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Primary: centerline Cosserat tree + live tapered-tube containment.

"""Deformable vessel backends."""

from .centerline_data import CenterlineData, VesselConfig, VesselTransform
from .centerline_runtime import CenterlineDynamicsParams, CenterlineVesselRuntime
from .centerline_tree import (
    CenterlineTree,
    build_centerline_tree,
    build_spline_edge_topology,
    rooted_node_segments,
    transform_tree,
)
from .vessel_skinning import VesselSkinner

__all__ = [
    "CenterlineData",
    "CenterlineDynamicsParams",
    "CenterlineTree",
    "CenterlineVesselRuntime",
    "VesselConfig",
    "VesselSkinner",
    "VesselTransform",
    "build_centerline_tree",
    "build_spline_edge_topology",
    "rooted_node_segments",
    "transform_tree",
    "CosseratRod",
    "predict",
    "project",
    "finalize",
    "step",
]


def __getattr__(name: str):
    if name in ("CosseratRod", "predict", "project", "finalize", "step"):
        from . import pbd_cosserat

        return getattr(pbd_cosserat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
