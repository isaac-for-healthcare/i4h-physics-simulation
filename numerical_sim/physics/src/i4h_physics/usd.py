# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD for asset loaders. Standalone installs get ``pxr`` from ``usd-core`` (the ``usd`` extra); inside Isaac Lab it
comes from Lab's own USD (``usd-exchange``), so the packages do not require ``usd-core``: both ship ``pxr``."""


def usd():
    """``(Usd, UsdGeom)``, or a ``ModuleNotFoundError`` that says where ``pxr`` comes from."""
    try:
        from pxr import Usd, UsdGeom
    except ModuleNotFoundError as error:
        if error.name != "pxr":
            raise
        raise ModuleNotFoundError("USD assets need pxr: install the package's [usd] extra (usd-core), or run inside "
                                  "Isaac Lab, which provides USD", name="pxr") from error
    return Usd, UsdGeom
