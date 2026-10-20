# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_physics.loaders import LOADERS as GENERIC_LOADERS
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.loaders.tet_bundle import add_tet_bundle

from .presentation import with_presentation

LOADERS = {
    "tet_bundle": add_tet_bundle,
    "instrument": add_instrument,
}



LOADERS = {key: with_presentation(loader) for key, loader in {**GENERIC_LOADERS, **LOADERS}.items()}
