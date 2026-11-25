# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read optional domain presentation metadata without importing a domain."""
def appearance(record):
    sidecar = getattr(record, "appearance", None)
    return (sidecar.visual or {}) if sidecar is not None else {}
