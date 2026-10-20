# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


def copy(dst, src, stream=None):
    if dst.ptr != src.ptr:
        wp.copy(dst, src, stream=stream)
