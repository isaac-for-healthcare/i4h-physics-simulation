# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XR: the scene in a headset through OpenXR (pyopenxr) and a CloudXR runtime, the controllers as input devices, their
vibration as force feedback. Ported from omnisurg ``feature-cloudxr`` (``rendering/openxr.py``); needs the ``xr``
extra (pyopenxr, isaacteleop's CloudXR launcher) and a qualified SlangPy build with native Vulkan handles."""
