# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The private Newton capability needed during model authoring."""

import inspect

NEWTON_REVISION = "7f0c13af6e4dd75253709c31dc65c0edcaf22b44"


def require_model_finalizer(builder):
    """Check the bound registration call, without registering or running a callback."""
    hook = getattr(builder, "_add_custom_attribute_model_finalizer", None)
    message = (
        f"i4h_physics requires Newton @ {NEWTON_REVISION} (1.9.0.dev0): callable "
        "ModelBuilder._add_custom_attribute_model_finalizer(key, finalizer), with "
        "finalizer(builder, model, custom_attribute) called during model finalization"
    )
    if not callable(hook):
        raise RuntimeError(message)
    try:
        inspect.signature(hook).bind("i4h:capability_probe", lambda builder, model, attribute: None)
    except (TypeError, ValueError) as error:
        raise RuntimeError(message) from error
    return hook
