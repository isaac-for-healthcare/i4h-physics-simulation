# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The controllers' latest samples, from the XR renderer (render thread) to the ``openxr`` input sources (physics
thread): every headset frame ``render/xr.py`` publishes each hand in world coordinates (through the head anchor), the
sources read the latest. One headset per process, so a module-level board under a lock."""

from __future__ import annotations

import threading
import time

HANDS = ("left", "right")
STALE_S = 0.1  # a vibration command older than this is dropped (a stopped or hung servo never leaves a controller buzzing)
SAMPLE_STALE_S = 0.1  # a controller sample older than this reads inactive (a stalled render thread releases jaw, pedal, fire)
_lock = threading.Lock()
_samples: dict[str, dict] = {h: {"active": False} for h in HANDS}
_haptics: dict[str, tuple[float, float]] = {h: (0.0, 0.0) for h in HANDS}  # hand -> (amplitude 0..1, perf_counter)


def publish(hand: str, sample: dict) -> None:
    """``sample``: ``{active, position (3,) world, rotation xyzw (4,) world, trigger, squeeze, thumbstick, buttons}``."""
    with _lock:
        _samples[hand] = {**sample, "t": time.perf_counter()}


def read(hand: str) -> dict:
    """The hand's latest sample; inactive once it is older than ``SAMPLE_STALE_S`` (the input devices then keep the pose
    and release the jaw, the energy and the trigger, as for any device that stops reporting)."""
    with _lock:
        s = dict(_samples[hand])
    if s.get("active", False) and time.perf_counter() - s.get("t", 0.0) > SAMPLE_STALE_S:
        return {"active": False}
    return s


def set_haptic(hand: str, amplitude: float) -> None:
    """The servo's vibration command for a hand (``XrHandSource.io``), 0..1."""
    with _lock:
        _haptics[hand] = (min(max(float(amplitude), 0.0), 1.0), time.perf_counter())


def haptic(hand: str) -> float:
    """The hand's current vibration amplitude for the XR render thread; 0 once the command is older than ``STALE_S``."""
    with _lock:
        amplitude, at = _haptics[hand]
    return amplitude if time.perf_counter() - at <= STALE_S else 0.0


def clear() -> None:
    """No headset: every hand inactive and still (a released session, a closed renderer)."""
    with _lock:
        for h in HANDS:
            _samples[h] = {"active": False}
            _haptics[h] = (0.0, 0.0)
