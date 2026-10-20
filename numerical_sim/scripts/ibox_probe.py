# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument-box probe: a small ImGui window that shows every channel of the IBController (opening, opto force, opto
voltage, motor PWM, handle id, the force sent) and drives one chosen channel with ``max force × slider`` (N), to find
which channels carry a handle's motor. Every other channel is sent 0, so switching channels releases the previous one.

    uv run --with /path/to/follou-sdk python scripts/ibox_probe.py [--device 0] [--channels 6] [--rate 1000]

All box I/O runs on one poll thread at ``--rate`` Hz (the SDK's subscriptions count updates: PWM refreshes every 41,
ids every 997); the window only reads its snapshot and writes the command. The probe owns the box's serial port, so it
cannot run alongside ``i4h-sim``. Closing the window zeroes every channel. The slider starts at 0 and the max force
field at 0.5 N; positive pushes the handle open (device qualification evidence retained privately).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from i4h_standalone.input.avatar import InstrumentBox


class Probe(threading.Thread):
    def __init__(self, box: InstrumentBox, rate_hz: float = 1000.0):
        super().__init__(daemon=True)
        from follou_sdk.devices.ibcontroller import Cmd

        self._opto_command = Cmd.GET_OPTO_VOLTAGES
        self.box, self.period = box, 1.0 / float(rate_hz)
        self.lock = threading.Lock()
        self.channel, self.max_n, self.scale = 0, 0.5, 0.0  # the command, written by the window
        self.gain = None  # (channel, kp, kd) pending for the poll thread
        self.snap = None
        self.hz = 0.0  # the measured poll rate
        self.stop = threading.Event()

    def tick(self) -> None:
        with self.lock:
            channel, force, gain, self.gain = self.channel, self.max_n * self.scale, self.gain, None
        if gain is not None:
            self.box.controller.set_loop_gain(*gain)
        for c in range(self.box.channels):
            self.box.queue_handle_force(c, force if c == channel else 0.0)
        self.box.poll()
        volts = [float(self.box.controller.get_return(self._opto_command, c)) for c in range(self.box.channels)]
        snap = {"active": self.box.active, "openings": self.box.openings.copy(), "measured": self.box.measured.copy(),
                "volts": volts, "pwm": self.box.pwm.copy(), "ids": self.box.ids.copy(), "sent": self.box.queued.copy(),
                "pedals": self.box.pedals, "hz": self.hz}
        with self.lock:
            self.snap = snap

    def run(self) -> None:
        next_t = t0 = time.perf_counter()
        ticks = 0
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:  # a write to a box that went away: shown as not answering, retried next tick
                with self.lock:
                    if self.snap is not None:
                        self.snap = dict(self.snap, active=False)
            ticks += 1
            if next_t - t0 >= 1.0:
                self.hz, ticks, t0 = ticks / (time.perf_counter() - t0), 0, time.perf_counter()
            next_t += self.period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()


def gui(probe: Probe):
    from imgui_bundle import imgui

    def frame():
        with probe.lock:
            snap, channel, max_n, scale = probe.snap, probe.channel, probe.max_n, probe.scale
        if snap is None:
            imgui.text("waiting for the box...")
            return
        imgui.text(f"box {'connected' if snap['active'] else 'NOT ANSWERING'}   poll {snap['hz']:.0f} Hz   "
                   f"pedals {snap['pedals'][0]} {snap['pedals'][1]}")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("channels", 8, flags):
            for name in ("drive", "ch", "open %", "opto force", "opto V", "pwm", "id", "sent N"):
                imgui.table_setup_column(name)
            imgui.table_headers_row()
            for c in range(len(snap["openings"])):
                imgui.table_next_row()
                imgui.table_next_column()
                if imgui.radio_button(f"##drive{c}", channel == c):
                    channel = c
                row = (str(c), f"{100 * (1 - snap['openings'][c]):.0f}", f"{snap['measured'][c]:.3f}", f"{snap['volts'][c]:.3f}",
                       str(snap["pwm"][c]), str(snap["ids"][c]), f"{snap['sent'][c]:.3f}")
                for value in row:
                    imgui.table_next_column()
                    imgui.text(value)
            imgui.end_table()
        imgui.separator()
        imgui.set_next_item_width(120)
        _, max_n = imgui.input_float("max force (N)", max_n, 0.1, 0.5, "%.3f")
        imgui.set_next_item_width(300)
        _, scale = imgui.slider_float("scale", scale, 0.0, 1.0, "%.3f")
        imgui.same_line()
        if imgui.button("zero"):
            scale = 0.0
        imgui.text(f"channel {channel}: {max_n * scale:.3f} N")
        imgui.separator()
        _, frame.kp = imgui.input_float("kp", frame.kp, 0.0, 0.0, "%.3f")
        _, frame.kd = imgui.input_float("kd", frame.kd, 0.0, 0.0, "%.3f")
        gain = None
        if imgui.button(f"set loop gain on channel {channel}"):  # the SDK's start-up sets kp 4 on channels 0 and 2 only
            gain = (channel, frame.kp, frame.kd)
        with probe.lock:
            probe.channel, probe.max_n, probe.scale = channel, max_n, scale
            if gain is not None:
                probe.gain = gain

    frame.kp, frame.kd = 4.0, 0.0
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=0, help="IBController device index")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--rate", type=float, default=1000.0, help="poll rate (Hz)")
    args = parser.parse_args(argv)

    from imgui_bundle import immapp

    box = InstrumentBox.open(args.device, args.channels)
    probe = Probe(box, args.rate)
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)  # the poll thread cannot wait the default 5 ms for the GIL
    probe.start()
    try:
        immapp.run(gui(probe), window_title="instrument box probe", window_size=(720, 420), fps_idle=0, ini_disable=True)
    finally:
        probe.stop.set()
        probe.join(timeout=1.0)
        sys.setswitchinterval(switch_interval)
        if not probe.is_alive():
            box.close()  # zeroes every channel and releases the SDK manager
    return 0


if __name__ == "__main__":
    sys.exit(main())
