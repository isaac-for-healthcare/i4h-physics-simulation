# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-window panels of the standalone app: a Systems window built from every system's ``panel()``, a
Simulation window (pause, step, reset, env, tool, gravity, rate, substeps, iterations, graph, real time,
overrides, metrics), a separate Render window (camera, exposure, tonemap and postprocessing, folded at startup), a Procedure window for task status and, when the haptic servo runs, a Haptics window beside it (folded at start behind its ``expanded``
switch, since slangpy cannot start a window collapsed; the terminal status printout switch, the current force of each
instrument-box handle, output on/off, an on/off switch above each filter stage's slider, the conditioning values applied
to the servo directly, per-device status, and a ``handles`` group with the same controls for the handles' force). ``SimControls`` is the key handler the panel shares its state with
(P, N, Backspace, G).

Threading: the widgets are slangpy's retained ImGui tree on ``renderer.ui.screen``; their callbacks fire inside
``Context.begin_frame`` on the render thread while the frame is encoded but not submitted, so a callback only writes
a host mirror or a pending entry. ``update()``, called by the render loop before ``ring.acquire()``, applies them:
render-side changes directly (``set_env``, exposure), simulator changes through the worker's queue (a live array
is ``assign``ed from a copy of its mirror between steps; structural fields and the simulator's own are set and
``prepare()`` recaptures, one recapture in flight at a time), and refreshes what the widgets show from host data
(the drawn env's row after PgUp, the pause label, Tab's tool). The mirrors are read once at construction, before the
physics thread starts: the render thread never reads a device array afterwards.

Overrides: ``save`` writes what differs from the values at construction (the drawn env's row) as a small YAML;
``load_overrides(sim, data)`` applies such a file before ``prepare()`` (the CLI's ``--overrides``) or inside an edit.
omnisurg's UI (``rendering/ui/*``) reflected the config schema into widgets and wrote registry overrides; ours
binds the knobs each system lists, and the file names systems by their YAML id and knobs by attribute.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from i4h_isaaclab.camera.camera import key_event
from i4h_physics.stepper import Stepper
from i4h_physics.systems import SYSTEMS
from i4h_physics.systems.base import System
from i4h_standalone.app.haptics import FLAGS as HAPTIC_FLAGS
from i4h_standalone.app.haptics import NUMERIC as HAPTIC_NUMERIC
from i4h_standalone.app.haptics import RANGES as HAPTIC_RANGES
from i4h_standalone.app.panel_adapters import Knob, panel
from i4h_standalone.render.post import RANGES as POST_RANGES
from i4h_standalone.render.post import SKIP as POST_SKIP
from i4h_standalone.render.post import TONEMAPS, Post

SIM_FIELDS = Stepper.structural  # ("dt", "substeps", "iterations", "cuda_graph")
RENDER_ATTRS = {"spheres": "debug_spheres", "exposure": "exposure", "tonemap": "tonemap"}  # pending key -> renderer attribute
HAPTICS_FOLDED_HEIGHT = 60.0  # the title bar and the expanded switch
METRIC_SCOPES = ("gpu", "submit", "interval", "frame.age", "frame.draw")



class SimControls:
    """P pause, N step, Backspace reset all, G gravity; the Simulation panel shares the gravity state."""

    def __init__(self, worker, gravity):
        self.worker = worker
        self.gravity = np.asarray(gravity, np.float32).reshape(3)
        self.gravity_on = True

    def set_gravity(self, on: bool) -> None:
        self.gravity_on = bool(on)
        g = self.gravity if self.gravity_on else np.zeros(3, np.float32)
        self.worker.edit(lambda sim: sim.scene.gravity.assign(np.tile(g, (sim.scene.envs, 1))))

    def handle_key(self, event) -> None:
        name, press, _, _ = key_event(event)
        if not press:
            return
        if name == "p":
            self.worker.pause()
        elif name == "n":
            self.worker.step_once()
        elif name == "backspace":
            self.worker.reset()
        elif name == "g":
            self.set_gravity(not self.gravity_on)


@dataclass
class Row:
    """A knob's host mirror ``(E,)`` or ``(E, n)`` and its slider per column (structural knobs: one widget, no mirror)."""

    system: System
    knob: Knob
    mirror: np.ndarray | None
    initial: object = None  # the mirror (or the structural value) at construction: the overrides baseline
    widgets: list = field(default_factory=list)
    mixed_widgets: list = field(default_factory=list)


def kind_of(system) -> str:
    return next((k for k, cls in SYSTEMS.items() if cls is type(system)), type(system).__name__)


def load_overrides(sim, data: dict) -> None:
    """Apply an overrides mapping: ``sim`` fields, then per system id ``enabled``, knob values (all envs) and
    structural fields. Call before ``prepare()`` or inside a worker edit that ends with it. Unknown names raise."""
    data = check_overrides(sim, data)  # every name and shape first: a bad file must not half-apply
    for name, value in dict(data.get("sim") or {}).items():
        setattr(sim, name, type(getattr(sim, name))(value))
    from i4h_physics.regions import region_override_tables
    regional = region_override_tables(sim, data.get("regions"))
    by_id = {s.id: s for s in sim.systems}
    E = sim.scene.envs
    for sid, values in dict(data.get("systems") or {}).items():
        s = by_id[sid]
        knobs = {k.label: k for k in panel(s)}
        for label, value in dict(values or {}).items():
            if label == "enabled":
                s.enabled = bool(value)
            elif knobs[label].array is not None:
                arr = np.asarray(value, np.float32)
                knobs[label].array.assign(np.ascontiguousarray(np.broadcast_to(arr, (E,) + arr.shape)))
            else:
                setattr(s, knobs[label].field, type(getattr(s, knobs[label].field))(value))

    for props, tables in regional:
        for name, table in tables.items():
            props.tables[name].assign(table)


def check_overrides(sim, data, mirrors=None) -> dict:
    """Validate an overrides mapping against the simulator's systems and knobs (no writes): names, array shapes,
    finite values, structural types and ranges (``dt > 0``, counts at least 1, int knobs within lo..hi)."""
    data = dict(data or {})
    from i4h_standalone.app.panel_adapters import canonical_knob
    canonical, seen = {}, {}
    known = {s.id for s in sim.systems}
    for sid, values in dict(data.get('systems') or {}).items():
        if sid not in known:
            raise ValueError(f'overrides.systems: unknown system {sid!r}')
        for label, value in dict(values or {}).items():
            target, name = canonical_knob(sim, sid, label)
            key = target, name
            if key in seen and not np.array_equal(seen[key], value):
                raise ValueError(f'conflicting overrides for {target}/{name}')
            seen[key] = value
            canonical.setdefault(target, {})[name] = value
    if 'systems' in data:
        data['systems'] = canonical
    unknown = set(data) - {"sim", "systems", "regions"}
    if unknown:
        raise ValueError(f"overrides: unknown top-level keys {sorted(unknown)}; expected sim, systems")
    for name, value in dict(data.get("sim") or {}).items():
        if name not in SIM_FIELDS:
            raise ValueError(f"overrides.sim: unknown field {name!r}; known {SIM_FIELDS}")
        if name == "cuda_graph":
            if not isinstance(value, bool):
                raise ValueError("overrides.sim.cuda_graph must be a bool")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"overrides.sim.{name} must be a finite number, got {value!r}")
        elif name == "dt" and value <= 0.0:
            raise ValueError("overrides.sim.dt must be positive")
        elif name != "dt" and int(value) < 1:
            raise ValueError(f"overrides.sim.{name} must be at least 1")
    by_id = {s.id: s for s in sim.systems}
    for sid, values in dict(data.get("systems") or {}).items():
        if sid not in by_id:
            raise ValueError(f"overrides.systems: unknown system {sid!r}; known {sorted(by_id)}")
        knobs = {k.label: k for k in panel(by_id[sid])}
        for label, value in dict(values or {}).items():
            if label == "enabled":
                if not isinstance(value, bool):
                    raise ValueError(f"overrides.systems.{sid}.enabled must be a bool")
                continue
            if label not in knobs:
                raise ValueError(f"overrides.systems.{sid}: unknown knob {label!r}; known {sorted(knobs)} and enabled")
            k = knobs[label]
            if k.scope is not None:
                raise ValueError("regional overrides use the regions section")
            if k.array is not None:
                arr = np.asarray(value, np.float64)
                if arr.shape != k.array.shape[1:] or not np.isfinite(arr).all():
                    raise ValueError(f"overrides.systems.{sid}.{label}: expected finite values of shape {k.array.shape[1:]}")
            elif isinstance(getattr(by_id[sid], k.field), bool):
                if not isinstance(value, bool):
                    raise ValueError(f"overrides.systems.{sid}.{label} must be a bool")
            elif not isinstance(value, int) or isinstance(value, bool) or not k.lo <= value <= k.hi:
                raise ValueError(f"overrides.systems.{sid}.{label} must be an int in [{int(k.lo)}, {int(k.hi)}], got {value!r}")
    from i4h_physics.regions import region_override_tables
    region_override_tables(sim, data.get("regions"), mirrors)
    return data


class Panels:
    def __init__(self, worker, renderer, controls: SimControls, hub=None, *, timings=None, overrides_path=None, loaded=None,
                 ui=None, screen=None, haptics=None, procedure=None):
        """Build the windows on ``screen`` (``renderer.ui.screen``) with the widget classes of ``ui`` (``renderer.spy.ui``);
        tests pass fakes. Runs on the main thread before the physics thread starts (reads every live knob once).
        ``haptics``: the ``HapticsDriver`` when the servo runs (adds the Haptics window)."""
        self.worker, self.renderer, self.controls, self.hub, self.timings = worker, renderer, controls, hub, timings
        self.procedure = procedure
        self.haptics = haptics
        self.sim = sim = worker.sim
        self.ui = ui if ui is not None else renderer.spy.ui
        self.screen = screen if screen is not None else renderer.ui.screen
        self.env = int(renderer.env)
        self.all_envs = True
        self.overrides_path = str(overrides_path or "overrides.yaml")
        self.loaded = dict(loaded or {})  # what --overrides applied: saved again even when unchanged
        self.pending: dict = {}
        self.pending_knobs: dict[int, Row] = {}
        self.pending_structural: dict[tuple, object] = {}
        self.recapturing = False  # a structural edit is queued or running: later ones wait
        self.rows: list[Row] = []
        self.pending_regions = []
        self.workspace_widgets = []
        self.readout_widgets = {}
        self.workspace_window = None
        self.region_index = 0
        self.region_body_index = 0
        self.rows_by_system: dict[int, list[Row]] = {id(s): [] for s in sim.systems}
        self.enabled_boxes: dict[int, object] = {}
        self.initial_sim = {name: getattr(sim, name) for name in SIM_FIELDS}
        self.initial_enabled = {id(s): s.enabled for s in sim.systems}
        self._shown_enabled = dict(self.initial_enabled)
        self.status_text = ""
        width, height = float(getattr(renderer, "width", 1280)), float(getattr(renderer, "height", 720))
        self.systems_window = self._window("Systems", (width - 470.0, 10.0), (460.0, height - 20.0))
        self.sim_window = self._window("Simulation", (10.0, 10.0), (380.0, min(760.0, height - 20.0)))
        self.haptics_window = None
        self._build_systems()
        self._build_simulation()
        from i4h_standalone.app.workspace import resolve_workspace
        self.workspace = resolve_workspace(sim, getattr(sim, "workspace", getattr(sim.scene, "workspace", None)))
        if self.workspace is not None:
            self._build_workspace()
        if haptics is not None:  # beside the Simulation window, its height when expanded
            self.haptics_sizes = {True: (380.0, min(760.0, height - 20.0)), False: (380.0, HAPTICS_FOLDED_HEIGHT)}
            self.haptics_window = self._window("Haptics", (400.0, 10.0), self.haptics_sizes[False])
            self._build_haptics()
        self.render_sizes = {True: (380.0, min(760.0, height - 20.0)), False: (380.0, 60.0)}
        self.render_window = self._window("Render", (400.0, 80.0 if haptics is not None else 10.0), self.render_sizes[False])
        self._build_render()
        self.procedure_window = self.task_text = None
        if procedure is not None:
            self.procedure_window = self._window("Procedure", (400.0, 150.0 if haptics is not None else 80.0), (420.0, 300.0))
            self.task_text = self.ui.Text(self.procedure_window, "")

    @property
    def windows(self):
        return [w for w in (self.systems_window, self.sim_window, self.render_window, self.procedure_window, self.haptics_window, self.workspace_window) if w is not None]

    # construction -----------------------------------------------------------------------------------------------

    def _f2(self, x, y):
        spy = getattr(self.renderer, "spy", None)
        return spy.float2(float(x), float(y)) if spy is not None else (x, y)

    def _window(self, title, position, size):
        return self.ui.Window(self.screen, title, position=self._f2(*position), size=self._f2(*size))

    def _flags(self, log: bool):
        flags = getattr(self.ui, "SliderFlags", None)
        if flags is None:
            return 0
        return flags.always_clamp | (flags.logarithmic if log else flags.none)

    def _build_systems(self):
        """A list of the systems and, below it, the selected system's group: slangpy's Group cannot start collapsed,
        so one group is visible at a time and the rest are hidden."""
        win = self.systems_window
        self.all_env_box = self.ui.CheckBox(win, "edit all envs", True, self._set_all_envs)
        names = [f"{s.id or type(s).__name__}" + ("" if (s.id or type(s).__name__) == kind_of(s) else f" ({kind_of(s)})") for s in self.sim.systems]
        self.groups: list = []
        self.selected = 0
        self.ui.ListBox(win, "##systems", 0, lambda v: self._pend("show", int(v)), names, min(len(names), 14))  # ##: no label text
        for s, name in zip(self.sim.systems, names):
            sid = s.id or type(s).__name__
            group = self.ui.Group(win, name)
            group.visible = not self.groups  # the first system open, the others folded
            self.groups.append(group)
            self.enabled_boxes[id(s)] = self.ui.CheckBox(group, f"enabled##{sid}", bool(s.enabled),
                                                         lambda v, s=s: self._set_structural(s, "enabled", bool(v)))
            for knob in panel(s):
                if knob.array is not None:
                    mirror = np.array(knob.array.numpy(), dtype=np.float32)  # once, before the worker starts
                    row = Row(s, knob, mirror, mirror.copy())
                    for j, col in enumerate(knob.labels or (None,)):
                        text = knob.label if col is None else f"{knob.label} {col}"
                        value = float(mirror[self.env] if mirror.ndim == 1 else mirror[self.env, j])
                        row.mixed_widgets.append(self.ui.Text(group, ""))
                        row.widgets.append(self.ui.SliderFloat(group, f"{text}##{sid}", value, lambda v, row=row, j=j: self._set_knob(row, j, v),
                                                               float(knob.lo), float(knob.hi), knob.format, self._flags(knob.log)))
                else:
                    current = getattr(s, knob.field)
                    row = Row(s, knob, None, current)
                    cb = lambda v, s=s, f=knob.field, t=type(current): self._set_structural(s, f, t(v))  # noqa: E731
                    if isinstance(current, bool):
                        row.widgets.append(self.ui.CheckBox(group, f"{knob.label}##{sid}", current, cb))
                    else:
                        row.widgets.append(self.ui.SliderInt(group, f"{knob.label}##{sid}", int(current), cb, int(knob.lo), int(knob.hi)))
                self.rows.append(row)
                self.rows_by_system[id(s)].append(row)

    def _build_workspace(self):
        ui = self.ui
        self.workspace_window = win = self._window(self.workspace.get('title', 'Workspace'), (10.0, 10.0), (460.0, 680.0))
        self.sim_window.visible = False
        self.workspace_all_box = ui.CheckBox(win, 'edit all environments', True, self._set_all_envs)
        self.workspace_env_slider = ui.SliderInt(win, 'displayed environment', self.env, lambda v: self._pend('env', int(v)), 0, max(self.sim.scene.envs - 1, 0))
        self.region_bodies = self.workspace.get('region_bodies', [])
        self.region_choices = [(name, j) for name in self.region_bodies
                               for j in range(len(self.sim.scene.body(name).regions.names))]
        if self.region_choices:
            labels = [f"{name} / {self.sim.scene.body(name).regions.names[j]}" for name, j in self.region_choices]
            initial_choice = next((i for i, (name, j) in enumerate(self.region_choices)
                                   if np.any(self.sim.scene.body(name).regions.edge == j)), 0)
            ui.ComboBox(win, 'body / region', initial_choice, self._select_region, labels)
            from importlib.resources import files
            template_path = files('i4h_endoluminal') / 'data/vascular_templates.yaml'  # an uncalibrated UI preset, not case data
            self.vascular_templates = yaml.safe_load(template_path.read_text())['templates']
            ui.ComboBox(win, 'vascular template (uncalibrated)', 0, self._apply_template, list(self.vascular_templates))
        ui.Text(win, self.workspace.get('instructions', ''))
        for group_index, declaration in enumerate(self.workspace.get('groups', [])):
            group = ui.Group(win, declaration.get('title', 'Controls'))
            for knob_index, (sid, label) in enumerate(declaration.get('knobs', [])):
                row = next(r for r in self.rows if r.system.id == sid and r.knob.label == label)
                if row.mirror is None:
                    current = getattr(row.system, row.knob.field)
                    def cb(v, row=row, t=type(current)):
                        return self._set_structural(row.system, row.knob.field, t(v))
                    if isinstance(current, bool):
                        widget = ui.CheckBox(group, f'{label}##workspace-{group_index}-{knob_index}-{sid}', current, cb)
                    else:
                        widget = ui.SliderInt(group, f'{label}##workspace-{group_index}-{knob_index}-{sid}', current, cb, int(row.knob.lo), int(row.knob.hi))
                    self.workspace_widgets.append((row, 0, widget, ui.Text(group, '')))
                    continue
                for col in ([0] if row.knob.scope is not None else range(row.mirror.shape[1] if row.mirror.ndim > 1 else 1)):
                    caption = label.removeprefix('region_').replace('_', ' ')
                    if row.knob.scope is None and row.knob.labels:
                        caption += ' ' + str(row.knob.labels[col])
                    widget = ui.SliderFloat(group, f'{caption}##workspace-{group_index}-{knob_index}-{sid}-{col}', 0.0,
                        lambda v, row=row, col=col: self._set_knob(row, self.region_index if row.knob.scope else col, v),
                        float(row.knob.lo), float(row.knob.hi), row.knob.format, self._flags(row.knob.log))
                    mixed = ui.Text(group, '')
                    self.workspace_widgets.append((row, col, widget, mixed))
            for key in declaration.get('readouts', []):
                self.readout_widgets[tuple(key)] = ui.Text(group, f'{key[1]}: waiting for snapshot')
        self.derived_text = ui.Text(win, '')
        for action in self.workspace.get('actions', []):
            if action == 'camera':
                ui.ComboBox(win, 'camera', 0, lambda v: self._pend('camera', int(v)), ['free', 'tip'])
            else:
                ui.Button(win, action.replace('_', ' ').title(), lambda a=action: self._pend(a, True))
        ui.InputText(win, 'overrides path', self.overrides_path, self._set_path)
        self.workspace_status = ui.Text(win, '')
        if self.region_choices:
            self._select_region(initial_choice)

    def _select_region(self, index):
        if not self.region_choices:
            return
        name, self.region_index = self.region_choices[int(index)]
        self.region_body_index = self.region_bodies.index(name)
        if hasattr(self.renderer, 'select_region'):
            self.renderer.select_region(name, self.region_index)

    def _apply_template(self, index):
        name = self.region_bodies[self.region_body_index]
        rows = [r for r in self.rows if r.knob.scope == name]
        if not rows or 'region_stretch' not in {r.knob.label for r in rows}:
            self.status_text = 'Select a vessel region to apply a vascular template'
            return
        template = list(self.vascular_templates.values())[int(index)]
        props = self.sim.scene.body(name).region_properties
        tables = {r.knob.label.removeprefix('region_'): r.mirror.copy() for r in rows}
        ids = slice(None) if self.all_envs else [self.env]
        try:
            for prop, value in template.items():
                if prop not in tables:
                    raise ValueError(f'unknown template property {prop}')
                tables[prop][ids, self.region_index] = value
            props.validate(tables)
        except ValueError as exc:
            self.status_text = str(exc)
            return
        for row in rows:
            prop = row.knob.label.removeprefix('region_')
            if prop in template:
                self._set_knob(row, self.region_index, template[prop])

    def _refresh_workspace(self):
        if self.workspace_window is None:
            return
        self._show(self.workspace_all_box, 'value', self.all_envs)
        self._show(self.workspace_env_slider, 'value', self.env)
        body = self.region_bodies[self.region_body_index] if self.region_bodies else None
        for row, col, widget, mixed in self.workspace_widgets:
            widget.visible = mixed.visible = row.knob.scope in (None, body)
            widget.enabled = bool(row.system.enabled)
            if row.mirror is None:
                self._show(widget, 'value', getattr(row.system, row.knob.field))
                continue
            j = self.region_index if row.knob.scope is not None else col
            values = row.mirror if row.mirror.ndim == 1 else row.mirror[:, j]
            self._show(widget, 'value', float(values[self.env]))
            self._show(mixed, 'text', 'mixed environments' if self.all_envs and np.any(values != values[0]) else '')
        rows = {r.knob.label: r for r in self.rows if getattr(r.system, 'body', None) is not None
                and getattr(r.system.body, 'name', None) == body}
        if 'region_young_multiplier' in rows:
            young = rows['young'].mirror[self.env] * rows['region_young_multiplier'].mirror[self.env, self.region_index]
            torsion = rows['torsion'].mirror[self.env] * rows['region_torsion_multiplier'].mirror[self.env, self.region_index]
            self._show(self.derived_text, 'text', f'Young parameter: {young:.3e}; torsion parameter: {torsion:.3e} (uncalibrated)')
        else:
            self._show(self.derived_text, 'text', '')
        self._show(self.workspace_status, 'text', self.status_text)

    def publish_readouts(self, values):
        for key, widget in self.readout_widgets.items():
            if key in values:
                self._show(widget, 'text', f'{key[1]}: {values[key][self.env]:.6g}')

    def _build_simulation(self):
        ui, win, sim, E = self.ui, self.sim_window, self.sim, self.sim.scene.envs
        self.pause_button = ui.Button(win, "Pause", lambda: self._pend("pause", True))
        ui.Button(win, "Step", lambda: self._pend("step", True))
        ui.Button(win, "Reset all", lambda: self._pend("reset_all", True))
        ui.Button(win, "Reset env", lambda: self._pend("reset_env", True))
        self.env_slider = ui.SliderInt(win, "env", self.env, lambda v: self._pend("env", int(v)), 0, max(E - 1, 0))
        self.tool_combo = None
        if self.hub is not None and self.hub.order:
            index = self.hub.order.index(self.hub.active) if self.hub.active in self.hub.order else 0
            self.tool_combo = ui.ComboBox(win, "tool", index, lambda v: self._pend("tool", int(v)), list(self.hub.order))
        self.gravity_box = ui.CheckBox(win, "gravity", self.controls.gravity_on, lambda v: self._pend("gravity", bool(v)))
        self.hz_slider = ui.SliderInt(win, "physics Hz", int(round(1.0 / sim.dt)), lambda v: self._pend("hz", int(v)), 30, 240)
        self.substeps_slider = ui.SliderInt(win, "substeps", int(sim.substeps), lambda v: self._set_structural(sim, "substeps", int(v)), 1, 16)
        self.iterations_slider = ui.SliderInt(win, "iterations", int(sim.iterations), lambda v: self._set_structural(sim, "iterations", int(v)), 1, 32)
        self.graph_box = ui.CheckBox(win, "cuda graph", bool(sim.cuda_graph), lambda v: self._set_structural(sim, "cuda_graph", bool(v)))
        self.realtime_box = ui.CheckBox(win, "realtime", bool(self.worker.realtime), lambda v: self._pend("realtime", bool(v)))
        over = ui.Group(win, "overrides")
        ui.InputText(over, "path", self.overrides_path, self._set_path)
        ui.Button(over, "Save", lambda: self._pend("save", True))
        ui.Button(over, "Load", lambda: self._pend("load", True))
        self.status = ui.Text(over, "")
        self.metrics_text = ui.Text(ui.Group(win, "metrics"), "")

    def _build_render(self):
        """Rendering controls stay on the render thread; the window starts folded like Haptics."""
        ui, render = self.ui, self.render_window
        ui.CheckBox(render, "expanded", False, lambda v: self._pend("render_expanded", bool(v)))
        r = self.renderer
        ui.CheckBox(render, "spheres (X)", bool(getattr(r, "debug_spheres", False)), lambda v: self._pend("spheres", bool(v)))
        ui.SliderFloat(render, "exposure", float(getattr(r, "exposure", 1.0)), lambda v: self._pend("exposure", float(v)), 0.1, 8.0, "%.2f",
                       self._flags(True))
        ui.ComboBox(render, "tonemap", int(getattr(r, "tonemap", 0)), lambda v: self._pend("tonemap", int(v)), list(TONEMAPS))
        self.camera_combo = self.tip_combo = None
        rig = getattr(r, "rig", None)
        if rig is not None:  # free or tip camera and the tip's target (render/rig.py); the C key changes the mode behind the panel
            self.camera_combo = ui.ComboBox(render, "camera", 0 if r.camera_mode == "free" else 1, lambda v: self._pend("camera", int(v)), ["free", "tip"])
            self.tip_combo = ui.ComboBox(render, "tip target", rig.index, lambda v: self._pend("tip_target", int(v)), list(rig.names))
        post = getattr(r, "post", None)
        if post is not None:  # the post chain's toggles and knobs (render/post.py Post), applied on the render thread
            group = ui.Group(render, "post")
            for f in dataclasses.fields(Post):
                if f.name in POST_SKIP:
                    continue
                value = getattr(post, f.name)
                if isinstance(value, bool):
                    ui.CheckBox(group, f.name, value, lambda v, n=f.name: self._pend(("post", n), bool(v)))
                else:
                    lo, hi = POST_RANGES.get(f.name, (0.0, 1.0))
                    ui.SliderFloat(group, f.name, float(value), lambda v, n=f.name: self._pend(("post", n), float(v)), float(lo), float(hi),
                                   "%.3f", self._flags(False))
        self._fold_render(False)

    def _fold_render(self, expanded: bool) -> None:
        for widget in list(self.render_window.children)[1:]:
            widget.visible = expanded
        self.render_window.size = self._f2(*self.render_sizes[expanded])

    def _build_haptics(self):
        """The servo's switches and conditioning (host values, applied to the servo directly in ``update()``), the status."""
        ui, win, servo = self.ui, self.haptics_window, self.haptics.servo
        ui.CheckBox(win, "expanded", False, lambda v: self._pend(("haptics_ui", "expanded"), bool(v)))  # the fold: the window's first child
        ui.CheckBox(win, "console printout", bool(self.haptics.console), lambda v: self._pend(("haptics_ui", "console"), bool(v)))
        self.handle_force_sliders = {dev: ui.SliderFloat(win, f"{dev} handle", 0.0, lambda v: None, 0.0, float(servo.handle_ceilings[dev]),
                                                         "%.2f N", self._flags(False))
                                     for dev in servo.handle_forces()}  # display only: refreshed every frame from the servo
        self.haptics_box = ui.CheckBox(win, "output", bool(servo.enabled), lambda v: self._pend(("haptics", "enabled"), bool(v)))
        params = servo.params()
        self.haptics_flags = {}
        for name in HAPTIC_NUMERIC:  # axis_gains stays a YAML setting
            flag = HAPTIC_FLAGS.get(name)
            if flag is not None:  # the stage's on/off switch sits above its slider; off bypasses the stage, the value is kept
                self.haptics_flags[flag] = ui.CheckBox(win, f"{flag} on", bool(getattr(params, flag)), lambda v, f=flag: self._pend(("haptics", f), bool(v)))
            lo, hi, log = HAPTIC_RANGES[name]
            if name == "max_force_n":
                hi = servo.force_ceiling()
            ui.SliderFloat(win, name, float(getattr(params, name)), lambda v, n=name: self._pend(("haptics", n), float(v)), float(lo), float(hi),
                           "%.3f", self._flags(log))
        self.haptics_text = ui.Text(win, servo.status_text())
        self.handle_flags = {}
        self.handles_box = None
        if servo.has_handles():  # the instrument-box handles: their own switch, gain, cap and stage switches (the jaw load's conditioner)
            grp = ui.Group(win, "handles")
            self.handles_box = ui.CheckBox(grp, "handle output", bool(servo.handles_enabled), lambda v: self._pend(("haptics_handle", "enabled"), bool(v)))
            hp = servo.handle_params()
            for name in HAPTIC_NUMERIC:
                flag = HAPTIC_FLAGS.get(name)
                if flag is not None:
                    self.handle_flags[flag] = ui.CheckBox(grp, f"{flag} on##handle", bool(getattr(hp, flag)),
                                                          lambda v, f=flag: self._pend(("haptics_handle", f), bool(v)))
                lo, hi, log = HAPTIC_RANGES[name]
                hi = servo.force_ceiling(handle=True) if name == "max_force_n" else max(hi, float(getattr(hp, name)))
                ui.SliderFloat(grp, f"{name}##handle", float(getattr(hp, name)), lambda v, n=name: self._pend(("haptics_handle", n), float(v)),
                               float(lo), float(hi), "%.3f", self._flags(log))
        self._haptics_shown = 0.0
        self._fold_haptics(False)

    def _fold_haptics(self, expanded: bool) -> None:
        win = self.haptics_window
        for w in list(win.children)[1:]:
            w.visible = expanded
        win.size = self._f2(*self.haptics_sizes[expanded])

    # callbacks (render thread, inside begin_frame): host writes only ------------------------------------------

    def _pend(self, name, value):
        self.pending[name] = value

    def _set_all_envs(self, value):
        self.all_envs = bool(value)

    def _set_path(self, value):
        self.overrides_path = str(value)

    def _set_knob(self, row: Row, j: int, value):
        v = float(value)
        if row.knob.scope is not None:
            props = self.sim.scene.body(row.knob.scope).region_properties
            tables = {r.knob.label.removeprefix('region_'): r.mirror.copy() for r in self.rows
                      if r.knob.scope == row.knob.scope}
            name = row.knob.label.removeprefix('region_')
            ids = np.arange(self.sim.scene.envs) if self.all_envs else np.array([self.env])
            tables[name][ids, j] = v
            try:
                props.validate(tables)
            except ValueError as exc:
                self.status_text = str(exc)
                return
            self.pending_regions.append((row, j, v, None if self.all_envs else [self.env]))
            row.mirror = tables[name]
            return
        if row.mirror.ndim == 1:
            if self.all_envs:
                row.mirror[:] = v
            else:
                row.mirror[self.env] = v
        elif self.all_envs:
            row.mirror[:, j] = v
        else:
            row.mirror[self.env, j] = v
        self.pending_knobs[id(row)] = row

    def _set_structural(self, obj, name, value):
        self.pending_structural[(id(obj), name)] = (obj, value)

    def handle_key(self, event) -> None:
        name, press, repeat, _ = key_event(event)
        if press and not repeat and name == "f1":
            shown = not self.systems_window.visible
            for win in self.windows:
                win.visible = shown and not (self.workspace_window is not None and win is self.sim_window)

    # render thread, before ring.acquire() --------------------------------------------------------------------

    def publish_procedure(self, row, progress=None, history=None) -> None:
        """Called after physics snapshot publication, before drawing this frame's UI."""
        if self.task_text is not None:
            from i4h_standalone.render.procedure import task_text

            self.task_text.text = task_text(self.procedure, row, progress, history)

    def update(self) -> None:
        p, self.pending = self.pending, {}
        worker, r = self.worker, self.renderer
        if p.get("load"):
            self.load()
        if "show" in p and p["show"] != self.selected:
            self.groups[self.selected].visible = False
            self.selected = p["show"] % len(self.groups)
            self.groups[self.selected].visible = True
        if "env" in p and p["env"] != r.env:
            r.set_env(p["env"])
        if "tool" in p and self.hub is not None:
            self.hub.active = self.hub.order[p["tool"] % len(self.hub.order)]
        for key, attr in RENDER_ATTRS.items():
            if key in p:
                setattr(r, attr, p[key])
        if "camera" in p:
            r.set_camera_mode(("free", "tip")[p["camera"] % 2])
        if "tip_target" in p:
            r.rig.select(p["tip_target"])
        for key, value in p.items():
            if isinstance(key, tuple) and key[0] == "post":
                setattr(r.post, key[1], value)
            elif isinstance(key, tuple) and key[0] == "haptics":  # host values the servo reads once per tick
                if key[1] == "enabled":
                    self.haptics.servo.enabled = bool(value)
                else:
                    self.haptics.servo.set_param(key[1], value)
            elif isinstance(key, tuple) and key[0] == "haptics_handle":  # the handles group: the jaw load's conditioner
                if key[1] == "enabled":
                    self.haptics.servo.handles_enabled = bool(value)
                else:
                    self.haptics.servo.set_handle_param(key[1], value)
            elif isinstance(key, tuple) and key[0] == "haptics_ui":
                if key[1] == "expanded":
                    self._fold_haptics(bool(value))
                else:
                    self.haptics.console = bool(value)
        if "render_expanded" in p:
            self._fold_render(p["render_expanded"])
        if "gravity" in p:
            self.controls.set_gravity(p["gravity"])
        if p.get("pause"):
            worker.pause()
        if p.get("step"):
            worker.step_once()
        if p.get("reset_all"):
            worker.reset()
        if p.get("reset_env"):
            worker.reset([self.env])
        if "realtime" in p:
            worker.set_realtime(p["realtime"])
        if "hz" in p:
            self._set_structural(self.sim, "dt", 1.0 / float(p["hz"]))
        if p.get('restore_defaults'):
            body = self.sim.scene.body(self.region_bodies[self.region_body_index])
            ids = None if self.all_envs else [self.env]
            def restore(sim, body=body, ids=ids):
                body.region_properties.restore(ids)
                for row in self.rows:
                    if row.knob.scope == body.name:
                        row.mirror = body.region_properties.tables[row.knob.label.removeprefix('region_')].numpy().copy()
            worker.edit(restore)
        for row, j, value, ids in self.pending_regions:
            def edit_region(sim, row=row, j=j, value=value, ids=ids):
                props = sim.scene.body(row.knob.scope).region_properties
                props.edit({row.knob.label.removeprefix('region_'): value}, row.knob.labels[j], ids)
                row.mirror = props.tables[row.knob.label.removeprefix('region_')].numpy().copy()
            worker.edit(edit_region)
        self.pending_regions.clear()
        for row in self.pending_knobs.values():
            copy = row.mirror.copy()
            worker.edit(lambda sim, a=row.knob.array, c=copy: a.assign(c))
        self.pending_knobs.clear()
        if self.pending_structural and not self.recapturing:
            items = [(obj, name, value) for (_, name), (obj, value) in self.pending_structural.items()]
            self.pending_structural.clear()
            self.recapturing = True

            def recapture(sim, items=items):
                try:
                    for obj, name, value in items:
                        setattr(obj, name, value)
                    sim.prepare()
                finally:
                    self.recapturing = False

            worker.edit(recapture)
        if p.get("save"):
            self.save()
        self._refresh()
        self._show_mirrors()
        self._refresh_workspace()

    def _show(self, widget, attr: str, value) -> None:
        """Write a widget attribute when it differs from what the widget holds (a user edit changes that behind any cache)."""
        if getattr(widget, attr) != value:
            setattr(widget, attr, value)

    def _refresh(self) -> None:
        """What the widgets show, from host data only."""
        r, sim, worker, show = self.renderer, self.sim, self.worker, self._show
        show(self.all_env_box, "value", self.all_envs)
        if int(r.env) != self.env:
            self.env = int(r.env)
            show(self.env_slider, "value", self.env)
            self._show_mirrors()
        for s in sim.systems:
            if self._shown_enabled[id(s)] != s.enabled:
                self._shown_enabled[id(s)] = s.enabled
                show(self.enabled_boxes[id(s)], "value", bool(s.enabled))
                for row in self.rows_by_system[id(s)]:
                    for w in row.widgets:
                        show(w, "enabled", bool(s.enabled))
        show(self.pause_button, "label", "Resume" if worker.paused else "Pause")
        if self.tool_combo is not None and self.hub.active in self.hub.order:
            show(self.tool_combo, "value", self.hub.order.index(self.hub.active))
        if not self.pending_structural and not self.recapturing:  # a load or a recapture landed: show the live values
            show(self.hz_slider, "value", int(round(1.0 / sim.dt)))
            show(self.substeps_slider, "value", int(sim.substeps))
            show(self.iterations_slider, "value", int(sim.iterations))
            show(self.graph_box, "value", bool(sim.cuda_graph))
            for row in self.rows:
                if row.mirror is None:
                    show(row.widgets[0], "value", getattr(row.system, row.knob.field))
        show(self.gravity_box, "value", bool(self.controls.gravity_on))
        show(self.realtime_box, "value", bool(worker.realtime))
        if self.camera_combo is not None:
            show(self.camera_combo, "value", 0 if r.camera_mode == "free" else 1)
            show(self.tip_combo, "value", r.rig.index)
        show(self.status, "text", self.status_text)
        if self.haptics is not None:
            show(self.haptics_box, "value", bool(self.haptics.servo.enabled))
            params = self.haptics.servo.params()
            for flag, box in self.haptics_flags.items():
                show(box, "value", bool(getattr(params, flag)))
            if self.handles_box is not None:
                show(self.handles_box, "value", bool(self.haptics.servo.handles_enabled))
                hp = self.haptics.servo.handle_params()
                for flag, box in self.handle_flags.items():
                    show(box, "value", bool(getattr(hp, flag)))
            if self.handle_force_sliders:
                hp = self.haptics.servo.handle_params()
                for dev, force in self.haptics.servo.handle_forces().items():
                    slider = self.handle_force_sliders[dev]
                    show(slider, "max", float(hp.max_force_n))
                    show(slider, "value", float(force))
            now = time.perf_counter()
            if now - self._haptics_shown >= 1.0:
                self._haptics_shown = now
                show(self.haptics_text, "text", self.haptics.servo.status_text())

    def _show_mirrors(self) -> None:
        for row in self.rows:
            if row.mirror is None:
                continue
            for j, w in enumerate(row.widgets):
                self._show(w, "value", float(row.mirror[self.env] if row.mirror.ndim == 1 else row.mirror[self.env, j]))
                values = row.mirror if row.mirror.ndim == 1 else row.mirror[:, j]
                self._show(row.mixed_widgets[j], "text", "mixed environments" if self.all_envs and np.any(values != values[0]) else "")

    def set_metrics(self, text: str) -> None:
        self.metrics_text.text = str(text)

    # overrides ----------------------------------------------------------------------------------------------

    def overrides(self) -> dict:
        """What differs from the values at construction (the drawn env's row), plus what a loaded file set."""
        sim = self.sim
        out: dict = {"sim": {}, "systems": {}}
        for name in SIM_FIELDS:
            value = getattr(sim, name)
            if value != self.initial_sim[name] or name in (self.loaded.get("sim") or {}):
                out["sim"][name] = value
        for s in sim.systems:
            sid = s.id or type(s).__name__
            was = dict((self.loaded.get("systems") or {}).get(sid) or {})
            entry: dict = {}
            if s.enabled != self.initial_enabled[id(s)] or "enabled" in was:
                entry["enabled"] = bool(s.enabled)
            for row in self.rows_by_system[id(s)]:
                label = row.knob.label
                if row.knob.scope is not None:
                    continue
                if row.mirror is not None:
                    value = row.mirror[self.env]
                    if label in was or not np.array_equal(value, row.initial[self.env]):
                        entry[label] = float(value) if value.ndim == 0 else [float(v) for v in value]
                else:
                    value = getattr(s, row.knob.field)
                    if label in was or value != row.initial:
                        entry[label] = value
            if entry:
                out["systems"][sid] = entry
        regional = {}
        for row in self.rows:
            if row.knob.scope is None:
                continue
            sid = row.system.id
            for j, region in enumerate(row.knob.labels):
                values = row.mirror[:, j]
                value = float(values[0]) if np.all(values == values[0]) else values.tolist()
                regional.setdefault(row.knob.scope, {}).setdefault(sid, {}).setdefault(region, {})[row.knob.label.removeprefix('region_')] = value
        if regional:
            out['regions'] = regional
        return {k: v for k, v in out.items() if v}

    def save(self) -> None:
        try:
            data = self.overrides()
            Path(self.overrides_path).write_text(yaml.safe_dump(data, sort_keys=False) if data else "{}\n")
            self.status_text = f"saved {len(data.get('systems', {}))} systems to {self.overrides_path} ({time.strftime('%H:%M:%S')})"
        except Exception as exc:  # noqa: BLE001 - the panel reports, the app keeps running
            self.status_text = f"save failed: {exc}"

    def load(self) -> None:
        """Parse on the render thread, update the mirrors, apply everything in one edit ending in prepare()."""
        try:
            data = check_overrides(self.sim, yaml.safe_load(Path(self.overrides_path).read_text()),
                                   {(r.system.id, r.knob.label): r.mirror for r in self.rows if r.mirror is not None})
        except Exception as exc:  # noqa: BLE001
            self.status_text = f"load failed: {exc}"
            return
        for sid, values in dict(data.get("systems") or {}).items():
            for row in self.rows:
                if (row.system.id or type(row.system).__name__) == sid and row.knob.label in values and row.mirror is not None:
                    row.mirror[:] = np.asarray(values[row.knob.label], np.float32)
        from i4h_physics.regions import region_override_tables
        for props, tables in region_override_tables(self.sim, data.get('regions'),
                {(r.system.id, r.knob.label): r.mirror for r in self.rows if r.mirror is not None}):
            for row in self.rows:
                if row.knob.scope is not None and self.sim.scene.body(row.knob.scope).region_properties is props:
                    row.mirror = tables[row.knob.label.removeprefix('region_')].copy()
        self.loaded = data
        self.recapturing = True

        def apply(sim, data=data):
            try:
                load_overrides(sim, data)
                sim.prepare()
            except Exception as exc:  # noqa: BLE001 - validated data should not get here; report instead of killing physics
                self.status_text = f"load failed: {exc}"
                sim.prepare()  # whatever was applied is captured consistently
            finally:
                self.recapturing = False

        self.worker.edit(apply)
        self._show_mirrors()
        self.status_text = f"loaded {self.overrides_path}"

    def close(self) -> None:
        for win in self.windows:
            try:
                self.screen.remove_child(win)
            except Exception:  # noqa: BLE001 - already gone
                pass
        self.rows.clear()
        self.enabled_boxes.clear()
