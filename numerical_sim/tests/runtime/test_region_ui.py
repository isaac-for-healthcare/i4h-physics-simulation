# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped workspace edits, mirror sharing, overrides and snapshot staging."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
import yaml
from helpers import needs_external_data, regional_case
from i4h_standalone.app.ui import Panels, SimControls, load_overrides
from i4h_standalone.app.worker import PhysicsWorker
from i4h_standalone.config import load_case
from i4h_standalone.render.snapshots import Snapshots
from test_ui import FAKE_UI, _Widget, drain

pytestmark = needs_external_data


def setup(tmp_path, real=False):
    case = str(regional_case(tmp_path))
    sim = load_case(case, envs=2)
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    if real:
        from i4h_isaaclab.composition import read_case
        from i4h_standalone.render.slang import SlangRenderer
        cfg = read_case(case)['render'].copy()
        cfg.pop('backend')
        renderer = SlangRenderer(sim, **dict(cfg, size=(480, 320), window=False))
        renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    else:
        renderer = SimpleNamespace(env=0, camera=None, debug_spheres=False, exposure=1, tonemap=0, width=1280, height=720)
        renderer.set_env = lambda env: setattr(renderer, 'env', int(env))
    worker = PhysicsWorker(sim, [], ring, steps=0, realtime=False)
    kwargs = {} if real else {'ui': FAKE_UI, 'screen': _Widget(None)}
    panels = Panels(worker, renderer, SimControls(worker, (0, 0, 0)), overrides_path=tmp_path/'regions.yaml', **kwargs)
    return sim, worker, renderer, panels, ring


def test_workspace_scoping_duplicate_mirrors_and_roundtrip(tmp_path):
    sim, worker, renderer, panels, ring = setup(tmp_path)
    try:
        event = SimpleNamespace(key=SimpleNamespace(name='f1'), is_key_press=lambda: True, is_key_release=lambda: False)
        panels.handle_key(event)
        panels.handle_key(event)
        assert panels.workspace_window.visible and not panels.sim_window.visible
        tip = panels.region_choices.index(('catheter', 3))
        panels._select_region(tip)
        panels._set_all_envs(False)
        panels.env = renderer.env = 1
        row, col, widget, mixed = next(item for item in panels.workspace_widgets if item[0].knob.label == 'region_curvature_x')
        widget.fire(5)
        panels.update()
        drain(worker)
        panels.update()
        assert row.widgets[3].value == widget.value == 5
        table = sim.scene.body('catheter').region_properties.tables['curvature_x'].numpy()
        np.testing.assert_array_equal(table[0], 0)
        assert table[1, 3] == 5 and not table[1, :3].any()
        panels._set_all_envs(True)
        panels.update()
        assert mixed.text == 'mixed environments'
        panels.save()
        data = yaml.safe_load((tmp_path/'regions.yaml').read_text())
        assert data['regions']['catheter']['catheter_material']['tip']['curvature_x'] == [0, 5]
        panels._pend('restore_defaults', True)
        panels.update()
        drain(worker)
        assert not sim.scene.body('catheter').region_properties.tables['curvature_x'].numpy().any()
        panels.load()
        drain(worker)
        panels.update()
        assert row.mirror[1, 3] == 5 and widget.value == 5
        sim.reset([1])
        assert sim.scene.body('catheter').region_properties.tables['curvature_x'].numpy()[1, 3] == 5
        # Invalid complete edit leaves both the host and device values alone.
        widget.fire(100)
        panels.update()
        drain(worker)
        assert row.mirror[1, 3] == 5
    finally:
        if worker.ident is not None:
            worker.stop()
        panels.close()


def test_readouts_are_completed_snapshot_copies(tmp_path):
    sim, worker, renderer, panels, ring = setup(tmp_path)
    try:
        contact = next(s for s in sim.systems if s.id == 'lumen_containment')
        contact.contact_depth.fill_(.125)
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        contact.contact_depth.fill_(.75)
        panels.publish_readouts(snap.readout_host)
        np.testing.assert_allclose(snap.readout_host[('lumen_containment', 'contact_depth')], .125)
        assert '0.125' in panels.readout_widgets[('lumen_containment', 'contact_depth')].text
    finally:
        if worker.ident is not None:
            worker.stop()
        panels.close()


@pytest.mark.skipif(os.environ.get('I4H_RENDER_TESTS') != '1', reason='offscreen Slang opt-in')
def test_offscreen_region_workspace_highlights_and_steering(tmp_path):
    pytest.importorskip('slangpy')
    sim, worker, renderer, panels, ring = setup(tmp_path, real=True)
    try:
        # Real retained widget tree with queued edits, both body highlights, steering and reset.
        panels._select_region(panels.region_choices.index(('catheter', 3)))
        row = next(r for r in panels.rows if r.system.id == 'catheter_material' and r.knob.label == 'region_curvature_y')
        panels._set_all_envs(False)
        panels._set_knob(row, 3, 5)
        panels.update()
        drain(worker)
        if worker.ident is not None:
            worker.stop()
        control = next(s for s in sim.systems if s.id == 'catheter_control')
        control.actions.fill_(.2)
        for _ in range(4):
            sim.step()
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        renderer.publish(snap)
        panels.publish_readouts(snap.readout_host)
        renderer.draw()
        catheter = renderer.read_frame()
        panels._select_region(panels.region_choices.index(('aorta', 1)))
        renderer.publish(snap)
        renderer.draw()
        vessel = renderer.read_frame()
        assert np.isfinite(catheter).all() and np.isfinite(vessel).all()
        assert np.count_nonzero(catheter != vessel) > 20
        assert [h.selected for h in renderer.region_highlights] == [-1, 1]
        sim.reset()
        assert not control.targets.numpy().any() and not control.controls.bend_offset.numpy().any()
        assert row.mirror[0, 3] == 5
        panels.save()
        load_overrides(sim, yaml.safe_load((tmp_path/'regions.yaml').read_text()))
        renderer.publish()
        renderer.draw()
    finally:
        if worker.ident is not None:
            worker.stop()
        panels.close()
        renderer.close()


def test_legacy_panel_load_saves_canonical_material_ids(tmp_path):
    sim, worker, renderer, panels, ring = setup(tmp_path)
    try:
        legacy = {'systems': {'tree': {'radius_weighting': True}}, 'regions': {'catheter': {
            'rod': {'tip': {'curvature_x': [0, 2], 'bend_multiplier': [1, 3]}}}}}
        (tmp_path/'regions.yaml').write_text(yaml.safe_dump(legacy))
        panels.load()
        drain(worker)
        panels.update()
        material = next(s for s in sim.systems if s.id == 'vessel_material')
        assert material.radius_weighting
        row = next(r for r in panels.rows if r.system.id == 'catheter_material' and r.knob.label == 'region_curvature_x')
        np.testing.assert_array_equal(row.mirror[:, 3], [0, 2])
        panels.save()
        saved = yaml.safe_load((tmp_path/'regions.yaml').read_text())
        assert saved['systems']['vessel_material']['radius_weighting'] is True
        assert 'tree' not in saved['systems']
        assert saved['regions']['catheter']['catheter_material']['tip']['curvature_x'] == [0, 2]
        assert 'rod' not in saved['regions']['catheter']
    finally:
        if worker.ident is not None:
            worker.stop()
        panels.close()
