
from i4h_standalone.app.panel_adapters import panel

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional app presentation resolved against existing system panels at startup."""
ACTIONS = ('pause', 'step', 'reset_env', 'reset_all', 'restore_defaults', 'save', 'load', 'camera')


def resolve_workspace(sim, declaration):
    if declaration is None:
        return None
    if not isinstance(declaration, dict) or set(declaration) - {'title', 'region_bodies', 'groups', 'instructions', 'actions'}:
        raise ValueError('ui: invalid workspace declaration')
    if not isinstance(declaration.get('title', 'Workspace'), str):
        raise ValueError('ui.title must be a string')
    if 'restore_defaults' in declaration.get('actions', []) and not declaration.get('region_bodies'):
        raise ValueError('ui: restore_defaults requires a region body selection')
    import copy

    from i4h_standalone.app.panel_adapters import canonical_knob
    declaration = copy.deepcopy(declaration)
    for group in declaration.get('groups', []):
        if isinstance(group, dict):
            group['knobs'] = [list(canonical_knob(sim, sid, name)) for sid, name in group.get('knobs', [])]
    by_id = {s.id: s for s in sim.systems}
    for name in declaration.get('region_bodies', []):
        if sim.scene.body(name).region_properties is None:
            raise ValueError(f'ui: body {name} has no regional properties')
    for group in declaration.get('groups', []):
        if not isinstance(group, dict) or set(group) - {'title', 'knobs', 'readouts'}:
            raise ValueError('ui: invalid group')
        for sid, name in group.get('knobs', []):
            if sid not in by_id or name not in {k.label for k in panel(by_id[sid])}:
                raise ValueError(f'ui: unknown knob ({sid}, {name})')
        for sid, name in group.get('readouts', []):
            if sid not in by_id or name not in by_id[sid].readouts():
                raise ValueError(f'ui: unknown readout ({sid}, {name})')
    if any(a not in ACTIONS for a in declaration.get('actions', [])):
        raise ValueError('ui: unknown app action')
    return declaration
