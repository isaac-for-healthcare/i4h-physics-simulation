# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fail on use of another object's underscore-prefixed names in the package sources.

A reach is ``obj._name`` where ``obj`` is not ``self``, ``cls`` or ``super()``, a ``from module import _name``, or a
``getattr``/``hasattr``/``setattr`` with a ``"_name"`` string. Every reach must be listed in ``ALLOWED`` with its
reason, and every listed entry must still exist, so the list can only shrink deliberately.
"""
import ast
import sys
from pathlib import Path

PACKAGES = ("physics", "surgical", "endoluminal", "assessment", "isaaclab", "host")

# reason -> {source file relative to the workspace: names reached there}
ALLOWED = {'Newton WORLD row count on the pinned and current-stack assessment revisions': {'physics/src/i4h_physics/scene_storage.py': {'_attribute_frequency_count'}},
 'Newton internals at the pinned revision': {'physics/src/i4h_physics/newton_compat.py': {'_add_custom_attribute_model_finalizer'},
                                             'physics/src/i4h_physics/observables.py': {'_observable_fields'},
                                             'physics/src/i4h_physics/storage.py': {'_add_custom_attribute_model_finalizer',
                                                                                    '_attribute_frequency_count',
                                                                                    '_normalize_attribute_reference'},
                                             'isaaclab/src/i4h_isaaclab/newton_solver.py': {'_kinds'}},
 'Warp internals: array lifetime reference and dtype layout': {'physics/src/i4h_physics/commands.py': {'_ref'},
                                                               'physics/src/i4h_physics/continuation.py': {'_ref'},
                                                               'physics/src/i4h_physics/rod_storage.py': {'_shape_'},
                                                               'physics/src/i4h_physics/storage.py': {'_ref',
                                                                                                      '_shape_',
                                                                                                      '_wp_scalar_type_'},
                                                               'surgical/src/i4h_surgical/instrument_storage.py': {'_shape_'}},
 'Isaac Lab internals at the pinned revisions': {'isaaclab/src/i4h_isaaclab/env.py': {'_reset_idx'},
                                                 'isaaclab/src/i4h_isaaclab/versius/task.py': {'_reset_idx'},
                                                 'isaaclab/src/i4h_isaaclab/versius/qualification.py': {'_reset_idx'},
                                                 'isaaclab/src/i4h_isaaclab/lab30.py': {'_cfg',
                                                                                        '_contacts',
                                                                                        '_create_solver',
                                                                                        '_initialize_contacts',
                                                                                        '_invalidate_graph',
                                                                                        '_needs_collision_pipeline',
                                                                                        '_prepare_builder_for_finalize',
                                                                                        '_register_builder_attributes',
                                                                                        '_registers_builder_attributes_from_solver',
                                                                                        '_solver',
                                                                                        '_supports_contact_sensors',
                                                                                        '_supports_rigid_body_force_input',
                                                                                        '_use_single_state'},
                                                 'isaaclab/src/i4h_isaaclab/recording.py': {'_hdf5_data_group'}},
 'i4h attribute stored on a host or Newton object': {'physics/src/i4h_physics/authoring.py': {'_i4h_authoring'},
                                                     'isaaclab/src/i4h_isaaclab/lab30.py': {'__i4h_context'},
                                                     'isaaclab/src/i4h_isaaclab/newton_manager.py': {'__i4h_context'}},
 "own instance, set in the class's alternate constructor": {'physics/src/i4h_physics/bindings.py': {'_core',
                                                                                                    '_output_arrays',
                                                                                                    '_prepare',
                                                                                                    '_prepared',
                                                                                                    '_runtime',
                                                                                                    '_standalone'}}}


def private(name):
    return isinstance(name, str) and name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def own(node):
    """``self``, ``cls`` and ``super()`` reach the object's own (or inherited) names."""
    if isinstance(node, ast.Name):
        return node.id in ("self", "cls")
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"


def reaches(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and private(node.attr) and not own(node.value):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names if private(alias.name))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr", "setattr")
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and private(node.args[1].value)
              and not own(node.args[0])):
            found.add(node.args[1].value)
    return found


def scan(root):
    found = {}
    for package in PACKAGES:
        for path in sorted((root / package / "src").rglob("*.py")):
            names = reaches(ast.parse(path.read_text()))
            if names:
                found[str(path.relative_to(root))] = names
    return found


def verify(root, allowed=None):
    allowed = ALLOWED if allowed is None else allowed
    found, listed = scan(root), {}
    for files in allowed.values():
        for file, names in files.items():
            listed.setdefault(file, set()).update(names)
    errors = [f"{file}: {name} is not listed" for file, names in found.items() for name in sorted(names - listed.get(file, set()))]
    errors += [f"{file}: listed {name} no longer occurs" for file, names in listed.items() for name in sorted(names - found.get(file, set()))]
    return errors, sum(len(names) for names in found.values())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if "--list" in sys.argv:
        for file, names in scan(root).items():
            print(f'        "{file}": {{{", ".join(repr(n) for n in sorted(names))}}},')
        raise SystemExit(0)
    errors, count = verify(root)
    print("\n".join(errors) if errors else f"PASS: {count} listed private reaches in {len(scan(root))} files")
    raise SystemExit(bool(errors))
