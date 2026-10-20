# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture checks cover imports, lazy paths, extensions and explicit data setup."""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
FORBIDDEN = ("i4h_surgical", "i4h_endoluminal", "i4h_isaaclab", "i4h_standalone")


@pytest.mark.parametrize("member,package", [("physics", "i4h_physics"), ("assessment", "i4h_assessment")])
def test_neutral_imports_and_terminology(member, package):
    for path in (WORKSPACE / member / "src" / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name.startswith(FORBIDDEN + (("i4h_assessment",) if member == "physics" else ())) for name in imports), path
            if member == "physics":
                identifiers = ([node.id] if isinstance(node, ast.Name) else [node.arg] if isinstance(node, ast.arg)
                               else [node.name] if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                               else [node.attr] if isinstance(node, ast.Attribute) else [])
                assert not any(re.search(r"(?i)(catheter|anatomical|tissue|vessel|instrument)", name)
                               for name in identifiers), (path, identifiers)
                if isinstance(node, ast.Raise):
                    for text in ast.walk(node):
                        if isinstance(text, ast.Constant) and isinstance(text.value, str):
                            assert not re.search(r"(?i)\b(catheter|anatomical|tissue|vessel|instrument)\b", text.value), (path, text.value)


def test_core_and_assessment_lazy_imports_without_domains():
    script = '''
import importlib, pkgutil, sys
class BlockDomains:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("i4h_surgical", "i4h_endoluminal", "i4h_isaaclab", "i4h_standalone")):
            raise AssertionError("unexpected domain/host import: " + fullname)
sys.meta_path.insert(0, BlockDomains())
for name in ("i4h_physics", "i4h_assessment"):
    module = importlib.import_module(name)
    for child in pkgutil.walk_packages(module.__path__, name + "."):
        importlib.import_module(child.name)
'''
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", check=True)


def test_explicit_data_precedence_and_no_checkout_discovery(monkeypatch, tmp_path):
    from i4h_physics.config import data_dir, resolve_data_path
    monkeypatch.delenv("I4H_DATA", raising=False)
    packaged = data_dir()
    assert packaged != WORKSPACE / "data"
    assert not (packaged / "vessels").exists()  # physics ships no domain data; templates live in i4h_endoluminal
    monkeypatch.setenv("I4H_DATA", str(tmp_path / "env"))
    assert data_dir() == tmp_path / "env"
    assert data_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert resolve_data_path("${DATA}/mesh.vtu", tmp_path / "explicit") == str(tmp_path / "explicit/mesh.vtu")


def test_missing_production_assets_explain_checkout_command(tmp_path):
    from i4h_surgical import load_case
    with pytest.raises(FileNotFoundError) as error:
        load_case("chole_tet_med", data=tmp_path, device="cpu")
    assert str(tmp_path / "choly-mid") in str(error.value)
    assert "I4H_DATA" in str(error.value)
    assert "uv run i4h-sim CASE --data ./data" in str(error.value)


MODULE_PATH = re.compile(r"^i4h_\w+(\.\w+)*(:\w+)?$")  # "i4h_x.y" or a class path "i4h_x.y:Name"
DOMAINS = ("i4h_surgical", "i4h_endoluminal")


def violations(source: str, forbidden: tuple, guarded: tuple = ()) -> list[str]:
    """Names of ``forbidden`` packages that ``source`` imports or names as a module path: import statements,
    ``__import__``/``importlib.import_module`` arguments and module-path strings (Isaac Lab ``class_type`` values).
    An import of a ``guarded`` package must sit in a ``try`` that handles ``ModuleNotFoundError``, or in a loop over
    ``domains()`` (only installed domains); a function-local import is not enough (it fails when first called)."""
    tree, found = ast.parse(source), []
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def tolerated(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == "domains":
                return True
            if isinstance(node, ast.Try) and any(
                    "ModuleNotFoundError" in ast.unparse(h.type) for h in node.handlers if h.type is not None):
                return True
        return False

    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and \
                ast.unparse(node.func) in ("__import__", "importlib.import_module", "import_module"):
            names.append(str(node.args[0].value))
        found += [f"line {node.lineno}: imports {n}" for n in names if n.startswith(forbidden)]
        found += [f"line {node.lineno}: unguarded import of {n}" for n in names
                  if n.startswith(guarded) and not tolerated(node)]  # "unguarded" includes function-local
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_PATH.match(node.value) \
                and node.value.startswith(forbidden):
            found.append(f"line {node.lineno}: names {node.value!r}")
    return found


@pytest.mark.parametrize("member,forbidden,guarded", [
    ("surgical", ("i4h_endoluminal", "i4h_isaaclab", "i4h_standalone"), ()),
    ("endoluminal", ("i4h_surgical", "i4h_isaaclab", "i4h_standalone"), ()),
    ("isaaclab", ("i4h_standalone",), DOMAINS),
])
def test_domain_and_camera_lazy_import_boundaries(member, forbidden, guarded):
    for path in (WORKSPACE / member / "src").rglob("*.py"):
        found = violations(path.read_text(), forbidden, guarded)
        assert not found, (path, found)


@pytest.mark.parametrize("source,count", [
    ("import i4h_standalone.devices", 1),
    ("class_type = 'i4h_standalone.devices:I4hTeleopDevice'", 1),
    ("m = __import__('i4h_standalone.app')", 2),  # the call and its module-path string
    ("import importlib\nm = importlib.import_module('i4h_standalone')", 2),
    ("from i4h_surgical.systems import X", 1),  # module level, unguarded
    ("try:\n    from i4h_surgical.systems import X\nexcept ModuleNotFoundError:\n    X = ()", 0),
    ("def f():\n    from i4h_endoluminal.assessment import R", 1),  # function-local: still unguarded
    ("for package in domains():\n    from i4h_surgical.assessment import G", 0),
    ('"""See i4h_standalone for the device."""', 0),  # prose, not a module path
    ("if error.name != 'i4h_surgical':\n    raise", 0),
])
def test_boundary_rules_catch_planted_violations(source, count):
    found = violations(source, ("i4h_standalone",), DOMAINS)
    assert len(found) == count, found


def test_isaaclab_root_and_lab_pin_import_without_isaac_lab_or_standalone():
    """The package root stays lazy: without Isaac Lab (or standalone) only the modules that need neither import."""
    script = """
import importlib, sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("isaaclab", "isaaclab_newton", "isaaclab_tasks", "isaaclab_rl", "i4h_standalone"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
for name in ("i4h_isaaclab", "i4h_isaaclab.lab"):
    importlib.import_module(name)
try:
    importlib.import_module("i4h_isaaclab.env")
except ModuleNotFoundError as error:
    assert error.name.split(".")[0] in ("isaaclab", "isaaclab_newton", "gymnasium"), error.name  # Lab or its deps
else:
    raise AssertionError("i4h_isaaclab.env imported without Isaac Lab")
"""
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", check=True)


@pytest.mark.parametrize("member", ["surgical", "endoluminal"])
def test_domain_lazy_paths_without_hosts_or_other_domain(member):
    forbidden = ("i4h_standalone", "i4h_isaaclab", "i4h_" + ("surgical" if member == "endoluminal" else "endoluminal"))
    script = f"""
import importlib, pkgutil, sys
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith({forbidden!r}):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockImports())
module = importlib.import_module('i4h_{member}')
for child in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
    importlib.import_module(child.name)
"""
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", check=True)


def test_physics_lazy_paths_without_assessment():
    script = '''
import importlib, pkgutil, sys
class BlockAssessment:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("i4h_assessment"):
            raise AssertionError("physics imported assessment: " + fullname)
sys.meta_path.insert(0, BlockAssessment())
module = importlib.import_module("i4h_physics")
for child in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
    importlib.import_module(child.name)
'''
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", check=True)


@pytest.mark.parametrize("member", ["surgical", "endoluminal"])
def test_domains_without_assessment(member):
    """i4h-assessment is the domains' optional ``assessment`` extra: every module but the measurements imports
    without it, and a physics case still reads."""
    script = f"""
import importlib, pkgutil, sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "i4h_assessment":
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
module = importlib.import_module('i4h_{member}')
for child in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
    if child.name != 'i4h_{member}.assessment':
        importlib.import_module(child.name)
from i4h_{member}.config import read_case
case = read_case({"two_worlds" if member == "surgical" else "aorta_static"!r})
assert "procedure" not in case
try:
    read_case({"chole_grasp" if member == "surgical" else "aorta_static_reach"!r})
except ModuleNotFoundError as error:
    assert error.name == "i4h_assessment" and "[assessment]" in str(error), error
else:
    raise AssertionError("a procedure case read without i4h-assessment")
"""
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", check=True)
