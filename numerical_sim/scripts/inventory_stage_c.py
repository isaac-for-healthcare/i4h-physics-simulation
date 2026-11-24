# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Produce a source allocation census for manual Stage C ownership review.

This is an offline audit tool, never a declaration generator or runtime array
discovery mechanism. Sites are not fields: helpers, conditionals and loops must
be expanded and their ownership/next-use contracts reviewed by the packages.
Unreviewed entries deliberately cannot be mistaken for a complete inventory.
"""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

PACKAGES = ("physics", "surgical", "endoluminal", "isaaclab", "host", "assessment")
ALLOCATORS = {"array", "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like", "full", "full_like", "clone", "cat", "stack"}
STRUCTURES = {"HashGrid", "Bvh", "Mesh"}


def census(path, root):
    source = path.read_text()
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in node.names})
        elif isinstance(node, ast.ImportFrom):
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})

    def ancestry(node):
        while node in parents:
            node = parents[node]
            yield node

    def resolved(node):
        value = ast.unparse(node)
        prefix, dot, rest = value.partition(".")
        return aliases.get(prefix, prefix) + dot + rest

    entries = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        chain = list(ancestry(call))
        # A wp.array(dtype=...) annotation allocates nothing. Inspect executable
        # statements only, excluding annotations even inside function bodies.
        if any(isinstance(p, (ast.arg, ast.FunctionDef)) and
               (isinstance(p, ast.arg) or (p.returns is not None and call in ast.walk(p.returns))) for p in chain):
            continue
        if any(isinstance(p, ast.AnnAssign) and call in ast.walk(p.annotation) for p in chain):
            continue
        functions = [p for p in chain if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if any(resolved(d) in ("warp.kernel", "warp.func") for f in functions for d in f.decorator_list):
            continue
        name = resolved(call.func)
        prefix, _, leaf = name.rpartition(".")
        keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
        kind = None
        if prefix in ("warp", "torch") and leaf in ALLOCATORS:
            kind = "view" if "ptr" in keywords else "allocation"
            if leaf == "array" and not call.args and not ({"data", "shape", "ptr"} & keywords.keys()):
                continue
        elif prefix == "torch" and leaf in {"tensor", "as_tensor", "arange", "from_numpy"}:
            kind = "conversion_or_allocation"
        elif prefix == "warp" and leaf in {"from_torch", "to_torch"}:
            kind = "interop_view"
        elif name == "attribute.build_array":
            kind = "newton_backing_allocation"
        elif prefix == "warp" and leaf in STRUCTURES:
            kind = "acceleration_structure"
        elif leaf in {"to", "clone"} and isinstance(call.func, ast.Attribute):
            kind = "method_conversion_or_copy"
        elif leaf in {"compute_camera_rays_pinhole", "create_color_image_output",
                      "create_depth_image_output", "create_shape_index_image_output"}:
            kind = "external_renderer_allocation"
        elif leaf in {"create_buffer", "create_texture"}:
            kind = "graphics_resource"
        elif (leaf == "_live" and call.args) or name.endswith("systems.base.live"):
            kind = "live_helper"
        if kind is None:
            continue
        statement = next((p for p in chain if isinstance(p, ast.stmt)), None)
        target = None
        if isinstance(statement, ast.Assign):
            target = ", ".join(ast.unparse(t) for t in statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            target = ast.unparse(statement.target)
        entries.append({
            "path": str(path.relative_to(root)), "line": call.lineno,
            "scope": ".".join(p.name for p in reversed(chain) if isinstance(p, (ast.ClassDef, ast.FunctionDef))),
            "kind": kind, "target": target, "expression": ast.get_source_segment(source, call),
            "dtype_expression": keywords.get("dtype"),
            "shape_or_source_expression": keywords.get("shape", ast.unparse(call.args[0]) if call.args else None),
            "review": {"status": "pending", "owner": None, "indexing": None, "authored_value": None,
                       "reset": None, "aliases": None, "first_subsequent_reader_writer": None},
        })
    return sorted(entries, key=lambda row: row["line"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, hashes = [], {}
    for package in PACKAGES:
        for path in sorted((args.root / package / "src").rglob("*.py")):
            hashes[str(path.relative_to(args.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.extend(census(path, args.root))
    report = {"status": "source census only; field ownership and next-use review pending",
              "limitations": ["Helper call sites, nested containers and aliases require manual expansion.",
                              "A syntactic call is not proof of allocation, ownership, or overwrite before use.",
                              "External native Newton allocations are outside this source census."],
              "source_sha256": hashes, "counts": dict(Counter(row["kind"] for row in rows)), "sites": rows}
    # Preserve earlier evidence rather than silently replacing a census.
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"files": len(hashes), "sites": len(rows), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
