# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict disk continuation bundle for the standalone pinned-liver harness."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import warp as wp
from i4h_physics.storage import Owner

SCHEMA = 1


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def continuation_arrays(binding, state, control):
    result = {}
    for owner, key, *_ in binding.solver.continuation.entries:
        container = state if owner is Owner.STATE else control
        array = (
            getattr(container, key)
            if isinstance(key, str)
            else binding.scene.storage.backing(key, container)
        )
        result[f"{owner.name}:{key}"] = array
    return result


def identity(binding, cfg, data):
    """Hash content, configuration, immutable Model arrays and all domain/runtime source files."""
    import i4h_physics
    import i4h_surgical

    assets = {}
    # Exactly the fixture's numerical and instrument assets; unused organs/textures do not enter it.
    root = Path(data)
    for path in sorted((root / "choly-mid/liver").glob("*")):
        if path.is_file():
            assets[str(path.relative_to(root))] = digest(path.read_bytes())
    path = root / "instruments/pgrasp.usdc"
    assets[str(path.relative_to(root))] = digest(path.read_bytes())
    sources = {}
    for module in (i4h_physics, i4h_surgical):
        directory = Path(module.__file__).parent
        for path in sorted(directory.rglob("*.py")):
            sources[f"{module.__name__}/{path.relative_to(directory)}"] = digest(
                path.read_bytes()
            )
    dependencies = {}
    for distribution, module_name in (
        ("warp-lang", "warp"),
        ("newton", "newton"),
        ("numpy", "numpy"),
    ):
        module = importlib.import_module(module_name)
        directory = Path(module.__file__).parent
        files = {
            str(path.relative_to(directory)): digest(path.read_bytes())
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".so")
        }
        dist = importlib.metadata.distribution(distribution)
        dependencies[distribution] = dict(
            version=dist.version,
            source_hash=digest(encode(files)),
            direct_url=dist.read_text("direct_url.json"),
        )
    model_arrays = {}
    for name, value in vars(binding.model).items():
        if isinstance(value, wp.array):
            model_arrays[name] = digest(value.numpy().tobytes())
    for name, value in vars(binding.model.i4h).items():
        if isinstance(value, wp.array):
            model_arrays[f"i4h:{name}"] = digest(value.numpy().tobytes())
    return dict(
        schema=SCHEMA,
        config=cfg,
        assets=assets,
        sources=sources,
        dependencies=dependencies,
        worlds=binding.scene.envs,
        model=model_arrays,
    )


class HarnessSnapshot:
    def __init__(self, binding, configuration_identity, counters):
        self.binding = binding
        self.identity = configuration_identity
        self.counters = dict(counters)
        self.state = binding.model.state()
        self.control = binding.model.control(clone_variables=True)
        binding.save(self.state, self.control)
        self.arrays = continuation_arrays(binding, self.state, self.control)
        attachments = next(s for s in binding.systems if s.id == "attachments")
        self.attachments = attachments
        self.attachment_rest = attachments.rest.numpy().copy()
        self.attachment_index = attachments.index.numpy().copy()
        self.hash = self.metadata()["snapshot_hash"]

    def metadata(self):
        values = {
            k: dict(
                shape=list(a.numpy().shape),
                dtype=str(a.numpy().dtype),
                hash=digest(a.numpy().tobytes()),
            )
            for k, a in self.arrays.items()
        }
        content = dict(
            identity=self.identity,
            counters=self.counters,
            arrays=values,
            schema_hash=digest(
                encode(
                    {
                        k: {f: r[f] for f in ("shape", "dtype")}
                        for k, r in values.items()
                    }
                )
            ),
            attachment_rest=digest(self.attachment_rest.tobytes()),
            attachment_index=digest(self.attachment_index.tobytes()),
        )
        return {**content, "snapshot_hash": digest(encode(content))}

    def write(self, path):
        metadata = self.metadata()
        np.savez_compressed(
            path,
            metadata=np.frombuffer(encode(metadata), np.uint8),
            attachment_rest=self.attachment_rest,
            attachment_index=self.attachment_index,
            **{k: a.numpy() for k, a in self.arrays.items()},
        )

    def load(self, path):
        with np.load(path, allow_pickle=False) as bundle:
            metadata = json.loads(bundle["metadata"].tobytes())
            content = {k: v for k, v in metadata.items() if k != "snapshot_hash"}
            if (
                metadata["snapshot_hash"] != digest(encode(content))
                or metadata["identity"] != self.identity
            ):
                raise ValueError("snapshot identity/hash mismatch")
            if set(metadata["arrays"]) != set(self.arrays):
                raise ValueError("snapshot layout mismatch")
            for key, array in self.arrays.items():
                value = bundle[key]
                record = metadata["arrays"][key]
                current = array.numpy()
                if (
                    value.shape != current.shape
                    or value.dtype != current.dtype
                    or record
                    != dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        hash=digest(value.tobytes()),
                    )
                ):
                    raise ValueError(f"snapshot buffer mismatch: {key}")
            for name in ("attachment_rest", "attachment_index"):
                value = bundle[name]
                if digest(value.tobytes()) != metadata[name] or not np.array_equal(
                    value, getattr(self, name)
                ):
                    raise ValueError("attachment rest identity mismatch")
            # Validate everything before mutating even the checkpoint containers.
            for key, array in self.arrays.items():
                array.assign(bundle[key])
            self.counters = metadata["counters"]
            self.hash = metadata["snapshot_hash"]

    def restore(self, counters, probe=None, env_ids=None):
        if not np.array_equal(
            self.attachments.rest.numpy(), self.attachment_rest
        ) or not np.array_equal(self.attachments.index.numpy(), self.attachment_index):
            raise ValueError("constructed attachment rest changed")
        self.binding.restore(self.state, self.control, env_ids=env_ids)
        if env_ids is None:
            counters.clear()
            counters.update(self.counters)
        else:
            # Per-world harness counters are reset by callers; global physics time cannot be rewound selectively.
            for e in env_ids:
                counters[f"world_{e}_step"] = self.counters.get(f"world_{e}_step", 0)
        if probe is not None:
            probe.clear(env_ids)

    def exact(self):
        live = continuation_arrays(
            self.binding, self.binding.state, self.binding.control
        )
        return {
            k: live[k].numpy().tobytes() == a.numpy().tobytes()
            for k, a in self.arrays.items()
        }
