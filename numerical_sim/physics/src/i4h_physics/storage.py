# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit pre-finalization storage declarations; no runtime array discovery.

Packages supply authored host values and logical layouts. Newton allocates flat
backing arrays; bindings cache shaped aliases without initializing any history.
This module does not construct or schedule systems.
"""

from dataclasses import dataclass
from enum import Enum
from math import prod
from types import MappingProxyType
from uuid import uuid4

import newton
import numpy as np
import warp as wp


class Owner(Enum):
    MODEL = "model"
    STATE = "state"
    CONTROL = "control"


@dataclass(frozen=True)
class Domain:
    kind: str
    instance: str
    frequency: newton.Model.AttributeFrequency | str
    shape: tuple[int, ...]
    owner_reference: newton.Model.AttributeFrequency | str | None = None

    def __post_init__(self):
        if self.kind not in ("scene", "body", "system", "mount") or not isinstance(self.instance, str) or not self.instance:
            raise ValueError("a domain needs a scene/body/system/mount kind and a nonempty instance id")
        if not self.instance.isidentifier():
            raise ValueError("domain instance must be a Newton-compatible identifier")
        if self.kind == "scene" and self.instance not in ("particle", "pool"):
            raise ValueError("scene storage uses particle_* or pool_* names")
        _shape(self.shape)
        if isinstance(self.frequency, str):
            if not self.frequency.startswith("i4h:") or not self.frequency[4:].isidentifier():
                raise ValueError("custom domains require an i4h:frequency name")
            if self.owner_reference is None:
                raise ValueError("custom domains require an owner reference")
            _reference(self.owner_reference)
        elif not isinstance(self.frequency, newton.Model.AttributeFrequency):
            raise TypeError("domain frequency must be a Newton enum or an i4h custom frequency")
        elif self.owner_reference is not None:
            raise ValueError("native domains already identify their entity; no owner reference column")

    @property
    def count(self):
        return prod(self.shape)

    @property
    def custom(self):
        return isinstance(self.frequency, str)


def _shape(shape):
    if not isinstance(shape, tuple) or not 1 <= len(shape) <= 4 or any(
        not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape
    ):
        raise ValueError("shape must contain one to four nonnegative integer dimensions")


def _reference(reference):
    if reference is not None and not isinstance(reference, (newton.Model.AttributeFrequency, str)):
        raise TypeError("references must be a Newton enum or string reference")
    if isinstance(reference, str) and not reference:
        raise ValueError("reference cannot be empty")


@dataclass(frozen=True)
class Field:
    domain: Domain
    key: str
    owner: Owner
    dtype: type
    shape: tuple[int, ...]
    indexing: str
    reset: str
    # Only genuine Newton entity indices may request merge offsets. Body-local
    # indices and the reserved native_tet=-1 mapping must leave this unset.
    references: newton.Model.AttributeFrequency | str | None = None
    # Several scalar columns of one domain row (e.g. six local tet-edge ids).
    # Newton stores a vector per row; Views exposes the original scalar layout.
    row_width: int = 1
    # Fields on a shared row domain retain the declaring system namespace.
    attribute_name: str | None = None
    # Existing native Model array: expose a checked alias, without a custom copy.
    native_attribute: str | None = None

    def __post_init__(self):
        if not isinstance(self.domain, Domain) or not isinstance(self.owner, Owner):
            raise TypeError("field domain and owner must be explicit Domain and Owner values")
        if not isinstance(self.key, str) or not self.key.isidentifier() or not self.indexing or not self.reset:
            raise ValueError("field key, indexing and reset contracts are required")
        if self.attribute_name is not None and not self.attribute_name.isidentifier():
            raise ValueError("attribute_name must be a Newton-compatible identifier")
        if self.native_attribute is not None and (
            not self.native_attribute.isidentifier() or self.owner is not Owner.MODEL
            or self.domain.custom or self.row_width != 1 or self.references is not None
        ):
            raise ValueError("native aliases require a Model native domain and scalar rows")
        _shape(self.shape)
        _reference(self.references)
        if not isinstance(self.row_width, int) or isinstance(self.row_width, bool) or self.row_width < 1:
            raise ValueError("row_width must be a positive integer")
        if prod(self.shape) != self.domain.count * self.row_width:
            raise ValueError("field shape must match its domain row count and row_width")
        if self.row_width != 1 and not wp.types.type_is_scalar(self.dtype):
            raise ValueError("row_width applies only to scalar fields")
        if self.key == "native_tet" and self.references is not None:
            raise ValueError("native_tet is a reserved local sentinel, not a Newton reference")

    @property
    def name(self):
        return self.attribute_name or f"{self.domain.instance}_{self.key}"

    @property
    def backing_dtype(self):
        return self.dtype if self.row_width == 1 else wp.types.vector(self.row_width, self.dtype)


class Declarations:
    """One explicit schema per authoring. Freeze it before builder finalization."""

    def __init__(self):
        self._fields = {}
        self._values = {}
        self._names = {}
        self._domains = {}
        self._builder = None
        self._authored = False
        self._identity_resolvers = {}
        self._value_resolvers = {}
        self._token = uuid4().hex
        self._initializers = []

    def register_package(self, *, records=()):
        """Publish host-only package records before schema registration.

        Each entry is ``(scene, attribute, {local_id: record})``. Existing ids
        cannot be overwritten. Runtime identities/checks use the same entrypoint
        on the bound storage Views. No finalized arrays belong in these records.
        """
        if self._builder is not None:
            raise RuntimeError("package records are frozen; reconstruct the model")
        pending = {}
        for container, name, entries in records:
            key = (id(container), name)
            if key not in pending:
                pending[key] = (container, dict(getattr(container, name, {})))
            merged = pending[key][1]
            if merged.keys() & entries.keys():
                raise ValueError(f"duplicate package record in {name}")
            merged.update(entries)
        for (_, name), (container, entries) in pending.items():
            # Keep existing host readers of the dictionary valid.
            if hasattr(container, name):
                getattr(container, name).update(entries)
            else:
                setattr(container, name, entries)

    def initializer(self, callback):
        """Register package initialization of finalized Model/template arrays."""
        if self._builder is not None:
            raise RuntimeError("initializers are frozen; reconstruct the model")
        if not callable(callback):
            raise TypeError("initializer must be callable")
        self._initializers.append(callback)

    @property
    def fields(self):
        return MappingProxyType(self._fields)

    def domain(self, domain, *, worlds=None, owners=None):
        """Declare custom row identities, or deferred host-only identity resolvers.

        Deferred callables run in author_rows after native entities exist. Native
        rows use entity hooks and never receive custom owner/world columns.
        """
        if self._builder is not None:
            raise RuntimeError("persistent declarations are frozen; reconstruct the model for additions")
        if domain in self._domains:
            raise ValueError("duplicate domain")
        if domain.custom:
            if any(d.frequency == domain.frequency for d in self._domains):
                raise ValueError("each custom domain instance needs its own frequency")
            arrays = []
            for name, value in (("worlds", worlds), ("owners", owners)):
                if callable(value):
                    self._identity_resolvers[domain, name] = value
                    value = np.full(domain.shape, -1, np.int32)
                data = np.asarray(value)
                if data.shape != domain.shape or data.dtype.kind not in "iu" or np.any(data < -1) or np.any(data > np.iinfo(np.int32).max):
                    raise ValueError(f"{name} must be int32-compatible reference rows of shape {domain.shape}")
                data = np.array(data, dtype=np.int32, copy=True).reshape(-1)
                data.flags.writeable = False
                arrays.append(data)
            self._domains[domain] = tuple(arrays)
        else:
            if worlds is not None or owners is not None:
                raise ValueError("native domain values are authored through native entity hooks")
            self._domains[domain] = None

    def add(self, field, values=None):
        if self._builder is not None:
            raise RuntimeError("persistent declarations are frozen; reconstruct the model for additions")
        if not isinstance(field, Field):
            raise TypeError("expected a Field declaration")
        key = (field.domain, field.key)
        if field.domain not in self._domains:
            raise ValueError("declare the domain before its fields")
        if key in self._fields:
            raise ValueError(f"duplicate persistent field: {key}")
        if field.name in self._names.values() or field.name in self._metadata_names():
            raise ValueError(f"persistent attribute name collision: i4h:{field.name}")
        if field.native_attribute is not None:
            if values is not None:
                raise ValueError("native aliases do not author duplicate values")
            self._fields[key], self._names[key] = field, field.name
            return
        if callable(values):
            if not field.domain.custom:
                raise ValueError("deferred values require a custom field domain")
            self._value_resolvers[key] = values
            self._fields[key], self._names[key] = field, field.name
            return
        self._fields[key], self._values[key] = field, self._host_values(field, values)
        self._names[key] = field.name

    @staticmethod
    def _host_values(field, values):
        key = (field.domain, field.key)
        # Validate host values without allocating a device array. Vector/matrix
        # components are dtype dimensions, not additional logical array axes.
        scalar = getattr(field.dtype, "_wp_scalar_type_", field.dtype)
        dtype = wp.dtype_to_numpy(scalar)
        components = tuple(getattr(field.dtype, "_shape_", ()))
        data = np.asarray(values, dtype=dtype)
        expected = field.shape + components
        if data.shape != expected:
            raise ValueError(f"{key}: authored values have shape {data.shape}, expected {expected}")
        if field.key == "native_tet" and np.any(data != -1):
            raise ValueError("native_tet must contain only the reserved -1 sentinel")
        data = np.array(data, copy=True, order="C")
        data.flags.writeable = False
        return data

    def _metadata_names(self):
        return [f"{d.frequency[4:]}_{column}" for d in self._domains if d.custom for column in ("world", "owner")]

    def entity_values(self, domain, start=0, count=None):
        """Native rows for add_particles(custom_attributes=...) or another host hook.

        Never patch particle custom attributes after particle creation. ``start``
        indexes authored component rows, not the host's foreign particle prefix.
        """
        if domain not in self._domains or domain.custom:
            raise ValueError("entity_values requires a declared native domain")
        count = domain.count - start if count is None else count
        if start < 0 or count < 0 or start + count > domain.count:
            raise ValueError("native row selection outside the authored domain")
        return {f"i4h:{f.name}": self._rows(key, start, count) for key, f in self._fields.items()
                if f.domain == domain and f.native_attribute is None}

    def _rows(self, key, start=0, count=None):
        field = self._fields[key]
        components = tuple(getattr(field.backing_dtype, "_shape_", ()))
        rows = self._values[key].reshape((field.domain.count,) + components)
        return rows[start:start + count if count is not None else None].tolist()

    def register_schema(self, builder):
        """Register and freeze field definitions before native entity creation.

        Empty logical fields have one inaccessible padding element because the
        pinned Newton finalizer omits zero-frequency attributes altogether.
        Accounting includes that element; consumers see only their empty view.
        """
        if self._builder is not None:
            if self._builder is not builder:
                raise ValueError("a different builder requires fresh declarations")
            return
        names = [*self._names.values(), *self._metadata_names(), "storage_model_id", "storage_state_id", "storage_control_id",
                 "storage_initialized"]
        if len(names) != len(set(names)):
            raise ValueError("persistent attribute name collision")
        if any(f"i4h:{name}" in builder.custom_attributes for name in names):
            raise ValueError("builder already contains an i4h storage schema")
        known_frequencies = set(builder.custom_frequencies) | {d.frequency for d in self._domains if d.custom}
        references = [d.owner_reference for d in self._domains] + [f.references for f in self._fields.values()]
        for reference in references:
            normalized = newton.Model._normalize_attribute_reference(reference)
            if isinstance(normalized, str) and normalized not in known_frequencies:
                raise ValueError(f"unknown Newton reference frequency: {reference}")
        for owner in Owner:
            builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                name=f"storage_{owner.value}_id", namespace="i4h", dtype=str,
                frequency=newton.Model.AttributeFrequency.ONCE,
                assignment=getattr(newton.Model.AttributeAssignment, owner.name), default=self._token))
        # Register all frequencies before attributes: owner columns can refer to
        # another declared i4h domain (instrument, rod, etc.).
        for domain in self._domains:
            if domain.custom:
                builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=domain.frequency[4:], namespace="i4h"))
                for column, reference in (("world", "world"), ("owner", domain.owner_reference)):
                    builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                        name=f"{domain.frequency[4:]}_{column}", namespace="i4h", dtype=wp.int32,
                        frequency=domain.frequency, references=reference, default=-1))
        for key, field in self._fields.items():
            if field.native_attribute is not None:
                continue
            name = self._names[key]
            builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                name=name, namespace="i4h", dtype=field.backing_dtype, frequency=field.domain.frequency,
                assignment=getattr(newton.Model.AttributeAssignment, field.owner.name), references=field.references,
                # begin_world(attributes=...) is metadata-only in pinned Newton.
                # WORLD values use its documented enum-indexed value dictionary.
                values=dict(enumerate(self._rows(key))) if field.domain.frequency == newton.Model.AttributeFrequency.WORLD else None))
            if not field.domain.custom and field.domain.count == 0:
                # The normal finalizer skips empty native frequencies. Preserve
                # the genuine frequency and reserve one backing element without
                # inventing a native particle/tet/world.
                builder._add_custom_attribute_model_finalizer(f"i4h:{name}", _finalize_empty_native)
        if self._initializers:
            from i4h_physics.newton_compat import require_model_finalizer
            register = require_model_finalizer(builder)
            builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                name="storage_initialized", namespace="i4h", dtype=str,
                frequency=newton.Model.AttributeFrequency.ONCE,
                assignment=newton.Model.AttributeAssignment.MODEL, default=self._token))
            def initialize(builder, model, attribute):
                if not self._authored:
                    raise RuntimeError("complete storage declarations before finalization")
                for callback in self._initializers:
                    callback(model)
                # Finish all initialization launches before callers can clone templates.
                wp.synchronize_device(model.device)
                model.add_attribute(attribute.name, [self._token], attribute.frequency,
                                    attribute.assignment, attribute.namespace, attribute.references)
            register("i4h:storage_initialized", initialize)
        self._builder = builder

    def author_rows(self, builder):
        """Resolve host identities and author each complete custom row exactly once."""
        if self._builder is not builder:
            raise ValueError("register schema on this builder before authoring rows")
        if self._authored:
            return
        # Resolve and validate every host value before emitting any custom row.
        resolved = {key: self._host_values(self._fields[key], resolve(builder))
                    for key, resolve in self._value_resolvers.items()}
        for (domain, column), resolve in self._identity_resolvers.items():
            data = np.asarray(resolve())
            if data.shape != domain.shape or data.dtype.kind not in "iu" or np.any(data < -1) or np.any(data > np.iinfo(np.int32).max):
                raise ValueError(f"invalid deferred {column} identities for {domain}")
            arrays = list(self._domains[domain])
            arrays[0 if column == "worlds" else 1] = np.array(data, dtype=np.int32).reshape(-1)
            self._domains[domain] = tuple(arrays)
        self._values.update(resolved)
        for domain, identities in self._domains.items():
            if not domain.custom:
                continue
            columns = {f"i4h:{f.name}": self._rows(key) for key, f in self._fields.items() if f.domain == domain}
            worlds, owners = identities
            for row in range(max(1, domain.count)):
                values = {key: values[row] if domain.count else None for key, values in columns.items()}
                values[f"i4h:{domain.frequency[4:]}_world"] = int(worlds[row]) if domain.count else -1
                values[f"i4h:{domain.frequency[4:]}_owner"] = int(owners[row]) if domain.count else -1
                builder.add_custom_values(**values)
        self._authored = True

    def register(self, builder):
        """Convenience for schemas whose host identities are already complete."""
        self.register_schema(builder)
        self.author_rows(builder)

    def bind(self, model, state, control, *, placements=None):
        if not self._authored:
            raise RuntimeError("complete declaration rows before binding")
        return Views(self, model, state, control, placements or {})


def _finalize_empty_native(builder, model, attribute):
    count = model._attribute_frequency_count(attribute.frequency)
    data = attribute.build_array(max(1, count), device=model.device, requires_grad=model.requires_grad)
    model.add_attribute(attribute.name, data, attribute.frequency, attribute.assignment, attribute.namespace, attribute.references)


class Views:
    """Cached zero-copy aliases and identity validation for exactly one binding."""

    def __init__(self, declarations, model, state, control, placements):
        self._declarations = declarations
        self._objects = {Owner.MODEL: model, Owner.STATE: state, Owner.CONTROL: control}
        self._identities, self._arrays, self._views = [], [], {}
        self._checks = []
        if any(d not in declarations._domains or d.custom for d in placements):
            raise ValueError("placements must name declared native domains")
        model_namespace = getattr(model, "i4h", None)
        for owner, container in self._objects.items():
            namespace = getattr(container, "i4h", None)
            name = f"storage_{owner.value}_id"
            marker = getattr(namespace, name, None)
            if marker != [declarations._token] or marker is not getattr(model_namespace, name, None):
                raise ValueError(f"foreign {owner.value} container for this finalized model")
            self._identities.extend(((container, "i4h", namespace), (namespace, name, marker),
                                     (model_namespace, name, marker)))
        for key, field in declarations.fields.items():
            container = self._objects[field.owner] if field.native_attribute else self._objects[field.owner].i4h
            name = field.native_attribute or declarations._names[key]
            array = getattr(container, name, None)
            count = model._attribute_frequency_count(field.domain.frequency)
            size = count if field.native_attribute else max(1, count)
            self._identities.append((container, name, array))
            if field.native_attribute and size == 0 and array is None:
                # Newton has no native array for zero particles. Zero-capacity
                # views allocate no backing storage and still validate None.
                array = wp.empty(0, dtype=field.dtype, device=model.device)
            if (not isinstance(array, wp.array) or array.dtype != field.backing_dtype or array.shape != (size,)
                    or array.device != model.device or not array.is_contiguous):
                raise ValueError(f"incompatible persistent buffer: {key}")
            self._arrays.append((array, (array.ptr, array.shape, array.strides, array.dtype, array.device)))
            if field.domain.custom and size != max(1, field.domain.count):
                raise ValueError(f"custom domain count differs from authoring: {field.domain}")
            # Reinterpret scalar row vectors without copying. Keep a strong
            # reference to the Newton backing array in _arrays.
            scalar = array.view(field.dtype).flatten()
            placement = placements.get(field.domain)
            if placement is None:
                if field.domain.count and size != field.domain.count:
                    raise ValueError("native domain with foreign rows requires an explicit placement")
                rows = scalar if size == 0 else scalar[:prod(field.shape)]
                self._views[key] = rows.reshape(field.shape)
            else:
                if field.domain.custom or len(field.shape) < 2:
                    raise ValueError("placements require a native world-major field")
                start, stride = placement  # native rows, including foreign rows
                if any(not isinstance(n, int) or isinstance(n, bool) for n in placement):
                    raise ValueError("native placement start/stride must be integers")
                width = prod(field.shape[1:])
                stride *= field.row_width
                start *= field.row_width
                if start < 0 or stride < width or (field.shape[0] - 1) * stride + start + width > scalar.size:
                    raise ValueError("native placement exceeds backing storage")
                rows = scalar[start:].reshape((-1,))
                item_bytes = wp.types.type_size_in_bytes(field.dtype)
                strides = tuple(prod(field.shape[i + 1:]) * item_bytes for i in range(len(field.shape)))
                self._views[key] = wp.array(ptr=rows.ptr, dtype=field.dtype, shape=field.shape,
                                           strides=(stride * item_bytes, *strides[1:]), device=array.device,
                                           capacity=array.capacity - start * item_bytes, copy=False)
                self._views[key]._ref = array
            view = self._views[key]
            self._arrays.append((view, (view.ptr, view.shape, view.strides, view.dtype, view.device)))
        for domain in declarations._domains:
            if domain.custom:
                for column in ("world", "owner"):
                    name = f"{domain.frequency[4:]}_{column}"
                    array = getattr(model.i4h, name)
                    if array.dtype != wp.int32 or array.shape != (max(1, domain.count),) or array.device != model.device or not array.is_contiguous:
                        raise ValueError(f"incompatible domain identity column: {name}")
                    self._identities.append((model.i4h, name, array))
                    self._arrays.append((array, (array.ptr, array.shape, array.strides, array.dtype, array.device)))

    def register_package(self, *, identities=(), checks=(), replace=False):
        """Register package aliases and layout checks without exposing internals.

        Identities are ``(container, attribute, expected_object)``. Array identities
        also freeze pointer/layout metadata. Repeated identical registrations are
        harmless. ``replace=True`` explicitly routes aliases during binding; all
        old identities are checked before any attribute is changed.
        """
        identities, checks = tuple(identities), tuple(checks)
        if any(not callable(check) for check in checks):
            raise TypeError("package checks must be callable")
        if replace:
            self.check()
        indexed = {(id(owner), name): i for i, (owner, name, _) in enumerate(self._identities)}
        pending = {}
        for owner, name, value in identities:
            key = (id(owner), name)
            if key in pending and pending[key][2] is not value:
                raise ValueError(f"conflicting package identity: {name}")
            pending[key] = (owner, name, value)
            if not replace:
                if getattr(owner, name, None) is not value:
                    raise RuntimeError(f"{name} alias replaced; bind a fresh runtime")
                if key in indexed and self._identities[indexed[key]][2] is not value:
                    raise RuntimeError(f"{name} alias already registered; bind a fresh runtime")
        for key, (owner, name, value) in pending.items():
            if replace:
                setattr(owner, name, value)
            if key in indexed:
                self._identities[indexed[key]] = (owner, name, value)
            else:
                self._identities.append((owner, name, value))
            if isinstance(value, wp.array) and not any(array is value for array, _ in self._arrays):
                self._arrays.append((value, (value.ptr, value.shape, value.strides, value.dtype, value.device)))
        self._checks.extend(checks)

    def container(self, owner):
        """The selected Newton Model, State or Control for this binding."""
        return self._objects[owner]

    def check(self):
        for check in self._checks:
            check()
        for container, name, expected in self._identities:
            if getattr(container, name, None) is not expected:
                raise RuntimeError(f"persistent {name} container or buffer was replaced; bind a fresh runtime")
        for array, expected in self._arrays:
            if (array.ptr, array.shape, array.strides, array.dtype, array.device) != expected:
                raise RuntimeError("persistent array layout was changed; bind a fresh runtime")

    def attr(self, domain, key):
        return self._views[domain, key]

    def allocated_bytes(self):
        """Declared backing bytes only; excludes native arrays and construction peak.

        Model-owned parameters and Newton's initialization templates are reported
        separately. Shared controls intentionally overlap their template bytes.
        Each additional State/Control must be accounted with its own binding.
        """
        totals = {"model": 0, "templates": 0, "state": 0, "control": 0, "control_shared_with_template": 0}
        model = self._objects[Owner.MODEL]
        totals["model"] += sum(max(1, d.count) * 8 for d in self._declarations._domains if d.custom)
        for key, field in self._declarations.fields.items():
            if field.native_attribute is not None:
                continue  # Native buffers are counted by their existing owner.
            name = self._declarations._names[key]
            backing = getattr(self._objects[field.owner].i4h, name)
            nbytes = backing.size * wp.types.type_size_in_bytes(backing.dtype)
            if field.owner is Owner.MODEL and any(
                isinstance(a, wp.array) and a.ptr == backing.ptr for a in vars(model).values()
            ):
                continue  # A Scene finalizer may alias native gravity; count its native owner once.
            totals[field.owner.value] += nbytes
            if field.owner is not Owner.MODEL:
                totals["templates"] += nbytes
                if field.owner is Owner.CONTROL and backing.ptr == getattr(model.i4h, name).ptr:
                    totals["control_shared_with_template"] += nbytes
        return totals


def defer_declaration(scene, key, callback, settings):
    """Queue a package's host-only declaration for direct Scene construction."""
    if hasattr(scene, "declarations") or scene.q is not None:
        raise RuntimeError("declare storage before allocation or schema registration; reconstruct the model")
    pending = scene.__dict__.setdefault("_deferred_storage", {})
    if key in pending:
        raise ValueError(f"duplicate storage declaration {key!r}")
    pending[key] = (callback, tuple(settings.items()))
