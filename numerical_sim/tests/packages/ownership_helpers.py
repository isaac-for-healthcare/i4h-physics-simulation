# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Container helpers shared by the ownership test modules: execution modes, whole-container snapshots and copies,
and a rigid test solver whose history is all native State."""
import newton
import warp as wp

MODES = [("cpu", False), ("cuda:0", False), ("cuda:0", True)]


def assert_native_alias(live, backing, declaration):
    """An exact logical alias, allowing only Newton's trailing, unowned WORLD row.

    Return that row's bytes so callers can prove their operations leave it untouched.
    """
    assert live.ptr == backing.ptr and live.device == backing.device and live.is_contiguous
    global_row = (declaration.domain.frequency == newton.Model.AttributeFrequency.WORLD
                  and backing.size == declaration.domain.count + 1)
    row_bytes = wp.types.type_size_in_bytes(backing.dtype)
    assert live.size * wp.types.type_size_in_bytes(live.dtype) == backing.size * row_bytes - (row_bytes if global_row else 0)
    return backing.numpy()[-1:].tobytes() if global_row else None


def arrays(container):
    """Every Warp array of a Newton Model, State or Control, including the i4h namespace."""
    found = {}
    for key, value in vars(container).items():
        if isinstance(value, wp.array):
            found[key] = value
        elif isinstance(value, newton.Model.AttributeNamespace):
            found.update({f"{key}:{k}": v for k, v in vars(value).items() if isinstance(v, wp.array)})
    return found


def snapshot(*containers):
    wp.synchronize()
    return [{key: a.numpy().tobytes() for key, a in arrays(c).items()} for c in containers]


def copy_container(source, destination):
    """Whole-container copy between containers of one Model: every array, nothing else (tests only; the public
    call is ``save_i4h`` / ``restore_i4h``)."""
    src, dst = arrays(source), arrays(destination)
    assert src.keys() == dst.keys()
    for key, a in src.items():
        if a.size:
            wp.copy(dst[key], a)
    wp.synchronize()


def first_difference(a, b):
    """Name the first array (in container order) whose bytes differ, for a readable resume failure."""
    for index, (x, y) in enumerate(zip(a, b)):
        for key in x:
            if x[key] != y[key]:
                return f"container {index}: {key}"
    return None


@wp.kernel
def rigid_advance(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
                  force: wp.array(dtype=wp.spatial_vector), inverse_mass: float, dt: float):
    i = wp.tid()
    v = wp.spatial_top(qd[i]) + wp.spatial_top(force[i]) * inverse_mass * dt
    qd[i] = wp.spatial_vector(v, wp.spatial_bottom(qd[i]))
    q[i] = wp.transform(wp.transform_get_translation(q[i]) + v * dt, wp.transform_get_rotation(q[i]))


class ExposedRigid(newton.solvers.SolverBase):
    """Test rigid solver: ALL integration history is native State.body_q and body_qd."""

    def __init__(self, model, inverse_mass=1.0):
        super().__init__(model)
        self.inverse_mass = inverse_mass

    def step(self, state_in, state_out, control, contacts, dt, observables=None):
        wp.launch(rigid_advance, self.model.body_count, [state_in.body_q, state_in.body_qd, state_in.body_f, self.inverse_mass, dt],
                  device=self.model.device)

    def reset(self, state, world_mask=None, flags=0):
        pass
