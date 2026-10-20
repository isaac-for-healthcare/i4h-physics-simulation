# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA interop for the Slang renderer.

slangpy cannot wrap a CUDA pointer, so sharing runs the other way: slangpy allocates a
buffer with ``BufferUsage.shared``, the CUDA driver imports its opaque file descriptor
and maps it, and Warp aliases the mapping with ``wp.array(ptr=...)``. Warp kernels and
copies then write straight into vertex buffers; no per-frame memcpy.

Ported from omnisurg/rendering/slang.py:562-700 (driver shim) and 3299-3369 (shared
buffer), trimmed to Linux/Vulkan (opaque FD handles).
"""

from __future__ import annotations

import ctypes

import warp as wp


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):  # keeps the size of CUDA's CUDA_EXTERNAL_MEMORY_HANDLE_DESC union
    _fields_ = [("fd", ctypes.c_int), ("win32", _Win32Handle), ("nv_sci_buf_object", ctypes.c_void_p)]


class _HandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _BufferDesc(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class _SemaphoreDesc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _HandleUnion), ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class _SemaphoreParams(ctypes.Structure):
    # CUDA_EXTERNAL_SEMAPHORE_{SIGNAL,WAIT}_PARAMS: both have a 72-byte params block.
    # Only params.fence.value is used for timeline semaphores; all other fields stay zero.
    _fields_ = [("value", ctypes.c_ulonglong), ("unused_params", ctypes.c_uint * 16),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class CudaDriver:
    OPAQUE_FD = 1
    DEDICATED = 0x1

    def __init__(self) -> None:
        cu = self._cu = ctypes.CDLL("libcuda.so.1")
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_HandleDesc)]
        cu.cuExternalMemoryGetMappedBuffer.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.POINTER(_BufferDesc)]
        self._mem_free = getattr(cu, "cuMemFree_v2", cu.cuMemFree)
        self._mem_free.argtypes = [ctypes.c_ulonglong]
        cu.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        cu.cuImportExternalSemaphore.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_SemaphoreDesc)]
        cu.cuDestroyExternalSemaphore.argtypes = [ctypes.c_void_p]
        for fn in (cu.cuSignalExternalSemaphoresAsync, cu.cuWaitExternalSemaphoresAsync):
            fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_SemaphoreParams), ctypes.c_uint, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        cu.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        cu.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        for fn in (cu.cuInit, cu.cuCtxSetCurrent, cu.cuImportExternalMemory,
                   cu.cuExternalMemoryGetMappedBuffer, self._mem_free, cu.cuDestroyExternalMemory,
                   cu.cuImportExternalSemaphore, cu.cuDestroyExternalSemaphore):
            fn.restype = ctypes.c_int
        self._check(cu.cuInit(0), "cuInit")

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            name, desc = ctypes.c_char_p(), ctypes.c_char_p()
            self._cu.cuGetErrorName(result, ctypes.byref(name))
            self._cu.cuGetErrorString(result, ctypes.byref(desc))
            text = (name.value or str(result).encode()).decode("utf-8", "replace")
            if desc.value:
                text += ": " + desc.value.decode("utf-8", "replace")
            raise RuntimeError(f"{operation} failed: {text}")

    def set_current_context(self, context: int) -> None:
        self._check(self._cu.cuCtxSetCurrent(ctypes.c_void_p(int(context))), "cuCtxSetCurrent")

    def current_context(self) -> int:
        """The calling thread's current CUDA context handle (0 when none)."""
        ctx = ctypes.c_void_p()
        self._cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._check(self._cu.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        return int(ctx.value or 0)

    def device_uuid(self, ordinal: int) -> bytes:
        """The 16-byte UUID of CUDA device ``ordinal`` (Vulkan's deviceUUID of the same GPU: the XR device match)."""
        device = ctypes.c_int()
        self._cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._check(self._cu.cuDeviceGet(ctypes.byref(device), int(ordinal)), "cuDeviceGet")
        uuid = (ctypes.c_ubyte * 16)()
        get_uuid = getattr(self._cu, "cuDeviceGetUuid_v2", None) or self._cu.cuDeviceGetUuid
        get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._check(get_uuid(ctypes.cast(uuid, ctypes.c_void_p), device), "cuDeviceGetUuid")
        return bytes(uuid)

    def import_fd(self, fd: int, size: int) -> ctypes.c_void_p:
        desc = _HandleDesc()
        desc.type = self.OPAQUE_FD
        desc.handle.fd = int(fd)
        desc.size = int(size)
        desc.flags = self.DEDICATED
        memory = ctypes.c_void_p()
        self._check(self._cu.cuImportExternalMemory(ctypes.byref(memory), ctypes.byref(desc)), "cuImportExternalMemory")
        return memory

    def map_buffer(self, memory: ctypes.c_void_p, size: int) -> int:
        desc = _BufferDesc()
        desc.offset = 0
        desc.size = int(size)
        ptr = ctypes.c_ulonglong()
        self._check(self._cu.cuExternalMemoryGetMappedBuffer(ctypes.byref(ptr), memory, ctypes.byref(desc)),
                    "cuExternalMemoryGetMappedBuffer")
        return int(ptr.value)

    def release(self, memory: ctypes.c_void_p, ptr: int) -> None:
        if ptr:
            self._check(self._mem_free(ctypes.c_ulonglong(int(ptr))), "cuMemFree")
        if memory:
            self._check(self._cu.cuDestroyExternalMemory(memory), "cuDestroyExternalMemory")


_DRIVER: CudaDriver | None = None


def cuda_driver() -> CudaDriver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = CudaDriver()
    return _DRIVER


class GraphicsInterop:
    """Two one-way timelines: CUDA finishes geometry, graphics finishes reading it.

    Keep these separate from Slang's submission/lifetime fence. Its automatic CUDA interop signals
    that same fence from both APIs; texture uploads during UI font updates can stall the handoff.
    Explicit fences also cover resource creation between frames without a host wait per frame.
    """

    def __init__(self, device, warp_device):
        self.device, self.context = device, int(warp_device.context)
        self.cuda = cuda_driver()
        self.cuda.set_current_context(self.context)
        self.ready = device.create_fence({"shared": True})
        self.done = device.create_fence({"shared": True})
        self.cu_ready, self.cu_done = ctypes.c_void_p(), ctypes.c_void_p()
        self.value = 0
        try:
            for fence, semaphore in ((self.ready, self.cu_ready), (self.done, self.cu_done)):
                desc = _SemaphoreDesc()
                desc.type = 9  # CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
                desc.handle.fd = int(fence.shared_handle.value)
                self.cuda._check(self.cuda._cu.cuImportExternalSemaphore(ctypes.byref(semaphore), ctypes.byref(desc)),
                                 "cuImportExternalSemaphore")
        except Exception:
            self.close()
            raise

    def submit(self, command_buffer, stream):
        self.cuda.set_current_context(self.context)
        self.value += 1
        params = _SemaphoreParams(value=self.value)
        cu = self.cuda._cu
        self.cuda._check(cu.cuSignalExternalSemaphoresAsync(ctypes.byref(self.cu_ready), ctypes.byref(params),
                                                         1, stream.cuda_stream), "cuSignalExternalSemaphoresAsync")
        # No cuda_stream argument: it would re-enable Slang's shared timeline handoff.
        result = self.device.submit_command_buffers([command_buffer], wait_fences=[self.ready],
                                                   wait_fence_values=[self.value], signal_fences=[self.done],
                                                   signal_fence_values=[self.value])
        self.cuda._check(cu.cuWaitExternalSemaphoresAsync(ctypes.byref(self.cu_done), ctypes.byref(params),
                                                       1, stream.cuda_stream), "cuWaitExternalSemaphoresAsync")
        return result

    def close(self):
        """After graphics and CUDA have drained, release imported handles before their fences/device."""
        self.cuda.set_current_context(self.context)
        for semaphore in (self.cu_ready, self.cu_done):
            if semaphore.value:
                self.cuda._check(self.cuda._cu.cuDestroyExternalSemaphore(semaphore), "cuDestroyExternalSemaphore")
                semaphore.value = None
        self.ready = self.done = self.device = None


class SharedBuffer:
    """A slangpy buffer and the Warp array aliasing the same device memory."""

    def __init__(self, spy, device, cuda: CudaDriver, warp_device, count: int, dtype, stride: int, usage, label: str):
        size = max(1, int(count) * int(stride))
        self.buffer = device.create_buffer(
            size=size,
            usage=spy.BufferUsage.shared | usage | spy.BufferUsage.shader_resource | spy.BufferUsage.copy_destination,
            label=label,
        )
        self._context = int(warp_device.context)
        cuda.set_current_context(self._context)
        fd = int(self.buffer.shared_handle.value)
        if fd == 0:
            raise RuntimeError(f"shared buffer {label!r}: slangpy returned no shared handle")
        self._cuda = cuda
        self._memory = cuda.import_fd(fd, size)
        try:
            self._ptr = cuda.map_buffer(self._memory, size)
        except Exception:
            cuda.release(self._memory, 0)
            raise
        self.array = wp.array(ptr=self._ptr, dtype=dtype, shape=(int(count),), capacity=size, device=warp_device, copy=False)

    def close(self) -> None:
        memory, ptr = self._memory, self._ptr
        self._memory, self._ptr = ctypes.c_void_p(), 0
        if memory or ptr:
            self._cuda.set_current_context(self._context)
            self._cuda.release(memory, ptr)
        self.buffer = None
