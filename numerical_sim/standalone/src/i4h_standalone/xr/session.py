# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenXR session (pyopenxr, Vulkan): port of omnisurg's ``PyOpenXRRuntimeAdapter`` (``rendering/openxr.py``
249-1548), without the D3D12 path and with the full Quest controller action set.

Lifecycle, all on the render thread:

1. ``start()`` creates the instance and the action set (instance creation needs the runtime selected:
   ``xr.runtime``).
2. ``begin_frame()`` polls events and, until a headset is there, tries to create the session. It then creates the
   Slang Vulkan device on Warp's CUDA context with the Vulkan extensions OpenXR asks for, and checks that OpenXR
   chose the same physical GPU (CUDA device UUID). ``on_device(device)`` lets the renderer build on it. The session
   gets one swapchain per eye, whose images are wrapped as Slang textures (no copy).
3. With the session running, each frame is ``begin_frame()`` -> ``Frame`` (wait/begin, the eye views, the head pose,
   the controller actions at the predicted display time) and then ``end_frame(encode)``: acquire both eyes' images,
   ``encode(encoder, textures, frame)`` records the renderer's work, submit, release, and ``end_frame`` with one
   projection layer. ``end_frame`` always runs, from a ``finally``.
4. A disconnect (STOPPING) or a lost session/instance tears the session down (``on_release()`` first: the renderer
   drops everything it made on the device) and waits for the headset again; ``system`` mode ends instead.

Controller samples (``hand_sample``) and haptics (``apply_haptic``) are read and written here, on this thread only;
the haptic servo hands its command over through ``set_haptic`` (a plain assignment, applied at the next frame).
"""

from __future__ import annotations

import ctypes
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

HANDS = ("left", "right")
_VULKAN_FORMATS = (  # (VkFormat, slangpy format): the first the runtime offers wins
    (43, "rgba8_unorm_srgb"),  # VK_FORMAT_R8G8B8A8_SRGB
    (50, "bgra8_unorm_srgb"),  # VK_FORMAT_B8G8R8A8_SRGB
    (37, "rgba8_unorm"),  # VK_FORMAT_R8G8B8A8_UNORM
    (44, "bgra8_unorm"),  # VK_FORMAT_B8G8R8A8_UNORM
)
# face buttons: action name -> (hand, component) on the Quest 3 touch controllers
BUTTONS = {"a": ("right", "a"), "b": ("right", "b"), "x": ("left", "x"), "y": ("left", "y")}


@dataclass
class EyeSwapchain:
    handle: Any
    images: tuple
    width: int
    height: int
    textures: tuple = ()  # the images wrapped as Slang textures


@dataclass
class Frame:
    """One display frame: ``views`` per eye ``(position, rotation xyzw, fov (left, right, up, down))`` in LOCAL space;
    ``period`` the predicted display period (s)."""

    should_render: bool
    views: tuple = ()
    head: tuple | None = None
    period: float = 0.0
    display_time: int = 0
    samples: dict = field(default_factory=dict)


def _pose(pose) -> tuple:
    p, o = pose.position, pose.orientation
    return (float(p.x), float(p.y), float(p.z)), (float(o.x), float(o.y), float(o.z), float(o.w))


def _location_valid(xr, flags) -> bool:
    required = int(xr.SpaceLocationFlags.ORIENTATION_VALID_BIT | xr.SpaceLocationFlags.POSITION_VALID_BIT)
    return (int(flags) & required) == required


def _xr_handle(xr_struct, field_name: str, value: int):
    handle_type = dict(getattr(xr_struct, "_fields_", ())).get(field_name)
    return value if handle_type is None else ctypes.cast(ctypes.c_void_p(value), handle_type)


def _pointer_value(value) -> int:
    return int(value) if isinstance(value, int) else int(ctypes.cast(value, ctypes.c_void_p).value or 0)


def _native_handle(handle, expected_type, description: str) -> int:
    if getattr(handle, "type", None) != expected_type:
        raise RuntimeError(f"Slang {description} is not a {getattr(expected_type, 'name', expected_type)} native handle")
    value = int(getattr(handle, "value", 0) or 0)
    if not value:
        raise RuntimeError(f"Slang {description} native handle is null")
    return value


def _close(resource) -> None:
    for name in ("close", "destroy", "release"):
        method = getattr(resource, name, None)
        if callable(method):
            method()
            return


def _call_proc(xr, instance, name: str, proc_type, *args):
    function = ctypes.cast(xr.get_instance_proc_addr(instance, name), proc_type)
    result = xr.check_result(function(*args))
    if result.is_exception():
        raise result


def vulkan_requirements(xr, instance, system_id):
    requirements = xr.GraphicsRequirementsVulkanKHR()
    _call_proc(xr, instance, "xrGetVulkanGraphicsRequirementsKHR", xr.PFN_xrGetVulkanGraphicsRequirementsKHR,
               instance, system_id, ctypes.byref(requirements))
    return requirements


def _vulkan_extensions(xr, instance, system_id, name: str, proc_type) -> tuple:
    count = ctypes.c_uint32()
    _call_proc(xr, instance, name, proc_type, instance, system_id, 0, ctypes.byref(count), None)
    buffer = ctypes.create_string_buffer(max(1, int(count.value)))
    _call_proc(xr, instance, name, proc_type, instance, system_id, int(count.value), ctypes.byref(count),
               ctypes.cast(buffer, ctypes.c_char_p))
    return tuple(buffer.value.decode("utf-8").split())


def vulkan_instance_extensions(xr, instance, system_id) -> tuple:
    return _vulkan_extensions(xr, instance, system_id, "xrGetVulkanInstanceExtensionsKHR", xr.PFN_xrGetVulkanInstanceExtensionsKHR)


def vulkan_device_extensions(xr, instance, system_id) -> tuple:
    return _vulkan_extensions(xr, instance, system_id, "xrGetVulkanDeviceExtensionsKHR", xr.PFN_xrGetVulkanDeviceExtensionsKHR)


def vulkan_graphics_device(xr, instance, system_id, vk_instance: int) -> int:
    handle_type = dict(xr.GraphicsBindingVulkanKHR._fields_)["physical_device"]
    physical_device = handle_type()
    _call_proc(xr, instance, "xrGetVulkanGraphicsDeviceKHR", xr.PFN_xrGetVulkanGraphicsDeviceKHR, instance, system_id,
               _xr_handle(xr.GraphicsBindingVulkanKHR, "instance", vk_instance), ctypes.byref(physical_device))
    return _pointer_value(physical_device)


class XrSession:
    def __init__(self, xr_module, spy, cuda, warp_device, *, runtime_mode: str = "managed", shader_dir=None,
                 on_device: Callable | None = None, on_release: Callable | None = None, debug_layers: bool = False,
                 requirements_getter=vulkan_requirements, instance_extensions_getter=vulkan_instance_extensions,
                 device_extensions_getter=vulkan_device_extensions, graphics_device_getter=vulkan_graphics_device):
        self.xr, self.spy, self.cuda, self.warp_device = xr_module, spy, cuda, warp_device
        if not bool(getattr(warp_device, "is_cuda", False)):
            raise RuntimeError("XR needs a CUDA scene device (the Slang device shares Warp's CUDA context)")
        self.runtime_mode, self.shader_dir, self.debug_layers = runtime_mode, shader_dir, bool(debug_layers)
        self.on_device, self.on_release = on_device, on_release
        self.resource_lock = None
        self._requirements, self._instance_ext = requirements_getter, instance_extensions_getter
        self._device_ext, self._graphics_device = device_extensions_getter, graphics_device_getter
        self.instance = self.system_id = self.session = self.device = None
        self.local_space = self.view_space = self.action_set = None
        self.actions: dict[str, Any] = {}
        self.hand_paths: dict[str, Any] = {}
        self.hand_spaces: dict[str, Any] = {}
        self.swapchains: list[EyeSwapchain] = []
        self.color_format = None
        self.running = False  # the session is running (READY .. STOPPING)
        self.alive = True  # False once system mode's session ended or close()
        self.state = None
        self.reconnects = self.focus_misses = 0
        self._samples = self._inactive()
        self._haptics: dict[str, tuple[float, float, float] | None] = {h: None for h in HANDS}  # (amplitude, frequency, duration)
        self._haptic_on = {h: False for h in HANDS}
        self._uuid: bytes | None = None
        self._frame_state = None

    # instance and actions -------------------------------------------------------------------------------------

    def start(self) -> None:
        self._create_instance()

    def _create_instance(self) -> None:
        xr = self.xr
        available = {bytes(p.extension_name).split(b"\0", 1)[0].decode("utf-8") for p in xr.enumerate_instance_extension_properties()}
        if "XR_KHR_vulkan_enable" not in available:
            raise RuntimeError("the OpenXR runtime does not support XR_KHR_vulkan_enable")
        application = {"application_name": "i4h-sim", "application_version": 1, "engine_name": "i4h-sim", "engine_version": 1}
        api_11 = getattr(xr, "XR_API_VERSION_1_1", None)
        extensions = ["XR_KHR_vulkan_enable"]
        openxr_11 = api_11 is not None
        if openxr_11:
            application["api_version"] = api_11
        try:
            self.instance = xr.create_instance(xr.InstanceCreateInfo(application_info=xr.ApplicationInfo(**application),
                                                                     enabled_extension_names=extensions))
        except Exception as exc:
            unsupported = getattr(xr, "ApiVersionUnsupportedError", None)
            if unsupported is None or not isinstance(exc, unsupported):
                raise
            application.pop("api_version", None)
            if "XR_META_touch_controller_plus" in available:
                extensions.append("XR_META_touch_controller_plus")
            self.instance = xr.create_instance(xr.InstanceCreateInfo(application_info=xr.ApplicationInfo(**application),
                                                                     enabled_extension_names=extensions))
            openxr_11 = False
        profiles = ["/interaction_profiles/oculus/touch_controller"]
        if openxr_11:
            profiles.append("/interaction_profiles/meta/touch_plus_controller")
        elif "XR_META_touch_controller_plus" in extensions:
            profiles.append("/interaction_profiles/meta/touch_controller_plus")
        self._create_actions(profiles)

    def _create_actions(self, profiles) -> None:
        xr = self.xr
        self.hand_paths = {h: xr.string_to_path(self.instance, f"/user/hand/{h}") for h in HANDS}
        subaction = list(self.hand_paths.values())
        self.action_set = xr.create_action_set(self.instance, xr.ActionSetCreateInfo(
            action_set_name="i4h_input", localized_action_set_name="i4h-sim input", priority=0))

        def action(name, kind, paths=subaction):
            self.actions[name] = xr.create_action(self.action_set, xr.ActionCreateInfo(
                action_name=name, localized_action_name=name.replace("_", " "), action_type=kind, subaction_paths=paths))

        T = xr.ActionType
        action("grip_pose", T.POSE_INPUT)
        action("trigger", T.FLOAT_INPUT)
        action("squeeze", T.FLOAT_INPUT)
        action("thumbstick", T.VECTOR2F_INPUT)
        action("thumbstick_click", T.BOOLEAN_INPUT)
        action("haptic", T.VIBRATION_OUTPUT)
        for name, (hand, _) in BUTTONS.items():
            action(name, T.BOOLEAN_INPUT, [self.hand_paths[hand]])
        bind = []
        for h in HANDS:
            for name, component in (("grip_pose", "input/grip/pose"), ("trigger", "input/trigger/value"),
                                    ("squeeze", "input/squeeze/value"), ("thumbstick", "input/thumbstick"),
                                    ("thumbstick_click", "input/thumbstick/click"), ("haptic", "output/haptic")):
                bind.append((name, f"/user/hand/{h}/{component}"))
        for name, (hand, button) in BUTTONS.items():
            bind.append((name, f"/user/hand/{hand}/input/{button}/click"))
        bindings = [xr.ActionSuggestedBinding(action=self.actions[name], binding=xr.string_to_path(self.instance, path))
                    for name, path in bind]
        for profile in profiles:
            try:
                xr.suggest_interaction_profile_bindings(self.instance, xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(self.instance, profile), suggested_bindings=bindings))
            except Exception as exc:
                unsupported = getattr(xr, "PathUnsupportedError", None)
                if unsupported is None or not isinstance(exc, unsupported):
                    raise
                print(f"OpenXR interaction profile unsupported: {profile}")

    # session and device -------------------------------------------------------------------------------------------

    def _try_create_session(self) -> bool:
        with self.resource_lock or nullcontext():
            return self._create_session_resources()

    def _create_session_resources(self) -> bool:
        xr = self.xr
        try:
            system_id = xr.get_system(self.instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        except xr.FormFactorUnavailableError:
            return False  # no headset connected yet
        view_type = xr.ViewConfigurationType.PRIMARY_STEREO
        views = tuple(xr.enumerate_view_configuration_views(self.instance, system_id, view_type))
        if len(views) != 2:
            raise RuntimeError(f"OpenXR PRIMARY_STEREO must expose two views; found {len(views)}")
        dimensions = {(int(v.recommended_image_rect_width), int(v.recommended_image_rect_height)) for v in views}
        if len(dimensions) != 1:
            raise RuntimeError("XR rendering currently requires equal left and right eye image dimensions")
        self._requirements(xr, self.instance, system_id)  # required before the session (XR_KHR_vulkan_enable)
        spy = self.spy
        context = int(self.warp_device.context)
        self.cuda.set_current_context(context)
        uuid = bytes(self.cuda.device_uuid(int(self.warp_device.ordinal)))
        if self._uuid is not None and uuid != self._uuid:
            raise RuntimeError("OpenXR reconnected on a different GPU")
        self._uuid = uuid
        options = spy.SlangCompilerOptions()
        options.include_paths = [spy.SHADER_PATH] + ([str(self.shader_dir)] if self.shader_dir else [])
        self.device = spy.Device(  # DeviceDesc fields: create_device takes no extra Vulkan extensions
            type=spy.DeviceType.vulkan, enable_debug_layers=self.debug_layers, compiler_options=options,
            enable_cuda_interop=True, existing_device_handles=spy.get_cuda_current_context_native_handles(), adapter_luid=list(uuid),
            additional_vulkan_instance_extensions=list(self._instance_ext(xr, self.instance, system_id)),
            additional_vulkan_device_extensions=list(self._device_ext(xr, self.instance, system_id)))
        if self.cuda.current_context() != context:
            raise RuntimeError("Slang device creation changed the current CUDA context")
        if not bool(getattr(self.device, "supports_cuda_interop", False)):
            raise RuntimeError("the Slang Vulkan device does not support CUDA interop")
        native_api = ("get_native_command_queue_info", "create_texture_from_native_handle")
        if not all(callable(getattr(self.device, name, None)) for name in native_api):
            raise RuntimeError("slangpy lacks the native command queue / texture handle API: install a qualified SlangPy build with native Vulkan handles (see docs/xr.md)")
        native = {h.type: h for h in self.device.native_handles}
        H = spy.NativeHandleType
        vk_instance = _native_handle(native.get(H.VkInstance), H.VkInstance, "Vulkan instance")
        vk_physical = _native_handle(native.get(H.VkPhysicalDevice), H.VkPhysicalDevice, "Vulkan physical device")
        vk_device = _native_handle(native.get(H.VkDevice), H.VkDevice, "Vulkan device")
        if vk_physical != int(self._graphics_device(xr, self.instance, system_id, vk_instance)):
            raise RuntimeError("the Slang VkPhysicalDevice is not the one OpenXR selected")
        queue = self.device.get_native_command_queue_info()
        vk_queue = _native_handle(queue.handle, H.VkQueue, "Vulkan graphics queue")
        del vk_queue  # validated; the binding names the queue by family and index
        B = xr.GraphicsBindingVulkanKHR
        binding = B(instance=_xr_handle(B, "instance", vk_instance), physical_device=_xr_handle(B, "physical_device", vk_physical),
                    device=_xr_handle(B, "device", vk_device), queue_family_index=int(queue.family_index),
                    queue_index=int(queue.queue_index))
        self.session = xr.create_session(self.instance, xr.SessionCreateInfo(system_id=system_id, next=binding))
        self.system_id = system_id
        xr.attach_session_action_sets(self.session, xr.SessionActionSetsAttachInfo(action_sets=[self.action_set]))
        for h, path in self.hand_paths.items():
            self.hand_spaces[h] = xr.create_action_space(self.session, xr.ActionSpaceCreateInfo(
                action=self.actions["grip_pose"], subaction_path=path, pose_in_action_space=xr.Posef()))
        R = xr.ReferenceSpaceType
        self.local_space = xr.create_reference_space(self.session, xr.ReferenceSpaceCreateInfo(reference_space_type=R.LOCAL))
        self.view_space = xr.create_reference_space(self.session, xr.ReferenceSpaceCreateInfo(reference_space_type=R.VIEW))
        supported = {int(v) for v in xr.enumerate_swapchain_formats(self.session)}
        chosen = next((f for f in _VULKAN_FORMATS if f[0] in supported), None)
        if chosen is None:
            raise RuntimeError("OpenXR offers no RGBA/BGRA 8-bit swapchain format")
        native_format, name = chosen
        self.color_format = getattr(spy.Format, name)
        for eye, view in enumerate(views):
            width, height = int(view.recommended_image_rect_width), int(view.recommended_image_rect_height)
            handle = xr.create_swapchain(self.session, xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT, format=native_format, sample_count=1,
                width=width, height=height, face_count=1, array_size=1, mip_count=1))
            images = tuple(xr.enumerate_swapchain_images(handle, xr.SwapchainImageVulkanKHR))
            wrapped = []
            for i, image in enumerate(images):
                desc = spy.TextureDesc()
                desc.type, desc.format, desc.width, desc.height = spy.TextureType.texture_2d, self.color_format, width, height
                desc.sample_count, desc.usage = 1, spy.TextureUsage.render_target
                desc.default_state = spy.ResourceState.render_target
                desc.label = f"i4h-xr-{HANDS[eye]}-{i}"
                wrapped.append(self.device.create_texture_from_native_handle(_pointer_value(image.image), desc))
            self.swapchains.append(EyeSwapchain(handle, images, width, height, tuple(wrapped)))
        if self.on_device is not None:
            self.on_device(self)
        print(f"OpenXR headset connected: {self.swapchains[0].width}x{self.swapchains[0].height} per eye")
        return True

    # events ---------------------------------------------------------------------------------------------------------

    def _poll_events(self) -> None:
        xr = self.xr
        while self.instance is not None:
            try:
                event = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return
            kind = xr.StructureType(event.type)
            if kind == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self._lost_instance()
                return
            if kind != xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                continue
            if not hasattr(event, "state"):  # the generic buffer: view its memory as the state-changed event
                event = xr.EventDataSessionStateChanged.from_buffer(event)
            self._session_state(xr.SessionState(event.state))

    def _session_state(self, state) -> None:
        xr = self.xr
        print(f"OpenXR session state: {getattr(state, 'name', state)} ({int(state)})", flush=True)
        self.state = state
        if state == xr.SessionState.READY and self.session is not None:
            xr.begin_session(self.session, xr.SessionBeginInfo(primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO))
            self.running = True
            print("OpenXR session running")
        elif state == xr.SessionState.STOPPING:
            if self.session is not None and self.running:
                xr.end_session(self.session)
            self.running = False
            if self.runtime_mode != "system":
                self._destroy_session()
                self.reconnects += 1
                print("OpenXR headset disconnected; waiting for it to reconnect")
        elif state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
            if self.runtime_mode == "system":
                self.alive = False
                self._destroy_all()
            else:
                self._recreate_instance()

    def _lost_instance(self) -> None:
        if self.runtime_mode == "system":
            self.alive = False
            self._destroy_all()
        else:
            self._recreate_instance()

    def _recreate_instance(self) -> None:
        self._destroy_all()
        self._create_instance()
        self.reconnects += 1

    def _recover(self, exc: Exception) -> bool:
        """A runtime loss mid-frame: tear down and wait (not in system mode). False: not a loss, re-raise."""
        if self.runtime_mode == "system":
            return False
        xr = self.xr
        lost_instance = tuple(c for c in (getattr(xr, "InstanceLostError", None),) if isinstance(c, type))
        lost_session = tuple(c for c in (getattr(xr, "SessionLostError", None), getattr(xr, "SessionLossPending", None))
                             if isinstance(c, type))
        if lost_instance and isinstance(exc, lost_instance):
            self._recreate_instance()
            return True
        if lost_session and isinstance(exc, lost_session):
            self._destroy_session()
            self.reconnects += 1
            return True
        return False

    # per frame ------------------------------------------------------------------------------------------------------

    def begin_frame(self) -> Frame | None:
        """Poll, connect, and when the session runs: wait for and begin a frame. None: nothing to render this call (no
        headset or not running yet; the caller paces itself)."""
        try:
            return self._begin_frame()
        except Exception as exc:  # noqa: BLE001 - a runtime loss is recovered, anything else re-raised
            if self._recover(exc):
                return None
            raise

    def _begin_frame(self) -> Frame | None:
        if not self.alive:
            return None
        self._poll_events()
        if self.session is None and (self.instance is None or not self._try_create_session()):
            return None
        if not self.running:
            return None
        xr = self.xr
        state = xr.wait_frame(self.session, xr.FrameWaitInfo())
        xr.begin_frame(self.session, xr.FrameBeginInfo())
        self._frame_state = state
        view_state, views = xr.locate_views(self.session, xr.ViewLocateInfo(
            view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO, display_time=state.predicted_display_time,
            space=self.local_space))
        valid = int(xr.ViewStateFlags.ORIENTATION_VALID_BIT | xr.ViewStateFlags.POSITION_VALID_BIT)
        views_ok = len(views) == 2 and (int(view_state.view_state_flags) & valid) == valid
        eye_views = tuple((*_pose(v.pose), (float(v.fov.angle_left), float(v.fov.angle_right), float(v.fov.angle_up),
                                            float(v.fov.angle_down))) for v in views) if views_ok else ()
        self._raw_views = tuple(views)
        head = None
        location = xr.locate_space(self.view_space, self.local_space, state.predicted_display_time)
        if _location_valid(xr, getattr(location, "location_flags", 0)):
            head = _pose(location.pose)
        self._sync_actions(state.predicted_display_time)
        return Frame(should_render=bool(state.should_render) and views_ok, views=eye_views, head=head,
                     period=int(state.predicted_display_period) * 1e-9, display_time=int(state.predicted_display_time),
                     samples={h: dict(s) for h, s in self._samples.items()})

    def end_frame(self, encode: Callable | None, cuda_stream=None) -> bool:
        """Acquire both eyes' swapchain images, ``encode(encoder, textures)`` (returns False to submit nothing), submit
        (ordered with ``cuda_stream``, a slangpy ``NativeHandle``: the renderer's snapshot copies), release, end the
        frame. Always ends a begun frame."""
        if self._frame_state is None or self.session is None:
            return False
        xr = self.xr
        layers, acquired = [], []
        state = self._frame_state
        try:
            if encode is not None:
                textures = []
                for eye in self.swapchains:
                    index = int(xr.acquire_swapchain_image(eye.handle, xr.SwapchainImageAcquireInfo()))
                    acquired.append(eye)
                    xr.wait_swapchain_image(eye.handle, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                    textures.append(eye.textures[index])
                encoder = self.device.create_command_encoder()
                if encode(encoder, textures):
                    for texture in textures:  # XR_KHR_vulkan_enable: released in the colour-attachment layout
                        encoder.set_texture_state(texture, self.spy.ResourceState.render_target)
                    if cuda_stream is not None:
                        self.device.submit_command_buffer(encoder.finish(), cuda_stream=cuda_stream)
                    else:
                        self.device.submit_command_buffer(encoder.finish())
                    views = []
                    for eye, view in zip(self.swapchains, self._raw_views):
                        views.append(xr.CompositionLayerProjectionView(pose=view.pose, fov=view.fov, sub_image=xr.SwapchainSubImage(
                            swapchain=eye.handle, image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(eye.width, eye.height)),
                            image_array_index=0)))
                    layer = xr.CompositionLayerProjection(space=self.local_space, views=views)
                    layers.append(ctypes.pointer(layer) if hasattr(xr, "CompositionLayerBaseHeader") else layer)
            self._apply_haptics()
        except Exception as exc:  # noqa: BLE001
            if not self._recover(exc):
                raise
            return False
        finally:
            for eye in acquired:
                if self.session is not None:
                    xr.release_swapchain_image(eye.handle, xr.SwapchainImageReleaseInfo())
            if self.session is not None:
                xr.end_frame(self.session, xr.FrameEndInfo(display_time=state.predicted_display_time,
                                                           environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE, layers=layers))
            self._frame_state = None
        return bool(layers)

    # controllers ------------------------------------------------------------------------------------------------------

    @staticmethod
    def _inactive() -> dict:
        return {h: {"active": False} for h in HANDS}

    def _state(self, getter: str, name: str, path):
        xr = self.xr
        return getattr(xr, getter)(self.session, xr.ActionStateGetInfo(action=self.actions[name], subaction_path=path))

    def _sync_actions(self, display_time) -> None:
        """Every hand's grip pose (LOCAL space, metres), trigger, squeeze, thumbstick and buttons at the display time."""
        self._samples = self._inactive()
        xr = self.xr
        try:
            xr.sync_actions(self.session, xr.ActionsSyncInfo(active_action_sets=[xr.ActiveActionSet(action_set=self.action_set)]))
        except xr.SessionNotFocused:
            self.focus_misses += 1
            return
        for h, path in self.hand_paths.items():
            if not bool(self._state("get_action_state_pose", "grip_pose", path).is_active):
                continue
            location = xr.locate_space(self.hand_spaces[h], self.local_space, display_time)
            if not _location_valid(xr, getattr(location, "location_flags", 0)):
                continue

            def value(name, getter="get_action_state_float", h=h, path=path):
                s = self._state(getter, name, path)
                return s.current_state if bool(s.is_active) else None

            trigger, squeeze = value("trigger") or 0.0, value("squeeze") or 0.0
            stick = value("thumbstick", "get_action_state_vector2f")
            buttons = {"thumbstick": bool(value("thumbstick_click", "get_action_state_boolean"))}
            for name, (hand, _) in BUTTONS.items():
                if hand == h:
                    buttons[name] = bool(value(name, "get_action_state_boolean"))
            position, rotation = _pose(location.pose)
            clean = lambda v: min(max(float(v), 0.0), 1.0) if math.isfinite(float(v)) else 0.0  # noqa: E731
            self._samples[h] = {"active": True, "position": position, "rotation": rotation, "trigger": clean(trigger),
                                "squeeze": clean(squeeze), "thumbstick": (float(stick.x), float(stick.y)) if stick is not None else (0.0, 0.0),
                                "buttons": buttons}

    def hand_sample(self, hand: str) -> dict:
        """The hand's latest sample (LOCAL space; the renderer maps it into the world through its anchor)."""
        return dict(self._samples[hand])

    def set_haptic(self, hand: str, amplitude: float, frequency: float = 0.0, duration: float = 0.0) -> None:
        """Queue a vibration for the next frame (any thread: a tuple assignment); amplitude 0 stops it."""
        self._haptics[hand] = (float(amplitude), float(frequency), float(duration))

    def _apply_haptics(self) -> None:
        xr = self.xr
        for h, command in self._haptics.items():
            if command is None or self.session is None:
                continue
            self._haptics[h] = None
            amplitude, frequency, duration = command
            info = xr.HapticActionInfo(action=self.actions["haptic"], subaction_path=self.hand_paths[h])
            if not amplitude > 0.0 or not math.isfinite(amplitude):
                if self._haptic_on[h]:
                    xr.stop_haptic_feedback(self.session, info)
                    self._haptic_on[h] = False
                continue
            vibration = xr.HapticVibration(duration=int(duration * 1e9) if duration > 0.0 else xr.MIN_HAPTIC_DURATION,
                                           frequency=float(frequency) if frequency > 0.0 else xr.FREQUENCY_UNSPECIFIED,
                                           amplitude=min(float(amplitude), 1.0))
            xr.apply_haptic_feedback(self.session, info, ctypes.cast(ctypes.pointer(vibration), ctypes.POINTER(xr.HapticBaseHeader)).contents)
            self._haptic_on[h] = True

    # teardown ---------------------------------------------------------------------------------------------------------

    def _destroy_session(self) -> None:
        with self.resource_lock or nullcontext():
            self._destroy_session_resources()

    def _destroy_session_resources(self) -> None:
        self._frame_state = None
        self.running = False
        self._samples = self._inactive()
        self._haptic_on = {h: False for h in HANDS}
        if self.on_release is not None and self.device is not None:
            self.on_release(self)  # the renderer drops what it made on the device first
        xr = self.xr
        for eye in self.swapchains:
            for texture in eye.textures:
                _close(texture)
            eye.textures = ()
        texture = eye = None
        for eye in reversed(self.swapchains):
            xr.destroy_swapchain(eye.handle)
        self.swapchains.clear()
        for space in self.hand_spaces.values():
            xr.destroy_space(space)
        self.hand_spaces.clear()
        for name in ("view_space", "local_space"):
            space = getattr(self, name)
            if space is not None:
                xr.destroy_space(space)
                setattr(self, name, None)
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.device is not None:
            _close(self.device)
        self.device = self.system_id = None

    def _destroy_all(self) -> None:
        self._destroy_session()
        if self.action_set is not None:
            self.xr.destroy_action_set(self.action_set)
        self.action_set = None
        self.actions.clear()
        self.hand_paths.clear()
        if self.instance is not None:
            self.xr.destroy_instance(self.instance)
            self.instance = None

    def close(self) -> None:
        if not self.alive and self.instance is None:
            return
        for h in HANDS:  # the controllers stop vibrating
            self._haptics[h] = (0.0, 0.0, 0.0)
        try:
            if self.session is not None and self.running:
                self._apply_haptics()
        except Exception:  # noqa: BLE001 - shutting down
            pass
        self._destroy_all()
        self.alive = False
