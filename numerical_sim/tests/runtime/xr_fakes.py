# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake pyopenxr (and fake Slang/CUDA for the session tests) shared by the XR tests (test_xr.py, the Isaac Lab XR
teleop test). Adapted from omnisurg's tests/test_openxr.py. Not a test module."""

import ctypes
import math
from enum import IntEnum, IntFlag
from types import SimpleNamespace


class Rec(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        if args:
            kwargs["value"] = args[0]
        super().__init__(**kwargs)


class SessionState(IntEnum):
    READY = 1
    STOPPING = 2
    EXITING = 3
    LOSS_PENDING = 4


class Flags(IntFlag):
    ORIENTATION_VALID_BIT = 1
    POSITION_VALID_BIT = 2


class ActionType(IntEnum):
    BOOLEAN_INPUT = 1
    FLOAT_INPUT = 2
    VECTOR2F_INPUT = 3
    POSE_INPUT = 4
    VIBRATION_OUTPUT = 100


class HapticBaseHeader(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int)]


class HapticVibration(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("duration", ctypes.c_int64), ("frequency", ctypes.c_float), ("amplitude", ctypes.c_float)]


class FakeXr:
    """The pyopenxr surface XrSession uses; records calls in ``events``. ``images`` (optional) returns real swapchain
    images: a callable ``(width, height) -> list of VkImage handles``."""

    class ApiVersionUnsupportedError(Exception):
        pass

    class PathUnsupportedError(Exception):
        pass

    class FormFactorUnavailableError(Exception):
        pass

    class EventUnavailable(Exception):
        pass

    class SessionNotFocused(Exception):
        pass

    class SessionLostError(Exception):
        pass

    FormFactor = SimpleNamespace(HEAD_MOUNTED_DISPLAY=1)
    ViewConfigurationType = SimpleNamespace(PRIMARY_STEREO=2)
    ReferenceSpaceType = SimpleNamespace(LOCAL=1, VIEW=2)
    SwapchainUsageFlags = SimpleNamespace(COLOR_ATTACHMENT_BIT=1)
    ViewStateFlags = SpaceLocationFlags = Flags
    ActionType = ActionType
    SessionState = SessionState
    EnvironmentBlendMode = SimpleNamespace(OPAQUE=1)
    StructureType = IntEnum("StructureType", {"EVENT_DATA_INSTANCE_LOSS_PENDING": 1, "EVENT_DATA_SESSION_STATE_CHANGED": 2})
    SwapchainImageVulkanKHR = type("SwapchainImageVulkanKHR", (), {})
    INFINITE_DURATION = 2**63 - 1
    XR_API_VERSION_1_1 = "1.1"
    MIN_HAPTIC_DURATION = -1
    FREQUENCY_UNSPECIFIED = 0
    HapticVibration, HapticBaseHeader = HapticVibration, HapticBaseHeader
    ApplicationInfo = InstanceCreateInfo = SystemGetInfo = GraphicsBindingVulkanKHR = ActionSetCreateInfo = Rec
    ActionCreateInfo = InteractionProfileSuggestedBinding = ActionSuggestedBinding = SessionActionSetsAttachInfo = Rec
    ActionSpaceCreateInfo = ActionsSyncInfo = ActiveActionSet = ActionStateGetInfo = Posef = SessionCreateInfo = Rec
    ReferenceSpaceCreateInfo = SwapchainCreateInfo = SessionBeginInfo = FrameWaitInfo = FrameBeginInfo = ViewLocateInfo = Rec
    SwapchainImageAcquireInfo = SwapchainImageWaitInfo = SwapchainImageReleaseInfo = CompositionLayerProjectionView = Rec
    SwapchainSubImage = Rect2Di = Offset2Di = Extent2Di = CompositionLayerProjection = FrameEndInfo = HapticActionInfo = Rec

    def __init__(self, images=None, size=(64, 48)):
        self.events, self.images, self.size = [], images, size
        self.client = False
        self.pending_events = []
        self.should_render = True
        self.display_time = 1_000_000_000
        self.buttons = {"a": True, "x": False}
        self.trigger = {"left": 0.25, "right": 0.75}

    def enumerate_instance_extension_properties(self):
        return [SimpleNamespace(extension_name=b"XR_KHR_vulkan_enable")]

    def create_instance(self, info):
        self.events.append("create_instance")
        return "instance"

    def string_to_path(self, instance, value):
        return value

    def create_action_set(self, instance, info):
        return "action-set"

    def create_action(self, action_set, info):
        self.events.append(("action", info.action_name, info.action_type))
        return f"action-{info.action_name}"

    def suggest_interaction_profile_bindings(self, instance, info):
        self.events.append(("bindings", info.interaction_profile, len(info.suggested_bindings)))

    def get_system(self, instance, info):
        if not self.client:
            raise self.FormFactorUnavailableError()
        return 7

    def enumerate_view_configuration_views(self, instance, system_id, view_type):
        w, h = self.size
        return [SimpleNamespace(recommended_image_rect_width=w, recommended_image_rect_height=h) for _ in range(2)]

    def create_session(self, instance, info):
        self.events.append("create_session")
        self.binding = info.next
        if getattr(self, "auto_ready", False):  # a runtime that starts the session right away (render-thread tests)
            self.pending_events.append(SessionState.READY)
        return "session"

    def attach_session_action_sets(self, session, info):
        pass

    def create_action_space(self, session, info):
        return f"hand-space-{str(info.subaction_path).rsplit('/', 1)[-1]}"

    def create_reference_space(self, session, info):
        return f"space-{info.reference_space_type}"

    def enumerate_swapchain_formats(self, session):
        return [43, 37]

    def create_swapchain(self, session, info):
        self.events.append(("create_swapchain", info.format, info.width, info.height))
        return f"swapchain-{len([e for e in self.events if e[0] == 'create_swapchain'])}"

    def enumerate_swapchain_images(self, swapchain, image_type):
        handles = self.images(*self.size) if self.images is not None else [10_000 + int(swapchain[-1])]
        return [SimpleNamespace(image=ctypes.cast(ctypes.c_void_p(h), ctypes.POINTER(ctypes.c_int))) for h in handles]

    def poll_event(self, instance):
        if self.pending_events:
            state = self.pending_events.pop(0)
            return SimpleNamespace(type=2, state=state)
        raise self.EventUnavailable()

    def begin_session(self, session, info):
        self.events.append("begin_session")

    def end_session(self, session):
        self.events.append("end_session")

    def wait_frame(self, session, info):
        self.display_time += 11_000_000
        return SimpleNamespace(predicted_display_time=self.display_time, predicted_display_period=11_000_000,
                               should_render=self.should_render)

    def begin_frame(self, session, info):
        self.events.append("begin_frame")

    def locate_views(self, session, info):
        half = math.radians(40.0)
        views = [SimpleNamespace(pose=self._pose((x, 0.0, 0.0)), fov=SimpleNamespace(angle_left=-half, angle_right=half,
                                                                                    angle_up=half, angle_down=-half))
                 for x in (-0.032, 0.032)]
        return SimpleNamespace(view_state_flags=Flags.ORIENTATION_VALID_BIT | Flags.POSITION_VALID_BIT), views

    @staticmethod
    def _pose(p):
        return SimpleNamespace(position=SimpleNamespace(x=p[0], y=p[1], z=p[2]), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))

    def sync_actions(self, session, info):
        pass

    def _hand(self, info):
        return str(info.subaction_path).rsplit("/", 1)[-1]

    def get_action_state_pose(self, session, info):
        return SimpleNamespace(is_active=True)

    def get_action_state_float(self, session, info):
        hand = self._hand(info)
        value = self.trigger[hand] if info.action == "action-trigger" else 0.5
        return SimpleNamespace(is_active=True, current_state=value)

    def get_action_state_vector2f(self, session, info):
        return SimpleNamespace(is_active=True, current_state=SimpleNamespace(x=0.1, y=-0.2))

    def get_action_state_boolean(self, session, info):
        name = str(info.action).removeprefix("action-")
        return SimpleNamespace(is_active=True, current_state=self.buttons.get(name, False))

    def locate_space(self, space, base_space, display_time):
        position = (0.0, 0.0, 0.0) if space == "space-2" else ((-0.2, -0.3, -0.4) if space.endswith("left") else (0.2, -0.3, -0.4))
        return SimpleNamespace(location_flags=Flags.ORIENTATION_VALID_BIT | Flags.POSITION_VALID_BIT, pose=self._pose(position))

    def acquire_swapchain_image(self, swapchain, info):
        self.events.append(("acquire", swapchain))
        return 0

    def wait_swapchain_image(self, swapchain, info):
        pass

    def release_swapchain_image(self, swapchain, info):
        self.events.append(("release", swapchain))

    def end_frame(self, session, info):
        self.events.append(("end_frame", len(info.layers)))

    def apply_haptic_feedback(self, session, info, header):
        vibration = ctypes.cast(ctypes.byref(header), ctypes.POINTER(HapticVibration)).contents  # as the runtime reads it
        self.events.append(("haptic", self._hand(info), round(float(vibration.amplitude), 3)))

    def stop_haptic_feedback(self, session, info):
        self.events.append(("haptic_stop", self._hand(info)))

    def destroy_swapchain(self, swapchain):
        self.events.append(("destroy_swapchain", swapchain))

    def destroy_space(self, space):
        pass

    def destroy_session(self, session):
        self.events.append("destroy_session")

    def destroy_instance(self, instance):
        self.events.append("destroy_instance")

    def destroy_action_set(self, action_set):
        pass


class FakeHandleType(IntEnum):
    VkInstance = 4
    VkPhysicalDevice = 5
    VkDevice = 6
    VkQueue = 7


class FakeDevice:
    supports_cuda_interop = True

    def __init__(self, events):
        self.events = events
        self.native_handles = [SimpleNamespace(type=FakeHandleType.VkInstance, value=300),
                               SimpleNamespace(type=FakeHandleType.VkPhysicalDevice, value=301),
                               SimpleNamespace(type=FakeHandleType.VkDevice, value=302)]

    def get_native_command_queue_info(self):
        return SimpleNamespace(handle=SimpleNamespace(type=FakeHandleType.VkQueue, value=303), family_index=4, queue_index=0)

    def create_texture_from_native_handle(self, handle, desc):
        self.events.append(("wrap", handle, desc.width, desc.height))
        return SimpleNamespace(handle=handle, close=lambda: self.events.append(("texture_close", handle)))

    def create_command_encoder(self):
        return SimpleNamespace(set_texture_state=lambda t, s: None, finish=lambda: "commands")

    def submit_command_buffer(self, commands, **kwargs):
        self.events.append(("submit", "cuda_stream" in kwargs))

    def close(self):
        self.events.append("device_close")


class FakeSpy:
    DeviceType = SimpleNamespace(vulkan="vulkan")
    NativeHandleType = FakeHandleType
    Format = SimpleNamespace(rgba8_unorm_srgb="rgba-srgb", bgra8_unorm_srgb="bgra-srgb", rgba8_unorm="rgba", bgra8_unorm="bgra")
    TextureType = SimpleNamespace(texture_2d="2d")
    TextureUsage = SimpleNamespace(render_target=1)
    ResourceState = SimpleNamespace(render_target="render-target")
    TextureDesc = Rec
    SHADER_PATH = "/slangpy/shaders"

    class SlangCompilerOptions:
        include_paths = []

    def __init__(self, events):
        self.events = events

    def get_cuda_current_context_native_handles(self):
        return ["ctx"]

    def Device(self, **kwargs):
        self.events.append(("device", kwargs["adapter_luid"][:2], tuple(kwargs["additional_vulkan_device_extensions"])))
        return FakeDevice(self.events)


class FakeCuda:
    def __init__(self):
        self.context = 0

    def set_current_context(self, context):
        self.context = context

    def current_context(self):
        return self.context

    def device_uuid(self, ordinal):
        return bytes(range(16))


def install_fake_xr(monkeypatch, tmp_path, spy, size=(160, 120)):
    """The real XR renderer on the real GPU without a headset: ``sys.modules["xr"]`` becomes a ``FakeXr`` whose
    swapchain images are textures made on the session's own Slang device (collected in ``owned``); the Vulkan
    requirement getters are stubbed and the physical-device check learns the real one. Set ``holder.session`` to the
    renderer's session once it is built. Returns ``(xr, owned, holder)``."""
    import sys

    from i4h_standalone.xr import session as xr_session

    owned = []

    def images(width, height):
        device = holder.session.device
        textures = [device.create_texture(format=spy.Format.rgba8_unorm_srgb, width=width, height=height,
                                          usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource
                                          | spy.TextureUsage.copy_source, label="fake-swapchain")]
        owned.extend(textures)
        return [int(t.native_handle.value) for t in textures]

    xr = FakeXr(images=images, size=size)
    xr.client = True
    monkeypatch.setitem(sys.modules, "xr", xr)
    physical = {}

    def graphics_device(xr_module, instance, system_id, vk_instance):
        return physical["value"]

    orig_init = xr_session.XrSession.__init__

    def init(self, *args, **kwargs):
        kwargs.update(requirements_getter=lambda *a: None, instance_extensions_getter=lambda *a: (),
                      device_extensions_getter=lambda *a: (), graphics_device_getter=graphics_device)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(xr_session.XrSession, "__init__", init)
    orig_try = xr_session.XrSession._try_create_session

    def try_create(self):  # learn the real VkPhysicalDevice once the device exists: patch around device creation
        real_device = self.spy.Device

        def device(**kwargs):
            d = real_device(**kwargs)
            physical["value"] = next(h.value for h in d.native_handles if h.type == self.spy.NativeHandleType.VkPhysicalDevice)
            return d

        self.spy = SimpleNamespace(**{n: getattr(spy, n) for n in dir(spy) if not n.startswith("__")})
        self.spy.Device = device
        return orig_try(self)

    monkeypatch.setattr(xr_session.XrSession, "_try_create_session", try_create)
    holder = SimpleNamespace(session=None)
    monkeypatch.setenv("XR_RUNTIME_JSON", str(tmp_path / "unused.json"))
    return xr, owned, holder
