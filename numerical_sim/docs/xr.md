# XR: i4h-sim in a Quest 3 through CloudXR

The scene is rendered in stereo into the headset by the app's Slang renderer
(`render/xr.py`, `xr/session.py`). It runs on OpenXR (pyopenxr) and a CloudXR
runtime, with no Kit. The design is ported from omnisurg `feature-cloudxr`.

## Setup (once)

```bash
uv sync --extra xr --extra gl --group dev   # pyopenxr and isaacteleop CloudXR
uv pip install /path/to/qualified-slangpy.whl  # native Vulkan queue/texture handle support
.venv/bin/python -m isaacteleop.cloudxr                                        # accept the NVIDIA CloudXR EULA once, then Ctrl+C
```

The XR extra requires a separately supplied, qualified SlangPy wheel with native
Vulkan queue/texture handles. Public metadata does not reference a personal repository
or claim that the ordinary rendering build supports those APIs. The exact qualified
revision is retained privately. Install the wheel after syncing; use the environment's
commands directly or `uv run --no-sync` to preserve it. Do not combine the render and
XR extras in the same environment. XR currently requires equal image dimensions for
both eyes and rejects unequal sizes explicitly.

A qualified build may omit ImGui fonts; mirror panel text then requires its font assets.
The headset draws no UI.

## Run

```bash
.venv/bin/i4h-sim chole_tet_med --render xr --steps 0   # until closed (the default under xr); the case's render.xr block
```

The app starts CloudXR itself (`runtime.mode: managed`) and prints the URLs.
Chole sets `host_client: true`, so the web client is served from this machine
(downloaded once from NVIDIA, then local; the Quest needs no internet). Then,
in the Quest 3 browser:

1. Open `https://<this machine>:48322/` and accept the self-signed
   certificate. Otherwise the client cannot reach the proxy.
2. Open `https://<this machine>:48322/client/` and connect to this machine's
   address.

Without `host_client`, the client is NVIDIA's hosted page,
`https://nvidia.github.io/IsaacTeleop/client/v<isaacteleop version>/`.

`<this machine>` is `render.xr.runtime.host`, or `TELEOP_PROXY_HOST`, or a
guess that prefers the LAN over VPN and container interfaces (`ip -4 addr`).
The Quest must be on that network.

The physics runs from the start. The headset joins when the client connects.
Your first head pose is placed at the case's desktop camera, and
`world_units_per_meter` scales head motion (1.0 on chole: one scene unit per
metre).

The mirror window (`render.xr.mirror: left|right|off`) shows one eye and takes
the app's command keys (Space pauses, and so on). With `headless: true` there
is no window.

Firewall: the CloudXR WSS proxy (and the local client) listens on 48322/tcp.
The runtime uses 49100/tcp and 47998/udp for signalling and media.

```bash
sudo ufw status numbered
sudo ufw allow 48322/tcp && sudo ufw allow 49100/tcp && sudo ufw allow 47998/udp
ss -ltnup | grep -E '48322|49100|47998'      # while the app runs
```

## Quest browser: use NVIDIA's hosted client

The locally served client (`host_client: true`) works in a PC browser with the
emulator, but not on the Quest. The proxy's self-signed certificate is issued
for `localhost` only, and the Quest browser withholds WebXR from a page whose
certificate it had to click through, so Connect stays greyed out.

On the Quest:

1. Accept the proxy certificate once at `https://<this machine>:48322/`.
2. Open `https://nvidia.github.io/IsaacTeleop/client/v1.4.145/`, choose the
   Simulation profile, set the server to this machine and connect. That page's
   certificate is valid; it only opens a WebSocket to the proxy.

This needs internet on the Quest. A fully local alternative is USB with
`adb reverse` (not wired in yet).

## Controllers

`--render xr` defaults to `--input devices` with the controllers of
`render.xr.inputs`, which replace the case's desk devices. Chole binds the
right hand to `primary` and the left to `secondary`; `pickup.start` makes the right hand start on the hook and the left on the grasper (the desk bindings are untouched). Per hand:

| Control | Action |
| --- | --- |
| grip pose | the held tool follows the hand (the pickup's offsets: it moves from where it was taken) |
| trigger | the jaw (grip); `settings.grip: squeeze` uses the grip button instead |
| A / X | energy (the hook's cautery, the suction) |
| B / Y | fire (the clip applier) |
| thumbstick click | put the held tool down, or take the nearest one within `pickup.distance` |

The mapping is `settings.energy`, `fire` and `pickup`, each one of
`a b x y thumbstick`.

## Vibration (`--haptics`)

With `--haptics` the haptic servo drives the controllers like the MiniMous.
The tool force of the tool a hand holds is conditioned and capped by the
case's `haptics:` block, and `|F| / full_scale_n` sets the vibration strength
(`full_scale_n` defaults to 0.5 N, the pose cap). With `vibrate_on: force+jaw`,
the default, the grasp squeeze adds `jaw load / jaw_full_scale_n` (2 N by
default).

The vibration stops at zero force, on reset, pause and exit, and when the
servo stops refreshing for 0.1 s. The Haptics window's pose and handle groups
govern force and jaw as for the desk devices.

## Frame rate and physics

The headset paces rendering (72–120 Hz), and the physics keeps real time
beside it: chole on an emulated headset at 72 fps had step intervals with a
median of 16.7 ms and a p99 of 17.2 ms.

Connecting the headset stalls the physics once, for a few hundred ms up to a
few seconds, while the render thread builds the device and scene renderer. The
run's average rate in the exit summary includes that stall.

`render.xr.max_fps` caps the XR loop if needed; the runtime reprojects.

## Isaac Lab teleop in the headset

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export PY=$ISAACLAB_PATH/.venv/bin/python
uv pip install --python "$PY" physics assessment endoluminal surgical isaaclab "standalone[xr,gl]"
$PY scripts/isaaclab_teleop.py --case chole_tet_med --render xr --haptics [--record /tmp/demos]
```

The controllers of `render.xr.inputs` drive env 0 of the Isaac Lab env, on
either backend. A render thread draws the newest stepped state at the
headset's rate, and the env steps at `step_dt` on the main thread
(`i4h_standalone.teleop.XrView`). The `standalone[xr,gl]` install declares the pinned SlangPy fork, OpenXR and
IsaacTeleop dependencies. `--record` records the demos; the
mirror window's R, K and X keys control the takes.

## Runtime modes (`render.xr.runtime`)

| mode | what |
| --- | --- |
| `managed` (default) | isaacteleop's `CloudXRLauncher` starts the CloudXR 6.3 runtime and its WSS proxy in this process (`~/.cloudxr`), and stops them at exit. `device_profile` defaults to `auto-webrtc`. |
| `external` | A runtime someone else runs, such as `python -m isaacteleop.cloudxr` or omnisurg's `omnisurg-cloudxr-service` (CloudXR 6.2). The manifest comes from `runtime.directory/openxr_cloudxr.json`, from `I4H_CLOUDXR_DIR`, or from `XR_RUNTIME_JSON`. |
| `system` | The system's active OpenXR runtime, for example Quest Link. |

## What the headset shows

The opaque scene: tissue, vessels, instruments, rods and clips. It uses the
case's exposure and tonemap. Fluids, smoke, the desktop post chain (AO, bloom,
auto exposure) and the UI are desktop only, as in omnisurg.

A disconnect tears the session down and waits: reconnecting builds a new
device and scene renderer and anchors the head again.
