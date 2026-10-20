# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which OpenXR runtime the session talks to (``render.xr.runtime.mode``; omnisurg ``cloudxr/service.py`` modes).

- ``managed``: CloudXR started in this process by isaacteleop's ``CloudXRLauncher`` (the runtime as a subprocess plus
  its WSS proxy; ``~/.cloudxr``). The Quest 3 connects from its browser to the web client: NVIDIA's hosted one, or with
  ``host_client: true`` the same client served from this machine by the proxy (``https://<host>:48322/client/``,
  downloaded once; the headset needs no internet). ``host`` is the LAN address the headset reaches this machine at
  (default: guessed, preferring 192.168.x over VPN and container interfaces). The launcher writes ``XR_RUNTIME_JSON``
  and the ``NV_CXR_*`` variables into this process's environment. The CloudXR EULA must have been accepted once
  (``python -m isaacteleop.cloudxr``).
- ``external``: a runtime someone else runs (``python -m isaacteleop.cloudxr``, omnisurg's
  ``omnisurg-cloudxr-service`` for CloudXR 6.2, ...): ``XR_RUNTIME_JSON`` from ``runtime.directory`` (its
  ``openxr_cloudxr.json``), ``I4H_CLOUDXR_DIR``, or the environment as it is.
- ``system``: the system's active OpenXR runtime (e.g. Quest Link through a vendor runtime): ``XR_RUNTIME_JSON`` is
  cleared.

``start()`` runs before ``import xr``; ``close()`` stops a managed runtime and restores the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

MODES = ("managed", "external", "system")
RUNTIME_KEYS = {"mode", "directory", "device_profile", "env_config", "install_dir", "host_client", "host"}
SKIP_INTERFACES = ("docker", "br-", "veth", "virbr", "gpd", "tun", "tap", "wg", "zt", "lo")  # VPN and container links
_MANIFEST = "openxr_cloudxr.json"


def lan_ipv4() -> str | None:
    """This machine's LAN IPv4 for the headset: the global addresses of the interfaces that are not VPN or container
    links, 192.168.x first, then 172.16-31.x, then 10.x (``ip -4 addr``; None if none)."""
    import ipaddress
    import subprocess

    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True,
                             timeout=3, check=False).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1].startswith(SKIP_INTERFACES):
            continue
        found.append(ipaddress.ip_interface(parts[3]).ip)
    for net in ("192.168.0.0/16", "172.16.0.0/12", "10.0.0.0/8"):
        for ip in found:
            if ip in ipaddress.ip_network(net):
                return str(ip)
    return str(found[0]) if found else None


class XrRuntime:
    def __init__(self, mode: str = "managed", directory: str | None = None, device_profile: str = "auto-webrtc",
                 env_config: str | None = None, install_dir: str = "~/.cloudxr", host_client: bool = False,
                 host: str | None = None):
        if mode not in MODES:
            raise ValueError(f"render.xr.runtime.mode must be one of {MODES}, got {mode!r}")
        self.mode, self.directory, self.device_profile = mode, directory, str(device_profile)
        self.env_config, self.install_dir = env_config, install_dir
        self.host_client, self.host = bool(host_client), host
        self.launcher = None
        self._saved: dict[str, str | None] = {}

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "XrRuntime":
        cfg = dict(cfg or {})
        unknown = set(cfg) - RUNTIME_KEYS
        if unknown:
            raise ValueError(f"render.xr.runtime: unknown keys {sorted(unknown)}; known {sorted(RUNTIME_KEYS)}")
        return cls(**cfg)

    def _set(self, name: str, value: str | None) -> None:
        if name not in self._saved:
            self._saved[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def start(self) -> None:
        if self.mode == "system":
            self._set("XR_RUNTIME_JSON", None)
            return
        if self.mode == "external":
            directory = self.directory or os.environ.get("I4H_CLOUDXR_DIR")
            if directory:
                manifest = Path(directory).expanduser() / _MANIFEST
                if not manifest.is_file():
                    raise FileNotFoundError(f"render.xr.runtime: no {_MANIFEST} in {directory}")
                self._set("XR_RUNTIME_JSON", str(manifest))
            elif not os.environ.get("XR_RUNTIME_JSON"):
                raise RuntimeError("render.xr.runtime.mode external: set runtime.directory, I4H_CLOUDXR_DIR or "
                                   "XR_RUNTIME_JSON to the running runtime's manifest")
            return
        from isaacteleop.cloudxr.launcher import CloudXRLauncher
        from isaacteleop.cloudxr.oob_teleop_env import versioned_web_client_url

        for name in ("XR_RUNTIME_JSON", "NV_CXR_RUNTIME_DIR", "NV_CXR_OUTPUT_DIR", "XRT_NO_STDIN", "CXR_INSTALL_DIR"):
            self._set(name, os.environ.get(name))  # remembered: the launcher overwrites them
        host = self.host or os.environ.get("TELEOP_PROXY_HOST") or lan_ipv4()
        if host:
            self._set("TELEOP_PROXY_HOST", host)  # isaacteleop's address for the headset (it would guess the VPN one)
        self.launcher = CloudXRLauncher(install_dir=self.install_dir, env_config=self.env_config,
                                        device_profile=self.device_profile, host_client=self.host_client)
        proxy = f"https://{host or '<this machine>'}:48322"
        if self.host_client:
            url = f"{proxy}/client/"
        else:
            try:
                import isaacteleop

                url = versioned_web_client_url(getattr(isaacteleop, "__version__", ""))
            except Exception:  # noqa: BLE001 - the URL is a convenience
                url = "https://nvidia.github.io/IsaacTeleop/client/"
        print(f"CloudXR runtime running (profile {self.device_profile}). Quest 3 browser: first open {proxy}/ and accept "
              f"the certificate, then open {url} and connect to {host or 'this machine'}")

    def close(self) -> None:
        if self.launcher is not None:
            try:
                self.launcher.stop()
            finally:
                self.launcher = None
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved.clear()
