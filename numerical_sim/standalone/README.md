# i4h-standalone

Version 0.9.0, import `i4h_standalone`, Python `>=3.12,<3.13`.
Dependencies: the other five i4h packages; optional usd/render/gl/xr extras (`usd` installs `usd-core` on non-aarch64 platforms or `usd-exchange` on aarch64 for USD assets outside Isaac Lab; inside Lab, omit it). Install locally with `uv pip install ./numerical_sim/standalone`
or use the workspace lockfile for development.

```bash
uv run i4h-sim two_worlds --data /path/to/data --render noop --steps 20
```

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.

For Isaac Lab device registration, use `i4h_standalone.devices.configure_teleop(cfg)`
after choosing `cfg.case`. Registered tasks include a lazy `i4h` device configuration. Selecting it requires
standalone; importing Isaac Lab environments or cameras does not import standalone.
The XR extra requires a separately supplied qualified SlangPy wheel; see `docs/xr.md`
in the numerical workspace.

## Capture profile

Use `--capture DIR --capture-encoder ffv1` with the Slang renderer and ffmpeg on
PATH. The default is **10 fps of simulation time**, independently of `--fps`
(render pacing). Explicit `--capture-fps` must be finite and positive. Keep the
case's render resolution fixed during a session. The rc3 qualification report
records the tested resolution and host; higher rates such as 60 fps are overrides,
not qualified throughput promises.

```bash
uv run --locked --extra render i4h-sim chole_tet_med --input scripted --steps 600 --capture capture-10 --capture-max 300
```

`--capture-max` caps actual samples; it does not request that many samples from a
short run. Capture slots are indexed from the first observed simulation timestamp
at the selected rate. The expected count follows the observed simulation duration.
A full staging queue drops samples without back-pressure and logs one warning per
session, with rate/resolution and advice to reduce load. Counters remain in
`manifest.json`. Missed video slots before a later sample repeat the previous frame
in all four streams (black before the first). `video_frames` therefore can exceed
`captured_frames`; only `frames.jsonl` rows represent actual samples. Use each row's
`index` to align its metadata with the videos. Hold-fill is not a new observation.
