# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate capture rates before case loading, while preserving render pacing."""
import argparse

import pytest
from i4h_standalone.app import cli
from i4h_standalone.app.capture import DEFAULT_CAPTURE_FPS, CaptureSettings


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_rate_before_resources(monkeypatch, rate):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid rate reached runtime setup")
    monkeypatch.setattr(cli, "read_case", unexpected)
    monkeypatch.setattr(cli, "make_renderer", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["two_worlds", "--capture-fps=" + rate])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="finite and positive"):
        CaptureSettings("unused", fps=float(rate))


@pytest.mark.parametrize("render_fps,explicit,expected", [(60, None, 10), (24, None, 10), (24, 60, 60), (60, 2.5, 2.5)])
def test_cli_default_and_explicit_rate(monkeypatch, render_fps, explicit, expected):
    parsed = []
    original = argparse.ArgumentParser.parse_args

    def parse(parser, *args, **kwargs):
        result = original(parser, *args, **kwargs)
        parsed.append(result)
        return result

    def stop(case):
        raise RuntimeError("stop before runtime")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(cli, "read_case", stop)
    args = ["two_worlds", "--fps", str(render_fps)]
    if explicit is not None:
        args += ["--capture-fps", str(explicit)]
    with pytest.raises(RuntimeError, match="stop before runtime"):
        cli.main(args)
    assert parsed[0].capture_fps == expected and parsed[0].fps == render_fps
    assert CaptureSettings("unused").fps == DEFAULT_CAPTURE_FPS == 10.0
