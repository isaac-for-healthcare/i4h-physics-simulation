# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A successful printout cannot hide a failed qualification process or replace evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


SPEC = importlib.util.spec_from_file_location(
    "qualify_newton18", Path(__file__).parents[2] / "scripts/qualify_newton18.py")
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.mark.parametrize("failure", ["identity", "particles", "rigid", None])
def test_reports_actual_process_exit_and_does_not_overwrite(tmp_path, monkeypatch, failure):
    calls = []

    def run(command, **kwargs):
        probe = command[-1]
        calls.append(probe)
        kwargs["stdout"].write("probe body passed\n")
        return subprocess.CompletedProcess(command, -11 if probe == failure else 0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    out = tmp_path / "result"
    argv = ["--stack", "core", "--newton-source", str(tmp_path), "--out", str(out)]
    assert qualification.main(argv) == int(failure is not None)
    report = json.loads((out / "report.json").read_text())
    assert hashlib.sha256((out / "probe.py").read_bytes()).hexdigest() == report["script_sha256"]
    if failure is not None:
        assert report["checks"][failure]["status"] == "failed"
        assert report["checks"][failure]["exit_code"] == -11
    if failure == "identity":
        assert calls == ["identity"]
        assert report["checks"]["rigid"]["status"] == "not_run"
    else:
        assert calls == ["identity", "particles", "rigid"]
    frozen = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError):
        qualification.main(argv)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == frozen


def test_timeout_fails_but_other_probes_still_run(tmp_path, monkeypatch):
    def run(command, **kwargs):
        if command[-1] == "particles":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    out = tmp_path / "result"
    assert qualification.main(["--stack", "core", "--newton-source", str(tmp_path), "--out", str(out)]) == 1
    checks = json.loads((out / "report.json").read_text())["checks"]
    assert checks["particles"]["status"] == "failed"
    assert checks["particles"]["reason"] == "timeout"
    assert checks["rigid"]["status"] == "passed"


def test_wrong_source_revision_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification.subprocess, "check_output", lambda *a, **kw: "other-revision\n")
    with pytest.raises(ValueError, match="expected pinned-revision, found other-revision"):
        qualification.source_identity(tmp_path, "pinned-revision")


def test_dirty_source_is_rejected(tmp_path, monkeypatch):
    outputs = iter(["pinned-revision\n", " M newton/sim.py\n"])
    monkeypatch.setattr(qualification.subprocess, "check_output", lambda *a, **kw: next(outputs))
    with pytest.raises(ValueError, match="qualification source has tracked changes"):
        qualification.source_identity(tmp_path, "pinned-revision")
