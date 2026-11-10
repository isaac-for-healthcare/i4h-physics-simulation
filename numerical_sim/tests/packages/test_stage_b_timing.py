# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fake-worker control-flow tests; these create no GPU timing samples."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/qualify_stage_b_timing.py"
    spec = importlib.util.spec_from_file_location("stage_b_timing", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for source in ("newton18/lab-main", "stage-b-followup-20261104/lab30-cloner"):
        (tmp_path / ".tmp_tests" / source / "source").mkdir(parents=True)
    ref = tmp_path / ".tmp_tests/stage-b-host"
    ref.mkdir()
    for case in ("tissue_contact", "aorta_static"):
        for worlds in (1, 64):
            (ref / f"perf-main-{case}-{worlds}.json").write_text(json.dumps({"step_ms_p50": 1.0}))
    monkeypatch.setattr(module, "processes", lambda: [])
    return module, SimpleNamespace(workspace=tmp_path, out=tmp_path / "attempt", replacement_env_root=None)


def test_busy_gpu_launches_nothing(campaign, monkeypatch):
    module, args = campaign
    monkeypatch.setattr(module, "processes", lambda: [{"pid": 123, "row": "unrelated"}])
    def forbidden(*args, **kwargs):
        pytest.fail("busy GPU must not start workers")
    monkeypatch.setattr(module, "monitored_run", forbidden)
    assert module.campaign(args) == 2
    assert json.loads((args.out / "decision.json").read_text())["status"] == "pending"
    assert not (args.out / "results.json").exists()


@pytest.mark.parametrize("source", ["current", "preserved"])
def test_identity_comes_from_selected_implementation(campaign, monkeypatch, source):
    module, args = campaign
    root = args.workspace / source
    package = root / "physics/src/i4h_physics/__init__.py"
    package.parent.mkdir(parents=True)
    package.touch()
    recorder = root / "scripts/record_baseline.py"
    recorder.parent.mkdir()
    recorder.write_text(f"def source_identity():\n    return {{'selected': {source!r}}}\n")
    monkeypatch.setitem(sys.modules, "i4h_physics", SimpleNamespace(__file__=str(package)))
    assert module.implementation_identity() == {"selected": source}


@pytest.mark.parametrize("outcome,status", [("pass", "passed"), ("cadence_fail", "failed"),
                                           ("cadence_skip", "failed"), ("cadence_contaminated", "invalid"),
                                           ("timing_fail", "failed")])
def test_one_cadence_sample_in_same_campaign(campaign, monkeypatch, outcome, status):
    module, args = campaign
    args.replacement_env_root = Path("patched")
    calls = []
    def fake_worker(command, env, log_path, sample, observations, workspace):
        calls.append(command)
        log_path.touch()
        if "--worker" in command:
            output = Path(command[command.index("--out") + 1])
            median = 1.0 if output.name.startswith("prechange") else 1.01
            if outcome == "timing_fail" and output.name.startswith("lab30-"):
                median = 1.10
            output.write_text(json.dumps({"step_ms_p50": median, "replay_allocation_bytes": 0}))
            return 0, False
        assert "tests/runtime/test_app_loop.py::test_physics_rate_independent_of_render" in command
        assert "core-env/bin/python" in command[0]
        junit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        tag = "<failure/>" if outcome == "cadence_fail" else "<skipped/>" if outcome == "cadence_skip" else ""
        junit.write_text(f"<testsuites><testsuite><testcase>{tag}</testcase></testsuite></testsuites>")
        return int(outcome == "cadence_fail"), outcome == "cadence_contaminated"
    monkeypatch.setattr(module, "monitored_run", fake_worker)
    assert module.campaign(args) == {"passed": 0, "failed": 1, "invalid": 2}[status]
    decision = json.loads((args.out / "decision.json").read_text())
    assert decision["status"] == status and len(decision["comparisons"]) == 4
    assert len(calls) == 21 and sum("--worker" not in c for c in calls) == 1
    assert all(Path(c[0]).is_relative_to(args.workspace / ".tmp_tests/newton18") for c in calls[:4])
    assert all(Path(c[0]).is_relative_to(args.workspace / "patched") for c in calls[4:])
    assert decision["timing_passed"] == (outcome != "timing_fail")
    assert decision["cadence"]["passed"] == (outcome in ("pass", "timing_fail"))
    with pytest.raises(FileExistsError):
        module.campaign(args)


def test_monitored_child_contamination_is_retained(campaign, monkeypatch):
    module, args = campaign
    args.out.mkdir()
    polls = iter([None, None, 0])
    class Child:
        pid = 123
        returncode = 0
        def poll(self):
            return next(polls)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    observations = []
    def sample(allowed):
        observations.append(allowed)
        return len(observations) == 2
    code, contaminated = module.monitored_run([], {}, args.out / "worker.log", sample,
                                               args.out / "observations.jsonl", args.workspace)
    assert code == 0 and contaminated
    assert observations == [(123,), (123,), (123,)]


def test_reference_campaign_replaces_prechange_timing(campaign, monkeypatch):
    """Later stages gate on a completed campaign's own references; nothing pre-change is re-timed."""
    module, args = campaign
    args.replacement_env_root = Path("pins")
    args.lab_root = Path("pins")
    for lab in ("lab-main", "lab30"):
        (args.workspace / "pins" / lab / "source" / "isaaclab").mkdir(parents=True)
    args.reference_dir = Path("stage-b")
    reference = args.workspace / "stage-b"
    reference.mkdir()
    for case in ("tissue_contact", "aorta_static"):
        for worlds in (1, 64):
            (reference / f"prechange-lab30-{case}-{worlds}.json").write_text(json.dumps({"step_ms_p50": 1.0}))
            (reference / f"lab30-{case}-{worlds}.json").write_text(json.dumps({"step_ms_p50": 1.02}))
    calls = []
    def fake_worker(command, env, log_path, sample, observations, workspace):
        calls.append((command, env))
        log_path.touch()
        if "--worker" in command:
            Path(command[command.index("--out") + 1]).write_text(json.dumps({"step_ms_p50": 1.04, "replay_allocation_bytes": 0}))
            return 0, False
        junit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        junit.write_text("<testsuites><testsuite><testcase/></testsuite></testsuites>")
        return 0, False
    monkeypatch.setattr(module, "monitored_run", fake_worker)
    assert module.campaign(args) == 0
    workers = [c for c, _ in calls if "--worker" in c]
    assert len(workers) == 16 and not any("prechange" in " ".join(c) for c in workers)
    assert all(str(args.workspace / "pins") in env["PYTHONPATH"] for c, env in calls if "--worker" in c)
    decision = json.loads((args.out / "decision.json").read_text())
    row = decision["comparisons"]["aorta_static-64"]
    assert row["lab30_percent"] == pytest.approx(4.0) and row["lab30_reference"].endswith("prechange-lab30-aorta_static-64.json")
    assert row["lab30_percent_vs_reference_campaign_result"] == pytest.approx(100 * (1.04 / 1.02 - 1))
    assert "main_percent_vs_reference_campaign_result" not in row and decision["status"] == "passed"
