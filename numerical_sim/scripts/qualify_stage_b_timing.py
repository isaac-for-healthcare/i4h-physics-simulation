# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive Stage B campaign. Refuse busy GPUs; preserve every attempt, never retry.

The parent uses the standard library only. Workers use the preserved qualified
Lab environments. Run from numerical_sim after other compute workloads finish.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return [{"pid": int(line.split(",", 1)[0]), "row": line.strip()} for line in result.stdout.splitlines() if line.strip()]


def implementation_identity():
    """Use the imported implementation's recorder, including the preserved source tree."""
    import i4h_physics

    root = Path(i4h_physics.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("_timing_baseline_identity", root / "scripts/record_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_identity()


def worker(args):
    import numpy as np
    import warp as wp
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg
    from isaaclab.sim import SimulationContext

    identity = implementation_identity()
    native = Path(wp.__file__).parent / "bin/warp.so"
    identity["warp_native"] = {"path": str(native), "sha256": hashlib.sha256(native.read_bytes()).hexdigest()}
    cfg = I4hEnvCfg(case=args.case, decimation=1, stagger_episodes=False)
    cfg.scene.num_envs = args.worlds
    cfg.sim.physics.use_cuda_graph = True
    if args.exports:
        from i4h_physics.observables import KINDS
        cfg.sim.physics.solver_cfg.observables = tuple(sorted(KINDS))
    wp.init()
    wp.synchronize()
    base = wp.get_mempool_used_mem_current("cuda:0")
    start = time.perf_counter()
    env = I4hEnv(cfg)
    wp.synchronize()
    construction = time.perf_counter() - start
    try:
        for _ in range(20):
            env.sim.step(render=False)
        wp.synchronize()
        before = wp.get_mempool_used_mem_current("cuda:0")
        times = []
        for _ in range(200):
            start = time.perf_counter()
            env.sim.step(render=False)
            wp.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        after = wp.get_mempool_used_mem_current("cuda:0")
        assert np.isfinite(env.simulator.scene.q.numpy()).all()
        result = dict(source=identity, case=args.case, worlds=args.worlds, exports=args.exports,
                      step_ms_p50=float(np.median(times)), step_ms_p95=float(np.percentile(times, 95)),
                      construction_s=construction, retained_warp_bytes=after-base,
                      peak_warp_bytes=wp.get_mempool_used_mem_high("cuda:0")-base,
                      replay_allocation_bytes=after-before, warmup=20, steps=200, controls="zero", captured=True,
                      particle_grid_retained=env.simulator.scene.model.particle_grid is not None,
                      samples_ms=times)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()
        SimulationContext.clear_instance()


def monitored_run(command, env, log_path, sample, observations, workspace):
    """One child, sampled throughout; a monitor failure invalidates its evidence."""
    contaminated = False
    with log_path.open("x") as log:
        child = subprocess.Popen(command, env=env, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                contaminated = sample((child.pid,)) or contaminated
                time.sleep(0.25)
            contaminated = sample((child.pid,)) or contaminated
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            contaminated = True
            with observations.open("a") as stream:
                stream.write(json.dumps({"time_ns": time.time_ns(), "monitor_error": repr(error)}) + "\n")
            child.wait()  # retain products, but never qualify unmonitored execution
    return child.returncode, contaminated


def campaign(args):
    w = args.workspace.resolve()
    replacement = args.replacement_env_root or w / ".tmp_tests/newton18"
    if not replacement.is_absolute():
        replacement = w / replacement
    replacement = replacement.resolve()
    # Later stages gate against a completed campaign's references instead of re-timing the
    # preserved pre-change source: Lab 3.0 uses that campaign's prechange-lab30 results.
    reference = getattr(args, "reference_dir", None)
    reference = reference and (reference if reference.is_absolute() else w / reference).resolve()
    labs = getattr(args, "lab_root", None)
    labs = labs and (labs if labs.is_absolute() else w / labs).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment-roots.json").write_text(json.dumps({
        "prechange": str(w / ".tmp_tests/newton18"), "replacement": str(replacement),
        "reference_dir": reference and str(reference), "lab_root": labs and str(labs)}, indent=2) + "\n")
    observations = args.out / "process-observations.jsonl"
    def sample(allowed=()):
        procs = processes()
        with observations.open("a") as stream:
            stream.write(json.dumps({"time_ns": time.time_ns(), "allowed": list(allowed), "processes": procs}) + "\n")
        return any(p["pid"] not in allowed for p in procs)
    if sample():
        (args.out / "decision.json").write_text(json.dumps({"status": "pending", "reason": "GPU has unrelated compute processes; no timing launched"}, indent=2) + "\n")
        print("Pending: unrelated GPU compute workloads are active; left untouched.")
        return 2
    results = {}
    versions = ("prechange-lab30", "lab30", "main", "lab30-exports", "main-exports")
    for version in versions[1:] if reference else versions:
        main = version.startswith("main")
        exports = version.endswith("exports")
        source = w / ".tmp_tests/stage-b-v3/prechange" if version.startswith("prechange") else w
        lab = w / (".tmp_tests/newton18/lab-main" if main else ".tmp_tests/stage-b-followup-20261104/lab30-cloner")
        if labs:
            lab = labs / ("lab-main" if main else "lab30")
        runtime = w / ".tmp_tests/newton18" if version.startswith("prechange") else replacement
        py = runtime / f'{"main" if main else "lab30"}-env/bin/python'
        paths = [source / m / "src" for m in ("physics", "isaaclab", "surgical", "endoluminal", "assessment", "standalone")]
        paths += sorted(p for p in (lab / "source").iterdir() if p.is_dir())
        env = dict(os.environ, PYTHONPATH=":".join(map(str, paths)), I4H_DATA=str(w / "data"))
        for case in ("tissue_contact", "aorta_static"):
            for worlds in (1, 64):
                name = f"{version}-{case}-{worlds}"
                result_path = args.out / (name + ".json")
                command = [str(py), str(Path(__file__).resolve()), "--worker", "--case", case, "--worlds", str(worlds), "--out", str(result_path)]
                if exports:
                    command.append("--exports")
                contaminated = sample()
                if contaminated:
                    (args.out / "decision.json").write_text(json.dumps({"status": "invalid", "reason": "compute workload appeared between runs", "results": results}, indent=2) + "\n")
                    return 2
                exit_code, contaminated = monitored_run(command, env, args.out / (name + ".log"),
                                                        sample, observations, w)
                row = json.loads(result_path.read_text()) if result_path.exists() else {}
                row.update(exit_code=exit_code, contaminated=contaminated, valid=not contaminated and exit_code == 0)
                results[name] = row
                (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                if not row["valid"]:
                    (args.out / "decision.json").write_text(json.dumps({"status": "invalid" if contaminated else "failed", "run": name}, indent=2) + "\n")
                    return 2 if contaminated else 1
    comparisons = {}
    passed = True
    for case in ("tissue_contact", "aorta_static"):
        for worlds in (1, 64):
            suffix = f"{case}-{worlds}"
            lab30_ref = reference / f"prechange-lab30-{suffix}.json" if reference else None
            old = (json.loads(lab30_ref.read_text()) if reference else results[f"prechange-lab30-{suffix}"])["step_ms_p50"]
            new = results[f"lab30-{suffix}"]["step_ms_p50"]
            main_ref = w / f".tmp_tests/stage-b-host/perf-main-{suffix}.json"
            main_old = json.loads(main_ref.read_text())["step_ms_p50"]
            comparisons[suffix] = {
                "lab30_percent": 100 * (new / old - 1),
                "main_percent_vs_host_reference": 100 * (results[f"main-{suffix}"]["step_ms_p50"] / main_old - 1),
                "main_reference": str(main_ref), "main_reference_sha256": hashlib.sha256(main_ref.read_bytes()).hexdigest(),
                **({"lab30_reference": str(lab30_ref), "lab30_reference_sha256": hashlib.sha256(lab30_ref.read_bytes()).hexdigest(),
                    **{f"{host}_percent_vs_reference_campaign_result":
                       100 * (results[f"{host}-{suffix}"]["step_ms_p50"] / json.loads((reference / f"{host}-{suffix}.json").read_text())["step_ms_p50"] - 1)
                       for host in ("lab30", "main") if (reference / f"{host}-{suffix}.json").is_file()}} if reference else {}),
                **{f"{host}_export_overhead_percent": 100 * (results[f"{host}-exports-{suffix}"]["step_ms_p50"] / results[f"{host}-{suffix}"]["step_ms_p50"] - 1)
                   for host in ("lab30", "main")}}
            passed &= comparisons[suffix]["lab30_percent"] <= 5.0
            passed &= comparisons[suffix]["main_percent_vs_host_reference"] <= 5.0
    passed &= all(r["replay_allocation_bytes"] == 0 for r in results.values())
    # The cadence failure is sampled once, in the same continuously monitored
    # window, even when a completed timing comparison exceeds its bound.
    decision = {"median_limit_percent": 5.0, "comparisons": comparisons,
                "main_reference_is_historical": True, "timing_passed": bool(passed)}
    if sample():
        decision.update(status="invalid", reason="compute workload appeared before cadence test")
        (args.out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        return 2
    paths = [w / m / "src" for m in ("physics", "isaaclab", "surgical", "endoluminal", "assessment", "standalone")]
    env = dict(os.environ, PYTHONPATH=":".join(map(str, paths)), I4H_DATA=str(w / "data"))
    node = "tests/runtime/test_app_loop.py::test_physics_rate_independent_of_render"
    junit = args.out / "cadence.xml"
    command = [str(replacement / "core-env/bin/python"), "-m", "pytest", node, "-q", f"--junitxml={junit}"]
    exit_code, contaminated = monitored_run(command, env, args.out / "cadence.log", sample, observations, w)
    # Exit zero may mean a skipped test. Require exactly one executed passing test.
    cases = list(ET.parse(junit).getroot().iter("testcase")) if junit.is_file() else []
    executed_pass = len(cases) == 1 and not any(cases[0].find(tag) is not None for tag in ("skipped", "failure", "error"))
    cadence = {"node": node, "exit_code": exit_code, "contaminated": contaminated,
               "passed": exit_code == 0 and not contaminated and executed_pass}
    (args.out / "cadence.json").write_text(json.dumps(cadence, indent=2) + "\n")
    passed = passed and cadence["passed"]
    decision.update(status="invalid" if contaminated else "passed" if passed else "failed", cadence=cadence)
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return 2 if contaminated else int(not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replacement-env-root", type=Path,
                        help="directory containing qualified replacement core/lab30/main-env; pre-change stays preserved")
    parser.add_argument("--reference-dir", type=Path,
                        help="completed campaign whose prechange-lab30 results are the Lab 3.0 gate reference; skips re-timing pre-change source")
    parser.add_argument("--lab-root", type=Path, help="directory containing the pinned lab-main and lab30 checkouts")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=("tissue_contact", "aorta_static"))
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--exports", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    return worker(args) if args.worker else campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
