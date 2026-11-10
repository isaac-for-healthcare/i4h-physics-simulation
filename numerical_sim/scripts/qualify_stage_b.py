# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen v3 recorder through standalone or the selected Lab host.

Run in the appropriate isolated environment. Each case uses a fresh process.
The test harness forwards step calls to Lab; production PhysicsBinding has no step.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import record_baseline as baseline

C30_NEWTON_REVISION = "7f0c13af6e4dd75253709c31dc65c0edcaf22b44"


def check_reference_identity(identity, reference, revision=None):
    """Allow only the explicitly requested C3.0 Newton transition; retain both identities."""
    software = dict(identity["software"])
    if revision is not None:
        origin = json.loads(importlib.metadata.distribution("newton").read_text("direct_url.json"))
        if (revision != C30_NEWTON_REVISION or origin.get("vcs_info", {}).get("commit_id") != revision
                or software.get("newton") != "1.9.0.dev0" or reference["software"].get("newton") != "1.8.0rc1"):
            raise ValueError("C3.0 requires the exact installed Newton git revision and version transition")
        software["newton"] = reference["software"]["newton"]
    if software != reference["software"] or identity["gpu"] != reference["gpu"]:
        raise ValueError("qualified software/hardware identity differs from the reference")


def hosted_case(case, *, device, envs, eager, exports=False):
    from i4h_isaaclab.composition import sim_settings
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg

    cfg = I4hEnvCfg(case=case, decimation=1, stagger_episodes=False)
    cfg.sim.device, cfg.scene.num_envs = device, envs
    cfg.sim.dt = sim_settings(case)["dt"]
    cfg.sim.physics.use_cuda_graph = not eager
    if exports:
        from i4h_physics.observables import KINDS
        cfg.sim.physics.solver_cfg.observables = tuple(sorted(KINDS))
    env = I4hEnv(cfg)

    class Harness:
        scene, systems = env.simulator.scene, env.simulator.systems

        def step(self):
            env.sim.step(render=False)

        def reset(self, ids=None):
            env.simulator.reset(ids)

        def close(self):
            from isaaclab.sim import SimulationContext
            env.close()
            SimulationContext.clear_instance()

    return Harness()


def run_one(args):
    import i4h_isaaclab.composition as composition

    handles = []
    if args.host != "standalone":
        from i4h_isaaclab.newton_manager import host_api
        actual = "main" if host_api().__name__.endswith("newton_solver") else "lab30"
        if actual != args.host:
            raise RuntimeError(f"requested {args.host}, imported {actual}")

        def load(case, device=None, envs=None):
            value = hosted_case(case, device=device, envs=envs, eager=args.eager, exports=args.exports)
            handles.append(value)
            return value
        composition.load_case = load
    elif args.eager or args.exports:
        original = composition.load_case

        def load(case, **kwargs):
            if args.eager:
                case = {**case, "sim": {**case.get("sim", {}), "cuda_graph": False}}
            sim = original(case, **kwargs)
            if args.exports:
                sim.observables = sim.solver.observables(sim.solver.supported_observables)
            return sim
        composition.load_case = load
    try:
        identity = baseline.source_identity()
        reference_dir = args.chole_candidate if args.case == "chole_tet_med" and args.chole_candidate else args.baseline
        manifest = json.loads((reference_dir / f"{args.case}.json").read_text())
        if (manifest["source"]["baseline_scope"] != baseline.BASELINE_SCOPE or manifest["repeat_count"] != 6
                or manifest["factor"] != 3 or manifest["scenario"] != baseline.scenario(args.case, manifest["steps"])):
            raise ValueError("scope/scenario/repetition/factor contract changed")
        check_reference_identity(identity, manifest["source"], args.newton_revision)
        reference = dict(np.load(reference_dir / f"{args.case}.npz"))
        if set(manifest["products"]) != {k for k in reference if not k.startswith("noise/")}:
            raise ValueError("reference product set differs from its manifest")
        for key, product in manifest["products"].items():
            value = reference[key]
            if (baseline.digest(value) != product["sha256"][0] or value.dtype.str != product["dtype"]
                    or list(value.shape) != product["shape"] or float(reference[f"noise/{key}"]) != product["noise"]):
                raise ValueError(f"reference integrity failure: {key}")
        data = baseline.run_case(args.case, manifest["envs"], manifest["steps"], args.device)
        failures = baseline.compare(reference, data, 0, 0, 3)
        np.savez_compressed(args.out / f"{args.case}.npz", **data)
        report = {"exports": args.exports, "reference": str(reference_dir),
                  "reference_software": manifest["source"]["software"], "candidate_newton_revision": args.newton_revision,
                  "reference_disposition": "unpromoted candidate" if reference_dir == args.chole_candidate else "approved",
                  "host": args.host, "eager": args.eager, "source": identity, "failures": failures,
                  "passed": not failures,
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if args.host != "standalone":
            import isaaclab
            report["lab_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(isaaclab.__file__).parent, text=True).strip()
        (args.out / f"{args.case}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.case}: {'FAIL' if failures else 'PASS'} ({len(failures)} failures)", flush=True)
        for failure in failures:
            print(failure)
        return int(bool(failures))
    finally:
        for value in handles:
            value.close()


def qualify_case(args):
    reference_dir = args.chole_candidate if args.case == "chole_tet_med" and args.chole_candidate else args.baseline
    manifest = json.loads((reference_dir / f"{args.case}.json").read_text())
    policy = baseline.comparison_policy(args.case, manifest)
    if args.single_replay or policy["replays"] == 1:
        return run_one(args)
    trials = []
    for k in range(policy["replays"]):
        out = args.out / f"{args.case}.trial{k}"
        command = [sys.executable, str(Path(__file__).resolve()), "--host", args.host,
                   "--baseline", str(args.baseline), "--out", str(out), "--device", args.device,
                   "--case", args.case, "--single-replay"]
        for flag in ("exports", "eager"):
            if getattr(args, flag):
                command.append(f"--{flag}")
        if args.chole_candidate:
            command += ["--chole-candidate", str(args.chole_candidate)]
        if getattr(args, "newton_revision", None):
            command += ["--newton-revision", args.newton_revision]
        with (args.out / f"{args.case}.trial{k}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report = out / f"{args.case}.json"
        detail = json.loads(report.read_text()) if report.exists() else {"failures": ["worker did not complete"]}
        trials.append({"trial": k, "exit_code": result.returncode, "report": str(report),
                       "valid": report.exists() and result.returncode == int(not detail.get("passed", False)),
                       "passed": result.returncode == 0 and detail.get("passed") is True,
                       "failures": detail["failures"]})
        print(f"{args.case} replay {k}: exit {result.returncode}", flush=True)
    summary = baseline.summarize_replays(policy, trials)
    summary.update(host=args.host, exports=args.exports, eager=args.eager, reference=str(reference_dir),
                   reference_disposition="approved", harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / f"{args.case}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(not summary["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("standalone", "main", "lab30"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--exports", action="store_true", help="enable all i4h outputs; comparator and scenario stay unchanged")
    parser.add_argument("--newton-revision", choices=(C30_NEWTON_REVISION,),
                        help="qualify the exact C3.0 Newton upgrade against preserved 1.8 references")
    parser.add_argument("--chole-candidate", type=Path, help="report chole against an explicitly unpromoted candidate")
    parser.add_argument("--case", choices=baseline.CASES)
    parser.add_argument("--single-replay", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.case and args.out.exists():
        raise FileExistsError("use a fresh qualification directory; prior results are preserved")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.case:
        if (args.out / f"{args.case}.json").exists():
            raise FileExistsError("qualification products cannot be overwritten")
        return qualify_case(args)
    failures = []
    for case in baseline.CASES:
        command = [sys.executable, __file__, *sys.argv[1:], "--case", case]
        with (args.out / f"{case}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(case)
        print(f"{case}: exit {result.returncode}", flush=True)
    (args.out / "summary.json").write_text(json.dumps({"host": args.host, "failures": failures}, indent=2) + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
