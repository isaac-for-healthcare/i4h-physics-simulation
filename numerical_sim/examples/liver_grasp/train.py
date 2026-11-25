# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded liver experiment CLI. All outputs remain in the liver-grasp worktree."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from training_env import ROOT, SETTINGS, LiverEnv, schedule, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "stage",
        choices=["benchmark", "qualify", "collect", "bc", "ppo", "evaluate", "play"],
    )
    parser.add_argument(
        "--data", default=str(ROOT.parent.parent / "i4h-isaaclab/numerical_sim/data")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/liver-policy-20261119"
    )
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Evaluation seeds are reserved before any training stage.
    reserved = args.output / "reserved_100.json"
    if not reserved.exists():
        write_json(reserved, schedule(100, 981304))
    if args.stage == "bc":
        behavior_clone(args)
        return
    env = LiverEnv(
        args.data,
        1 if args.stage == "play" else args.worlds,
        args.output / "snapshots",
        args.seed,
    )
    if args.stage == "benchmark":
        benchmark(env, args)
        return
    if args.stage == "qualify":
        specs = schedule(args.episodes, 271828)
        for i in range(5):
            specs[i] = dict(target_index=i, seed=0, jitter_m=[0, 0, 0])
        evaluate(env, args, specs, "qualification_controller")
        return
    if args.stage == "collect":
        collect(env, args)
        return
    if args.stage == "ppo":
        train_ppo(env, args)
        return
    if args.stage == "play":
        play(env, args)
        return
    evaluate(
        env,
        args,
        json.loads(reserved.read_text())[: args.episodes],
        "controller" if args.checkpoint is None else args.checkpoint.stem,
    )


def benchmark(env, args):
    from i4h_surgical.grasp_snapshot import continuation_arrays

    before = {
        k: v.numpy().copy()
        for k, v in continuation_arrays(
            env.sim.physics, env.sim.state, env.sim.control
        ).items()
    }
    peer_probe = {k: getattr(env.probe, k).numpy().copy() for k in env.probe.fields}
    env.reset([0])
    after = continuation_arrays(env.sim.physics, env.sim.state, env.sim.control)
    for (owner, key, world, *_), (name, old) in zip(
        env.sim.solver.continuation.entries, before.items()
    ):
        peer = world.numpy() != 0
        np.testing.assert_array_equal(after[name].numpy()[peer], old[peer])
    for k in env.probe.fields:
        np.testing.assert_array_equal(
            getattr(env.probe, k).numpy()[1:], peer_probe[k][1:]
        )
    start = time.monotonic()
    for _ in range(64):
        env.step(env.teacher_actions())
    elapsed = time.monotonic() - start
    import subprocess

    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    report = dict(
        worlds=env.num_envs,
        elapsed_s=elapsed,
        environment_control_steps_per_s=64 * env.num_envs / elapsed,
        reset_isolation="passed",
        gpu_memory_used_free_mib=memory,
        contention="unrelated GPU workloads preserved",
    )
    write_json(args.output / f"benchmark_{env.num_envs}.json", report)
    print(report, flush=True)


def evaluate(env, args, specs, label, policy=None):
    if policy is None and args.checkpoint is not None:
        policy = load_policy(env, args.checkpoint)
    specs = [
        dict(s, evaluation_id=i, canonical_row=i % env.num_envs)
        for i, s in enumerate(specs)
    ]
    pending = [iter(specs[e :: env.num_envs]) for e in range(env.num_envs)]
    active = {}
    starts = []
    for e in range(env.num_envs):
        s = next(pending[e], None)
        if s is not None:
            active[e] = s
            env.reset([e], [s])
            starts.append(start_record(env, e, s))
    results = []
    prior_completed_count = len(env.completed)
    previous_diagnostics = env.diagnostics_every_physics_step
    env.diagnostics_every_physics_step = True
    trace = []
    started = time.monotonic()
    while active:
        obs = env.get_observations()
        with torch.no_grad():
            actions = env.teacher_actions() if policy is None else policy(obs)
        _, _, dones, extra = env.step(actions, auto_reset=False)
        if args.episodes <= 5:
            trace.append(
                dict(
                    positions=env.sim.scene.field("q", env.probe.body)
                    .numpy()[0]
                    .copy(),
                    action=env.probe.ic.actions.numpy()[0, 0].copy(),
                    jaw_angle=float(env.probe.ic.jaw_angle.numpy()[0, 0]),
                    phase=env.tasks[0].phase,
                )
            )
        for e, row in extra["results"]:
            if e not in active:
                continue
            results.append(row)
            print(
                label,
                len(results),
                row["target"],
                row["status"],
                row["failure"],
                flush=True,
            )
            del active[e]
            s = next(pending[e], None)
            if s is not None:
                env.reset([e], [s])
                active[e] = s
                starts.append(start_record(env, e, s))
        if len(results) % 5 == 0:
            write_json(args.output / f"{label}_evaluation.json", results)
    write_json(args.output / f"{label}_evaluation.json", results)
    write_json(
        args.output / f"{label}_starts.json",
        sorted(starts, key=lambda r: r["evaluation_id"]),
    )
    summary = dict(
        episodes=len(results),
        success=sum(r["status"] == "passed" for r in results) / len(results),
        elapsed_s=time.monotonic() - started,
        per_target={
            str(ex["particle_ids"][0]): dict(
                episodes=sum(r["target"] == ex["particle_ids"][0] for r in results),
                passed=sum(
                    r["target"] == ex["particle_ids"][0] and r["status"] == "passed"
                    for r in results
                ),
            )
            for ex in env.examples
        },
    )
    write_json(args.output / f"{label}_summary.json", summary)
    if trace:
        np.savez_compressed(
            args.output / f"{label}_trace.npz",
            positions=[t["positions"] for t in trace],
            actions=[t["action"] for t in trace],
            jaw_angle=[t["jaw_angle"] for t in trace],
            phase=[t["phase"] for t in trace],
        )
    env.diagnostics_every_physics_step = previous_diagnostics
    del env.completed[prior_completed_count:]
    return summary


def start_record(env, e, spec):
    q = env.sim.scene.field("q", env.probe.body).numpy()[e]
    action = env.probe.ic.actions.numpy()[e]
    return dict(
        **spec,
        tissue_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        action_sha256=hashlib.sha256(action.tobytes()).hexdigest(),
        root_m=env.probe.ic.root.numpy()[e, 0].tolist(),
        snapshot_hash=env.snapshot.hash,
    )


def collect(env, args):
    folder = args.output / "dataset"
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    buffers = [[] for _ in range(env.num_envs)]
    records = []
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    source_examples = (
        ROOT
        / "recordings/liver/20261119_141840/physics_replay/successful_examples.json"
    )
    frozen = folder / "source"
    frozen.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        (frozen / source.name).write_bytes(source.read_bytes())
    manifest = dict(
        settings=SETTINGS,
        seed=args.seed,
        sources=source_hashes,
        examples_sha256=hashlib.sha256(source_examples.read_bytes()).hexdigest(),
        examples=env.examples,
        snapshot_hash=env.snapshot.hash,
        identity=env.snapshot.identity,
        reserved_seeds="reserved_100.json",
        episodes=records,
    )
    requested_specs = schedule(min(args.episodes, 1000), args.seed)
    pending = iter(requested_specs)
    active = {}
    for e in range(env.num_envs):
        spec = next(pending, None)
        if spec is not None:
            env.reset([e], [spec])
            active[e] = spec
    while active and time.monotonic() - started < args.minutes * 60:
        obs = env.get_observations()["policy"].numpy().copy()
        actions = env.teacher_actions()
        _, rewards, dones, extra = env.step(actions, auto_reset=False)
        for e in active:
            buffers[e].append(
                (
                    obs[e],
                    extra["requested"][e],
                    extra["applied"][e],
                    float(rewards[e]),
                    bool(dones[e]),
                    extra["terminal_observation"]["policy"][e].numpy().copy(),
                )
            )
        for e, row in extra["results"]:
            if e not in active:
                continue
            number = len(records)
            path = folder / f"episode_{number:04d}.npz"
            b = buffers[e]
            np.savez_compressed(
                path,
                observations=[x[0] for x in b],
                requested_actions=[x[1] for x in b],
                applied_actions=[x[2] for x in b],
                rewards=[x[3] for x in b],
                dones=[x[4] for x in b],
                next_observations=[x[5] for x in b],
            )
            buffers[e] = []
            records.append(dict(file=path.name, **row, split=None))
            print(
                "DEMO",
                number + 1,
                row["target"],
                row["status"],
                row["failure"],
                flush=True,
            )
            del active[e]
            spec = next(pending, None)
            if spec is not None:
                env.reset([e], [spec])
                active[e] = spec
            manifest["elapsed_s"] = time.monotonic() - started
            write_json(folder / "manifest.json", manifest)
    # Split successful complete episodes, stratified by target, never individual time steps.
    rng = np.random.default_rng(5531)
    for target in range(5):
        ids = [
            i
            for i, r in enumerate(records)
            if r["target_index"] == target and r["status"] == "passed"
        ]
        rng.shuffle(ids)
        cut = max(1, int(len(ids) * 0.8))
        for j, i in enumerate(ids):
            records[i]["split"] = "train" if j < cut else "validation"
    manifest["elapsed_s"] = time.monotonic() - started
    manifest["incomplete_discarded_steps"] = [len(b) for b in buffers]
    write_json(folder / "manifest.json", manifest)


def algorithm_cfg():
    return dict(
        actor=dict(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=[128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=dict(
                class_name="rsl_rl.modules:GaussianDistribution", init_std=0.02
            ),
        ),
        critic=dict(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=[128, 128],
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=dict(
            class_name="rsl_rl.algorithms:PPO",
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            schedule="fixed",
            entropy_coef=0.001,
        ),
        obs_groups=dict(actor=["policy"], critic=["policy"]),
        num_steps_per_env=64,
        multi_gpu=None,
    )


def make_actor(obs):
    from rsl_rl.models import MLPModel

    cfg = algorithm_cfg()["actor"].copy()
    cfg.pop("class_name")
    return MLPModel(obs, dict(actor=["policy"]), "actor", 4, **cfg)


def save_checkpoint(path, actor, optimizer, **extra):
    state = dict(
        actor=actor.state_dict(),
        optimizer=optimizer.state_dict(),
        config=algorithm_cfg(),
        invocation=sys.argv,
        rsl_rl_version=importlib.metadata.version("rsl-rl-lib"),
        normalization="fixed physical scales in observation schema; no running statistics",
        settings=SETTINGS,
        source_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        **extra,
    )
    manifest_path = path.parent / "dataset/manifest.json"
    if manifest_path.exists():
        state["dataset_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_policy(env, path):
    actor = make_actor(env.get_observations())
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor"])
    actor.eval()
    return actor


def behavior_clone(args):
    from tensordict import TensorDict

    manifest = json.loads((args.output / "dataset/manifest.json").read_text())
    arrays = {}
    for split in ("train", "validation"):
        episodes = [
            np.load(args.output / "dataset" / r["file"])
            for r in manifest["episodes"]
            if r["split"] == split
        ]
        if not episodes:
            raise RuntimeError(
                f"no successful complete {split} episodes; BC cannot be fitted"
            )
        arrays[split] = (
            torch.tensor(np.concatenate([e["observations"] for e in episodes])),
            torch.tensor(np.concatenate([e["applied_actions"] for e in episodes])),
        )
    obs = TensorDict({"policy": arrays["train"][0][:1]}, batch_size=[1])
    actor = make_actor(obs)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    started, last_save, best = time.monotonic(), time.monotonic(), float("inf")
    curves = []
    start_epoch = 0
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(state["actor"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = state["epoch"] + 1
        best = state.get("best_validation_loss", state["validation_loss"])
    for epoch in range(start_epoch, 10000):
        actor.train()
        x, y = arrays["train"]
        order = torch.randperm(len(x))
        losses = []
        for ids in order.split(1024):
            pred = actor(TensorDict({"policy": x[ids]}, batch_size=[len(ids)]))
            loss = ((pred - y[ids]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        actor.eval()
        with torch.no_grad():
            x, y = arrays["validation"]
            loss = float(
                (
                    (actor(TensorDict({"policy": x}, batch_size=[len(x)])) - y) ** 2
                ).mean()
            )
        curves.append(
            dict(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                validation_loss=loss,
                elapsed_s=time.monotonic() - started,
            )
        )
        if loss < best:
            best = loss
            save_checkpoint(
                args.output / "bc_best.pt",
                actor,
                optimizer,
                epoch=epoch,
                validation_loss=loss,
                best_validation_loss=best,
            )
        if epoch % 10 == 0:
            print("BC", curves[-1], flush=True)
            write_json(args.output / "bc_curves.json", curves)
        if time.monotonic() - last_save >= 300:
            save_checkpoint(
                args.output / "bc_latest.pt",
                actor,
                optimizer,
                epoch=epoch,
                validation_loss=loss,
                best_validation_loss=best,
            )
            last_save = time.monotonic()
        if time.monotonic() - started > args.minutes * 60:
            break
    save_checkpoint(
        args.output / "bc_latest.pt",
        actor,
        optimizer,
        epoch=epoch,
        validation_loss=loss,
        best_validation_loss=best,
    )
    write_json(args.output / "bc_curves.json", curves)


def train_ppo(env, args):
    from rsl_rl.algorithms import PPO

    cfg = algorithm_cfg()
    alg = PPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    state = torch.load(
        args.checkpoint or args.output / "bc_best.pt",
        weights_only=False,
        map_location="cpu",
    )
    alg.actor.load_state_dict(state["actor"])
    start_iteration = 0
    if "ppo" in state:
        alg.load(state["ppo"], None, True)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        env.cursor = state.get("schedule_cursor", env.cursor)
        env.reset(list(range(env.num_envs)))
        start_iteration = state["iteration"] + 1
    started = last_save = last_validation = time.monotonic()
    curves = []
    validation_history = []
    best_success = state.get("best_validation_success", -1.0)
    obs = env.get_observations()
    for iteration in range(start_iteration, 100000):
        with torch.inference_mode():
            for _ in range(64):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        loss = alg.update()
        curves.append(
            dict(
                iteration=iteration,
                elapsed_s=time.monotonic() - started,
                loss={k: float(v) for k, v in loss.items()},
                episodes=len(env.completed),
                recent_success=float(
                    np.mean([r["status"] == "passed" for r in env.completed[-100:]])
                )
                if env.completed
                else 0.0,
            )
        )
        print("PPO", curves[-1], flush=True)
        write_json(args.output / "ppo_curves.json", curves)

        def save(name):
            save_checkpoint(
                args.output / name,
                alg.actor,
                alg.optimizer,
                ppo=alg.save(),
                iteration=iteration,
                schedule_cursor=env.cursor,
                best_validation_success=best_success,
                environment_resume="fresh canonical episodes at iteration boundary",
            )

        if iteration == start_iteration or time.monotonic() - last_save >= 300:
            save("ppo_latest.pt")
            last_save = time.monotonic()
        # Separate validation seeds, never the final reserved 100.
        if time.monotonic() - last_validation >= 600 or iteration == start_iteration:
            alg.eval_mode()
            summary = evaluate(
                env,
                args,
                schedule(20, 88201),
                f"ppo_validation_{iteration:06d}",
                alg.actor,
            )
            validation_history.append(
                dict(
                    iteration=iteration,
                    ppo_elapsed_s=time.monotonic() - started,
                    **summary,
                )
            )
            write_json(args.output / "ppo_validation_history.json", validation_history)
            if summary["success"] >= best_success:
                best_success = summary["success"]
                save("ppo_best.pt")
            env.reset(list(range(env.num_envs)))
            obs = env.get_observations()
            alg.train_mode()
            last_validation = time.monotonic()
        if time.monotonic() - started >= args.minutes * 60:
            break
    save("ppo_latest.pt")
    write_json(args.output / "ppo_training_episodes.json", env.completed)


def play(env, args):
    from i4h_standalone.render import make_renderer
    from i4h_standalone.render.slang import Markers
    from i4h_surgical.config import read_case

    policy = None if args.checkpoint is None else load_policy(env, args.checkpoint)
    cfg = read_case(ROOT / "examples/liver_grasp/minimou.yaml", data=args.data)[
        "render"
    ]
    cfg["camera"]["position"] = [0.04, 0.075, -0.22]
    renderer = make_renderer(cfg, env.sim)
    renderer.markers = Markers(renderer, 2, idle_color=(1.0, 0.8, 0.05))
    deadline = time.monotonic() + args.minutes * 60
    pause_until = time.monotonic() + 0.5
    end_until = None
    try:
        while renderer.is_running() and time.monotonic() < deadline:
            started = time.monotonic()
            if started >= pause_until and not env.tasks[0].done:
                with torch.no_grad():
                    action = (
                        env.teacher_actions()
                        if policy is None
                        else policy(env.get_observations())
                    )
                env.step(action, auto_reset=False)
                if env.tasks[0].done:
                    end_until = time.monotonic() + 0.25
            target = env.measurement["geometry"][0, 0]
            renderer.markers.set(0, target, 0.0015, False)
            dest = env.tasks[0].destination
            renderer.markers.set(
                1,
                target if dest is None else dest,
                0.0015 if dest is not None else 0.0,
                True,
            )
            renderer.set_status(
                f"{'CONTROLLER' if policy is None else args.checkpoint.stem} | {env.tasks[0].phase} | lift 20 mm | target {env.probe.target.numpy()[0]}"
            )
            renderer.publish()
            renderer.draw()
            if end_until is not None and time.monotonic() >= end_until:
                env.reset([0])
                pause_until = time.monotonic() + 0.5
                end_until = None
            time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
