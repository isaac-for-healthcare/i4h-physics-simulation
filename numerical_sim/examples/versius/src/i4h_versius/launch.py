"""Single scripted / PPO / playback / input-only teleoperation launcher."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "scripted",
            "train",
            "play",
            "teleop",
            "construct",
            "lifecycle",
            "mechanics",
        ),
    )
    parser.add_argument(
        "--scene", choices=("block", "chole", "chole-1x"), default="block"
    )
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--coupling", choices=("one_way", "two_way"), default="two_way")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path(
            "/home/pkorzeniowsk/Projects/i4h-v2/i4h-v2-physics-simulation/training/generated/usd/versius_poke_2/versius_poke.usda"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path(".tmp_tests/versius"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--input",
        choices=("keyboard", "replay", "follou", "openhaptics"),
        default="keyboard",
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--axis-map")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--follou-device-index", type=int, default=0)
    parser.add_argument("--hd-device", default="Default Device")
    parser.add_argument("--hd-library")
    parser.add_argument("--no-real-time", action="store_true")
    args = parser.parse_args()
    if args.headless:
        import pyglet

        pyglet.options["headless"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    from .provenance import manifest

    manifest(args.output / "before.json", args.usd)
    from .task import PokeEnv, configuration

    cfg, physical = configuration(
        args.scene,
        args.usd,
        envs=args.envs,
        graph=args.graph,
        coupling=args.coupling,
        teleop=args.mode in ("teleop", "mechanics", "lifecycle"),
    )
    (args.output / "scene.json").write_text(
        json.dumps(
            {
                "profile": vars(physical),
                "graph": args.graph,
                "coupling": args.coupling,
                "envs": args.envs,
            },
            default=str,
            indent=2,
        )
        + "\n"
    )
    env = None
    try:
        env = PokeEnv(cfg, physical=physical)
        env.reset()
        from .workflows import execute

        result = execute(env, args)
        if args.mode != "construct":
            if env.graph_seen != args.graph:
                raise AssertionError(
                    "Actual backend graph use differs from requested mode"
                )
            result["backend_graph_observed"] = env.graph_seen
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result.get("passed") is False:
            raise SystemExit(2)
    except BaseException as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "type": type(error).__name__}, indent=2)
            + "\n"
        )
        raise
    finally:
        if env is not None:
            env.close()
        manifest(args.output / "after.json", args.usd)


if __name__ == "__main__":
    main()
