"""Compare retained mechanical runs; report numerical drift without bitwise claims."""

import json
from pathlib import Path

import numpy as np


def compare(root):
    def trace(name, key):
        return np.asarray(
            [row[key] for row in json.loads((root / name / "mechanics.json").read_text())]
        )


    report = {}
    for scene, two, one, repeat, graph in (
        (
            "block",
            "mech-block-two",
            "mech-block-one",
            "mech-block-repeat",
            "mech-block-graph",
        ),
        (
            "chole",
            "mech-chole-eager",
            "mech-chole-one",
            "mech-chole-repeat",
            "mech-chole-graph",
        ),
    ):
        tcp = trace(two, "tcp")[:, 0]
        response = float(np.linalg.norm(tcp - trace(one, "tcp")[:, 0], axis=1).max())
        noise = float(np.linalg.norm(tcp - trace(repeat, "tcp")[:, 0], axis=1).max())
        mode = float(np.linalg.norm(tcp - trace(graph, "tcp")[:, 0], axis=1).max())
        depth_drift = float(abs(trace(two, "depth") - trace(graph, "depth")).max())
        force = trace(two, "force")[:, 0]
        opposing = float(
            np.median((-force[:, 2] if scene == "block" else force[:, 1])[55:95])
        )
        assert response > noise * 3, (scene, response, noise)
        assert opposing > 0, (scene, opposing)
        if scene == "block":
            np.testing.assert_allclose(tcp, trace(graph, "tcp")[:, 0], rtol=1e-4, atol=1e-5)
        report[scene] = {
            "max_reaction_tcp_response": response,
            "max_repeat_tcp_noise": noise,
            "max_mode_tcp_drift": mode,
            "max_mode_depth_drift": depth_drift,
            "median_opposing_hold_force_world_z": opposing,
        }
    for scene, contact, baseline in (
        ("block", "ownership-block-contact", "ownership-block-baseline"),
        ("chole", "mech-chole-graph", "ownership-chole-baseline"),
    ):
        a = np.load(root / contact / "state.npz")["q"]
        b = np.load(root / baseline / "state.npz")["q"]
        peer_force = trace(contact, "force")[:, 1:]
        assert not peer_force.any()
        assert not trace(contact, "count")[:, 1:].any()
        report[scene]["independent_rollout_peer_q_difference"] = float(
            abs(a[1:] - b[1:]).max()
        )
        report[scene]["independent_rollout_peer_patch_difference"] = float(
            abs(trace(contact, "patch")[:, 1:] - trace(baseline, "patch")[:, 1:]).max()
        )
        report[scene]["peer_contact_force_and_count_zero"] = True
        if scene == "block":
            np.testing.assert_allclose(a[1:], b[1:], rtol=0, atol=1e-6)
    (root / "comparisons.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(compare(parser.parse_args().root), indent=2))
