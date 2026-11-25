# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versius configuration and qualification options consumed by the shared scripts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def add_arguments(parser):
    parser.add_argument('--usd', help='prepared robot USD (or I4H_VERSIUS_USD)')
    parser.add_argument('--graph', action=argparse.BooleanOptionalAction, default=True,
                        help='CUDA graphs (default); --no-graph selects eager stepping')
    parser.add_argument('--coupling', choices=('one_way', 'two_way'), default='two_way')
    parser.add_argument('--no-contact', action='store_true', help='mechanical qualification paired baseline')
    parser.add_argument('--headless', action='store_true', help='offscreen GL')
    parser.add_argument('--output', type=Path, default=Path('.tmp_tests/versius-current'))


def make_cfg(args, *, teleop=False):
    from i4h_isaaclab.tasks import VERSIUS_TASKS

    from .task import configuration

    if args.task not in VERSIUS_TASKS:
        raise ValueError(f"Cartesian control requires a Versius task; known: {sorted(VERSIUS_TASKS)}")
    cfg, _ = configuration(VERSIUS_TASKS[args.task], args.usd,
                           envs=args.num_envs if args.num_envs is not None else 1, graph=args.graph, coupling=args.coupling,
                           teleop=teleop or getattr(args, 'validation', None) in ('mechanics', 'lifecycle'))
    cfg.sim.device = args.device or 'cuda:0'
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def workflow_args(args):
    return SimpleNamespace(**{**vars(args), 'viewer': args.render == 'gl',
                             'mode': getattr(args, 'validation', None),
                             'viewer_factory': getattr(args, 'viewer_factory', None),
                             'updates': getattr(args, 'train', 0),
                             'checkpoint': Path(args.policy.split(':', 1)[1])
                             if getattr(args, 'policy', '').startswith('checkpoint:') else None})


def run_validation(env, args):
    from .qualification import execute

    options = workflow_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    env = env.unwrapped
    env.reset()
    try:
        result = execute(env, options)
        if args.validation != 'construct' and env.graph_seen != args.graph:
            raise AssertionError('Actual backend graph use differs from requested mode')
        result['backend_graph_observed'] = env.graph_seen
        result['backend_graph_present'] = bool(env.sim.backend_manager.backend.graphs)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        if result.get('passed') is False:
            raise SystemExit(2)
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(
            {'error': str(error), 'type': type(error).__name__}, indent=2) + '\n')
        raise
