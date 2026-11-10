# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The accepted chole policy counts complete replays without changing numeric gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[2] / 'scripts'
spec = importlib.util.spec_from_file_location('chole_baseline', SCRIPTS / 'record_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def accepted_manifest():
    return {'comparison_policy': {'kind': 'majority-of-three', 'replays': 3, 'required_passes': 2,
                                  'fresh_processes': True, 'retain_all_trials': True},
            'reference_disposition': {'status': 'approved', 'capacity': 256}}


def test_policy_is_explicit_and_chole_only():
    for name in baseline.CASES:
        assert baseline.comparison_policy(name, {})['replays'] == 1
    assert baseline.comparison_policy('chole_tet_med', accepted_manifest())['replays'] == 3
    for name in set(baseline.CASES) - {'chole_tet_med'}:
        with pytest.raises(ValueError):
            baseline.comparison_policy(name, accepted_manifest())
    for path, value in [('required_passes', 1), ('replays', 5), ('fresh_processes', False)]:
        manifest = accepted_manifest()
        manifest['comparison_policy'][path] = value
        with pytest.raises(ValueError):
            baseline.comparison_policy('chole_tet_med', manifest)
    for key, value in [('status', 'candidate'), ('capacity', 8)]:
        manifest = accepted_manifest()
        manifest['reference_disposition'][key] = value
        with pytest.raises(ValueError):
            baseline.comparison_policy('chole_tet_med', manifest)


@pytest.mark.parametrize('passes,accepted', [([True, False, True], True), ([False, True, False], False),
                                           ([True, True, True], True), ([False, False, False], False)])
def test_whole_replay_majority(passes, accepted):
    policy = baseline.comparison_policy('chole_tet_med', accepted_manifest())
    trials = [{'passed': result, 'failures': [] if result else [f'product-{i}']} for i, result in enumerate(passes)]
    assert baseline.summarize_replays(policy, trials)['passed'] is accepted
    with pytest.raises(ValueError):
        baseline.summarize_replays(policy, trials[:2])


def test_worker_error_cannot_be_hidden_by_a_numerical_majority():
    policy = baseline.comparison_policy('chole_tet_med', accepted_manifest())
    trials = [{'passed': True}, {'passed': True}, {'passed': False, 'valid': False}]
    result = baseline.summarize_replays(policy, trials)
    assert not result['valid'] and not result['passed']


@pytest.mark.parametrize('name,values,expected_exit', [
    ('chole_tet_med', [0., 0., 4.], 0),  # still run and retain the third, failing replay
    ('chole_tet_med', [4., 4., 0.], 1),  # no fourth attempt after a failed majority
    ('two_worlds', [4.], 1),
])
def test_recorder_runs_fixed_count_and_retains_failures(tmp_path, monkeypatch, name, values, expected_exit):
    ref = tmp_path / 'reference'
    ref.mkdir()
    identity = {'software': {}, 'gpu': 'fixture', 'baseline_scope': baseline.BASELINE_SCOPE}
    array = np.array([0.], np.float32)
    manifest = accepted_manifest() if name == 'chole_tet_med' else {}
    manifest.update(source=identity, envs=2, steps=60, repeat_count=6, factor=3,
                    scenario=baseline.scenario(name, 60),
                    products={'x': {'sha256': [baseline.digest(array)], 'shape': [1], 'dtype': array.dtype.str, 'noise': 1.}})
    (ref / f'{name}.json').write_text(json.dumps(manifest))
    np.savez(ref / f'{name}.npz', x=array, **{'noise/x': np.array(1.)})
    original = {p.name: p.read_bytes() for p in ref.iterdir()}
    calls = []

    def replay(name, args, k, *, retained_path):
        calls.append(k)
        data = {'x': np.array([values[k]], np.float32)}
        np.savez(retained_path, **data)
        retained_path.with_suffix('.json').write_text(json.dumps(identity))
        retained_path.with_suffix('.log').write_text('retained worker output')
        return data, identity

    monkeypatch.setattr(baseline, 'source_identity', lambda: identity)
    monkeypatch.setattr(baseline, 'repeat_in_process', replay)
    monkeypatch.setattr(baseline.wp, 'init', lambda: None)
    out = tmp_path / 'results'
    assert baseline.main(['--compare', str(ref), '--cases', name, '--results', str(out)]) == expected_exit
    assert calls == list(range(len(values)))
    report = next(p for p in out.glob('*.json') if '.trial' not in p.name)
    result = json.loads(report.read_text())
    assert len(result['trials']) == len(values)
    assert result['passed'] is (expected_exit == 0)
    assert len(list(out.glob('*.npz'))) == len(values)
    assert {p.name: p.read_bytes() for p in ref.iterdir()} == original
    assert result['trials'][values.index(4.)]['failures']


@pytest.mark.parametrize('passes,expected_exit', [([True, True, False], 0), ([False, False, True], 1)])
def test_host_harness_uses_three_fresh_workers(tmp_path, monkeypatch, passes, expected_exit):
    import sys
    monkeypatch.setitem(sys.modules, 'record_baseline', baseline)
    spec = importlib.util.spec_from_file_location('chole_qualification', SCRIPTS / 'qualify_stage_b.py')
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    ref, out = tmp_path / 'reference', tmp_path / 'results'
    ref.mkdir()
    out.mkdir()
    (ref / 'chole_tet_med.json').write_text(json.dumps(accepted_manifest()))
    args = SimpleNamespace(case='chole_tet_med', baseline=ref, chole_candidate=None, single_replay=False,
                           out=out, host='lab30', exports=True, eager=False, device='cuda:0')
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        assert '--single-replay' in command and '--exports' in command
        destination = Path(command[command.index('--out') + 1])
        destination.mkdir()
        passed = passes[len(calls) - 1]
        (destination / 'chole_tet_med.json').write_text(json.dumps({'passed': passed, 'failures': [] if passed else ['x']}))
        return SimpleNamespace(returncode=0 if passed else 1)

    monkeypatch.setattr(qualification.subprocess, 'run', worker)
    assert qualification.qualify_case(args) == expected_exit
    assert len(calls) == 3
    result = json.loads((out / 'chole_tet_med.json').read_text())
    assert len(result['trials']) == 3
    assert result['passed'] is (expected_exit == 0)
    assert len(list(out.glob('*.log'))) == 3
