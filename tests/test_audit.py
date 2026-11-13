"""Regression tests for data provenance, observation timing and evaluation integrity."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from physwm.agent import exec as executor
from physwm.eval.metrics import MISSING_PENALTY, per_object_rows, validate_coverage
from physwm.schema import load_split
from physwm.sim import push
from physwm.sim.generate import prepare_split
from physwm.sim.scene import load_config
from physwm.vlm.parse import parse_answer

CFG = load_config(Path(__file__).parents[1] / 'configs/sim_train.yaml')


class NoRender:
    """Use real MuJoCo physics without needing an OpenGL context for these checks."""
    def __init__(self, model, width, height):
        self.shape = (height, width)

    def rgb(self, data):
        return np.zeros((*self.shape, 3), dtype=np.uint8)

    def depth(self, data):
        return np.zeros(self.shape)

    def seg(self, data):
        return np.zeros(self.shape, dtype=np.uint8)

    def close(self):
        pass


@pytest.fixture
def episode(monkeypatch):
    monkeypatch.setattr(push, 'CameraRenderer', NoRender)
    return push.simulate(CFG, 41, 'audit').episode


def test_observations_match_integrated_state(monkeypatch):
    monkeypatch.setattr(push, 'CameraRenderer', NoRender)
    original = push._states
    checked = []

    def checked_states(m, d, body_ids, *args):
        for body in body_ids:
            adr = m.jnt_qposadr[m.body_jntadr[body]]
            np.testing.assert_allclose(d.xpos[body], d.qpos[adr:adr + 3], atol=1e-12)
        checked.append(d.time)
        return original(m, d, body_ids, *args)

    monkeypatch.setattr(push, '_states', checked_states)
    data = push.simulate(CFG, 41, 'audit')
    assert len(checked) == 3
    for t, states in zip(data.episode.times, data.episode.states):
        k = np.flatnonzero(np.isclose(data.traj_t, t))[0]
        np.testing.assert_allclose(data.traj_pos[k], [s.pos for s in states])


@pytest.mark.parametrize('field,value', [('timestep', 0.003), ('traj_hz', 1000),
                                        ('store_times', [0, 0.5, 2]), ('horizon', 0.5)])
def test_invalid_timing_rejected(field, value):
    cfg = copy.deepcopy(CFG['sim'])
    cfg[field] = value
    with pytest.raises(ValueError):
        push.validate_timing(cfg)


def test_toppled_rest_pose_is_not_redefined_as_upright(monkeypatch):
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body quat="0.70710678 0.70710678 0 0" pos="0 0 .1"><freejoint/><geom type="box" size=".03 .03 .08"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert push._states(m, d, [1], [np.array([0, 0, 1])], ['box'], [.5, .35])[0].toppled


def test_generation_provenance_rejects_changed_config_and_legacy(tmp_path):
    identity = {'config': CFG, 'source_sha256': {'push.py': 'abc'}, 'mujoco_version': 'test'}
    split = tmp_path / 'new'
    prepare_split(split, identity, 0, 2)
    prepare_split(split, identity, 0, 2)
    changed = copy.deepcopy(identity)
    changed['config']['push']['speed'] = [2, 3]
    with pytest.raises(ValueError, match='provenance differs'):
        prepare_split(split, changed, 0, 2)
    with pytest.raises(ValueError, match='provenance differs'):
        prepare_split(split, identity, 0, 3)
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'index.jsonl').write_text('')
    with pytest.raises(ValueError, match='no provenance'):
        prepare_split(legacy, identity, 0, 2)


def test_split_uses_index_not_leftover_directories(tmp_path, episode):
    split = tmp_path / episode.split
    path = split / episode.episode_id
    path.mkdir(parents=True)
    episode.save(path / 'meta.json')
    (split / 'index.jsonl').write_text(json.dumps({'episode_id': episode.episode_id}) + '\n')
    stale = split / 'stale'
    stale.mkdir()
    (stale / 'meta.json').write_text('invalid json must not be read')
    assert len(load_split(tmp_path, episode.split)) == 1
    (split / 'index.jsonl').write_text(json.dumps({'episode_id': 'missing'}) + '\n')
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, episode.split)


@pytest.mark.parametrize('xyz', [[float('nan'), 0, 1], [0, 0, float('inf')], [1, 2], 'garbage'])
def test_invalid_coordinates_count_as_missing(episode, xyz):
    rows = per_object_rows({'task': 'B', 'objects': {'0': xyz}}, episode)
    assert rows[0]['missing'] and rows[0]['err'] == MISSING_PENALTY


def test_parser_rejects_nonfinite_coordinates(episode):
    text = json.dumps([{'label': episode.objects[0].name, 'center': [float('nan'), 0, 1]}])
    objects, parsed = parse_answer(text, episode)
    assert not parsed and all(v is None for v in objects.values())


def test_coverage_rejects_missing_and_duplicate_records(episode):
    episodes = {episode.episode_id: episode}
    a = {'episode_id': episode.episode_id, 'task': 'A'}
    b = {**a, 'task': 'B'}
    validate_coverage([a, b], episodes, ('A', 'B'))
    with pytest.raises(ValueError, match='incomplete'):
        validate_coverage([a], episodes, ('A', 'B'))
    with pytest.raises(ValueError, match='duplicate'):
        validate_coverage([a, b, b], episodes, ('A', 'B'))


def test_spec_rejects_seed_mismatch_and_diagnostic_training(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    (tmp_path / 'pyproject.toml').write_text((root / 'pyproject.toml').read_text())
    cfg = {'data': str(executor.DATA), 'train_split': 'train', 'seed': 0}
    cfg_path = tmp_path / 'train.yaml'
    spec = dict(id='exp999_audit', hypothesis='test', change='test', baseline='static',
                kind='train', budget_hours=1, seed=1, train_config='train.yaml')
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(executor.Denied, match='seed'):
        executor.validate_spec(spec, tmp_path)
    spec['seed'] = 0
    cfg['train_split'] = executor.DIAG_SPLITS[0]
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(executor.Denied, match='training on'):
        executor.validate_spec(spec, tmp_path)


def test_executor_scores_matched_limit_and_rejects_missing_records(tmp_path, monkeypatch, episode):
    from physwm.eval.metrics import gt_centers_cam

    policy = copy.deepcopy(executor.POLICY)
    policy['data']['validation_splits'] = ['val_id', 'val_ood_objects']
    policy['data']['ablation_split'] = 'val_id'
    monkeypatch.setattr(executor, 'POLICY', policy)
    monkeypatch.setattr(executor, 'DIAG_SPLITS', [])
    monkeypatch.setattr(executor, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(executor, 'AGENT', tmp_path / 'agent')
    monkeypatch.setattr(executor, 'EXPS', tmp_path / 'experiments')
    monkeypatch.setattr(executor, 'check_pinned', lambda: None)
    baseline, normal, none = [], [], []
    for split in policy['data']['validation_splits']:
        for i in range(2):
            ep = copy.deepcopy(episode)
            ep.split, ep.episode_id = split, f'{split}_{i}'
            path = executor.DATA / split / ep.episode_id
            path.mkdir(parents=True)
            ep.save(path / 'meta.json')
            xyz = gt_centers_cam(ep, 1)
            baseline.append(dict(episode_id=ep.episode_id, task='B', parsed=True,
                                 objects={str(j): (p + 100 * i).tolist() for j, p in enumerate(xyz)}))
            if i == 0:
                for task, t in [('A', 0), ('B', 1)]:
                    rec = dict(episode_id=ep.episode_id, task=task, parsed=True,
                               objects={str(j): p.tolist() for j, p in enumerate(gt_centers_cam(ep, t))})
                    normal.append(rec)
                    if split == 'val_id':
                        none.append(rec)
    base_dir = executor.AGENT / 'val_baselines'
    base_dir.mkdir(parents=True)
    exp = 'exp999_audit'
    exp_dir = executor.EXPS / exp
    (exp_dir / 'pred').mkdir(parents=True)
    (exp_dir / 'spec.json').write_text(json.dumps({'eval_limit': 1}))

    def write(path, records):
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))

    for name in executor.BASELINE_NAMES:
        write(base_dir / f'{name}.jsonl', baseline)
    write(exp_dir / 'pred/normal.jsonl', normal)
    write(exp_dir / 'pred/none.jsonl', none)
    metrics = executor.evaluate_experiment(exp)
    assert metrics['value'] == 0 and metrics['static_ood_target_mean'] == 0
    write(exp_dir / 'pred/normal.jsonl', normal[:-1])
    with pytest.raises(ValueError, match='incomplete'):
        executor.evaluate_experiment(exp)
