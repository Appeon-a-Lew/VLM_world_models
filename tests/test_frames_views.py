"""Frame-schedule and multi-view coverage for the episode generator and the schema helpers.

Rendering is stubbed (no OpenGL context needed here); physics, file layout and metadata are real.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from physwm.schema import Episode
from physwm.sim import generate, push
from physwm.sim.scene import load_config

CFG = load_config(Path(__file__).parents[1] / 'configs/sim_train.yaml')
WIDTH, HEIGHT = CFG['camera']['width'], CFG['camera']['height']


class NoRender:
    """Use real MuJoCo physics without needing an OpenGL context for these checks."""
    def __init__(self, model, width, height, camera='cam'):
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
def gen_cfg():
    cfg = copy.deepcopy(CFG)
    cfg['sim']['frame_times'] = [0.0, 0.5, 1.0]
    cfg['sim']['n_views'] = 2
    return cfg


@pytest.mark.parametrize('field,value', [
    ('frame_times', [0.0, 0.15, 1.0]),   # 0.15 s is not a stored time
    ('frame_times', [0.1, 0.5, 1.0]),    # missing t = 0
    ('frame_times', [0.0, 0.5]),         # missing t = 1
    ('frame_times', [0.0, 1.0, 0.5]),    # not increasing
    ('frame_times', [0.0, 2.5]),         # beyond the horizon
    ('n_views', 0),
    ('n_views', 2.5),
])
def test_invalid_frame_grid_rejected(field, value):
    cfg = copy.deepcopy(CFG['sim'])
    cfg[field] = value
    with pytest.raises(ValueError):
        push.validate_timing(cfg)


def test_n_views_defaults_to_one():
    cfg = copy.deepcopy(CFG['sim'])
    del cfg['n_views']
    push.validate_timing(cfg)


def test_generation_writes_every_view_and_frame(tmp_path, monkeypatch, gen_cfg):
    monkeypatch.setattr(push, 'CameraRenderer', NoRender)
    out = tmp_path / 'data'
    episode_id, error = generate._work((gen_cfg, 41, 'test', str(out)))
    assert error is None
    ep_dir = out / 'test' / episode_id
    assert {p.name for p in ep_dir.iterdir()} == set(generate.frame_artifacts(gen_cfg['sim'])) | {'meta.json'}

    ep = Episode.load(ep_dir / 'meta.json')
    assert ep.n_views == 2 and ep.frame_times == [0.0, 0.5, 1.0]
    assert ep.camera == ep.views[0]
    assert len({tuple(map(tuple, cam.cam2world)) for cam in ep.views}) == 2
    assert ep.frame_index_at(0.0) == 0 and ep.frame_index_at(0.5) == 1 and ep.frame_index_at(1.0) == 2
    with pytest.raises(KeyError):
        ep.frame_index_at(0.2)

    for view in range(2):
        for k in range(3):
            assert ep.frame_file('rgb', view, k) == f'rgb_v{view}f{k:02d}.png'
            assert ep.frame_file('depth', view, k) == f'depth_v{view}f{k:02d}.npz'
            assert ep.frame_file('seg', view, k) == f'seg_v{view}f{k:02d}.png'
            with np.load(ep_dir / ep.frame_file('depth', view, k)) as depth:
                assert depth['depth'].dtype == np.float16
                assert depth['depth'].shape == (HEIGHT, WIDTH)
            seg = np.array(Image.open(ep_dir / ep.frame_file('seg', view, k)))
            assert seg.shape == (HEIGHT, WIDTH)
            with Image.open(ep_dir / ep.frame_file('rgb', view, k)) as rgb:
                assert rgb.size == (WIDTH, HEIGHT)


def _legacy_dict() -> dict:
    """Episode as written before frames/views: single camera, t0/t1 file names."""
    state = {'pos': [0.0, 0.0, 0.03], 'quat': [1.0, 0.0, 0.0, 0.0], 'toppled': False, 'on_table': True}
    return {
        'episode_id': 'legacy_0000001', 'split': 'legacy', 'config': 'train', 'seed': 1,
        'table_half_size': [0.5, 0.35],
        'objects': [{
            'id': 0, 'name': 'red box', 'shape': 'box', 'color': 'red',
            'rgba': [0.85, 0.12, 0.12, 1.0], 'half_extents': [0.03, 0.03, 0.03],
            'mass': 0.2, 'density': 600.0, 'friction': 0.5,
        }],
        'camera': {
            'width': 64, 'height': 48,
            'K': [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]],
            'cam2world': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            'fovy_deg': 50.0, 'azimuth_deg': 0.0, 'elevation_deg': 45.0, 'distance': 1.0,
        },
        'action': {
            'type': 'push', 'target': 0, 'direction_label': 'left',
            'direction_world': [-1.0, 0.0, 0.0], 'impulse': 0.1, 'speed': 0.5,
            'text': 'Push the red box to the left with an impulse of 0.1 N·s.',
        },
        'times': [0.0, 1.0, 2.0],
        'states': [[dict(state)] for _ in range(3)],
        'files': {'rgb_t0': 'rgb_t0.png', 'rgb_t1': 'rgb_t1.png', 'depth_t0': 'depth_t0.npz',
                  'seg_t0': 'seg_t0.png', 'traj': 'traj.npz'},
    }


def test_legacy_episode_defaults_and_frame_fallback(tmp_path):
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(_legacy_dict()))
    ep = Episode.load(path)
    assert ep.n_views == 1 and ep.views == [ep.camera]
    assert ep.frame_times == [0.0, 1.0]
    assert ep.frame_index_at(1.0) == 1
    assert ep.frame_file('rgb', 0, 0) == 'rgb_t0.png'
    assert ep.frame_file('rgb', 0, 1) == 'rgb_t1.png'
    assert ep.frame_file('depth', 0, 0) == 'depth_t0.npz'
    assert ep.frame_file('seg', 0, 0) == 'seg_t0.png'
    with pytest.raises(KeyError):
        ep.frame_file('rgb', 1, 0)  # legacy episodes have no second view

    saved = tmp_path / 'again.json'
    ep.save(saved)
    again = Episode.load(saved)
    assert again == ep and again.frame_times == [0.0, 1.0] and again.views == [again.camera]