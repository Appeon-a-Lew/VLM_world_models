"""Simulator checks (need mujoco + EGL; run on a lab machine with MUJOCO_GL=egl)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from physwm import geometry as G
from physwm.eval.metrics import per_object_rows
from physwm.schema import Episode
from physwm.sim.push import simulate
from physwm.sim.scene import load_config

CFG = load_config(Path(__file__).parents[1] / "configs" / "sim_train.yaml")


@pytest.fixture(scope="module")
def episodes():
    return [simulate(CFG, s, "test") for s in range(12)]


def test_schema_roundtrip(episodes):
    ep = episodes[0].episode
    again = Episode.from_dict(json.loads(json.dumps(ep.to_dict())))
    assert again == ep


def test_projected_centres_inside_segmentation(episodes):
    hits, total = 0, 0
    for data in episodes:
        ep = data.episode
        uv = G.project(G.world_to_cam([s.pos for s in ep.state_at(0)], ep.camera.cam2world), ep.camera.K)
        for i, (u, v) in enumerate(uv):
            ys, xs = np.nonzero(data.seg[(0, 0)] == i + 1)
            if len(xs) < 30:  # fully occluded / out of view
                continue
            total += 1
            hits += xs.min() - 2 <= u <= xs.max() + 2 and ys.min() - 2 <= v <= ys.max() + 2
    assert total > 0 and hits / total > 0.9


def test_zero_impulse_keeps_objects_still():
    cfg = copy.deepcopy(CFG)
    cfg["push"]["speed"] = [0.0, 0.0]
    cfg["push"]["aim_at_other_prob"] = 0.0
    for s in range(5):
        ep = simulate(cfg, s, "test").episode
        p0 = np.array([x.pos for x in ep.state_at(0)])
        p1 = np.array([x.pos for x in ep.state_at(1)])
        assert np.abs(p1 - p0).max() < 1e-3


def test_left_push_moves_target_left_in_image():
    cfg = copy.deepcopy(CFG)
    cfg["push"]["directions"] = ["left"]
    cfg["push"]["aim_at_other_prob"] = 0.0
    cfg["push"]["speed"] = [0.6, 0.8]
    cfg["objects"]["shapes"] = ["box"]
    cfg["objects"]["count"] = [1, 1]
    for s in range(5):
        ep = simulate(cfg, s, "test").episode
        t = ep.action.target
        u0 = G.project(G.world_to_cam(ep.state_at(0)[t].pos, ep.camera.cam2world), ep.camera.K)[0]
        u1 = G.project(G.world_to_cam(ep.state_at(1)[t].pos, ep.camera.cam2world), ep.camera.K)[0]
        assert u1 < u0


def test_static_prediction_is_exact_for_unmoved(episodes):
    for data in episodes:
        ep = data.episode
        p0 = G.world_to_cam([s.pos for s in ep.state_at(0)], ep.camera.cam2world)
        pred = {"task": "B", "objects": {str(i): p0[i].tolist() for i in range(len(ep.objects))}}
        for r in per_object_rows(pred, ep):
            assert abs(r["err"] - r["err_static"]) < 1e-9
            if not r["moved"]:
                assert r["err"] < 0.01
