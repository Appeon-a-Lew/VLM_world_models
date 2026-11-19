"""Unit tests for Stage-1 action-free multi-frame prompts and chain3d parsing."""

import json
import numpy as np

from physwm.schema import Action, Camera, Episode, ObjectSpec, ObjectState
from physwm.vlm.parse import parse_chain3d
from physwm.vlm.prompts import prompt_text, target_text


def _ep():
    cam = Camera(
        width=448, height=448,
        K=[[480.0, 0.0, 224.0], [0.0, 480.0, 224.0], [0.0, 0.0, 1.0]],
        cam2world=np.eye(4).tolist(),
        fovy_deg=50.0, azimuth_deg=0.0, elevation_deg=45.0, distance=1.0,
    )
    obj = ObjectSpec(
        id=0, name="red box", shape="box", color="red", rgba=[1, 0, 0, 1],
        half_extents=[0.04, 0.03, 0.02], mass=0.1, density=600.0, friction=0.16,
    )
    s0 = [ObjectState(pos=[0.0, 0.0, 1.0], quat=[1, 0, 0, 0], toppled=False, on_table=True)]
    s1 = [ObjectState(pos=[0.05, 0.0, 1.0], quat=[1, 0, 0, 0], toppled=False, on_table=True)]
    s2 = [ObjectState(pos=[0.09, 0.0, 1.0], quat=[1, 0, 0, 0], toppled=False, on_table=True)]
    s3 = [ObjectState(pos=[0.18, 0.0, 1.0], quat=[1, 0, 0, 0], toppled=False, on_table=True)]
    act = Action(
        type="push", target=0, direction_label="right", direction_world=[1, 0, 0],
        impulse=0.08, speed=0.8, text="Push the red box to the right with an impulse of 0.08 N·s.",
    )
    return Episode(
        episode_id="test_0", split="val_id", config="simple_mf", seed=0,
        table_half_size=[0.5, 0.35], objects=[obj], camera=cam, action=act,
        times=[0.0, 0.1, 0.2, 1.0], states=[s0, s1, s2, s3], files={},
        views=[cam], frame_times=[0.0, 0.1, 0.2, 1.0],
    )


def test_chain3d_prompt_and_target_roundtrip():
    ep = _ep()
    p_noact = prompt_text(ep, "B", style="chain3d", frames=(0, 1, 2), include_action=False)
    assert "Action:" not in p_noact
    assert "impulse" not in p_noact
    assert "t = 0.0, 0.1, 0.2" in p_noact
    assert "t = 0.2 s" in p_noact and "t = 1.0 s" in p_noact

    p_act = prompt_text(ep, "B", style="chain3d", frames=(0, 1, 2), include_action=True)
    assert "Action: Push the red box" in p_act

    tgt = target_text(ep, "B", style="chain3d", frames=(0, 1, 2))
    data = json.loads(tgt)
    assert np.allclose(data["current"][0]["center"], [0.09, 0.0, 1.0])
    assert np.allclose(data["future"][0]["center"], [0.18, 0.0, 1.0])
    assert np.allclose(data["current"][0]["extents"], [0.08, 0.06, 0.04])

    cc, ce, fc, fe, pc, pf = parse_chain3d(tgt, ep)
    assert pc and pf
    assert np.allclose(cc["0"], [0.09, 0.0, 1.0])
    assert np.allclose(fc["0"], [0.18, 0.0, 1.0])
    assert np.allclose(ce["0"], [0.08, 0.06, 0.04])
    assert np.allclose(fe["0"], [0.08, 0.06, 0.04])


def test_parse_chain3d_handles_truncated_fallback():
    ep = _ep()
    broken = '{"current": [{"label": "red box", "center": [0.09, 0.0, 1.0], "extents": [0.08, 0.06, 0.04]}], "future": [{"label": "red box", "center": [0.18, 0.0, 1.0], "extents": [0.08, 0.0'
    cc, ce, fc, fe, pc, pf = parse_chain3d(broken, ep)
    assert pc and pf
    assert np.allclose(cc["0"], [0.09, 0.0, 1.0])
    assert np.allclose(fc["0"], [0.18, 0.0, 1.0])
