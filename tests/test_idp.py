import json
from pathlib import Path

import numpy as np
import pytest

from physwm import geometry as G
from physwm.schema import Action, Camera, Episode, ObjectSpec, ObjectState
from physwm.vlm import parse
from physwm.vlm.context import action_context
from physwm.vlm.parse import parse_ctx3d, parse_idp
from physwm.vlm.prompts import (IDP_AMBIGUOUS_M, IDP_TASK, NO_MOTION_LABEL, idp_frame,
                                idp_target, prompt_text, target_text)


def v3_episode(pos0=(0.05, 0, 0.05), pos01=(0.15, 0, 0.05), pos1=(0.3, 0, 0.05), speed=1.0):
    obj = ObjectSpec(0, 'red box', 'box', 'red', [1, 0, 0, 1], [.05, .05, .05], .1, 500, .4)
    states = [[ObjectState(list(pos0), [1, 0, 0, 0], False, True)],
              [ObjectState(list(pos01), [1, 0, 0, 0], False, True)],
              [ObjectState(list(pos1), [1, 0, 0, 0], False, True)]]
    cam = Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1)
    act = Action('push', 0, 'right', [1, 0, 0], .1, speed,
                'Push the red box to the right with an impulse of 0.1 N·s.')
    return Episode('e', 'val', 'test', 1, [.5, .35], [obj], cam, act,
                   [0.0, 0.1, 1.0], states, {}, [], [0.0, 0.1, 1.0])


def legacy_episode():
    obj = ObjectSpec(0, 'red box', 'box', 'red', [1, 0, 0, 1], [.05, .05, .05], .1, 500, .4)
    states = [[ObjectState([.05, 0, .05], [1, 0, 0, 0], False, True)],
              [ObjectState([.3, 0, .05], [1, 0, 0, 0], False, True)]]
    cam = Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1)
    act = Action('push', 0, 'right', [1, 0, 0], .1, 1.0, 'Push the red box to the right.')
    return Episode('e', 'val', 'test', 1, [.5, .35], [obj], cam, act,
                   [0.0, 1.0], states, {}, [], [0.0, 1.0])


def test_idp_frame_is_fixed():
    assert idp_frame(v3_episode()) == 1
    with pytest.raises(KeyError):
        idp_frame(legacy_episode())


def test_idp_prompt_carries_no_action_text():
    ep = v3_episode()
    p = prompt_text(ep, IDP_TASK, 'box3d', (0, idp_frame(ep)))
    assert 'impulse' not in p and 'Push the' not in p
    assert 't = 0.0, 0.1 seconds after the action' in p
    assert 'weak|medium|strong' in p


def test_idp_target_strength_bins():
    for speed, want in ((0.5, 'weak'), (1.0, 'medium'), (1.5, 'strong')):
        assert idp_target(v3_episode(speed=speed)) == {
            'target': 'red box', 'direction': 'right', 'strength': want}


def test_idp_no_motion_class_is_measured_not_predicted():
    ep = v3_episode(pos0=(0.05, 0, 0.05), pos01=(0.055, 0, 0.05))  # 5 mm << IDP_AMBIGUOUS_M
    assert idp_target(ep) == {'action': NO_MOTION_LABEL}


def test_idp_legacy_episode_has_no_gt():
    with pytest.raises(KeyError):
        idp_target(legacy_episode())


def test_idp_target_text_round_trips():
    ep = v3_episode()
    assert json.loads(target_text(ep, IDP_TASK, 'box3d', (0, idp_frame(ep)))) == idp_target(ep)


def test_parse_idp_full_triple_and_sync():
    ep = v3_episode()
    ans, ok = parse_idp('{"target": "red box", "direction": "right", "strength": "medium"}', ep)
    assert ok and ans == {'target': 0, 'direction': 'right', 'strength': 'medium'}
    assert parse._NO_MOTION == NO_MOTION_LABEL  # parse vocabulary stays in sync with prompts


def test_parse_idp_fenced_and_no_motion():
    ep = v3_episode()
    ans, ok = parse_idp('```json\n{"target": "red box", "direction": "left",'
                        ' "strength": "weak"}\n```', ep)
    assert ok and ans == {'target': 0, 'direction': 'left', 'strength': 'weak'}
    ans, ok = parse_idp(f'{{"action": "{NO_MOTION_LABEL}"}}', ep)
    assert ok and ans == {'action': NO_MOTION_LABEL}


def test_parse_idp_rejects_unusable_answers():
    ep = v3_episode()
    assert parse_idp('the red box moved right', ep) == (None, False)
    assert parse_idp('{"target": "red box", "direction": "rig", "strength": "medium"}', ep)[1] is False
    assert parse_idp('{"target": "blue box", "direction": "right", "strength": "medium"}', ep)[1] is False


def test_ctx3d_target_shape_and_content():
    ep = v3_episode()
    d = json.loads(target_text(ep, 'B', 'ctx3d', (0,)))
    assert set(d) == {'current', 'action_context', 'future'}
    assert d['current'][0]['support'] == 'table'
    assert d['action_context'] == action_context(ep)
    assert d['action_context']['edge_distance_m'] == pytest.approx(0.45)
    want = G.world_to_cam([ep.state_at(1)[0].pos], ep.camera.cam2world)[0]
    assert d['future'][0]['center'] == [round(float(v), 3) for v in want]


def test_ctx3d_target_parses_with_the_chain_parser():
    ep = v3_episode()
    t = target_text(ep, 'B', 'ctx3d', (0,))
    from physwm.vlm.parse import parse_chain3d
    c_c, _ce, f_c, _fe, ok_c, ok_f = parse_chain3d(t, ep)
    assert ok_c and ok_f
    want = G.world_to_cam([ep.state_at(1)[0].pos], ep.camera.cam2world)[0]
    assert f_c['0'] == pytest.approx(list(want), abs=1e-3)


def test_ctx3d_parse_extracts_emitted_context_and_support():
    ep = v3_episode()
    t = target_text(ep, 'B', 'ctx3d', (0,))
    c_c, _ce, f_c, _fe, ctx, sup, _pc, _pf = parse_ctx3d(t, ep)
    assert ctx == action_context(ep)
    assert sup == {'0': 'table'}
    assert parse_ctx3d('{"current": []}', ep)[4] is None
    assert parse_ctx3d('{"current": []}', ep)[5] is None


def test_oracle_context_injects_support_and_action_only_for_forward_tasks():
    ep = v3_episode()
    pb = prompt_text(ep, 'B', 'box3d', (0,), oracle_context=True)
    pa = prompt_text(ep, 'A', 'box3d', (0,), oracle_context=True)
    assert 'Physical context:' in pb
    assert '"support"' in pb and '"table"' in pb       # GT support travels with the payload
    assert json.dumps(action_context(ep)) in pb       # the action payload is nested verbatim
    assert 'Physical context:' not in pa
    assert 'Physical context:' not in prompt_text(ep, 'B', 'box3d', (0,))


def test_direction_prompt_uses_camera_frame_and_preserves_existing_task_inputs():
    ep = v3_episode()
    transform = G.look_at([1., 2., 1.], [0., 0., 0.])
    ep.camera.cam2world = transform.tolist()
    ep.action.direction_world = [1., 0., 0.]

    def extract(prompt):
        line = next(line for line in prompt.splitlines() if line.startswith("Push direction as"))
        return line, np.array(json.loads(line.rsplit(": ", 1)[-1].rstrip(".")))

    text = prompt_text(ep, "B")
    nominal = prompt_text(ep, "B", direction_input="nominal")
    recorded = prompt_text(ep, "B", direction_input="recorded")
    line, recorded_vector = extract(recorded)
    np.testing.assert_allclose(extract(nominal)[1], [1., 0., 0.], atol=0.001)
    np.testing.assert_allclose(transform[:3, :3] @ recorded_vector, [1., 0., 0.], atol=0.001)
    assert recorded.replace(line + "\n", "") == text
    for task, frames in (("A", (0,)), ("I", (0, 1))):
        assert prompt_text(ep, task, frames=frames, direction_input="recorded") == prompt_text(
            ep, task, frames=frames)
    assert prompt_text(ep, "B", include_action=False, direction_input="recorded") == prompt_text(
        ep, "B", include_action=False)


def test_ctx3d_prompt_describes_the_context_step():
    p = prompt_text(v3_episode(), 'B', 'ctx3d', (0,))
    assert 'action_context' in p and 'support' in p


def test_summarize_idp_matches_gt():
    from physwm.eval.idpmetrics import summarize_idp
    ep, epn = v3_episode(), v3_episode(pos0=(0.05, 0, 0.05), pos01=(0.055, 0, 0.05))
    ep.episode_id, epn.episode_id = 'a', 'b'
    preds = [
        {'episode_id': 'a', 'task': 'I', 'parsed': True,
         'idp': {'target': 0, 'direction': 'right', 'strength': 'medium'}},
        {'episode_id': 'b', 'task': 'I', 'parsed': True, 'idp': {'action': NO_MOTION_LABEL}},
    ]
    rows = summarize_idp(preds, {'a': ep, 'b': epn})
    assert len(rows) == 1
    r = rows[0]
    assert r['episodes'] == 2 and r['exact_acc'] == 1.0
    assert r['target_acc'] == 1.0 and r['direction_acc'] == 1.0 and r['strength_acc'] == 1.0
    assert r['no_motion_acc'] == 1.0 and r['n_action_gt'] == 1 and r['n_no_motion_gt'] == 1


def test_summarize_ctx_scores_emitted_context():
    from physwm.eval.idpmetrics import summarize_ctx
    ep = v3_episode()
    gt = action_context(ep)
    good = {'episode_id': 'a', 'task': 'B', 'action_context': dict(gt)}
    bad = {'episode_id': 'b', 'task': 'B',
           'action_context': {'target': 'red box', 'direction': 'left',
                               'edge_distance_m': 0.1, 'obstacles': []}}
    rows = summarize_ctx([good, bad], {'a': ep, 'b': ep})
    r = rows[0]
    assert r['episodes'] == 2 and r['ctx_parse_rate'] == 1.0
    assert r['ctx_target_acc'] == 1.0
    assert r['ctx_direction_acc'] == pytest.approx(0.5)
    assert r['ctx_edge_mae_m'] == pytest.approx(0.35 / 2)


def test_sft_data_task_i_skips_legacy_episodes():
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from physwm.vlm.sft import SFTData
    ep, legacy = v3_episode(), legacy_episode()
    ds = SFTData([(Path('a'), ep), (Path('b'), legacy)], ['A', IDP_TASK],
                 frames=(0,), views=(0,), style='box3d', processor=None)
    # v3 contributes A + I; the legacy episode has no 0.1 s state, so it contributes A only
    assert sorted(t for _, _, t in ds.samples) == ['A', 'A', IDP_TASK]


@pytest.mark.parametrize("episodes, expected_batches, expected_steps", [(1, 1, 1), (63, 47, 4), (64, 48, 4)])
def test_sft_plan_counts_batches_before_gradient_accumulation(monkeypatch, capsys, episodes,
                                                              expected_batches, expected_steps):
    import sys

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from physwm.vlm import sft

    config = Path(__file__).parents[1] / "configs/sft_qwen3vl2b.yaml"
    monkeypatch.setattr(sft, "load_split", lambda *_: [(Path("data"), v3_episode())] * episodes)
    monkeypatch.setattr(sys, "argv", ["sft", "--config", str(config), "--tasks", "A,B,I",
                                      "--epochs", "0.3", "--plan"])
    sft.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["batches_per_epoch"] == expected_batches
    assert plan["natural_optimizer_steps"] == expected_steps


def two_object_episode():
    objs = [ObjectSpec(0, 'red box', 'box', 'red', [1, 0, 0, 1], [.05, .05, .05], .1, 500, .4),
            ObjectSpec(1, 'green box', 'box', 'green', [0, 1, 0, 1], [.05, .05, .05], .1, 500, .4)]

    def states(x0):
        return [
            [ObjectState([x0, 0, .05], [1, 0, 0, 0], False, True),
             ObjectState([.25, 0, .05], [1, 0, 0, 0], False, True)],
            [ObjectState([x0 + .05, 0, .05], [1, 0, 0, 0], False, True),
             ObjectState([.25, 0, .05], [1, 0, 0, 0], False, True)],
            [ObjectState([x0 + .2, 0, .05], [1, 0, 0, 0], False, True),
             ObjectState([.25, 0, .05], [1, 0, 0, 0], False, True)],
        ]

    cam = Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1)
    act = Action('push', 0, 'right', [1, 0, 0], .1, 1.0, 'Push the red box to the right.')
    return Episode('e', 'val', 'test', 1, [.5, .35], objs, cam, act,
                   [0.0, 0.1, 1.0], states(.05), {}, [], [0.0, 0.1, 1.0])


def test_summarize_ctx_tolerates_malformed_model_output():
    import math

    from physwm.eval.idpmetrics import summarize_ctx
    ep = two_object_episode()
    assert action_context(ep)['obstacles']  # GT obstacle set is non-empty: null cannot match
    bad = {'episode_id': 'a', 'task': 'B',
           'action_context': {'target': 3.5, 'direction': 7, 'edge_distance_m': 'far',
                              'obstacles': None},
           'support': None}
    good = {'episode_id': 'a', 'task': 'B',
            'action_context': {'target': 'red box', 'direction': 'right',
                               'edge_distance_m': 0.45, 'obstacles': None},
            'support': {'0': 'table', '1': 'table'}}
    rows = summarize_ctx([bad, good], {'a': ep})   # must not raise
    r = rows[0]
    assert r['episodes'] == 2 and r['ctx_parse_rate'] == 1.0
    assert r['ctx_target_acc'] == pytest.approx(0.5)   # the float "target" counts as missing
    assert r['ctx_direction_acc'] == pytest.approx(0.5)
    assert r['ctx_obstacles_acc'] == 0.0               # null vs a non-empty GT set
    assert r['ctx_edge_mae_m'] == pytest.approx(0.0)  # only the one parsable distance
    assert r['ctx_support_acc'] == pytest.approx(1.0) and r['ctx_support_scored'] == 2
    assert rows[0]['split'] == 'val'
