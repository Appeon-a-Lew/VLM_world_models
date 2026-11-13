import copy
import json

import numpy as np
import pytest

from physwm import geometry as G
from physwm.schema import Action, Camera, Episode, ObjectSpec, ObjectState
from physwm.eval.counterfactual import VARIANTS, analyze, load_families, static_predictions


@pytest.fixture
def families(tmp_path):
    obj = ObjectSpec(0, 'red box', 'box', 'red', [1, 0, 0, 1], [.03]*3, .1, 500, .4)
    state = ObjectState([.1, 0, .1], [1, 0, 0, 0], False, True)
    base = Episode('val_1_original', 'val', 'test', 1, [.5, .35], [obj],
                   Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1),
                   Action('push', 0, 'right', [1, 0, 0], .1, 1, 'Push right'),
                   [0., 1.], [[state], [ObjectState([.4, 0, .1], [1, 0, 0, 0], False, True)]])
    positions = {'original': .4, 'reverse': -.2, 'half': .2, 'double': .8, 'recolor': .4, 'elevation': .4}
    items = []
    for variant in VARIANTS:
        ep = copy.deepcopy(base)
        ep.episode_id = f'val_1_{variant}'
        ep.states[1][0].pos[0] = positions[variant]
        if variant == 'reverse':
            ep.action.direction_world = [-1, 0, 0]
            ep.action.direction_label = 'left'
        if variant in ('half', 'double'):
            factor = .5 if variant == 'half' else 2
            ep.action.impulse *= factor
            ep.action.speed *= factor
        if variant == 'elevation':
            ep.camera.cam2world = G.look_at([2, -1, 2], [0, 0, 0]).tolist()
        path = tmp_path / ep.episode_id
        path.mkdir()
        meta = dict(family_id='f1', variant=variant, split='val', seed=1,
                    reference_episode_id=base.episode_id, state_sha256='s', model_sha256='m')
        (path/'counterfactual.json').write_text(json.dumps(meta))
        items.append((path, ep))
    return load_families(items), items


def perfect(families):
    preds = []
    for family in families.values():
        for ep, _ in family.values():
            pos = G.world_to_cam([s.pos for s in ep.state_at(1)], ep.camera.cam2world)
            preds.append(dict(episode_id=ep.episode_id, task='B',
                              objects={str(o.id): p.tolist() for o, p in zip(ep.objects, pos)}))
    return preds


def test_perfect_world_predictions_are_camera_invariant(families):
    data, _ = families
    result = analyze(data, perfect(data), n_boot=30)
    assert result['n_families'] == 1
    for variant, scopes in result['variants'].items():
        assert scopes['target']['endpoint_l2_m']['mean'] < 1e-12
        if variant != 'original':
            assert scopes['target']['effect_l2_m']['mean'] < 1e-12
    assert result['variants']['elevation']['target']['predicted_change_valid_families_m']['mean'] < 1e-12


def test_static_can_be_invariant_but_wrong(families):
    data, _ = families
    result = analyze(data, static_predictions(data), n_boot=30)
    assert result['variants']['recolor']['target']['effect_l2_m']['mean'] < 1e-12
    assert result['variants']['original']['target']['endpoint_l2_m']['mean'] == pytest.approx(.3)
    assert result['variants']['reverse']['target']['effect_l2_m']['mean'] == pytest.approx(.6)


def test_missing_reference_reported_and_penalized(families):
    data, _ = families
    preds = perfect(data)
    preds[0]['objects']['0'] = None
    result = analyze(data, preds, n_boot=30)
    for variant in VARIANTS[1:]:
        record = result['variants'][variant]['target']
        assert record['missing_pair_rate'] == 1
        assert record['effect_l2_m']['mean'] == 1
        assert record['effect_l2_valid_families_m']['n_families'] == 0


def test_incomplete_predictions_fail(families):
    data, _ = families
    with pytest.raises(ValueError, match='incomplete prediction'):
        analyze(data, perfect(data)[:-1], n_boot=30)


def test_family_integrity_checks(families):
    _, items = families
    with pytest.raises(ValueError, match='incomplete family'):
        load_families(items[:-1])
    with pytest.raises(ValueError, match='duplicate variant'):
        load_families(items + items[:1])
    items[-1][1].states[1][0].pos[0] += .1
    with pytest.raises(ValueError, match='changed physics'):
        load_families(items)


def test_intervention_semantics_checked(families):
    _, items = families
    items[1][1].action.direction_world = [1, 0, 0]
    with pytest.raises(ValueError, match='does not match intervention'):
        load_families(items)


def test_all_objects_are_averaged_within_family_before_bootstrap(families):
    data, _ = families
    second = copy.deepcopy(data['f1'])
    for ep, meta in second.values():
        ep.episode_id = ep.episode_id.replace('val_1_', 'val_2_')
        for oid in (1, 2):
            obj = copy.deepcopy(ep.objects[0])
            obj.id = oid
            ep.objects.append(obj)
            for states in ep.states:
                states.append(copy.deepcopy(states[0]))
    data['f2'] = second
    preds = perfect(data)
    for record in preds:
        if record['episode_id'].startswith('val_2_'):
            ep = next(ep for ep, _ in second.values() if ep.episode_id == record['episode_id'])
            for oid, p in record['objects'].items():
                world = G.cam_to_world(p, ep.camera.cam2world) + [1, 0, 0]
                record['objects'][oid] = G.world_to_cam(world, ep.camera.cam2world).tolist()
    result = analyze(data, preds, n_boot=200)
    endpoint = result['variants']['original']['all']['endpoint_l2_m']
    assert endpoint['n_families'] == 2
    assert endpoint['mean'] == pytest.approx(.5)  # object pooling would instead give .75
    assert endpoint['ci95'] == pytest.approx([0, 1])
