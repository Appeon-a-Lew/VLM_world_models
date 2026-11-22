import copy
import json

import numpy as np
import pytest

from physwm import geometry as G
from physwm.schema import Action, Camera, Episode, ObjectSpec, ObjectState
from physwm.eval.counterfactual import VARIANTS, load_families, static_predictions
from physwm.eval.paired_arms import (family_reductions, paired_stats, per_family_variant,
                                    seed_flags)


@pytest.fixture
def families(tmp_path):
    obj = ObjectSpec(0, 'red box', 'box', 'red', [1, 0, 0, 1], [.03] * 3, .1, 500, .4)
    state = ObjectState([.1, 0, .1], [1, 0, 0, 0], False, True)
    base = Episode('val_1_original', 'val', 'test', 1, [.5, .35], [obj],
                   Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1),
                   Action('push', 0, 'right', [1, 0, 0], .1, 1, 'Push right'),
                   [0., 1.], [[state], [ObjectState([.4, 0, .1], [1, 0, 0, 0], False, True)]])
    positions = {'original': .4, 'reverse': -.2, 'half': .2, 'double': .8,
                 'recolor': .4, 'elevation': .4}
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
        (path / 'counterfactual.json').write_text(json.dumps(meta))
        items.append((path, ep))
    return load_families(items)


def perfect(families):
    preds = []
    for family in families.values():
        for ep, _ in family.values():
            pos = G.world_to_cam([s.pos for s in ep.state_at(1)], ep.camera.cam2world)
            preds.append(dict(episode_id=ep.episode_id, task='B',
                              objects={str(o.id): p.tolist() for o, p in zip(ep.objects, pos)}))
    return preds


def with_missing_double(families):
    preds = perfect(families)
    for p in preds:
        if p['episode_id'].endswith('_double'):
            p['objects']['0'] = None
    return preds


E_STATIC = {'reverse': .6, 'half': .2, 'double': .4}


def test_static_effect_errors_match_the_hand_computed_values(families):
    err, miss = per_family_variant(families, static_predictions(families))
    for variant, want in E_STATIC.items():
        assert err[('f1', variant)] == pytest.approx(want)
    assert all(m == 0 for m in miss.values())


def test_paired_stats_perfect_vs_static(families):
    err_a, miss_a = per_family_variant(families, static_predictions(families))
    err_b, miss_b = per_family_variant(families, perfect(families))
    stats = paired_stats(families, err_a, err_b, n_boot=30, seed=0)
    assert stats['reduction_mean'] == pytest.approx(np.mean(list(E_STATIC.values())))
    assert stats['reduction_ci95'][0] == pytest.approx(stats['reduction_mean'])  # single family
    for variant, want in E_STATIC.items():
        assert stats['per_variant_reduction_m'][variant] == pytest.approx(want)
    flags = seed_flags(stats, threshold=.04, miss_rate_a=0.0, miss_rate_b=0.0)
    assert flags['earned'] and flags['seed_cleared']


def test_flat_contrast_never_clears(families):
    err, _ = per_family_variant(families, static_predictions(families))
    stats = paired_stats(families, err, err, n_boot=30, seed=0)
    flags = seed_flags(stats, threshold=.04, miss_rate_a=0.0, miss_rate_b=0.0)
    assert stats['reduction_mean'] == pytest.approx(0.0)
    assert not flags['earned'] and not flags['seed_cleared']


def test_missing_guard_blocks_a_penalty_gamed_gain(families):
    err_a, miss_a = per_family_variant(families, static_predictions(families))
    err_b, miss_b = per_family_variant(families, with_missing_double(families))
    stats = paired_stats(families, err_a, err_b, n_boot=30, seed=0)
    threshold = max(.10 * np.mean(list(E_STATIC.values())), .02)
    flags = seed_flags(stats, threshold=threshold,
                      miss_rate_a=float(np.mean(list(miss_a.values()))),
                      miss_rate_b=float(np.mean(list(miss_b.values()))))
    # the reduction itself clears the threshold, but the gain is bought by refusing to
    # answer, so neither seed 2 nor the gate may be authorized
    assert stats['reduction_mean'] >= threshold
    assert flags['missing_guard_ok'] is False
    assert flags['earned'] is False
    assert flags['seed_cleared'] is False


def test_half_threshold_earns_but_does_not_clear(families):
    err_a, _ = per_family_variant(families, static_predictions(families))
    err_b, _ = per_family_variant(families, perfect(families))
    # construct a point estimate above half but below the full threshold
    stats = paired_stats(families, err_a, err_b, n_boot=30, seed=0)
    stats['reduction_mean'] = 0.75 * .04
    flags = seed_flags(stats, threshold=.04, miss_rate_a=0.0, miss_rate_b=0.0)
    assert flags['earned'] and not flags['seed_cleared']


def test_gate_requires_two_verified_distinct_seed_pairs():
    from physwm.eval.paired_arms import REQUIRED_SEED_PAIRS, gate_verdict
    assert REQUIRED_SEED_PAIRS == 2
    cleared = {'seed_cleared': True}
    one = gate_verdict([cleared], verification_ok=True)
    assert not one['gate_cleared'] and 'insufficient seed pairs' in one['reason']
    two = gate_verdict([cleared, dict(cleared)], verification_ok=True)
    assert two['gate_cleared'] and two['reason'] == 'cleared'
    unverified = gate_verdict([cleared, dict(cleared)], verification_ok=False)
    assert not unverified['gate_cleared'] and unverified['reason'] == 'seed verification failed'
    mixed = gate_verdict([cleared, {'seed_cleared': False}], verification_ok=True)
    assert not mixed['gate_cleared'] and 'did not clear' in mixed['reason']


def test_run_seed_reads_recorded_training_seed(tmp_path):
    from physwm.eval.paired_arms import run_seed
    d = tmp_path / 'fwd_s1'
    d.mkdir()
    (d / 'pred.jsonl').write_text('{}')
    (d / 'config.json').write_text(json.dumps({'seed': 1}))
    assert run_seed(str(d / 'pred.jsonl')) == 1
    d2 = tmp_path / 'other'
    d2.mkdir()
    (d2 / 'pred.jsonl').write_text('{}')
    assert run_seed(str(d2 / 'pred.jsonl')) is None


def test_family_reductions_average_variants(families):
    err_a, _ = per_family_variant(families, static_predictions(families))
    err_b, _ = per_family_variant(families, perfect(families))
    red = family_reductions(err_a, err_b)
    assert red.shape == (1,)
    assert red[0] == pytest.approx(.4)
