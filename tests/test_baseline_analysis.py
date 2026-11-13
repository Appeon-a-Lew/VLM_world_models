"""Independent synthetic checks of episode pairing, weighting and missingness."""
from types import SimpleNamespace
import numpy as np
import pytest
from physwm.eval.analyze_baseline import analyze, episode_row, paired_bootstrap, summarize_rows


def episode(eid='one', split='val_id'):
    return SimpleNamespace(episode_id=eid, split=split, action=SimpleNamespace(target=0, speed=2.),
        objects=[SimpleNamespace(shape='box')], table_half_size=[.5, .4],
        camera=SimpleNamespace(cam2world=np.eye(4), elevation_deg=70),
        state_at=lambda t: [SimpleNamespace(pos=[.2+t*.1, .1, .05])])


def pred(eid, task, value):
    return dict(episode_id=eid, task=task, objects={'0': value}, parsed=True)


def test_displacement_cancels_shared_translation_without_claiming_causality():
    ep = episode()
    row = episode_row(ep, pred('one', 'A', [.4, .1, .05]), pred('one', 'B', [.5, .1, .05]))
    assert row['A_error'] == pytest.approx(.2)
    assert row['B_error'] == pytest.approx(.2)
    assert row['own_static_error'] == pytest.approx(.1)
    assert row['displacement_error'] == pytest.approx(0, abs=1e-15)
    assert row['B_abs_x'] == pytest.approx(.2)
    assert row['edge_distance'] == pytest.approx(.3)
    assert row['speed_group'] == 'strong'


def test_missing_penalties_are_separate_from_valid_only_axes():
    row = episode_row(episode(), pred('one', 'A', None), pred('one', 'B', [float('nan'), 0, 0]))
    summary = summarize_rows([row])
    assert row['B_error'] == row['own_static_error'] == 1
    assert summary['B_error']['n'] == 1
    assert summary['missing_B'] == 1
    assert summary['B_abs_x']['n'] == 0
    assert summary['displacement_error']['missing_rate'] == 1


def test_bootstrap_pairs_episodes_and_weights_splits_equally():
    groups = [[dict(B_error=1., static_error=0.)],
              [dict(B_error=0., static_error=1.) for _ in range(9)]]
    first = paired_bootstrap(groups, 'static_error', 100, 17)
    assert first['difference'] == 0
    assert first['ci95'] == [0, 0]
    assert first == paired_bootstrap(groups, 'static_error', 100, 17)


def test_bootstrap_pairing_retains_covariance():
    group = [dict(B_error=float(i)+.25, static_error=float(i)) for i in range(10)]
    result = paired_bootstrap([group], 'static_error', 200, 2)
    assert result['ci95'] == [.25, .25]


def test_analysis_rejects_incomplete_and_duplicate_records():
    ep = episode()
    a = pred('one', 'A', [.2, .1, .05])
    b = pred('one', 'B', [.3, .1, .05])
    with pytest.raises(ValueError, match='incomplete'):
        analyze({'one': ep}, [a], 10)
    with pytest.raises(ValueError, match='duplicate'):
        analyze({'one': ep}, [a, a, b], 10)
    result = analyze({'one': ep}, [a, b], 10)
    assert result['overall_descriptive']['static_error']['difference'] == pytest.approx(-.1)
    assert result['strata_by_split']['val_id']['shape']['box']['episodes'] == 1


def test_primary_is_only_three_main_ood_splits_and_not_overall():
    from physwm.eval.analyze_baseline import PRIMARY_OOD, SPLITS
    episodes, predictions = {}, []
    for j, split in enumerate(SPLITS):
        # Unequal sizes test equal-split aggregation, not episode pooling.
        for k in range(j+1):
            eid = f'{split}_{k}'
            ep = episode(eid, split)
            episodes[eid] = ep
            predictions += [pred(eid, 'A', [.2, .1, .05]),
                            pred(eid, 'B', [.3 + (1 if split in PRIMARY_OOD else 0), .1, .05])]
    result = analyze(episodes, predictions, 100, 7)
    assert result['primary_ood']['available']
    assert result['primary_ood']['paired']['static_error']['difference'] == pytest.approx(.9)
    assert result['overall_descriptive']['static_error']['difference'] == pytest.approx(3/8-.1)
    assert result['primary_ood']['paired']['static_error']['ci95'] == pytest.approx([.9, .9])
    sensitivity = result['complete_pair_sensitivity']['own_static_error']
    assert sensitivity['primary_ood']['split_count'] == 3
    assert sensitivity['primary_ood']['difference'] == pytest.approx(.9)
    assert sensitivity['primary_ood']['ci95'] == pytest.approx([.9, .9])
    missing_eid = 'val_ood_objects_0'
    predictions = [pred(p['episode_id'], p['task'], None) if p['episode_id'] == missing_eid and p['task'] == 'A' else p for p in predictions]
    incomplete = analyze(episodes, predictions, 100, 7)['complete_pair_sensitivity']['own_static_error']
    assert incomplete['counts_by_split']['val_ood_objects'] == 1
    assert incomplete['primary_ood']['difference'] == pytest.approx(.9)
    assert incomplete['primary_ood']['ci95'] == pytest.approx([.9, .9])
    one = analyze({'one': episode()}, [pred('one','A',[.2,.1,.05]), pred('one','B',[.3,.1,.05])], 10)
    assert not one['primary_ood']['available']
    assert one['primary_ood']['paired'] is None
    assert one['primary_ood']['missing_splits'] == list(PRIMARY_OOD)
