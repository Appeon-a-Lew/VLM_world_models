import numpy as np
import pytest

from physwm.schema import Action, Camera, Episode, ObjectSpec, ObjectState
from physwm.vlm.context import action_context, edge_distance, obstacles, support_map


def make_episode(specs, states, action, table=(0.5, 0.35)):
    cam = Camera(448, 448, np.eye(3).tolist(), np.eye(4).tolist(), 50, 0, 45, 1)
    return Episode('e', 'val', 'test', 1, list(table), specs, cam, action,
                   [0.0], [list(states)], {}, [], [0.0])


def box_spec(i, name, half, pos):
    return (ObjectSpec(i, name, 'box', 'red', [1, 0, 0, 1], list(half), .1, 500, .4),
            ObjectState(list(pos), [1, 0, 0, 0], False, True))


def push_action(target=0, direction=(1.0, 0.0)):
    return Action('push', target, 'right', [direction[0], direction[1], 0.0], .1, 1.0,
                  'Push the red box to the right.')


def test_edge_distance_axes_and_diagonal():
    ep = make_episode([], [], push_action())
    assert edge_distance(ep, (0.1, 0.0), (1, 0)) == pytest.approx(0.4)
    assert edge_distance(ep, (0.1, 0.2), (0, 1)) == pytest.approx(0.15)
    assert edge_distance(ep, (0.0, 0.0), (1, 1)) == pytest.approx(0.35 * np.sqrt(2))


def test_edge_distance_rejects_degenerate_direction():
    ep = make_episode([], [], push_action())
    with pytest.raises(ValueError, match='non-zero'):
        edge_distance(ep, (0.0, 0.0), (0, 0))


def test_support_flat_and_stacked():
    a, sa = box_spec(0, 'red box', [.03, .03, .05], (0, 0, .05))
    b, sb = box_spec(1, 'green box', [.03, .03, .05], (0, 0, .15))
    ep = make_episode([a, b], [sa, sb], push_action())
    sup = support_map(ep)
    assert sup[0] == 'table'
    assert sup[1] == 'red box'


def test_support_broken_stack_is_none():
    a, sa = box_spec(0, 'red box', [.03, .03, .05], (0, 0, .05))
    b, sb = box_spec(1, 'green box', [.03, .03, .05], (0.2, 0, .15))  # no footprint overlap
    ep = make_episode([a, b], [sa, sb], push_action())
    sup = support_map(ep)
    assert sup[0] == 'table'
    assert sup[1] == 'none'


def test_obstacles_corridor_order_and_gaps():
    specs, states = zip(*[
        box_spec(0, 'red box', [.03, .03, .05], (0, 0, .05)),
        box_spec(1, 'near box', [.03, .03, .05], (0.1, 0.05, .05)),   # perp 0.05 <= 0.06
        box_spec(2, 'far box', [.03, .03, .05], (0.25, 0, .05)),
        box_spec(3, 'behind box', [.03, .03, .05], (-0.1, 0, .05)),
    ])
    ep = make_episode(list(specs), list(states), push_action())
    obs = obstacles(ep)
    assert [o['label'] for o in obs] == ['near box', 'far box']
    assert obs[0]['distance_m'] == pytest.approx(0.04)   # along 0.10 - radii 0.06
    assert obs[1]['distance_m'] == pytest.approx(0.19)  # along 0.25 - radii 0.06


def test_action_context_payload():
    a, sa = box_spec(0, 'red box', [.03, .03, .05], (0.05, 0, .05))
    b, sb = box_spec(1, 'green box', [.03, .03, .05], (0.25, 0, .05))
    ep = make_episode([a, b], [sa, sb], push_action())
    ctx = action_context(ep)
    assert ctx['target'] == 'red box'
    assert ctx['direction'] == 'right'
    assert ctx['edge_distance_m'] == pytest.approx(0.45)
    assert ctx['obstacles'] == [{'label': 'green box', 'distance_m': 0.14}]  # along .20 - radii .06
