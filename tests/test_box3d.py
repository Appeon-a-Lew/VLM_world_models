"""Centre + extents answer format: target/parse round trip and body-frame box IoU."""

import glob
import json
from pathlib import Path

import numpy as np
import pytest

from physwm.eval.boxmetrics import aligned_iou, per_object_box_rows, summarize_boxes
from physwm.eval.metrics import MISSING_PENALTY, summarize
from physwm.schema import Action, Camera, Episode, ObjectState, ObjectSpec
from physwm.vlm.parse import parse_box3d
from physwm.vlm.prompts import target_text

SAMPLES = Path("/tmp/opencode/wm-samples")


def _state(pos) -> ObjectState:
    return ObjectState(pos=list(pos), quat=[1.0, 0.0, 0.0, 0.0], toppled=False, on_table=True)


def _ep() -> Episode:
    objects = [
        ObjectSpec(0, "red box", "box", "red", [1, 0, 0, 1], [0.05, 0.04, 0.03], 0.1, 500, 0.4),
        ObjectSpec(1, "blue ball", "sphere", "blue", [0, 0, 1, 1], [0.02, 0.02, 0.02], 0.1, 500, 0.4),
    ]
    cam = Camera(448, 448, [[480.0, 0.0, 224.0], [0.0, 480.0, 224.0], [0.0, 0.0, 1.0]],
                 np.eye(4).tolist(), 50.0, 0.0, 45.0, 1.0)
    action = Action("push", 0, "left", [1.0, 0.0, 0.0], 0.1, 1.0, "Push the red box to the left.")
    times = [0.0, 0.1, 1.0]
    states = [
        [_state([0.10, 0.02, 1.20]), _state([-0.15, -0.03, 1.10])],
        [_state([0.12, 0.02, 1.20]), _state([-0.14, -0.03, 1.10])],
        [_state([0.30, 0.02, 1.20]), _state([-0.12, -0.03, 1.10])],
    ]
    return Episode("e", "s", "c", 0, [0.5, 0.35], objects, cam, action, times, states,
                   frame_times=times)


@pytest.mark.parametrize("task,t", [("A", 0.0), ("B", 1.0), ("N", 0.1)])
def test_target_text_parse_roundtrip(task, t):
    ep = _ep()
    text = target_text(ep, task, "box3d")
    data = json.loads(text)
    assert [d["label"] for d in data] == ["red box", "blue ball"]
    centres, extents, parsed = parse_box3d(text, ep)
    assert parsed
    for i, o in enumerate(ep.objects):
        expected_c = [round(float(v), 3) for v in ep.state_at(t)[i].pos]  # identity cam2world
        assert centres[str(o.id)] == expected_c
        assert extents[str(o.id)] == [round(2 * v, 3) for v in o.half_extents]


def test_roundtrip_on_real_sample_if_present():
    metas = sorted(glob.glob(str(SAMPLES / "val_id" / "*" / "meta.json")))
    if not metas:
        pytest.skip("no local sample corpus")
    ep = Episode.load(metas[0])
    for task in ("A", "B"):
        centres, extents, parsed = parse_box3d(target_text(ep, task, "box3d"), ep)
        assert parsed and all(v is not None for v in centres.values())
        assert all(v is not None for v in extents.values())


def test_parse_box3d_tolerates_fences_and_extra_keys():
    ep = _ep()
    answer = ('```json\n[{"label": "Red Box", "center": [0.1, 0.2, 1.0], '
              '"extents": [0.1, 0.08, 0.06], "confidence": 0.9}, '
              '{"label": "blue sphere", "center": [0.0, 0.0, 1.0]}]\n```')
    centres, extents, parsed = parse_box3d(answer, ep)
    assert parsed
    assert centres == {"0": [0.1, 0.2, 1.0], "1": [0.0, 0.0, 1.0]}
    assert extents == {"0": [0.1, 0.08, 0.06], "1": None}


def test_parse_box3d_truncated_and_garbage():
    ep = _ep()
    centres, extents, parsed = parse_box3d(
        '[{"label": "red box", "center": [0.1, 0.2, 1.0], "extents": [0.1, 0.08, 0.0', ep)
    assert parsed and centres["0"] == [0.1, 0.2, 1.0] and extents["0"] is None
    centres, extents, parsed = parse_box3d("I cannot determine this.", ep)
    assert not parsed and all(v is None for v in centres.values())


def test_iou_identical_disjoint_and_half_overlap():
    centre = np.array([0.0, 0.0, 1.0])
    ext = np.array([2.0, 2.0, 2.0])
    cam = np.eye(4)
    quat = [1.0, 0.0, 0.0, 0.0]
    assert aligned_iou(centre, ext, centre, ext, quat, cam) == pytest.approx(1.0)
    assert aligned_iou(centre + [5.0, 0.0, 0.0], ext, centre, ext, quat, cam) == 0.0
    # shift by half an edge along one axis: overlap 1/3 of the union
    assert aligned_iou(centre + [1.0, 0.0, 0.0], ext, centre, ext, quat, cam) == pytest.approx(1 / 3)


def test_iou_uses_gt_body_frame():
    # A 90 deg rotation about world z maps a body +x offset onto world +y; the box IoU must
    # still see a half-overlap along the body x axis.
    quat = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    centre = np.array([0.0, 0.0, 2.0])
    ext = np.array([2.0, 2.0, 2.0])
    iou = aligned_iou(centre + [0.0, 1.0, 0.0], ext, centre, ext, quat, np.eye(4))
    assert iou == pytest.approx(1 / 3)


def test_metrics_summarize_scores_next_frame_task():
    ep = _ep()
    pred = {"episode_id": "e", "task": "N", "parsed": True,
            "objects": {"0": [0.12, 0.02, 1.20], "1": [-0.14, -0.03, 1.10]}}
    rows = summarize([pred], {"e": ep})
    assert len(rows) == 1 and rows[0]["task"] == "N"
    assert rows[0]["all_mean"] == pytest.approx(0.0)  # t = frame_times[1] = 0.1 s
    assert rows[0]["all_median"] == pytest.approx(0.0)


def test_box_rows_and_summary_penalise_missing_extents():
    ep = _ep()
    pred = {"task": "B", "objects": {"0": [0.30, 0.02, 1.20], "1": None},
            "extents": {"0": [0.1, 0.08, 0.06]}}
    rows = per_object_box_rows(pred, ep)
    assert rows[0]["centre_l2"] == pytest.approx(0.0) and rows[0]["ext_l2"] == pytest.approx(0.0)
    assert rows[0]["iou"] == pytest.approx(1.0)
    assert rows[1]["missing"] and rows[1]["ext_l2"] == MISSING_PENALTY and rows[1]["iou"] == 0.0

    summary = summarize_boxes([{**pred, "episode_id": "e", "parsed": True}], {"e": ep},
                              repetitions=50)
    assert len(summary) == 1 and summary[0]["split"] == "s" and summary[0]["task"] == "B"
    assert 0.0 < summary[0]["iou"] < 1.0 and np.isfinite(summary[0]["ext_l2_ci95"]).all()
