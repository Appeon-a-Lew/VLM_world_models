"""Centre + extents answer format: target/parse round trip and body-frame box IoU."""

import glob
import json
from pathlib import Path

import numpy as np
import pytest

from physwm.eval.boxmetrics import aligned_iou, per_object_box_rows, summarize_boxes
from physwm.eval.metrics import MISSING_PENALTY, summarize, task_time
from physwm.schema import Action, Camera, Episode, ObjectState, ObjectSpec
from physwm.vlm.parse import parse_box3d
from physwm.vlm.prompts import prompt_text, target_text

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


def _grid_ep(n_frames: int = 11) -> Episode:
    """Episode on the standard 0.1 s frame grid (``frame_times = 0.0 .. 0.1 * (n-1)``)."""
    times = [round(0.1 * k, 1) for k in range(n_frames)]
    objects = [
        ObjectSpec(0, "red box", "box", "red", [1, 0, 0, 1], [0.05, 0.04, 0.03], 0.1, 500, 0.4),
        ObjectSpec(1, "blue ball", "sphere", "blue", [0, 0, 1, 1], [0.02, 0.02, 0.02], 0.1, 500, 0.4),
    ]
    cam = Camera(448, 448, [[480.0, 0.0, 224.0], [0.0, 480.0, 224.0], [0.0, 0.0, 1.0]],
                 np.eye(4).tolist(), 50.0, 0.0, 45.0, 1.0)
    action = Action("push", 0, "left", [1.0, 0.0, 0.0], 0.1, 1.0, "Push the red box to the left.")
    states = [
        [_state([0.10 + 0.01 * k, 0.02, 1.20]), _state([-0.15 - 0.01 * k, -0.03, 1.10])]
        for k in range(n_frames)
    ]
    return Episode("g", "s", "c", 0, [0.5, 0.35], objects, cam, action, times, states,
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


def test_task_time_n_follows_last_input_frame():
    ep = _grid_ep()
    assert task_time(ep, "N") == pytest.approx(ep.frame_times[1])  # legacy default
    assert task_time(ep, "N", (0,)) == pytest.approx(0.1)
    assert task_time(ep, "N", (0, 1, 2)) == pytest.approx(0.3)
    assert task_time(ep, "A", (0, 1, 2)) == 0.0
    assert task_time(ep, "B", (0, 1, 2)) == 1.0


def test_task_time_n_undefined_when_inputs_cover_all_frames():
    ep = _grid_ep(3)
    with pytest.raises(ValueError, match="inputs cover all stored frames"):
        task_time(ep, "N", (0, 1, 2))


def test_prompt_resolves_next_frame_after_last_input():
    ep = _grid_ep()
    single = prompt_text(ep, "N", "box3d")
    assert ("at the next stored frame after the last input image "
            "(0.1 second after the action)") in single
    assert single == prompt_text(ep, "N", "box3d", (0,))
    triple = prompt_text(ep, "N", "box3d", (0, 1, 2))
    assert "0.3 second after the action" in triple
    assert "The images show t = 0.0, 0.1, 0.2 seconds after the action." in triple
    assert "0.1 second after the action" not in triple


def test_multi_frame_prompt_lists_input_times():
    ep = _grid_ep()
    b_single, b_multi = prompt_text(ep, "B", "center"), prompt_text(ep, "B", "center", (0, 1, 2))
    assert "The images show t = 0.0, 0.1, 0.2 seconds after the action." in b_multi
    assert "1.0 second after the action" in b_multi
    assert "The images show" not in b_single
    a_multi = prompt_text(ep, "A", "center", (0, 1, 2))
    assert "as it is in the first image (t = 0.0 s)" in a_multi
    assert "The images show t = 0.0, 0.1, 0.2 seconds after the action." in a_multi
    assert "The images show" not in prompt_text(ep, "A", "center")


def test_target_text_uses_resolved_next_frame_time():
    ep = _grid_ep()
    data = json.loads(target_text(ep, "N", "center", (0, 1, 2)))
    assert data == [
        {"label": o.name,
         "center": [round(float(v), 3) for v in ep.state_at(0.3)[i].pos]}
        for i, o in enumerate(ep.objects)
    ]


def test_metrics_summarize_scores_task_n_from_record_frames():
    ep = _grid_ep()
    pred = {"episode_id": "g", "task": "N", "parsed": True, "frames": [0, 1, 2],
            "objects": {str(i): list(ep.state_at(0.3)[i].pos) for i in range(len(ep.objects))}}
    out = summarize([pred], {"g": ep})[0]
    assert out["all_mean"] == pytest.approx(0.0)  # identity cam2world, t = 0.3 s


def test_prediction_record_keeps_frames():
    pytest.importorskip("torch")
    from physwm.vlm.infer import prediction_record

    rec = prediction_record(_grid_ep(), "N", {"0": [0.0, 0.0, 1.0]}, None, True, "raw",
                            [0, 1, 2], image_mode="normal", style="center", lora=None)
    again = json.loads(json.dumps(rec))
    assert again["frames"] == [0, 1, 2] and again["task"] == "N"
    assert "extents" not in again


def test_summary_obs_columns_exclude_missing_and_count_target_missing():
    ep = _ep()
    non_target_missing = {"episode_id": "e", "task": "B", "parsed": True,
                          "objects": {"0": [0.30, 0.02, 1.20], "1": None},
                          "extents": {"0": [0.1, 0.08, 0.06]}}
    s = summarize_boxes([non_target_missing], {"e": ep}, repetitions=20)[0]
    assert s["ext_l2_obs"] == pytest.approx(0.0)  # object 1 excluded, not penalised
    assert s["iou_obs"] == pytest.approx(1.0)
    assert s["ext_l2"] == pytest.approx(MISSING_PENALTY / 2)  # penalized mean unchanged
    assert s["iou"] == pytest.approx(0.5)
    assert s["target_missing"] == 0.0  # only the non-target object is missing

    target_missing = {"episode_id": "e", "task": "B", "parsed": True,
                      "objects": {"0": None, "1": [-0.12, -0.03, 1.10]},
                      "extents": {"1": [0.04, 0.04, 0.04]}}
    s = summarize_boxes([target_missing], {"e": ep}, repetitions=20)[0]
    assert s["target_missing"] == 1.0
    assert s["ext_l2_obs"] == pytest.approx(0.0)
    assert s["iou_obs"] == pytest.approx(1.0)
