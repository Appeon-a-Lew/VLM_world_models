from physwm.schema import Action, Camera, Episode, ObjectSpec
from physwm.vlm.parse import parse_answer


def _ep():
    objs = [
        ObjectSpec(0, "red box", "box", "red", [1, 0, 0, 1], [0.03] * 3, 0.1, 500, 0.4),
        ObjectSpec(1, "blue ball", "sphere", "blue", [0, 0, 1, 1], [0.03] * 3, 0.1, 500, 0.4),
    ]
    cam = Camera(448, 448, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                 50, 0, 45, 1)
    act = Action("push", 0, "left", [1, 0, 0], 0.1, 1.0, "Push the red box to the left.")
    return Episode("e", "s", "c", 0, [0.5, 0.35], objs, cam, act, [0.0], [[]])


def test_parse_json_center():
    objs, ok = parse_answer('```json\n[{"label": "red box", "center": [0.1, 0.2, 1.0]},'
                            ' {"label": "Blue sphere", "center": [0, 0, 1]}]\n```', _ep())
    assert ok and objs == {"0": [0.1, 0.2, 1.0], "1": [0.0, 0.0, 1.0]}


def test_parse_qwen3d_and_missing():
    objs, ok = parse_answer('[{"bbox_3d": [0.1, 0.2, 1.0, 0.1, 0.1, 0.1, 0, 0, 0], "label": "red box"}]', _ep())
    assert ok and objs == {"0": [0.1, 0.2, 1.0], "1": None}


def test_parse_truncated_falls_back_to_regex():
    objs, ok = parse_answer('[{"label": "red box", "center": [0.1, 0.2, 1.0]}, {"label": "blue ba', _ep())
    assert ok and objs["0"] == [0.1, 0.2, 1.0] and objs["1"] is None


def test_parse_garbage():
    objs, ok = parse_answer("I cannot determine this.", _ep())
    assert not ok and all(v is None for v in objs.values())
