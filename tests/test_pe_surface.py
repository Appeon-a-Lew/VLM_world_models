"""Synthetic tests for the IDEA-103 surface-vs-centre read-out probe (CPU only)."""
import json

import numpy as np

from physwm.eval import analyze_pe_surface as A


def test_centre_prediction_has_zero_radial_and_tangential_error():
    gt = np.array([0.0, 0.0, 1.0])
    err, rad, tan = A.error_components(gt, gt)
    assert err < 1e-12
    assert abs(rad) < 1e-12
    assert abs(tan) < 1e-12


def test_tangential_shift_leaves_radial_error_at_zero():
    gt = np.array([0.0, 0.0, 1.0])
    err, rad, tan = A.error_components(gt + np.array([0.05, 0.0, 0.0]), gt)
    assert abs(rad) < 1e-12
    assert abs(tan - 0.05) < 1e-12
    assert abs(err - 0.05) < 1e-12


def test_axis_aligned_surface_shift_gives_rad_equals_minus_d_near():
    # Identity camera pose: world == camera frame, camera at the origin looking +z.
    gt = np.array([0.0, 0.0, 1.0])
    quat = [1.0, 0.0, 0.0, 0.0]
    h = [0.1, 0.2, 0.3]
    d_near = A.near_surface_offset(quat, h, np.eye(4), gt)
    assert abs(d_near - 0.3) < 1e-12
    r = gt / np.linalg.norm(gt)
    pred = gt - d_near * r  # nearest visible surface point along the view ray
    _, rad, tan = A.error_components(pred, gt)
    assert abs(rad - (-d_near)) < 1e-12
    assert abs(tan) < 1e-12


def test_rotated_box_projects_the_rotated_half_extent():
    # 90 deg about camera x maps body z -> -camera y; the ray along +z then sees the
    # body y half-extent (0.2), not the z half-extent (0.3).
    gt = np.array([0.0, 0.0, 1.0])
    q90x = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    d_near = A.near_surface_offset(q90x, [0.1, 0.2, 0.3], np.eye(4), gt)
    assert abs(d_near - 0.2) < 1e-9


def test_oblique_ray_matches_box_support_function():
    # Direct check against the eight transformed corners: d_near is the distance from
    # the centre to the nearest corner-supported point along the ray.
    gt = np.array([0.3, -0.4, 0.8])
    r = gt / np.linalg.norm(gt)
    h = np.array([0.05, 0.12, 0.07])
    quat = [0.9, 0.1, -0.2, 0.3]
    d_near = A.near_surface_offset(quat, h, np.eye(4), gt)
    corners = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    r_body = A.G.quat_to_mat(quat).T @ r
    assert abs(d_near - (-np.min(corners @ r_body))) < 1e-12


def _episode_dict(eid: str, split: str) -> dict:
    """Minimal valid episode: camera at world (0,0,-1) looking +z, one box at origin."""
    cam2world = np.eye(4)
    cam2world[:3, 3] = [0.0, 0.0, -1.0]
    return {
        "episode_id": eid,
        "split": split,
        "config": "id",
        "seed": 0,
        "table_half_size": [0.5, 0.5],
        "objects": [{
            "id": 0, "name": "red box", "shape": "box", "color": "red",
            "rgba": [1.0, 0.0, 0.0, 1.0], "half_extents": [0.1, 0.2, 0.3],
            "mass": 0.5, "density": 700.0, "friction": 0.5,
        }],
        "camera": {
            "width": 448, "height": 448, "K": [[480.0, 0.0, 224.0],
                                                [0.0, 480.0, 224.0], [0.0, 0.0, 1.0]],
            "cam2world": cam2world.tolist(), "fovy_deg": 50.0, "azimuth_deg": 0.0,
            "elevation_deg": 20.0, "distance": 1.0,
        },
        "action": {
            "type": "push", "target": 0, "direction_label": "left",
            "direction_world": [1.0, 0.0, 0.0], "impulse": 0.1, "speed": 0.2,
            "text": "push the red box to the left",
        },
        "times": [0.0, 1.0],
        "states": [
            [{"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
              "toppled": False, "on_table": True}],
            [{"pos": [0.1, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
              "toppled": False, "on_table": True}],
        ],
        "files": {},
        "views": [],
        "frame_times": [0.0, 1.0],
    }


def test_end_to_end_centre_vs_surface(tmp_path):
    eid, split, arm = "val_id_9000000", "val_id", "s_f1"
    meta = tmp_path / "metas" / split / f"{eid}.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps(_episode_dict(eid, split)))
    ep = A.Episode.load(meta)
    gt_cam = A.G.world_to_cam(np.array([s.pos for s in ep.state_at(0.0)]),
                              ep.camera.cam2world)[0]
    assert np.allclose(gt_cam, [0.0, 0.0, 1.0])  # camera at (0,0,-1) looking +z
    d_near = A.near_surface_offset(ep.state_at(0.0)[0].quat,
                                   ep.objects[0].half_extents,
                                   ep.camera.cam2world, gt_cam)
    assert abs(d_near - 0.3) < 1e-12

    pred_dir = tmp_path / "pred"
    for name, centre in (("centre", gt_cam),
                         ("surface", gt_cam - np.array([0.0, 0.0, d_near]))):
        d = pred_dir / f"{arm}_{name}" / split
        d.mkdir(parents=True)
        (d / f"{eid}.json").write_text(json.dumps([{
            "episode_id": eid, "task": "A",
            "objects": {"0": [float(x) for x in centre]}, "parsed": True,
        }]))
        rows = A.load_arm_rows(pred_dir, [split], {split: {eid: meta}},
                              f"{arm}_{name}")
        assert len(rows) == 1
        expect = 0.0 if name == "centre" else -d_near
        assert abs(rows[0]["rad"] - expect) < 1e-9
        assert rows[0]["tan"] < 1e-9