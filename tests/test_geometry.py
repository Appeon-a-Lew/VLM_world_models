import numpy as np

from physwm import geometry as G


def test_look_at_projects_target_to_principal_point():
    K = G.intrinsics_from_fovy(50, 448, 448)
    T = G.look_at([1.0, -0.5, 0.8], [0.1, 0.2, 0.0])
    uv = G.project(G.world_to_cam([0.1, 0.2, 0.0], T), K)
    assert np.allclose(uv, [224, 224], atol=1e-6)


def test_world_cam_roundtrip():
    T = G.look_at([0.3, 1.2, 0.9], [0, 0, 0])
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(G.cam_to_world(G.world_to_cam(p, T), T), p)


def test_camera_axes_up_is_negative_v():
    # a point above the target must appear higher in the image (smaller v)
    K = G.intrinsics_from_fovy(50, 448, 448)
    T = G.look_at([1.0, 0.0, 1.0], [0, 0, 0])
    v_target = G.project(G.world_to_cam([0, 0, 0], T), K)[1]
    v_above = G.project(G.world_to_cam([0, 0, 0.1], T), K)[1]
    assert v_above < v_target


def test_rotation_metrics():
    q = [1, 0, 0, 0]
    q90 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # 90 deg about x
    assert abs(G.rot_geodesic_deg(q, q)) < 1e-6
    assert abs(G.rot_geodesic_deg(q, q90) - 90) < 1e-6
    assert abs(G.tilt_deg(q90) - 90) < 1e-6
