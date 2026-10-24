"""Small geometry helpers (numpy only). Quaternions are wxyz."""

from __future__ import annotations

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_geodesic_deg(q1, q2) -> float:
    """Angle of the relative rotation between two quaternions, in degrees."""
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    q2 = np.asarray(q2, float) / np.linalg.norm(q2)
    d = min(1.0, abs(float(np.dot(q1, q2))))
    return float(np.degrees(2 * np.arccos(d)))


def tilt_deg(q) -> float:
    """Angle between the body z-axis and world up, in degrees."""
    return float(np.degrees(np.arccos(np.clip(quat_to_mat(q)[2, 2], -1.0, 1.0))))


def world_to_cam(p_world, cam2world) -> np.ndarray:
    """Transform world points (..., 3) into the OpenCV camera frame."""
    T = np.asarray(cam2world, float)
    R, t = T[:3, :3], T[:3, 3]
    return (np.asarray(p_world, float) - t) @ R


def cam_to_world(p_cam, cam2world) -> np.ndarray:
    T = np.asarray(cam2world, float)
    return np.asarray(p_cam, float) @ T[:3, :3].T + T[:3, 3]


def project(p_cam, K) -> np.ndarray:
    """Pinhole projection of camera-frame points (..., 3) to pixels (..., 2)."""
    p = np.asarray(p_cam, float)
    uvw = p @ np.asarray(K, float).T
    return uvw[..., :2] / uvw[..., 2:3]


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV-convention cam2world (x right, y down, z forward) looking from eye to target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, eye
    return T
