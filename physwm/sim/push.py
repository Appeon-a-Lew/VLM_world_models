"""Simulate one push episode: sample scene + camera + action, settle, push, roll out, render."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from physwm import geometry as G
from physwm.schema import (FRAME_KINDS, Action, Camera, Episode, ObjectSpec, ObjectState,
                           frame_file_name, frame_key)
from physwm.sim.render import CameraRenderer
from physwm.sim.scene import COLORS, SHAPE_NOUN, build_mjcf, sample_objects

DIRECTION_PHRASE = {
    "left": "to the left",
    "right": "to the right",
    "away": "away from the camera",
    "toward": "toward the camera",
}


class Rejected(Exception):
    """Sampled scene is unusable (placement failed or did not settle); resample."""


@dataclass
class EpisodeData:
    """One simulated episode plus every rendered observation, keyed by (view, frame index)."""

    episode: Episode
    rgb: dict[tuple[int, int], np.ndarray]
    depth: dict[tuple[int, int], np.ndarray]
    seg: dict[tuple[int, int], np.ndarray]
    traj_t: np.ndarray  # (T,)
    traj_pos: np.ndarray  # (T, N, 3)
    traj_quat: np.ndarray  # (T, N, 4)


def _sample_camera(rng, cfg: dict, lookat: np.ndarray) -> tuple[Camera, np.ndarray]:
    c = cfg["camera"]
    az = np.radians(rng.uniform(*c["azimuth"]))
    el = np.radians(rng.uniform(*c["elevation"]))
    dist = rng.uniform(*c["distance"])
    target = lookat + np.r_[rng.uniform(-1, 1, 2) * c["lookat_jitter"], 0.0]
    eye = target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    T = G.look_at(eye, target)
    K = G.intrinsics_from_fovy(c["fovy"], c["width"], c["height"])
    cam = Camera(
        width=c["width"], height=c["height"], K=K.tolist(), cam2world=T.tolist(), fovy_deg=c["fovy"],
        azimuth_deg=float(np.degrees(az)), elevation_deg=float(np.degrees(el)), distance=float(dist),
    )
    return cam, T


def _sample_views(rng, cfg: dict, lookat: np.ndarray, n_views: int) -> tuple[list[Camera], list[np.ndarray]]:
    """Sample V cameras with distinct poses.

    View 0 draws from the main stream (keeping single-view episodes seed-identical to
    dataset v2); each further view uses its own child stream derived from it.
    """
    cams, mats = [], []
    for v in range(n_views):
        view_rng = rng if v == 0 else np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(10):
            cam, T = _sample_camera(view_rng, cfg, lookat)
            if all(not np.allclose(T, other, rtol=0, atol=1e-9) for other in mats):
                break
        else:
            raise Rejected(f"could not sample a distinct camera for view {v}")
        cams.append(cam)
        mats.append(T)
    assert len({m.tobytes() for m in mats}) == n_views, "views must have distinct cam2world"
    return cams, mats


def _camera_xml(T: np.ndarray, fovy: float, name: str = "cam") -> str:
    # MuJoCo cameras look along -z with y up: x = right, y = -down (OpenCV y)
    x, y = T[:3, 0], -T[:3, 1]
    p = T[:3, 3]
    xy = " ".join(f"{v:.6f}" for v in np.r_[x, y])
    return f'<camera name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" xyaxes="{xy}" fovy="{fovy}"/>'


def _sample_push(rng, cfg: dict, T: np.ndarray, target_xy, other_xy) -> tuple[str, np.ndarray, float]:
    p = cfg["push"]
    right = T[:3, 0].copy()
    right[2] = 0
    right /= np.linalg.norm(right)
    fwd = T[:3, 2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    bases = {"left": -right, "right": right, "away": fwd, "toward": -fwd}
    label = str(rng.choice(p["directions"]))
    if len(other_xy) and rng.uniform() < p.get("aim_at_other_prob", 0.0):
        # aim roughly at a neighbour to create collisions; the label stays camera-relative
        to = np.r_[other_xy[int(rng.integers(len(other_xy)))] - target_xy, 0.0]
        label = max(p["directions"], key=lambda k: float(bases[k] @ to))
    base = bases[label]
    a = np.radians(rng.uniform(-1, 1) * p["direction_jitter_deg"])
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    direction = rot @ base
    if "speed_choices" in p:
        lo, hi = p["speed_choices"][int(rng.integers(len(p["speed_choices"])))]
    else:
        lo, hi = p["speed"]
    return label, direction, float(rng.uniform(lo, hi))


def _states(m, d, body_ids, up_local, shapes, table_half) -> list[ObjectState]:
    out = []
    for b, up, shape in zip(body_ids, up_local, shapes):
        pos, quat = d.xpos[b].copy(), d.xquat[b].copy()
        up_now = G.quat_to_mat(quat) @ up
        tilt = np.degrees(np.arccos(np.clip(up_now[2], -1, 1)))
        on_table = bool(pos[2] > -0.02 and abs(pos[0]) < table_half[0] and abs(pos[1]) < table_half[1])
        out.append(ObjectState(
            pos=pos.tolist(), quat=quat.tolist(),
            toppled=bool(shape != "sphere" and tilt > 45), on_table=on_table,
        ))
    return out


def simulate(cfg: dict, seed: int, split: str, max_attempts: int = 20) -> EpisodeData:
    validate_timing(cfg["sim"])
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        try:
            return _simulate_once(rng, cfg, seed, split)
        except Rejected:
            continue
    raise RuntimeError(f"seed {seed}: no valid scene after {max_attempts} attempts")


def validate_timing(s: dict) -> None:
    """Reject schedules and view counts which would silently omit or mislabel observations."""
    dt = float(s["timestep"])
    hz = float(s["traj_hz"])
    times = np.asarray(s["store_times"], dtype=float)
    frames = np.asarray(s["frame_times"], dtype=float)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(hz) or hz <= 0:
        raise ValueError("timestep and traj_hz must be positive and finite")
    if (len(times) < 2 or not np.all(np.isfinite(times)) or times[0] != 0
            or np.any(np.diff(times) <= 0) or not np.any(np.isclose(times, 1, atol=1e-9, rtol=0))
            or times[-1] > s["horizon"]):
        raise ValueError("store_times must increase from 0, include 1 s, and fit within horizon")
    if (len(frames) < 2 or not np.all(np.isfinite(frames)) or frames[0] != 0
            or np.any(np.diff(frames) <= 0) or not np.any(np.isclose(frames, 1, atol=1e-9, rtol=0))
            or frames[-1] > s["horizon"]):
        raise ValueError("frame_times must increase from 0, include 1 s, and fit within horizon")
    if not all(np.any(np.isclose(times, f, atol=1e-9, rtol=0)) for f in frames):
        raise ValueError("frame_times must be a subset of store_times")
    steps = frames / dt
    if np.any(~np.isfinite(steps)) or np.any(steps < 0) or np.any(~np.isclose(steps, np.round(steps), atol=1e-8, rtol=0)):
        raise ValueError("frame_times must be non-negative timestep multiples")
    n_views = s.get("n_views", 1)
    if not isinstance(n_views, int) or isinstance(n_views, bool) or n_views < 1:
        raise ValueError("n_views must be an integer >= 1")
    intervals = [s["settle"], s["horizon"], 1 / hz, *times[1:]]
    for interval in intervals:
        steps = interval / dt
        if not np.isfinite(steps) or steps < 1 or not np.isclose(steps, round(steps), atol=1e-8, rtol=0):
            raise ValueError("settle, horizon, trajectory period and stored times must be positive timestep multiples")


def _simulate_once(rng, cfg: dict, seed: int, split: str) -> EpisodeData:
    try:
        objs = sample_objects(rng, cfg)
    except RuntimeError as e:
        raise Rejected(str(e))
    lookat = np.r_[cfg["objects"].get("region_center", [0.0, 0.0]), 0.0]
    s = cfg["sim"]
    cams, mats = _sample_views(rng, cfg, lookat, int(s.get("n_views", 1)))
    cam, T = cams[0], mats[0]
    table_rgba = list(rng.uniform(0.55, 0.85) * np.array([1.0, 0.85, 0.65])) + [1.0]
    floor_rgba = list(rng.uniform(0.15, 0.35, 3)) + [1.0]
    xml = build_mjcf(objs, cfg, table_rgba, floor_rgba)
    xml = xml.replace("</worldbody>", "".join(
        _camera_xml(T_view, c.fovy_deg, f"cam{v}") for v, (c, T_view) in enumerate(zip(cams, mats))
    ) + "\n  </worldbody>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(len(objs))]
    table_half = cfg["table"]["half_size"]

    # settle
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=int(round(s["settle"] / s["timestep"])))
    if np.abs(d.qvel).max() > 0.05:
        raise Rejected("did not settle")
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    # Use the canonical body up-axis: redefining 'up' after settling hides already toppled objects.
    up_local = [np.array([0, 0, 1.0]) for b in body_ids]
    shapes = [o.shape for o in objs]
    s0 = _states(m, d, body_ids, up_local, shapes, table_half)
    if any(st.toppled or not st.on_table for st in s0):
        raise Rejected("unstable rest pose")

    specs = [
        ObjectSpec(
            id=i, name=f"{o.color} {SHAPE_NOUN[o.shape]}", shape=o.shape, color=o.color,
            rgba=COLORS[o.color] + [1.0], half_extents=o.half_extents.tolist(),
            mass=float(m.body_mass[b]), density=o.density, friction=o.friction,
        )
        for i, (o, b) in enumerate(zip(objs, body_ids))
    ]

    # action
    target = int(rng.integers(len(objs)))
    xy = np.array([st.pos[:2] for st in s0])
    label, direction, speed = _sample_push(rng, cfg, T, xy[target], np.delete(xy, target, axis=0))
    impulse = specs[target].mass * speed
    text = (
        f"Push the {specs[target].name} {DIRECTION_PHRASE[label]} "
        f"with an impulse of {impulse:.6g} N·s."
    )
    action = Action(
        type="push", target=target, direction_label=label, direction_world=direction.tolist(),
        impulse=float(impulse), speed=speed, text=text,
    )

    # observations at every configured view and frame time (frame 0 is before the push)
    frame_times = [float(t) for t in s["frame_times"]]
    n_views = len(cams)
    renderers = [CameraRenderer(m, cam.width, cam.height, camera=f"cam{v}") for v in range(n_views)]
    rgb: dict[tuple[int, int], np.ndarray] = {}
    depth: dict[tuple[int, int], np.ndarray] = {}
    seg: dict[tuple[int, int], np.ndarray] = {}

    def capture(frame: int) -> None:
        for v, rend in enumerate(renderers):
            rgb[(v, frame)] = rend.rgb(d)
            depth[(v, frame)] = rend.depth(d)
            seg[(v, frame)] = rend.seg(d)

    try:
        capture(frame_times.index(0.0))
        # apply the impulse as an instantaneous change of the target's linear velocity
        dof = m.jnt_dofadr[m.body_jntadr[body_ids[target]]]
        d.qvel[dof:dof + 3] += speed * direction

        store_times = s["store_times"]
        every = int(round(1.0 / (s["traj_hz"] * s["timestep"])))
        n_steps = int(round(s["horizon"] / s["timestep"]))
        traj_t, traj_pos, traj_quat = [0.0], [d.xpos[body_ids].copy()], [d.xquat[body_ids].copy()]
        states = [s0]
        pending = {k: t for k, t in enumerate(frame_times) if t > 0}
        for k in range(1, n_steps + 1):
            mujoco.mj_step(m, d)
            t = k * s["timestep"]
            store = any(abs(t - st) < 1e-9 for st in store_times[1:])
            if k % every == 0 or store:
                # mj_step integrates qpos after computing xpos/xquat. Refresh geometry before
                # recording or rendering, without another dynamics/constraint solve.
                mujoco.mj_fwdPosition(m, d)
            if k % every == 0:
                traj_t.append(t)
                traj_pos.append(d.xpos[body_ids].copy())
                traj_quat.append(d.xquat[body_ids].copy())
            if store:
                states.append(_states(m, d, body_ids, up_local, shapes, table_half))
            for frame, ft in list(pending.items()):
                if abs(t - ft) < 1e-9:
                    capture(frame)
                    del pending[frame]
        assert not pending, f"frames not rendered: {sorted(pending.values())}"
    finally:
        for rend in renderers:
            rend.close()

    files = {"traj": "traj.npz"}
    for v in range(n_views):
        for frame in range(len(frame_times)):
            for kind in FRAME_KINDS:
                files[frame_key(kind, v, frame)] = frame_file_name(kind, v, frame)
    ep = Episode(
        episode_id=f"{split}_{seed:07d}", split=split, config=cfg["name"], seed=int(seed),
        table_half_size=list(table_half), objects=specs, camera=cam, action=action,
        times=list(store_times), states=states, files=files,
        views=cams, frame_times=frame_times,
    )
    return EpisodeData(
        episode=ep, rgb=rgb, depth=depth, seg=seg,
        traj_t=np.array(traj_t), traj_pos=np.array(traj_pos), traj_quat=np.array(traj_quat),
    )
