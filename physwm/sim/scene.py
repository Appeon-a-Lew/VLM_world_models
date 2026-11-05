"""Config loading, random scene sampling and MJCF construction.

Appearance (colour), geometry (shape / size) and physics (density, friction) are sampled
independently so that no appearance cue predicts a physical property unless a config asks for it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

COLORS = {
    "red": [0.85, 0.12, 0.12],
    "green": [0.15, 0.65, 0.2],
    "blue": [0.15, 0.3, 0.85],
    "yellow": [0.95, 0.85, 0.1],
    "purple": [0.55, 0.2, 0.7],
    "orange": [0.95, 0.5, 0.1],
    "white": [0.95, 0.95, 0.95],
    "black": [0.08, 0.08, 0.08],
    "cyan": [0.1, 0.8, 0.85],
    "pink": [0.95, 0.5, 0.7],
    "brown": [0.45, 0.28, 0.12],
    "gray": [0.5, 0.5, 0.5],
}

# prompt nouns per shape
SHAPE_NOUN = {
    "box": "box",
    "cylinder": "cylinder",
    "sphere": "ball",
    "capsule": "capsule",
    "cone": "cone",
    "hexprism": "hexagonal prism",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    return cfg


@dataclass
class ObjSample:
    shape: str
    color: str
    half_extents: np.ndarray  # body-frame half extents (x, y, z)
    density: float
    friction: float
    pos: np.ndarray  # initial xyz (body origin = geometric centre)
    yaw: float
    mjcf_size: str  # geom size attribute (or "" for meshes)


def _uniform(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _sample_object(rng, ocfg: dict, shape: str, color: str) -> ObjSample:
    r = _uniform(rng, ocfg["half_size"])
    aspect = _uniform(rng, ocfg["aspect"])
    h = r * aspect  # half height
    if shape == "box":
        rx, ry = r, r * float(rng.uniform(0.6, 1.0))
        ext, size = [rx, ry, h], f"{rx:.4f} {ry:.4f} {h:.4f}"
    elif shape == "cylinder":
        ext, size = [r, r, h], f"{r:.4f} {h:.4f}"
    elif shape == "sphere":
        ext, size = [r, r, r], f"{r:.4f}"
    elif shape == "capsule":
        hl = max(h - r, 0.2 * r)  # half length of the cylindrical part
        ext, size = [r, r, hl + r], f"{r:.4f} {hl:.4f}"
    elif shape in ("cone", "hexprism"):
        ext, size = [r, r, h], ""
    else:
        raise ValueError(f"unknown shape {shape}")
    return ObjSample(
        shape=shape,
        color=color,
        half_extents=np.array(ext),
        density=_uniform(rng, ocfg["density"]),
        friction=_uniform(rng, ocfg["friction"]),
        pos=np.zeros(3),
        yaw=float(rng.uniform(-np.pi, np.pi)),
        mjcf_size=size,
    )


def sample_objects(rng, cfg: dict, max_tries: int = 200) -> list[ObjSample]:
    ocfg = cfg["objects"]
    n = int(rng.integers(ocfg["count"][0], ocfg["count"][1] + 1))
    colors = rng.choice(ocfg["colors"], size=n, replace=False)
    shapes = rng.choice(ocfg["shapes"], size=n, replace=True)
    objs = [_sample_object(rng, ocfg, s, c) for s, c in zip(shapes, colors)]
    cx, cy = ocfg.get("region_center", [0.0, 0.0])
    rx, ry = ocfg["region"]
    placed: list[ObjSample] = []
    for o in objs:
        foot = float(np.hypot(o.half_extents[0], o.half_extents[1]))
        for _ in range(max_tries):
            p = np.array([cx + rng.uniform(-rx, rx), cy + rng.uniform(-ry, ry)])
            ok = all(
                np.linalg.norm(p - q.pos[:2]) > foot + np.hypot(*q.half_extents[:2]) + ocfg["min_gap"]
                for q in placed
            )
            if ok:
                o.pos = np.array([p[0], p[1], o.half_extents[2] + 0.002])
                placed.append(o)
                break
        else:
            raise RuntimeError("placement failed")
    return placed


def _mesh_vertices(shape: str, r: float, h: float) -> str:
    """Convex mesh vertices centred at the origin (MuJoCo builds the hull)."""
    if shape == "cone":
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        verts = [[r * np.cos(a), r * np.sin(a), -h] for a in ang] + [[0, 0, h]]
    elif shape == "hexprism":
        ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        verts = [[r * np.cos(a), r * np.sin(a), z] for z in (-h, h) for a in ang]
    else:
        raise ValueError(shape)
    return " ".join(f"{v:.5f}" for p in verts for v in p)


def build_mjcf(objs: list[ObjSample], cfg: dict, table_rgba, floor_rgba) -> str:
    # MuJoCo combines contact friction with max(); the table gets a low value so each object's own
    # sampled friction is what governs sliding.
    tx, ty = cfg["table"]["half_size"]
    th = cfg["table"]["height"]
    ocfg_roll = cfg["objects"]["rolling_friction"]
    ts = cfg["sim"]["timestep"]
    assets, bodies = [], []
    for i, o in enumerate(objs):
        rgba = " ".join(f"{c:.3f}" for c in COLORS[o.color] + [1.0])
        common = (
            f'rgba="{rgba}" density="{o.density:.2f}" friction="{o.friction:.3f} 0.005 {ocfg_roll:.4f}" '
            f'condim="6" name="g{i}"'
        )
        if o.shape in ("cone", "hexprism"):
            assets.append(
                f'<mesh name="m{i}" vertex="{_mesh_vertices(o.shape, o.half_extents[0], o.half_extents[2])}"/>'
            )
            geom = f'<geom type="mesh" mesh="m{i}" {common}/>'
        else:
            geom = f'<geom type="{o.shape}" size="{o.mjcf_size}" {common}/>'
        qz = f"{np.cos(o.yaw / 2):.6f} 0 0 {np.sin(o.yaw / 2):.6f}"
        bodies.append(
            f'<body name="obj{i}" pos="{o.pos[0]:.5f} {o.pos[1]:.5f} {o.pos[2]:.5f}" quat="{qz}">'
            f'<freejoint name="j{i}"/>{geom}</body>'
        )
    trgba = " ".join(f"{c:.3f}" for c in table_rgba)
    frgb = " ".join(f"{c:.3f}" for c in floor_rgba[:3])
    return f"""<mujoco model="tabletop">
  <option timestep="{ts}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1024" offheight="1024"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="{frgb}" rgb2="0.3 0.3 0.3" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 2.0" dir="-0.2 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 {-th:.3f}" material="floor"/>
    <geom name="table" type="box" size="{tx} {ty} 0.02" pos="0 0 -0.02" rgba="{trgba}" friction="0.1 0.005 0.0002"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
