"""Unified episode schema shared by the simulator, baselines, VLM pipeline and evaluation.

Conventions
-----------
- World frame: z up, table top at z = 0, metres.
- Camera frame: OpenCV convention (x right, y down, z forward), metres.
- Quaternions: (w, x, y, z).
- Object states are stored in world coordinates; camera-frame values are derived via
  :func:`physwm.geometry.world_to_cam` with ``Camera.cam2world``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Camera:
    width: int
    height: int
    K: list[list[float]]  # 3x3 intrinsics
    cam2world: list[list[float]]  # 4x4, OpenCV camera frame -> world
    fovy_deg: float
    azimuth_deg: float
    elevation_deg: float
    distance: float


@dataclass
class ObjectSpec:
    id: int
    name: str  # unique referring expression used in prompts, e.g. "red box"
    shape: str  # box | cylinder | sphere | capsule | cone | hexprism
    color: str
    rgba: list[float]
    half_extents: list[float]  # axis-aligned half extents in the body frame
    mass: float
    density: float
    friction: float


@dataclass
class ObjectState:
    pos: list[float]  # world xyz of the body origin (= geometric centre)
    quat: list[float]  # world orientation, wxyz
    toppled: bool  # body up-axis tilted > 45 deg (always False for spheres)
    on_table: bool


@dataclass
class Action:
    type: str  # "push" (tilt / drop reserved for later phases)
    target: int  # object id
    direction_label: str  # camera-relative: left | right | away | toward
    direction_world: list[float]  # unit vector in the table plane
    impulse: float  # N*s, exact value used in simulation
    speed: float  # resulting initial speed of the target, m/s (impulse / mass)
    text: str  # natural-language action given to the model


@dataclass
class Episode:
    episode_id: str
    split: str
    config: str
    seed: int
    table_half_size: list[float]
    objects: list[ObjectSpec]
    camera: Camera
    action: Action
    times: list[float]  # seconds after the action at which states are stored
    states: list[list[ObjectState]]  # states[i][j]: time times[i], object j
    files: dict[str, str] = field(default_factory=dict)  # relative to the episode dir

    def state_at(self, t: float) -> list[ObjectState]:
        for ti, s in zip(self.times, self.states):
            if abs(ti - t) < 1e-6:
                return s
        raise KeyError(f"no state stored at t={t}; available: {self.times}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        d = dict(d)
        d["objects"] = [ObjectSpec(**o) for o in d["objects"]]
        d["camera"] = Camera(**d["camera"])
        d["action"] = Action(**d["action"])
        d["states"] = [[ObjectState(**s) for s in row] for row in d["states"]]
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Episode:
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_split(root: str | Path, split: str) -> list[tuple[Path, Episode]]:
    """Load the indexed cohort, rejecting incomplete or inconsistent splits.

    Legacy splits without an index remain readable; unrelated directories are never
    included when an index is present.
    """
    split_dir = Path(root) / split
    generation = split_dir / "generation.json"
    if generation.exists() and json.loads(generation.read_text()).get("errors"):
        raise ValueError(f"split has generation failures; repair before use: {split_dir}")
    index = split_dir / "index.jsonl"
    if index.exists():
        ids = [json.loads(line)["episode_id"] for line in index.read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)) or any(Path(i).name != i or i in (".", "..") for i in ids):
            raise ValueError(f"invalid or duplicate episode IDs in {index}")
        metas = [split_dir / i / "meta.json" for i in sorted(ids)]
    else:
        metas = sorted(split_dir.glob("*/meta.json"))
    if not metas:
        raise ValueError(f"empty or missing split: {split_dir}")
    out = []
    for meta in metas:
        ep = Episode.load(meta)
        if ep.episode_id != meta.parent.name or ep.split != split:
            raise ValueError(f"episode identity does not match its split/path: {meta}")
        out.append((meta.parent, ep))
    return out
