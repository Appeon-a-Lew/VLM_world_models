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
    """Load all episodes of a split as (episode_dir, Episode), sorted by id."""
    split_dir = Path(root) / split
    out = []
    for meta in sorted(split_dir.glob("*/meta.json")):
        out.append((meta.parent, Episode.load(meta)))
    return out
