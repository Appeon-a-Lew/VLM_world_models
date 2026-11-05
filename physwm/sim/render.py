"""Offscreen rendering of the episode camera (RGB, metric depth, per-object segmentation).

Set ``MUJOCO_GL=egl`` before importing mujoco for headless GPU rendering.
"""

from __future__ import annotations

import mujoco
import numpy as np


class CameraRenderer:
    def __init__(self, m: mujoco.MjModel, width: int, height: int, camera: str = "cam"):
        self.m, self.camera = m, camera
        self.r = mujoco.Renderer(m, height=height, width=width)
        # geom id -> object index + 1 (0 = background / table / floor)
        self.geom_to_obj = np.zeros(m.ngeom + 1, dtype=np.uint8)
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("g") and name[1:].isdigit():
                self.geom_to_obj[g] = int(name[1:]) + 1

    def rgb(self, d: mujoco.MjData) -> np.ndarray:
        self.r.update_scene(d, camera=self.camera)
        return self.r.render().copy()

    def depth(self, d: mujoco.MjData) -> np.ndarray:
        self.r.enable_depth_rendering()
        self.r.update_scene(d, camera=self.camera)
        out = self.r.render().copy()
        self.r.disable_depth_rendering()
        return out

    def seg(self, d: mujoco.MjData) -> np.ndarray:
        self.r.enable_segmentation_rendering()
        self.r.update_scene(d, camera=self.camera)
        raw = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        ids, types = raw[..., 0], raw[..., 1]
        is_geom = (types == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0)
        out = np.zeros(ids.shape, dtype=np.uint8)
        out[is_geom] = self.geom_to_obj[ids[is_geom]]
        return out

    def close(self):
        self.r.close()
