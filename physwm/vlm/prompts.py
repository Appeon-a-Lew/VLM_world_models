"""Prompt / target construction for Task A (current 3D boxes), Task B (boxes 1 s after a push)
and Task N (boxes at the next stored frame).

Answer styles:
- "center": [{"label": ..., "center": [x, y, z]}]  -- used for SFT and as the default zero-shot prompt.
- "qwen3d": Qwen3-VL's native 3D-grounding format
            [{"bbox_3d": [x, y, z, xs, ys, zs, roll, pitch, yaw], "label": ...}]; only the centre is scored.
- "box3d":  [{"label": ..., "center": [x, y, z], "extents": [ex, ey, ez]}] -- centre + full
            body-axis sizes in metres, camera frame; the phase-2 answer format.

Image modes (image-dependence test): normal | none (text only) | blank (uniform gray) | swap
(image of another episode, chosen by the caller).

Frame/view order: images are listed frame-major, view-minor (``frames`` outer, ``views``
inner); that is also the order of ``image_grid_thw`` rows and therefore of the merged
tokens the xyz PE is indexed by.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from physwm.eval.metrics import (NEXT_FRAME_TASK, TASK_TIME, gt_centers_cam, gt_extents,
                                 task_time)
from physwm.schema import Episode

FRAME = (
    "Use the camera coordinate frame in metres: x points right, y points down, "
    "z points forward along the viewing direction."
)


def _intrinsics(ep: Episode) -> str:
    K = ep.camera.K
    return (
        f"The image is {ep.camera.width}x{ep.camera.height} pixels, camera intrinsics "
        f"fx={K[0][0]:.1f}, fy={K[1][1]:.1f}, cx={K[0][2]:.1f}, cy={K[1][2]:.1f}."
    )


def _ask(ep: Episode, task: str, what: str) -> str:
    if task == "A":
        return f"Report the {what} of each listed object as it is in the image."
    if task == "B":
        return (
            f"Action: {ep.action.text}\n"
            f"Predict the {what} of each listed object {TASK_TIME['B']:.1f} second after the action."
        )
    if task == NEXT_FRAME_TASK:
        return (
            f"Action: {ep.action.text}\n"
            f"Predict the {what} of each listed object at the next stored frame "
            f"({task_time(ep, task):.1f} second after the action)."
        )
    raise ValueError(f"unknown task {task!r}")


def prompt_text(ep: Episode, task: str, style: str = "center") -> str:
    names = ", ".join(o.name for o in ep.objects)
    if style == "box3d":
        ask = _ask(ep, task, "3D bounding box")
        fmt = (
            'Answer only with a JSON list, one entry per listed object: '
            '[{"label": "<object>", "center": [x, y, z], "extents": [ex, ey, ez]}, ...]. '
            "center is the object centre in metres and extents are the full body-axis sizes "
            "(ex, ey, ez) in metres."
        )
    else:
        ask = _ask(ep, task, "3D centre")
        if style == "center":
            fmt = 'Answer only with JSON: [{"label": "<object>", "center": [x, y, z]}, ...]'
        elif style == "qwen3d":
            fmt = (
                "For each object provide its 3D bounding box. The output format required is JSON: "
                '[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],'
                '"label":"<object>"}]'
            )
        else:
            raise ValueError(style)
    return (
        f"A camera looks at objects on a table. {_intrinsics(ep)} {FRAME}\n"
        f"Objects: {names}.\n{ask}\n{fmt}"
    )


def target_text(ep: Episode, task: str, style: str = "center") -> str:
    """Ground-truth answer string; "box3d" adds full body-axis extents to the centres."""
    t = task_time(ep, task)
    c = gt_centers_cam(ep, t)
    if style == "box3d":
        e = gt_extents(ep)
        return json.dumps([
            {"label": o.name,
             "center": [round(float(v), 3) for v in c[i]],
             "extents": [round(float(v), 3) for v in e[i]]}
            for i, o in enumerate(ep.objects)
        ])
    if style != "center":
        raise ValueError(f"target style {style!r} is not stored as text; use 'center' or 'box3d'")
    return json.dumps([
        {"label": o.name, "center": [round(float(v), 3) for v in c[o.id]]} for o in ep.objects
    ])


def load_image(ep_dir: Path, ep: Episode, mode: str = "normal",
               swap_from: tuple[Path, Episode] | None = None, frame: int = 0, view: int = 0):
    """Return the RGB image of one frame/view, or None (mode 'none')."""
    if mode == "none":
        return None
    if mode == "blank":
        return Image.new("RGB", (ep.camera.width, ep.camera.height), (128, 128, 128))
    if mode == "swap":
        assert swap_from is not None, "swap mode needs swap_from"
        swap_dir, swap_ep = swap_from
        return Image.open(swap_dir / swap_ep.frame_file("rgb", view, frame)).convert("RGB")
    return Image.open(ep_dir / ep.frame_file("rgb", view, frame)).convert("RGB")


def input_images(ep_dir: Path, ep: Episode, frames=(0,), views=(0,), mode: str = "normal",
                 swap_from: tuple[Path, Episode] | None = None) -> list[Image.Image]:
    """PIL images in message order (frames outer, views inner); [] for mode 'none'."""
    if mode == "none":
        return []
    return [load_image(ep_dir, ep, mode, swap_from, frame=f, view=v) for f in frames for v in views]


def build_messages(text: str, image, answer: str | None = None) -> list[dict]:
    """One user turn with every image (a single PIL image or a list, in token order)."""
    images = ([] if image is None else list(image) if isinstance(image, (list, tuple)) else [image])
    content = [{"type": "image", "image": im} for im in images] + [
        {"type": "text", "text": text}
    ]
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def swap_partners(items: list[tuple[Path, Episode]], seed: int = 0) -> dict[str, tuple[Path, Episode]]:
    """Deterministic image-swap pairing: each episode gets the image of a different episode with the
    same number of objects (falls back to any other episode if its group has a single member).

    Values are the partner's (directory, episode) pair so callers can resolve its frame files.
    """
    rng = np.random.default_rng(seed)
    groups: dict[int, list] = {}
    for d, ep in items:
        groups.setdefault(len(ep.objects), []).append((d, ep))
    out = {}
    for members in groups.values():
        order = rng.permutation(len(members))
        for k, idx in enumerate(order):
            d, ep = members[idx]
            if len(members) > 1:
                out[ep.episode_id] = members[order[(k + 1) % len(members)]]
            else:
                out[ep.episode_id] = next(x for x in items if x[0] != d)
    return out