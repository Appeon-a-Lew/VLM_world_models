"""Prompt / target construction for Task A (current 3D centres) and Task B (centres 1 s after a push).

Two answer styles:
- "center": [{"label": ..., "center": [x, y, z]}]  -- used for SFT and as the default zero-shot prompt.
- "qwen3d": Qwen3-VL's native 3D-grounding format
            [{"bbox_3d": [x, y, z, xs, ys, zs, roll, pitch, yaw], "label": ...}]; only the centre is scored.

Image modes (image-dependence test): normal | none (text only) | blank (uniform gray) | swap
(image of another episode, chosen by the caller).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from physwm.eval.metrics import TASK_TIME, gt_centers_cam
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


def prompt_text(ep: Episode, task: str, style: str = "center") -> str:
    names = ", ".join(o.name for o in ep.objects)
    if task == "A":
        ask = "Report the 3D centre of each listed object as it is in the image."
    else:
        ask = (
            f"Action: {ep.action.text}\n"
            f"Predict the 3D centre of each listed object {TASK_TIME['B']:.1f} second after the action."
        )
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


def target_text(ep: Episode, task: str) -> str:
    c = gt_centers_cam(ep, TASK_TIME[task])
    return json.dumps([
        {"label": o.name, "center": [round(float(v), 3) for v in c[o.id]]} for o in ep.objects
    ])


def load_image(ep_dir: Path, ep: Episode, mode: str = "normal", swap_from: Path | None = None):
    """Return a PIL image or None (mode 'none')."""
    if mode == "none":
        return None
    if mode == "blank":
        return Image.new("RGB", (ep.camera.width, ep.camera.height), (128, 128, 128))
    if mode == "swap":
        assert swap_from is not None, "swap mode needs swap_from"
        return Image.open(swap_from / "rgb_t0.png").convert("RGB")
    return Image.open(ep_dir / ep.files["rgb_t0"]).convert("RGB")


def build_messages(text: str, image, answer: str | None = None) -> list[dict]:
    content = ([{"type": "image", "image": image}] if image is not None else []) + [
        {"type": "text", "text": text}
    ]
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def swap_partners(items: list[tuple[Path, Episode]], seed: int = 0) -> dict[str, Path]:
    """Deterministic image-swap pairing: each episode gets the image of a different episode with the
    same number of objects (falls back to any other episode if its group has a single member)."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list] = {}
    for d, ep in items:
        groups.setdefault(len(ep.objects), []).append((d, ep))
    out = {}
    all_dirs = [d for d, _ in items]
    for members in groups.values():
        order = rng.permutation(len(members))
        for k, idx in enumerate(order):
            d, ep = members[idx]
            if len(members) > 1:
                out[ep.episode_id] = members[order[(k + 1) % len(members)]][0]
            else:
                out[ep.episode_id] = next(x for x in all_dirs if x != d)
    return out
