"""Prompt / target construction for Task A (current 3D boxes), Task B (boxes 1 s after a push),
Task N (boxes at the next stored frame) and Task I (inverse dynamics, research/PREREG.md).

Task I shows the pre-action frame and the *fixed* early post-action frame (t = 0.1 s) and
asks which object was acted on, in which camera-relative direction, and how strongly --
or the explicit no-motion class when the target moved less than IDP_AMBIGUOUS_M. The
input times are fixed, so the timestamps stated in the prompt carry no strength
information (an adaptive rule would leak the answer into the clock; PREREG.md pins it).

Answer styles:
- "center": [{"label": ..., "center": [x, y, z]}]  -- used for SFT and as the default zero-shot prompt.
- "qwen3d": Qwen3-VL's native 3D-grounding format
            [{"bbox_3d": [x, y, z, xs, ys, zs, roll, pitch, yaw], "label": ...}]; only the centre is scored.
- "box3d":  [{"label": ..., "center": [x, y, z], "extents": [ex, ey, ez]}] -- centre + full
            body-axis sizes in metres, camera frame; the phase-2 answer format.
- "chain3d": {"current": [...], "future": [...]} -- Stage-1 chained answers.
- "ctx3d":  {"current": [... with per-object "support" ...], "action_context": {...},
            "future": [...]} -- the PCB-style physical-context arm. The action_context
            payload is built by physwm/vlm/context.py and its format is frozen for the
            whole M0/M1 ladder; ``oracle_context=True`` injects the simulator-provided
            payload into forward-prediction prompts instead (never for task A, which is
            a perception task and the context names the action target).

Image modes (image-dependence test): normal | none (text only) | blank (uniform gray) | swap
(image of another episode, chosen by the caller).

Frame/view order: images are listed frame-major, view-minor (``frames`` outer, ``views``
inner); that is also the order of ``image_grid_thw`` rows and therefore of the merged
tokens the xyz PE is indexed by. Task N always targets the first stored frame after the
last input frame; multi-frame prompts state the input times explicitly. Task I always
uses its own fixed frame pair regardless of the arm's ``--frames``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from physwm.eval.metrics import (NEXT_FRAME_TASK, TASK_TIME, gt_centers_cam, gt_extents,
                                 task_time)
from physwm.schema import Episode
from physwm.vlm.context import action_context, oracle_payload, support_map

FRAME = (
    "Use the camera coordinate frame in metres: x points right, y points down, "
    "z points forward along the viewing direction."
)

IDP_TASK = "I"
IDP_TIME = 0.1                      # s; fixed early post-action input frame (research/PREREG.md)
IDP_AMBIGUOUS_M = 0.02              # m; target motion below this at t = IDP_TIME -> no-motion class
IDP_STRENGTH_BINS = ((0.75, "weak"), (1.2, "medium"), (None, "strong"))   # by action.speed, m/s
NO_MOTION_LABEL = "no clear motion"


def _intrinsics(ep: Episode) -> str:
    K = ep.camera.K
    return (
        f"The image is {ep.camera.width}x{ep.camera.height} pixels, camera intrinsics "
        f"fx={K[0][0]:.1f}, fy={K[1][1]:.1f}, cx={K[0][2]:.1f}, cy={K[1][2]:.1f}."
    )


def _input_times(ep: Episode, frames: tuple[int, ...], include_action: bool = True) -> str:
    times = ", ".join(f"{float(ep.frame_times[f]):.1f}" for f in frames)
    if include_action:
        return f"The images show t = {times} seconds after the action."
    return f"The images show the scene at t = {times} seconds."


def _ask(ep: Episode, task: str, what: str, frames: tuple[int, ...] = (0,),
         include_action: bool = True) -> str:
    """Question sentence for one task; ``frames`` are the input frame indices."""
    frames = tuple(frames) if frames else (0,)
    if task == "A":
        if len(frames) > 1:
            t0 = float(ep.frame_times[frames[0]])
            return (f"Report the {what} of each listed object as it is in the first image "
                    f"(t = {t0:.1f} s).")
        return f"Report the {what} of each listed object as it is in the image."
    if task == "B":
        if not include_action:
            return (
                f"Observe the motion across the frames and predict the {what} of each listed "
                f"object at t = {TASK_TIME['B']:.1f} seconds."
            )
        return (
            f"Action: {ep.action.text}\n"
            f"Predict the {what} of each listed object {TASK_TIME['B']:.1f} second after the action."
        )
    if task == NEXT_FRAME_TASK:
        tn = task_time(ep, task, frames)
        if not include_action:
            return (
                f"Observe the motion across the frames and predict the {what} of each listed "
                f"object at the next stored frame (t = {tn:.1f} seconds)."
            )
        return (
            f"Action: {ep.action.text}\n"
            f"Predict the {what} of each listed object at the next stored frame after the last "
            f"input image ({tn:.1f} second after the action)."
        )
    raise ValueError(f"unknown task {task!r}")


def idp_frame(ep: Episode) -> int:
    """Input frame index of the fixed early post-action image (t = 0.1 s on the v3 grid)."""
    return ep.frame_index_at(IDP_TIME)


def idp_target(ep: Episode) -> dict:
    """GT IDP answer: the target/direction/strength triple, or the explicit no-motion class.

    Strength bins are pinned to absolute initial-speed thresholds (m/s) so they stay
    meaningful on the OOD-force buckets (0.08-0.25 -> weak, 2.0-2.8 -> strong). Motion
    support is measured, not predicted: the target's own displacement between t = 0 and
    the fixed IDP frame decides the no-motion class.
    """
    moved = float(np.linalg.norm(
        np.asarray(ep.state_at(IDP_TIME)[ep.action.target].pos)
        - np.asarray(ep.state_at(0.0)[ep.action.target].pos)))
    if moved < IDP_AMBIGUOUS_M:
        return {"action": NO_MOTION_LABEL}
    for hi, name in IDP_STRENGTH_BINS:
        if hi is None or ep.action.speed < hi:
            return {"target": ep.objects[ep.action.target].name,
                    "direction": ep.action.direction_label, "strength": name}
    raise AssertionError("unreachable strength bin")


def _direction_line(ep: Episode, mode: str) -> str:
    rotation = np.asarray(ep.camera.cam2world, dtype=float)[:3, :3]
    if mode == "nominal":
        column, sign = {"left": (0, -1), "right": (0, 1),
                        "away": (2, 1), "toward": (2, -1)}[ep.action.direction_label]
        world = rotation[:, column].copy()
        world[2] = 0.
        world *= sign / np.linalg.norm(world)
    elif mode == "recorded":
        world = np.asarray(ep.action.direction_world, dtype=float)
    else:
        raise ValueError(f"unknown direction input {mode!r}")
    camera_vector = (rotation.T @ world).round(3)
    camera_vector[camera_vector == 0] = 0.
    coords = ", ".join(f"{component:.3f}" for component in camera_vector)
    return f"Push direction as a unit vector in the camera frame [x, y, z]: [{coords}].\n"


def prompt_text(ep: Episode, task: str, style: str = "center",
                frames: tuple[int, ...] = (0,), include_action: bool = True,
                oracle_context: bool = False, direction_input: str = "text") -> str:
    """Prompt for one (episode, task); with several input frames it also states their times.

    ``oracle_context`` injects the simulator-provided action context into forward-prediction
    prompts (tasks B/N, any style); task A never receives it (perception task; the context
    names the action target). The direction_input option adds a nominal or recorded
    camera-frame direction vector to forward prompts.
    """
    frames = tuple(frames) if frames else (0,)
    names = ", ".join(o.name for o in ep.objects)
    if task == IDP_TASK:
        ask = ("Which object was acted on between the first and the second image, in which "
               "camera-relative direction, and how strongly?")
        fmt = ('Answer only with JSON: {"target": "<object>", "direction": '
               '"left|right|away|toward", "strength": "weak|medium|strong"} -- or '
               '{"action": "no clear motion"} if no object clearly moved.')
        times = _input_times(ep, frames)
        return (
            f"A camera looks at objects on a table. {_intrinsics(ep)} {FRAME}\n"
            f"Objects: {names}.\n{times}{ask}\n{fmt}"
        )
    if style == "chain3d":
        t_cur = float(ep.frame_times[frames[-1]])
        t_fut = task_time(ep, task, frames)
        prefix = f"Action: {ep.action.text}\n" if include_action and task != "A" else ""
        ask = (
            f"{prefix}First report the 3D bounding box of each listed object at the last observed "
            f"image (t = {t_cur:.1f} s), then predict its 3D bounding box at t = {t_fut:.1f} s."
        )
        fmt = (
            'Answer only with a JSON object: '
            '{"current": [{"label": "<object>", "center": [x, y, z], "extents": [ex, ey, ez]}, ...], '
            '"future": [{"label": "<object>", "center": [x, y, z], "extents": [ex, ey, ez]}, ...]}. '
            "center is the object centre in metres and extents are the full body-axis sizes "
            "(ex, ey, ez) in metres."
        )
    elif style == "ctx3d":
        t_cur = float(ep.frame_times[frames[-1]])
        if task == "A":
            ask = ("Report the 3D bounding box of each listed object as it is in the image, "
                   "together with what each object rests on.")
            fmt = (
                'Answer only with a JSON object: {"current": [{"label": "<object>", '
                '"center": [x, y, z], "extents": [ex, ey, ez], "support": "table|<object name>"}, ...]}. '
                "center is the object centre in metres, extents are the full body-axis sizes "
                "(ex, ey, ez) in metres, and support names the object or the table it rests on."
            )
        else:
            t_fut = task_time(ep, task, frames)
            if include_action:
                prefix = f"Action: {ep.action.text}\n"
                ask = (
                    f"{prefix}First report the 3D bounding box of each listed object at the last "
                    f"observed image (t = {t_cur:.1f} s) together with what each object rests on, "
                    f"then describe the pre-action scene geometry -- the target's distance to the "
                    f"table edge along the action and any objects in the path -- then predict the "
                    f"3D bounding box of each listed object at t = {t_fut:.1f} s."
                )
            else:
                ask = (
                    f"Observe the motion across the frames. First report the 3D bounding box of "
                    f"each listed object at the last observed image (t = {t_cur:.1f} s) together "
                    f"with what each object rests on, then describe the pre-action scene geometry, "
                    f"then predict the 3D bounding box of each listed object at t = {t_fut:.1f} s."
                )
            fmt = (
                'Answer only with a JSON object: {"current": [{"label": "<object>", '
                '"center": [x, y, z], "extents": [ex, ey, ez], "support": "table|<object name>"}, ...], '
                '"action_context": {"target": "<object>", "direction": "left|right|away|toward", '
                '"edge_distance_m": <number>, "obstacles": [{"label": "<object>", '
                '"distance_m": <number>}, ...]}, "future": [{"label": "<object>", '
                '"center": [x, y, z], "extents": [ex, ey, ez]}, ...]}. center is the object '
                "centre in metres, extents are the full body-axis sizes (ex, ey, ez) in metres, "
                "support names what each object rests on, edge_distance_m is the distance from "
                "the target to the table edge along the action, and obstacles lists the objects "
                "in the action path with their surface-to-surface distance in metres."
            )
    elif style == "box3d":
        ask = _ask(ep, task, "3D bounding box", frames, include_action=include_action)
        fmt = (
            'Answer only with a JSON list, one entry per listed object: '
            '[{"label": "<object>", "center": [x, y, z], "extents": [ex, ey, ez]}, ...]. '
            "center is the object centre in metres and extents are the full body-axis sizes "
            "(ex, ey, ez) in metres."
        )
    else:
        ask = _ask(ep, task, "3D centre", frames, include_action=include_action)
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
    times = f"{_input_times(ep, frames, include_action=include_action)}\n" if len(frames) > 1 else ""
    # Oracle payload carries GT support alongside the action context, so a flat oracle
    # result cannot hide a support-prediction failure (predicted support is scored
    # separately by idpmetrics).
    ctx_line = ""
    if task != "A" and oracle_context:
        ctx_line = f"Physical context: {json.dumps(oracle_payload(ep, float(ep.frame_times[frames[-1]])))}\n"
    direction_line = ""
    if include_action and task in ("B", NEXT_FRAME_TASK) and direction_input != "text":
        direction_line = _direction_line(ep, direction_input)
    return (
        f"A camera looks at objects on a table. {_intrinsics(ep)} {FRAME}\n"
        f"Objects: {names}.\n{times}{ask}\n{direction_line}{ctx_line}{fmt}"
    )


def target_text(ep: Episode, task: str, style: str = "center",
                frames: tuple[int, ...] = (0,)) -> str:
    """Ground-truth answer string; "box3d" adds full body-axis extents to the centres,
    "chain3d" outputs {"current": [...at last input frame...], "future": [...at task_time...]},
    "ctx3d" additionally emits the frozen action_context payload and per-object support,
    and task I returns the GT inverse-dynamics answer.

    ``frames`` select the last input frame and task-N target time.
    """
    frames = tuple(frames) if frames else (0,)
    if task == IDP_TASK:
        return json.dumps(idp_target(ep))
    t = task_time(ep, task, frames)
    c = gt_centers_cam(ep, t)
    if style == "chain3d":
        t_cur = float(ep.frame_times[frames[-1]])
        c_cur = gt_centers_cam(ep, t_cur)
        e = gt_extents(ep)
        def _box_list(centers):
            return [
                {"label": o.name,
                 "center": [round(float(v), 3) for v in centers[i]],
                 "extents": [round(float(v), 3) for v in e[i]]}
                for i, o in enumerate(ep.objects)
            ]
        return json.dumps({"current": _box_list(c_cur), "future": _box_list(c)})
    if style == "ctx3d":
        t_cur = float(ep.frame_times[frames[-1]])
        cur = gt_centers_cam(ep, t_cur)
        e = gt_extents(ep)
        sup = support_map(ep, t_cur)

        def _cur(i, o):
            return {"label": o.name,
                    "center": [round(float(v), 3) for v in cur[i]],
                    "extents": [round(float(v), 3) for v in e[i]],
                    "support": sup[o.id]}

        if task == "A":
            return json.dumps({"current": [_cur(i, o) for i, o in enumerate(ep.objects)]})
        return json.dumps({
            "current": [_cur(i, o) for i, o in enumerate(ep.objects)],
            "action_context": action_context(ep),
            "future": [{"label": o.name,
                        "center": [round(float(v), 3) for v in c[i]],
                        "extents": [round(float(v), 3) for v in e[i]]}
                       for i, o in enumerate(ep.objects)],
        })
    if style == "box3d":
        e = gt_extents(ep)
        return json.dumps([
            {"label": o.name,
             "center": [round(float(v), 3) for v in c[i]],
             "extents": [round(float(v), 3) for v in e[i]]}
            for i, o in enumerate(ep.objects)
        ])
    if style != "center":
        raise ValueError(f"target style {style!r} is not stored as text; "
                         "use 'center', 'box3d', 'chain3d', or 'ctx3d'")
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
