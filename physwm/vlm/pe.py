"""OneCanvas-style metric-xyz position embeddings on the frozen Qwen3-VL visual path.

Per merged vision token (``t * h/2 * w/2`` per image, in ``image_grid_thw`` order) the
GT depth of the token's pixel block is unprojected to a camera-frame xyz. The xyz vector
is encoded with the OneCanvas 136-channel sinusoidal recipe, mapped by a 2-layer MLP to
the vision hidden size and added to the merger output through a zero-initialised scalar
gate. Reference: ``research/literature/2026_onecanvas.md`` ("3D position embedding").

Deepstack: v1 adds the PE to the main merger output (:class:`XYZEmbedding` only). The
deepstack features that Qwen3-VL injects into early LLM layers are left un-PE'd, so the
deepstack path carries no metric geometry in this baseline.

`merge_with_config_defaults`-era transformers (verified on 5.23) expose the vision tower as
``model.model.visual`` on :class:`Qwen3VLForConditionalGeneration`; PEFT wrappers are
unwrapped by :func:`merger_of`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

RECIPE = "onecanvas-136-v1"
PE_DIM = 136
MLP_HIDDEN = 64
AXIS_FREQS = tuple(float(v) for v in 10.0 ** np.linspace(-1.0, 2.0, 16))  # 0.1 .. 100 rad/m
RADIAL_FREQS = tuple(float(v) for v in 10.0 ** np.linspace(-1.0, 1.0, 8))  # 0.1 .. 10 rad/m
MAX_DEPTH = 50.0  # m; deeper / non-finite depth is treated as missing
PE_LR_SCALE = 50.0

_AXIS = torch.tensor(AXIS_FREQS, dtype=torch.float32)
_RADIAL = torch.tensor(RADIAL_FREQS, dtype=torch.float32)


def xyz_features(xyz: torch.Tensor) -> torch.Tensor:
    """Encode camera-frame points, (N, 3) metres -> (N, 136) float32.

    Channels: per axis (x, y, z) ``{sin, cos, raw}`` at 16 log-spaced frequencies in
    [0.1, 100] rad/m (99); ``||q||`` and ``r_xz = sqrt(x^2 + z^2)`` at 8 log-spaced
    frequencies in [0.1, 10] with ``{sin, cos, raw}`` (34); unit ray ``q / ||q||`` (3).
    """
    q = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3)
    angle = q[:, :, None] * _AXIS.to(q.device)[None, None, :]  # (N, 3, 16)
    per_axis = torch.cat([torch.sin(angle), torch.cos(angle), q[:, :, None]], dim=-1)  # (N, 3, 33)
    radii = torch.stack([q.norm(dim=-1), torch.sqrt(q[..., 0] ** 2 + q[..., 2] ** 2)], dim=-1)  # (N, 2)
    radial_angle = radii[:, :, None] * _RADIAL.to(q.device)[None, None, :]  # (N, 2, 8)
    radial = torch.cat([torch.sin(radial_angle), torch.cos(radial_angle), radii[:, :, None]], dim=-1)
    ray = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat([per_axis.flatten(1), radial.flatten(1), ray], dim=-1)


class XYZEmbedding(nn.Module):
    """136-ch xyz -> MLP -> vision-hidden delta, added through a learned scalar gate."""

    def __init__(self, hidden: int, pe_dim: int = PE_DIM, mlp_hidden: int = MLP_HIDDEN):
        super().__init__()
        self.hidden = int(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, self.hidden)
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(xyz_features(xyz).to(features.device))
        return features + (self.gate * delta).to(features.dtype)


def _block_edges(size: int, cells: int) -> np.ndarray:
    """Monotone pixel boundaries splitting ``size`` pixels into ``cells`` blocks."""
    return np.rint(np.linspace(0, size, cells + 1)).astype(int)


def token_xyz(depth: np.ndarray, K: np.ndarray, grid_thw_row, patch: int = 16,
              merge: int = 2, max_depth: float = MAX_DEPTH) -> torch.Tensor:
    """Camera-frame xyz of every merged token of ONE image, (N, 3) float32.

    ``grid_thw_row`` is the processor's ``(t, h, w)`` in patch units; the image has
    ``t * (h/merge) * (w/merge)`` merged tokens in row-major ``(t, block row, block col)``
    order. ``depth`` is the rendered depth map at the episode's camera resolution and
    ``K`` the matching intrinsics, so each token's block is ``depth.shape / (h/merge,
    w/merge)`` pixels, its centre (u, v) unprojects to ``((u-cx)/fx, (v-cy)/fy, 1) * m``
    with ``m`` the median valid depth in the block (finite, ``0 < d < max_depth``), then the
    image-wide median, then 0. ``patch`` is kept for the OneCanvas API but block sizes
    follow the depth map's own resolution (the one ``K`` refers to), not the resized
    ``(h*patch, w*patch)`` grid.
    """
    d = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=float)
    t, h, w = (int(v) for v in grid_thw_row[:3])
    rows, cols = max(h // merge, 1), max(w // merge, 1)
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    image_median = float(np.median(d[valid])) if valid.any() else 0.0
    row_edges = _block_edges(d.shape[0], rows)
    col_edges = _block_edges(d.shape[1], cols)
    out = np.zeros((t, rows, cols, 3), dtype=np.float32)
    for r in range(rows):
        v0, v1 = row_edges[r], max(row_edges[r] + 1, row_edges[r + 1])
        for c in range(cols):
            u0, u1 = col_edges[c], max(col_edges[c] + 1, col_edges[c + 1])
            block_valid = valid[v0:v1, u0:u1]
            m = float(np.median(d[v0:v1, u0:u1][block_valid])) if block_valid.any() else image_median
            u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            out[:, r, c] = ((u - cx) / fx * m, (v - cy) / fy * m, m)
    return torch.from_numpy(out.reshape(-1, 3))


def frame_token_xyz(ep_dir: Path, ep, frame: int = 0, view: int = 0, grid_thw_row=(1, 1, 1),
                    merge: int = 2) -> torch.Tensor:
    """Token xyz of one rendered frame/view; zeros when that depth file does not exist.

    Legacy episodes only store depth at t = 0, so later frames fall back to zeros (a
    constant LE input, no geometry) instead of failing the batch.
    """
    t, h, w = (int(v) for v in grid_thw_row[:3])
    n_tokens = t * max(h // merge, 1) * max(w // merge, 1)
    try:
        with np.load(Path(ep_dir) / ep.frame_file("depth", view, frame)) as z:
            depth = z["depth"]
    except (KeyError, FileNotFoundError):
        return torch.zeros(n_tokens, 3)
    return token_xyz(depth, ep.views[view].K, (t, h, w), merge=merge)


def image_grid_rows(processor, images, cache: dict | None = None) -> list[list[int]]:
    """Per-image ``(t, h, w)`` patch grid the processor will produce, cached by image size."""
    cache = {} if cache is None else cache
    rows = []
    for im in images:
        key = (int(im.height), int(im.width))
        if key not in cache:
            with torch.no_grad():
                out = processor.image_processor(images=[im], return_tensors="pt")
            cache[key] = [int(v) for v in out["image_grid_thw"][0].tolist()]
        rows.append(cache[key])
    return rows


def merger_of(model) -> nn.Module:
    """The Qwen3-VL main patch merger (unwrapping a PEFT model if needed)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = getattr(getattr(base, "model", base), "visual")
    return visual.merger


def install_pe(model, hidden: int | None = None, module: XYZEmbedding | None = None):
    """Attach the xyz PE to ``model``'s merger; returns ``(XYZEmbedding, hook handle)``.

    The forward hook consumes ``model._physwm_pe_xyz`` (N, 3) and must see exactly the
    merger's output rows, i.e. the concatenated merged-token count over all images of the
    call in ``image_grid_thw`` order. ``None`` leaves the features untouched (B5), so
    inference without the PE works on the same code path.
    """
    merger = merger_of(model)
    pe = module if module is not None else XYZEmbedding(
        hidden if hidden is not None else merger.linear_fc2.out_features
    )

    def hook(_module, _inputs, output):
        xyz = getattr(model, "_physwm_pe_xyz", None)
        if xyz is None:
            return output
        setattr(model, "_physwm_pe_xyz", None)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != output.shape[0]:
            raise ValueError(
                f"xyz/merger misalignment: xyz {tuple(xyz.shape)} vs merged visual tokens "
                f"{tuple(output.shape)}"
            )
        return pe(xyz.to(output.device), output)

    return pe, merger.register_forward_hook(hook)


def pe_param_groups(module: nn.Module, base_lr: float) -> list[dict]:
    """Optimizer group for the PE at OneCanvas's 50x learning rate, no weight decay."""
    return [{"params": list(module.parameters()), "lr": PE_LR_SCALE * base_lr, "weight_decay": 0.0}]


def save_pe(path: str | Path, module: XYZEmbedding) -> None:
    torch.save({"state_dict": module.state_dict(), "hidden": module.hidden, "recipe": RECIPE},
               Path(path))


def load_pe(path: str | Path, hidden: int | None = None) -> XYZEmbedding:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("recipe") != RECIPE:
        raise ValueError(f"{path}: unsupported PE recipe {ckpt.get('recipe')!r}")
    module = XYZEmbedding(hidden if hidden is not None else int(ckpt["hidden"]))
    module.load_state_dict(ckpt["state_dict"])
    return module