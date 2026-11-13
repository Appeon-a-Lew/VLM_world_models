"""Render annotated sample sheets: RGB t0/t1 with projected 3D boxes, trajectory, seg, depth.

Usage:
    python scripts/viz_samples.py --root <data-or-sample-root> [--out DIR] [episode_id ...]

Without episode ids, every episode under <root>/<split>/<id>/ with a meta.json
is rendered. Panels per episode: rgb_t0 with 3D boxes (target marked *),
rgb_t1 with 3D boxes, target trajectory 0-2 s, segmentation, metric depth.
Uses the repo geometry conventions (world z-up, OpenCV camera frame, wxyz).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physwm.geometry import quat_to_mat, world_to_cam, project  # noqa: E402

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
SEG_COLORS = [(230, 80, 80), (80, 160, 240), (110, 200, 110),
              (240, 200, 90), (200, 120, 230), (120, 220, 210)]


def box_corners(half_extents, pos, quat):
    hx, hy, hz = half_extents
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ quat_to_mat(quat).T + np.asarray(pos, float)


def _proj(pts_world, cam2world, K):
    pc = world_to_cam(np.asarray(pts_world, float).reshape(-1, 3), cam2world)
    ok = pc[:, 2] > 0.01
    uv = np.full((pc.shape[0], 2), np.nan)
    if ok.any():
        uv[ok] = project(pc[ok], K)
    return uv


def draw_box3d(dr, corners_w, cam2world, K, color, width=2):
    uv = _proj(corners_w, cam2world, K)
    if np.any(np.isnan(uv)):
        return
    for i, j in EDGES:
        dr.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=width)


def draw_center(dr, pos_w, cam2world, K, color, r=5):
    uv = _proj(pos_w, cam2world, K)[0]
    if np.isnan(uv[0]):
        return
    u, v = uv
    dr.line([u - r, v, u + r, v], fill=color, width=2)
    dr.line([u, v - r, u, v + r], fill=color, width=2)


def panel_for_episode(ep_dir: Path, out_dir: Path) -> Path:
    ep = json.loads((ep_dir / "meta.json").read_text())
    cam = ep["camera"]
    K, cam2world = np.array(cam["K"]), np.array(cam["cam2world"])
    rgb0 = Image.open(ep_dir / "rgb_t0.png").convert("RGB")
    rgb1 = Image.open(ep_dir / "rgb_t1.png").convert("RGB")
    seg_a = np.array(Image.open(ep_dir / "seg_t0.png"))
    depth = np.load(ep_dir / "depth_t0.npz")["depth"].astype(np.float32)

    panels = []
    for img, ti, tag in ((rgb0, 0, f"t=0  ({ep['episode_id']})"), (rgb1, 1, "t=1 s")):
        canvas = img.copy()
        dr = ImageDraw.Draw(canvas)
        for obj, st in zip(ep["objects"], ep["states"][ti]):
            col = SEG_COLORS[obj["id"] % len(SEG_COLORS)]
            is_target = obj["id"] == ep["action"]["target"]
            draw_box3d(dr, box_corners(obj["half_extents"], st["pos"], st["quat"]),
                       cam2world, K, col, width=3 if is_target else 2)
            draw_center(dr, st["pos"], cam2world, K, (255, 255, 255) if is_target else col)
            uv = _proj(st["pos"], cam2world, K)[0]
            if not np.isnan(uv[0]):
                dr.text((uv[0] + 7, uv[1] - 14),
                        f"{obj['name']}{' *' if is_target else ''}", fill=col)
        dr.text((6, 6), tag, fill=(255, 255, 255))
        panels.append(canvas)

    traj = np.load(ep_dir / "traj.npz")
    tpos = traj["pos"][:, ep["action"]["target"], :]
    canvas = rgb0.copy()
    dr = ImageDraw.Draw(canvas)
    uv = _proj(tpos, cam2world, K)
    good = uv[~np.isnan(uv[:, 0])]
    for a, b in zip(good[:-1], good[1:]):
        dr.line([tuple(a), tuple(b)], fill=(255, 60, 60), width=2)
    dr.text((6, 6), "target trajectory 0->2 s (50 Hz)", fill=(255, 255, 255))
    panels.append(canvas)

    arr = np.zeros((*seg_a.shape, 3), np.uint8)
    for oid in np.unique(seg_a):
        arr[seg_a == oid] = (30, 30, 30) if oid == 0 else SEG_COLORS[int(oid) % len(SEG_COLORS)]
    panels.append(Image.fromarray(arr))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    d = np.clip(depth, 0, 2.5) / 2.5
    d[depth > 80] = 1.0
    panels.append(Image.fromarray((cm.viridis(d)[..., :3] * 255).astype(np.uint8)))

    W, H = rgb0.size
    header = 74
    sheet = Image.new("RGB", (W * len(panels) + 8 * (len(panels) - 1) + 8, H + header + 8),
                      (18, 18, 22))
    dr = ImageDraw.Draw(sheet)
    a = ep["action"]
    lines = [
        f"{ep['episode_id']}  |  split={ep['split']}  config={ep['config']}  seed={ep['seed']}"
        f"  |  cam: elev {cam['elevation_deg']:.1f} deg, dist {cam['distance']:.2f} m",
        "objects: " + ", ".join(f"[{o['id']}] {o['name']} ({o['shape']}, m={o['mass']:.2f} kg)"
                                for o in ep["objects"]),
        f"action: {a['text']}",
    ]
    for i, ln in enumerate(lines):
        dr.text((8, 6 + i * 22), ln, fill=(235, 235, 235))
    for i, p in enumerate(panels):
        sheet.paste(p, (8 + i * (W + 8), header + 4))
    out = out_dir / f"{ep['episode_id']}.png"
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dir containing <split>/<episode_id>/meta.json")
    ap.add_argument("--out", default=None, help="output dir (default: <root>/../viz or ./viz)")
    ap.add_argument("episodes", nargs="*", help="episode ids; default: all under --root")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root.parent / "viz" if root.name != "viz" else root
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_dirs = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())
    if args.episodes:
        wanted = set(args.episodes)
        ep_dirs = [p for p in ep_dirs if p.name in wanted]
    for d in ep_dirs:
        print(panel_for_episode(d, out_dir))


if __name__ == "__main__":
    main()
