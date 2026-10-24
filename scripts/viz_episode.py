"""Overlay ground-truth (and optionally predicted) object centres on the episode images.

    python scripts/viz_episode.py --data /var/tmp/toyi/wm/data --split train --n 20 --out viz/
    python scripts/viz_episode.py ... --pred predictions.jsonl      # adds predicted t=1 s centres

Per episode it writes one side-by-side PNG: left = t0 image with t0 centres (circles),
right = t1 image with GT t1 centres (circles) and predictions (crosses).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from physwm import geometry as G
from physwm.schema import load_split


def _draw(img: Image.Image, uv_list, colors, marker: str):
    dr = ImageDraw.Draw(img)
    for (u, v), c in zip(uv_list, colors):
        c = tuple(int(255 * x) for x in c[:3])
        if marker == "o":
            dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 255, 255), width=3)
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=c, width=2)
        else:
            dr.line([u - 7, v - 7, u + 7, v + 7], fill=(255, 255, 255), width=3)
            dr.line([u - 7, v + 7, u + 7, v - 7], fill=(255, 255, 255), width=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred", default=None)
    args = ap.parse_args()

    preds = {}
    if args.pred:
        for line in open(args.pred):
            r = json.loads(line)
            if r.get("task") == "B":
                preds[r["episode_id"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep_dir, ep in load_split(args.data, args.split)[: args.n]:
        K, T = ep.camera.K, ep.camera.cam2world
        colors = [o.rgba for o in ep.objects]
        panels = []
        for key, t in (("rgb_t0", 0.0), ("rgb_t1", 1.0)):
            img = Image.open(ep_dir / ep.files[key]).convert("RGB")
            pos = np.array([s.pos for s in ep.state_at(t)])
            _draw(img, G.project(G.world_to_cam(pos, T), K), colors, "o")
            if t == 1.0 and ep.episode_id in preds:
                p = preds[ep.episode_id]["objects"]
                pts = [p[str(i)] for i in range(len(ep.objects)) if str(i) in p and p[str(i)] is not None]
                cols = [colors[i] for i in range(len(ep.objects)) if str(i) in p and p[str(i)] is not None]
                if pts:
                    _draw(img, G.project(np.array(pts), K), cols, "x")
            panels.append(img)
        w, h = panels[0].size
        canvas = Image.new("RGB", (2 * w, h + 40), (255, 255, 255))
        canvas.paste(panels[0], (0, 40))
        canvas.paste(panels[1], (w, 40))
        ImageDraw.Draw(canvas).text((6, 6), f"{ep.episode_id}: {ep.action.text}", fill=(0, 0, 0))
        canvas.save(out / f"{ep.episode_id}.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
