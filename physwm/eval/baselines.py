"""Trivial baselines for Task B (t = 1 s), written in the shared predictions.jsonl format.

- static:          every object stays where it is at t = 0.
- const_velocity:  extrapolate the motion observed between t = 0 and t = 0.1 s (needs a second
                   frame after the push, i.e. privileged motion information the single-image
                   models do not get -- reported as "where applicable").

    python -m physwm.eval.baselines --data $WM/data --splits id_test ood_objects --out $WM/runs/baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physwm import geometry as G
from physwm.eval.metrics import gt_centers_cam
from physwm.schema import Episode, load_split

CV_DT = 0.1


def static(ep: Episode, ep_dir: Path) -> np.ndarray:
    return gt_centers_cam(ep, 0.0)


def const_velocity(ep: Episode, ep_dir: Path) -> np.ndarray:
    tr = np.load(ep_dir / ep.files["traj"])
    k = int(np.argmin(np.abs(tr["t"] - CV_DT)))
    p0, p1 = tr["pos"][0], tr["pos"][k]
    pred = p0 + (p1 - p0) / tr["t"][k] * 1.0
    return G.world_to_cam(pred, ep.camera.cam2world)


BASELINES = {"static": static, "const_velocity": const_velocity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {name: open(out / f"{name}.jsonl", "w") for name in BASELINES}
    for split in args.splits:
        for ep_dir, ep in load_split(args.data, split):
            for name, fn in BASELINES.items():
                c = fn(ep, ep_dir)
                rec = {"episode_id": ep.episode_id, "task": "B", "parsed": True,
                       "objects": {str(i): c[i].tolist() for i in range(len(ep.objects))}}
                files[name].write(json.dumps(rec) + "\n")
    for f in files.values():
        f.close()
    print("wrote", ", ".join(str(out / f"{n}.jsonl") for n in BASELINES))


if __name__ == "__main__":
    main()
