"""Generate a split of push episodes.

    MUJOCO_GL=egl python -m physwm.sim.generate --config configs/sim_train.yaml \
        --split train --seed-start 0 --n 10000 --out /var/tmp/toyi/wm/data --workers 20

Each episode is fully determined by (config, seed). Output layout:
    <out>/<split>/<episode_id>/{meta.json, rgb_t0.png, rgb_t1.png, depth_t0.npz, seg_t0.png, traj.npz}
    <out>/<split>/index.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from physwm.sim.scene import load_config


def _work(args):
    cfg, seed, split, out = args
    from physwm.sim.push import simulate

    ep_dir = Path(out) / split / f"{split}_{seed:07d}"
    if (ep_dir / "meta.json").exists():
        return json.loads((ep_dir / "meta.json").read_text())["episode_id"], None
    try:
        data = simulate(cfg, seed, split)
    except Exception as e:  # noqa: BLE001 - log and continue, never kill the pool
        return None, f"seed {seed}: {e!r}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data.rgb_t0).save(ep_dir / "rgb_t0.png")
    Image.fromarray(data.rgb_t1).save(ep_dir / "rgb_t1.png")
    Image.fromarray(data.seg_t0).save(ep_dir / "seg_t0.png")
    np.savez_compressed(ep_dir / "depth_t0.npz", depth=data.depth_t0.astype(np.float16))
    np.savez_compressed(ep_dir / "traj.npz", t=data.traj_t, pos=data.traj_pos, quat=data.traj_quat)
    data.episode.save(ep_dir / "meta.json")  # written last: marks the episode complete
    return data.episode.episode_id, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = range(args.seed_start, args.seed_start + args.n)
    jobs = [(cfg, s, args.split, args.out) for s in seeds]
    ok, errors = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for ep_id, err in tqdm(pool.imap_unordered(_work, jobs, chunksize=4), total=len(jobs)):
            (ok.append(ep_id) if ep_id else errors.append(err))

    split_dir = Path(args.out) / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "index.jsonl", "w") as f:
        for ep_id in sorted(ok):
            f.write(json.dumps({"episode_id": ep_id}) + "\n")
    (split_dir / "generation.json").write_text(json.dumps(
        {"config": cfg, "seed_start": args.seed_start, "n": args.n, "ok": len(ok), "errors": errors},
        indent=1,
    ))
    print(f"{args.split}: {len(ok)} ok, {len(errors)} failed")
    for e in errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
