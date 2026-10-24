"""Versioned manifests for dataset splits: generation config, seed range, and a content hash.

    python -m physwm.agent.manifest --data $WM/data --splits id_test ood_objects ... --out manifests/final_test.json
    python -m physwm.agent.manifest --data $WM/data --verify manifests/final_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def split_hash(split_dir: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for ep_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(ep_dir.iterdir()):
            h.update(f"{ep_dir.name}/{f.name}".encode())
            h.update(f.read_bytes())
        n += 1
    return h.hexdigest(), n


def describe(data: Path, split: str) -> dict:
    gen = json.loads((data / split / "generation.json").read_text())
    digest, n = split_hash(data / split)
    return {
        "split": split,
        "config_name": gen["config"]["name"],
        "config": gen["config"],
        "seed_start": gen["seed_start"],
        "n": n,
        "sha256": digest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", default=None)
    args = ap.parse_args()
    data = Path(args.data)
    if args.verify:
        man = json.loads(Path(args.verify).read_text())
        bad = [s["split"] for s in man["splits"] if split_hash(data / s["split"])[0] != s["sha256"]]
        print("manifest OK" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    man = {"version": 1, "splits": [describe(data, s) for s in args.splits]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(man, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
