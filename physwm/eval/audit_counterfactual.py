"""Audit completed matched families; partial generation can be inspected with --partial."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from physwm.schema import Episode

VARIANTS = ("original", "reverse", "half", "double", "recolor", "elevation")


def _hash(b):
    return hashlib.sha256(b).hexdigest()


def audit(root: Path, split: str, partial: bool = False, render_check: bool = False):
    folder = root / split
    errors, groups = [], defaultdict(dict)
    for path in sorted(folder.glob("*/meta.json")):
        try:
            episode = Episode.load(path)
            sidecar = json.loads((path.parent / "counterfactual.json").read_text())
            family, variant = sidecar["family_id"], sidecar["variant"]
            if variant in groups[family]:
                raise ValueError(f"duplicate variant {variant}")
            groups[family][variant] = (path.parent, episode, sidecar)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    complete = {f: g for f, g in groups.items() if set(g) == set(VARIANTS)}
    incomplete = sorted(set(groups) - set(complete))
    if not partial and incomplete:
        errors.append(f"incomplete families: {incomplete}")
    index = folder / "index.jsonl"
    if not partial:
        generation = folder / "generation.json"
        if not generation.exists():
            errors.append("generation.json missing")
        else:
            record = json.loads(generation.read_text())
            if record.get("errors") or record.get("n") != len(complete) or record.get("ok") != 6 * len(complete):
                errors.append("generation record does not match complete families")
        if not complete:
            errors.append("no complete families")
        if not index.exists():
            errors.append("index.jsonl missing")
        else:
            ids = [json.loads(line)["episode_id"] for line in index.read_text().splitlines() if line.strip()]
            actual = [e.episode_id for g in complete.values() for _, e, _ in g.values()]
            if len(ids) != len(set(ids)) or set(ids) != set(actual):
                errors.append("index does not match complete episodes")
    visibility, effects, speeds = [], [], []
    for family, group in complete.items():
        try:
            refpath, ref, refmeta = group["original"]
            assert _hash((refpath / "model.xml").read_bytes()) == refmeta["model_sha256"], "model hash"
            with np.load(refpath / "initial_state.npz", allow_pickle=False) as snap:
                assert _hash(snap["state"].tobytes()) == refmeta["state_sha256"], "state hash"
                assert str(snap["state_spec"]) == refmeta["state_spec"], "state spec"
                assert str(snap["mujoco_version"]) == refmeta["mujoco_version"], "MuJoCo version"
            if render_check:
                import mujoco
                from physwm.sim.render import CameraRenderer
                model = mujoco.MjModel.from_xml_path(str(refpath / "model.xml"))
                data = mujoco.MjData(model)
                with np.load(refpath / "initial_state.npz", allow_pickle=False) as snap:
                    mujoco.mj_setState(model, data, snap["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
                for camera_variant in ("original", "elevation"):
                    camera_path, camera_ep, _ = group[camera_variant]
                    if camera_variant == "elevation":
                        T = np.asarray(camera_ep.camera.cam2world)
                        model.cam_pos[cid] = T[:3, 3]
                        mat = T[:3, :3] @ np.diag([1., -1., -1.])
                        q = np.empty(4)
                        mujoco.mju_mat2Quat(q, mat.ravel())
                        model.cam_quat[cid] = q
                    mujoco.mj_forward(model, data)
                    renderer = CameraRenderer(model, camera_ep.camera.width, camera_ep.camera.height)
                    try:
                        actual_seg = renderer.seg(data)
                    finally:
                        renderer.close()
                    names = ("original", "reverse", "half", "double", "recolor") if camera_variant == "original" else ("elevation",)
                    for name in names:
                        seg_path, seg_ep, _ = group[name]
                        with Image.open(seg_path / seg_ep.files["seg_t0"]) as image:
                            assert np.array_equal(actual_seg, np.asarray(image)), f"{name} segmentation does not match camera/model"
            assert ref.action.impulse > 0 and ref.action.speed > 0, "nonpositive impulse/speed"
            speeds.append(ref.action.speed)
            with np.load(refpath / ref.files["traj"], allow_pickle=False) as tr:
                ref_t, ref_pos, ref_quat = tr["t"].copy(), tr["pos"].copy(), tr["quat"].copy()
            for variant, (path, ep, meta) in group.items():
                assert ep.episode_id == path.name and ep.split == split and ep.seed == ref.seed, f"{variant} identity"
                assert meta["family_id"] == family and meta["variant"] == variant, f"{variant} sidecar"
                assert meta["reference_episode_id"] == ref.episode_id, f"{variant} reference"
                assert all(meta[key] == refmeta[key] for key in ("model_sha256", "state_sha256", "config_sha256")), f"{variant} provenance"
                assert ep.action.target == ref.action.target == meta["target_id"], f"{variant} target"
                assert ep.action.impulse > 0 and ep.action.speed > 0, f"{variant} impulse/speed"
                assert np.isclose(ep.action.impulse, ep.objects[ep.action.target].mass * ep.action.speed), f"{variant} impulse/mass"
                assert all((path / file).is_file() for file in ep.files.values()), f"{variant} missing files"
                assert np.array_equal([s.pos for s in ep.states[0]], [s.pos for s in ref.states[0]]), f"{variant} initial positions"
                assert np.array_equal([s.quat for s in ep.states[0]], [s.quat for s in ref.states[0]]), f"{variant} initial quaternions"
                with Image.open(path / ep.files["seg_t0"]) as im:
                    seg = np.array(im)
                assert seg.shape == (ep.camera.height, ep.camera.width), f"{variant} segmentation dimensions"
                assert set(np.unique(seg)).issubset(set(range(len(ep.objects) + 1))), f"{variant} segmentation IDs"
                for key in ("rgb_t0", "rgb_t1"):
                    with Image.open(path / ep.files[key]) as im:
                        assert im.size == (ep.camera.width, ep.camera.height), f"{variant} image dimensions"
                with np.load(path / ep.files["depth_t0"], allow_pickle=False) as depth:
                    assert depth["depth"].shape == seg.shape and np.isfinite(depth["depth"]).all(), f"{variant} depth"
                cam = np.array(ep.camera.cam2world)
                assert cam.shape == (4, 4) and np.allclose(cam[:3,:3].T @ cam[:3,:3], np.eye(3), atol=1e-5), f"{variant} camera"
                # Pixel visibility is a descriptive measurement; occlusion is not corruption.
                visibility.append({"family_id": family, "variant": variant, "target_pixels": int(np.sum(seg == ep.action.target + 1))})
                with np.load(path / ep.files["traj"], allow_pickle=False) as tr:
                    t, pos, quat = tr["t"], tr["pos"], tr["quat"]
                    assert np.array_equal(t, ref_t) and pos.shape == ref_pos.shape and quat.shape == ref_quat.shape, f"{variant} trajectory shape/times"
                    assert np.array_equal(pos[0], ref_pos[0]) and np.array_equal(quat[0], ref_quat[0]), f"{variant} trajectory initial"
                    assert np.isfinite(pos).all() and np.isfinite(quat).all(), f"{variant} trajectory finite"
                    if variant in ("recolor", "elevation"):
                        assert np.array_equal(pos, ref_pos) and np.array_equal(quat, ref_quat), f"{variant} physical invariance"
                    effects.append({"family_id": family, "variant": variant,
                                    "target_endpoint_effect_m": float(np.linalg.norm(pos[-1, ep.action.target] - ref_pos[-1, ep.action.target])),
                                    "all_objects_endpoint_effect_m": float(np.linalg.norm(pos[-1] - ref_pos[-1]))})
                if variant in ("original", "reverse", "half", "double", "recolor"):
                    assert ep.camera == ref.camera, f"{variant} camera changed"
                if variant == "elevation":
                    assert ep.camera.azimuth_deg == ref.camera.azimuth_deg and ep.camera.elevation_deg != ref.camera.elevation_deg, "elevation camera"
                factor = {"half": .5, "double": 2.}.get(variant, 1.)
                assert np.isclose(ep.action.speed, ref.action.speed * factor), f"{variant} speed"
                direction = -np.array(ref.action.direction_world) if variant == "reverse" else ref.action.direction_world
                assert np.allclose(ep.action.direction_world, direction), f"{variant} direction"
                if variant == "reverse":
                    assert ep.action.direction_label == {"left":"right", "right":"left", "toward":"away", "away":"toward"}[ref.action.direction_label], "reverse label"
                else:
                    assert ep.action.direction_label == ref.action.direction_label, f"{variant} label"
                if variant == "recolor":
                    assert ep.objects[ep.action.target].color != ref.objects[ref.action.target].color, "recolor unchanged"
                    assert ep.objects[ep.action.target].name in ep.action.text, "recolor prompt"
        except Exception as exc:
            errors.append(f"{family}: {exc}")
    return {"split": split, "partial": partial, "families_complete": len(complete), "families_incomplete": incomplete,
            "episodes": len(complete) * 6, "render_check": render_check, "errors": errors, "target_visibility": visibility,
            "target_invisible_count": sum(x["target_pixels"] == 0 for x in visibility), "ground_truth_effects": effects,
            "original_speed": {"count": len(speeds), "min": min(speeds) if speeds else None,
                               "max": max(speeds) if speeds else None, "values": speeds}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--partial", action="store_true"); p.add_argument("--render-check", action="store_true")
    a = p.parse_args()
    result = audit(a.root, a.split, a.partial, a.render_check)
    a.out.write_text(json.dumps(result, indent=2))
    print(f"{result['families_complete']} complete families, {len(result['errors'])} integrity errors; {a.out}")
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
