"""Generate matched push counterfactual families from one settled MuJoCo state."""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from physwm import geometry as G
from physwm.schema import Action, Episode, ObjectSpec
from physwm.sim.generate import generation_identity, prepare_split
from physwm.sim.push import (DIRECTION_PHRASE, EpisodeData, Rejected, _camera_xml,
                             _sample_camera, _sample_push, _states, validate_timing)
from physwm.sim.render import CameraRenderer
from physwm.sim.scene import COLORS, SHAPE_NOUN, build_mjcf, load_config, sample_objects

VARIANTS = ("original", "reverse", "half", "double", "recolor", "elevation")
OPPOSITE = {"left": "right", "right": "left", "toward": "away", "away": "toward"}
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(m, d):
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    mujoco.mj_getState(m, d, state, STATE)
    return state


def _prepare(cfg, seed):
    rng = np.random.default_rng(seed)
    s = cfg["sim"]
    for _ in range(20):
        try:
            objs = sample_objects(rng, cfg)
            lookat = np.r_[cfg["objects"].get("region_center", [0., 0.]), 0.]
            cam, T = _sample_camera(rng, cfg, lookat)
            table = list(rng.uniform(.55, .85) * np.array([1., .85, .65])) + [1.]
            floor = list(rng.uniform(.15, .35, 3)) + [1.]
            xml = build_mjcf(objs, cfg, table, floor).replace(
                "</worldbody>", _camera_xml(T, cam.fovy_deg) + "\n  </worldbody>")
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(len(objs))]
            mujoco.mj_forward(m, d)
            mujoco.mj_step(m, d, nstep=round(s["settle"] / s["timestep"]))
            if np.abs(d.qvel).max() > .05:
                raise Rejected("did not settle")
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            up = [np.array([0., 0., 1.]) for _ in objs]
            shapes = [o.shape for o in objs]
            states = _states(m, d, ids, up, shapes, cfg["table"]["half_size"])
            if any(st.toppled or not st.on_table for st in states):
                raise Rejected("unstable rest pose")
            specs = [ObjectSpec(id=i, name=f"{o.color} {SHAPE_NOUN[o.shape]}", shape=o.shape,
                                color=o.color, rgba=COLORS[o.color] + [1.],
                                half_extents=o.half_extents.tolist(), mass=float(m.body_mass[b]),
                                density=o.density, friction=o.friction)
                     for i, (o, b) in enumerate(zip(objs, ids))]
            target = int(rng.integers(len(objs)))
            xy = np.array([st.pos[:2] for st in states])
            label, direction, speed = _sample_push(rng, cfg, T, xy[target], np.delete(xy, target, axis=0))
            return m, d, xml, cam, T, ids, up, shapes, specs, target, label, direction, speed
        except (Rejected, RuntimeError):
            continue
    raise RuntimeError(f"seed {seed}: no valid scene after 20 attempts")


def family_split(seed):
    """Assign every seed to one split independently of the output directory."""
    return "val" if seed % 5 == 0 else "train"


class Family(dict):
    """Episodes together with their replayable pre-action model and state."""


def simulate_family(cfg, seed, split):
    if split != family_split(seed):
        raise ValueError(f"seed {seed} belongs to {family_split(seed)}, not {split}")
    validate_timing(cfg["sim"])
    m, d, xml, cam, T, ids, up, shapes, specs, target, label, direction, speed = _prepare(cfg, seed)
    initial = _snapshot(m, d)
    state_hash, model_hash = _hash(initial.tobytes()), _hash(xml.encode())
    original_rgba = m.geom_rgba.copy()
    original_cam_pos, original_cam_quat = m.cam_pos.copy(), m.cam_quat.copy()
    results = Family()
    results.model_xml = xml
    results.initial_state = initial.copy()
    for variant in VARIANTS:
        m.geom_rgba[:] = original_rgba
        m.cam_pos[:], m.cam_quat[:] = original_cam_pos, original_cam_quat
        variant_cam = copy.deepcopy(cam)
        variant_specs = copy.deepcopy(specs)
        vlabel, vdir, vspeed = label, direction.copy(), speed
        intervention = {}
        if variant == "reverse":
            vlabel, vdir = OPPOSITE[label], -direction
            intervention = {"direction": "reverse"}
        elif variant in ("half", "double"):
            factor = .5 if variant == "half" else 2.
            vspeed *= factor
            intervention = {"impulse_factor": factor}
        elif variant == "recolor":
            used = {o.color for o in specs}
            available = [c for c in COLORS if c not in used]
            color = available[seed % len(available)] if available else next(c for c in COLORS if c != specs[target].color)
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g{target}")
            m.geom_rgba[geom] = COLORS[color] + [1.]
            obj = variant_specs[target]
            intervention = {"object_id": target, "from": obj.color, "to": color, "rgba": COLORS[color] + [1.]}
            obj.color, obj.rgba, obj.name = color, COLORS[color] + [1.], f"{color} {SHAPE_NOUN[obj.shape]}"
        elif variant == "elevation":
            # Rotate eye about the exact look-at point, preserving azimuth and distance.
            el = 70. if cam.elevation_deg < 55. else 35.
            az = np.radians(cam.azimuth_deg)
            old_eye = T[:3, 3]
            old_el = np.radians(cam.elevation_deg)
            lookat = old_eye - cam.distance * np.array([np.cos(old_el)*np.cos(az), np.cos(old_el)*np.sin(az), np.sin(old_el)])
            new_el = np.radians(el)
            eye = lookat + cam.distance * np.array([np.cos(new_el)*np.cos(az), np.cos(new_el)*np.sin(az), np.sin(new_el)])
            new_T = G.look_at(eye, lookat)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
            m.cam_pos[cid] = eye
            # MuJoCo camera basis is OpenCV x, -y, -z.
            mat = new_T[:3, :3] @ np.diag([1., -1., -1.])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, mat.ravel())
            m.cam_quat[cid] = quat
            variant_cam.cam2world, variant_cam.elevation_deg = new_T.tolist(), el
            intervention = {"elevation_deg": el, "azimuth_deg": cam.azimuth_deg, "cam2world": new_T.tolist()}
        mujoco.mj_resetData(m, d)
        mujoco.mj_setState(m, d, initial, STATE)
        mujoco.mj_forward(m, d)
        s0 = _states(m, d, ids, up, shapes, cfg["table"]["half_size"])
        impulse = specs[target].mass * vspeed
        action = Action(type="push", target=target, direction_label=vlabel, direction_world=vdir.tolist(),
                        impulse=float(impulse), speed=float(vspeed), text=f"Push the {variant_specs[target].name} {DIRECTION_PHRASE[vlabel]} with an impulse of {impulse:.6g} N·s.")
        rend = CameraRenderer(m, cam.width, cam.height)
        try:
            rgb0, depth0, seg0 = rend.rgb(d), rend.depth(d), rend.seg(d)
            dof = m.jnt_dofadr[m.body_jntadr[ids[target]]]
            d.qvel[dof:dof+3] += vspeed * vdir
            s = cfg["sim"]
            every = round(1 / (s["traj_hz"] * s["timestep"]))
            traj_t, pos, quat = [0.], [d.xpos[ids].copy()], [d.xquat[ids].copy()]
            states, rgb1 = [s0], None
            for k in range(1, round(s["horizon"] / s["timestep"]) + 1):
                mujoco.mj_step(m, d)
                t = k * s["timestep"]
                store = any(abs(t - x) < 1e-9 for x in s["store_times"][1:])
                if k % every == 0 or store:
                    mujoco.mj_fwdPosition(m, d)
                if k % every == 0:
                    traj_t.append(t); pos.append(d.xpos[ids].copy()); quat.append(d.xquat[ids].copy())
                if store:
                    states.append(_states(m, d, ids, up, shapes, cfg["table"]["half_size"]))
                    if abs(t - 1.) < 1e-9:
                        rgb1 = rend.rgb(d)
        finally:
            rend.close()
        episode_id = f"{split}_{seed:07d}_{variant}"
        ep = Episode(episode_id=episode_id, split=split, config=cfg["name"], seed=seed,
                     table_half_size=list(cfg["table"]["half_size"]), objects=variant_specs, camera=variant_cam,
                     action=action, times=list(s["store_times"]), states=states,
                     files={"rgb_t0":"rgb_t0.png", "rgb_t1":"rgb_t1.png", "depth_t0":"depth_t0.npz", "seg_t0":"seg_t0.png", "traj":"traj.npz"})
        data = EpisodeData(ep, rgb0, rgb1, depth0, seg0, np.array(traj_t), np.array(pos), np.array(quat))
        if variant in ("recolor", "elevation"):
            base = results["original"][0]
            if not (np.array_equal(data.traj_pos, base.traj_pos) and np.array_equal(data.traj_quat, base.traj_quat)):
                raise RuntimeError("visual intervention changed physics")
            data.traj_t, data.traj_pos, data.traj_quat = base.traj_t, base.traj_pos, base.traj_quat
            data.episode.states = copy.deepcopy(base.episode.states)
        metadata = {"family_id": f"family_{seed:07d}", "variant": variant, "seed": seed, "split": split,
                    "reference_episode_id": f"{split}_{seed:07d}_original", "target_id": target,
                    "state_sha256": state_hash, "model_sha256": model_hash, "state_spec": "mjSTATE_INTEGRATION",
                    "mujoco_version": mujoco.__version__, "config_sha256": _hash(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()), "intervention": intervention}
        results[variant] = (data, metadata)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--split", required=True)
    p.add_argument("--seed-start", type=int, default=9901000); p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.n <= 0 or a.seed_start < 9901000 or a.split not in ("train", "val"):
        p.error("n must be positive, seed-start >= 9901000, and split must be train or val")
    cfg = load_config(a.config)
    identity = generation_identity(cfg)
    identity["counterfactual_source_sha256"] = _hash(Path(__file__).read_bytes())
    identity["variants"] = list(VARIANTS)
    split_dir = Path(a.out) / a.split
    prepare_split(split_dir, identity, a.seed_start, a.n)
    ids = []
    seed = a.seed_start
    selected = 0
    while selected < a.n:
        # The seed alone assigns the family to a split, regardless of invocation.
        assigned = family_split(seed)
        if assigned != a.split:
            seed += 1
            continue
        selected += 1
        family = simulate_family(cfg, seed, a.split)
        reference = split_dir / family["original"][0].episode.episode_id
        if reference.exists() and (reference / "meta.json").exists():
            model_path, state_path = reference / "model.xml", reference / "initial_state.npz"
            if model_path.exists() and _hash(model_path.read_bytes()) != family["original"][1]["model_sha256"]:
                raise ValueError(f"model snapshot mismatch: {model_path}")
            if state_path.exists():
                with np.load(state_path, allow_pickle=False) as saved:
                    if _hash(saved["state"].tobytes()) != family["original"][1]["state_sha256"]:
                        raise ValueError(f"state snapshot mismatch: {state_path}")
        for variant, (data, meta) in family.items():
            path = split_dir / data.episode.episode_id
            if path.exists() and any(path.iterdir()) and not (path / "meta.json").exists():
                raise ValueError(f"incomplete episode without identity marker: {path}")
            if path.exists() and (path / "meta.json").exists():
                if not (path / "counterfactual.json").exists() or json.loads((path / "counterfactual.json").read_text()) != meta:
                    raise ValueError(f"counterfactual identity mismatch: {path}")
                if json.loads((path / "meta.json").read_text()) != data.episode.to_dict():
                    raise ValueError(f"episode identity mismatch: {path}")
            path.mkdir(parents=True, exist_ok=True)
            for name, image in (("rgb_t0.png", data.rgb_t0), ("rgb_t1.png", data.rgb_t1), ("seg_t0.png", data.seg_t0)):
                if not (path / name).exists():
                    Image.fromarray(image).save(path / name)
            if not (path / "depth_t0.npz").exists():
                np.savez_compressed(path / "depth_t0.npz", depth=data.depth_t0.astype(np.float16))
            if not (path / "traj.npz").exists():
                np.savez_compressed(path / "traj.npz", t=data.traj_t, pos=data.traj_pos, quat=data.traj_quat)
            if variant == "original":
                if not (path / "model.xml").exists():
                    (path / "model.xml").write_text(family.model_xml)
                if not (path / "initial_state.npz").exists():
                    np.savez_compressed(path / "initial_state.npz", state=family.initial_state, state_spec="mjSTATE_INTEGRATION", mujoco_version=mujoco.__version__)
            if not (path / "counterfactual.json").exists():
                (path / "counterfactual.json").write_text(json.dumps(meta, indent=1))
            if not (path / "meta.json").exists():
                data.episode.save(path / "meta.json")
            ids.append(data.episode.episode_id)
        print(f"{a.split}: family {selected}/{a.n} seed={seed} complete", flush=True)
        seed += 1
    (split_dir / "index.jsonl").write_text("".join(json.dumps({"episode_id": i}) + "\n" for i in sorted(ids)))
    (split_dir / "generation.json").write_text(json.dumps({"config": cfg, "seed_start": a.seed_start, "n": a.n, "ok": len(ids), "errors": []}, indent=1))


if __name__ == "__main__":
    main()
