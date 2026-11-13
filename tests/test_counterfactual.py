import numpy as np

from physwm.sim.counterfactual import VARIANTS, simulate_family
from physwm.sim.scene import load_config


def test_matched_family():
    cfg = load_config("configs/sim_train.yaml")
    family = simulate_family(cfg, 9901000, "val")
    assert tuple(family) == VARIANTS
    base, meta = family["original"]
    assert all(m["state_sha256"] == meta["state_sha256"] and m["model_sha256"] == meta["model_sha256"] for _, m in family.values())
    assert all(m["reference_episode_id"] == base.episode.episode_id for _, m in family.values())
    reverse = family["reverse"][0].episode.action
    np.testing.assert_allclose(reverse.direction_world, -np.array(base.episode.action.direction_world))
    assert reverse.direction_label == {"left":"right", "right":"left", "toward":"away", "away":"toward"}[base.episode.action.direction_label]
    for name, factor in (("half", .5), ("double", 2)):
        assert np.isclose(family[name][0].episode.action.impulse, base.episode.action.impulse * factor)
    for name in ("recolor", "elevation"):
        variant = family[name][0]
        np.testing.assert_array_equal(variant.traj_pos, base.traj_pos)
        np.testing.assert_array_equal(variant.traj_quat, base.traj_quat)
        assert variant.episode.action.direction_world == base.episode.action.direction_world
    recolor = family["recolor"][0].episode
    assert recolor.objects[recolor.action.target].name in recolor.action.text
    assert recolor.objects[recolor.action.target].color != base.episode.objects[base.episode.action.target].color
    assert family["elevation"][0].episode.camera.azimuth_deg == base.episode.camera.azimuth_deg
    assert family["elevation"][0].episode.camera.elevation_deg != base.episode.camera.elevation_deg
    assert np.array_equal(family["reverse"][0].rgb[(0, 0)], base.rgb[(0, 0)])


def test_cli_dataset_resume_and_split(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    from physwm.schema import load_split
    from physwm.sim.counterfactual import main

    args = ["counterfactual", "--config", "configs/sim_train.yaml", "--split", "val",
            "--seed-start", "9901000", "--n", "1", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    main()
    episodes = load_split(tmp_path, "val")
    assert len(episodes) == 6
    for path, ep in episodes:
        meta = json.loads((path / "counterfactual.json").read_text())
        assert meta["family_id"] == "family_9901000"
        assert meta["reference_episode_id"] == "val_9901000_original"
        assert all((path / filename).is_file() for filename in ep.files.values())
    # Replay the original from the persisted model and complete integration state.
    import mujoco
    reference = tmp_path / "val" / "val_9901000_original"
    model = mujoco.MjModel.from_xml_path(str(reference / "model.xml"))
    data = mujoco.MjData(model)
    with np.load(reference / "initial_state.npz") as snapshot:
        assert str(snapshot["state_spec"]) == "mjSTATE_INTEGRATION"
        mujoco.mj_setState(model, data, snapshot["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    original = next(ep for _, ep in episodes if ep.episode_id.endswith("_original"))
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(len(original.objects))]
    dof = model.jnt_dofadr[model.body_jntadr[body_ids[original.action.target]]]
    data.qvel[dof:dof+3] += original.action.speed * np.array(original.action.direction_world)
    cfg = load_config("configs/sim_train.yaml")
    s = cfg["sim"]
    every = round(1 / (s["traj_hz"] * s["timestep"]))
    positions = [data.xpos[body_ids].copy()]
    for k in range(1, round(s["horizon"] / s["timestep"]) + 1):
        mujoco.mj_step(model, data)
        t = k * s["timestep"]
        if k % every == 0 or any(abs(t-x) < 1e-9 for x in s["store_times"][1:]):
            mujoco.mj_fwdPosition(model, data)
        if k % every == 0:
            positions.append(data.xpos[body_ids].copy())
    with np.load(reference / "traj.npz") as trajectory:
        np.testing.assert_array_equal(positions, trajectory["pos"])
    # An incomplete episode can be safely reconstructed with pinned provenance.
    (episodes[0][0] / "traj.npz").unlink()
    main()
    assert (episodes[0][0] / "traj.npz").is_file()
    # A different request cannot resume the same split.
    monkeypatch.setattr(sys, "argv", args[:-3] + ["2", "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="provenance differs"):
        main()
    cfg = load_config("configs/sim_train.yaml")
    with pytest.raises(ValueError, match="belongs to val"):
        simulate_family(cfg, 9901000, "train")
