"""OneCanvas-style xyz PE: 136-ch recipe, token unprojection and merger-hook alignment."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from physwm.vlm import pe as P  # noqa: E402


def test_xyz_features_recipe_and_determinism():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.1, 0.7], [-1.0, 2.0, 0.5]])
    f = P.xyz_features(xyz)
    assert f.shape == (3, 136) and f.dtype == torch.float32
    assert torch.equal(f, P.xyz_features(xyz))  # deterministic

    q = xyz.numpy()
    # per-axis {sin, cos, raw} blocks of 16 freqs: raw channels at 32, 65, 98
    assert np.allclose(f[:, 32].numpy(), q[:, 0])
    assert np.allclose(f[:, 65].numpy(), q[:, 1])
    assert np.allclose(f[:, 98].numpy(), q[:, 2])
    # first channel is sin(x * 0.1 rad/m)
    assert np.allclose(f[:, 0].numpy(), np.sin(q[:, 0] * 0.1), atol=1e-6)
    # radial block: ||q|| raw at 99 + 16, r_xz raw at 99 + 33
    assert np.allclose(f[:, 115].numpy(), np.linalg.norm(q, axis=-1), atol=1e-6)
    assert np.allclose(f[:, 132].numpy(), np.hypot(q[:, 0], q[:, 2]), atol=1e-6)
    # unit ray
    assert np.allclose(f[:, 133:].numpy(), q / np.linalg.norm(q, axis=-1, keepdims=True), atol=1e-6)
    assert torch.isfinite(f).all()


def test_token_xyz_on_constant_plane_matches_analytic_rays():
    depth = np.full((16, 20), 2.0, dtype=np.float32)
    K = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]])
    xyz = P.token_xyz(depth, K, (1, 8, 10))
    assert xyz.shape == (4 * 5, 3)  # merged grid: 8/2 x 10/2
    uv = np.array([(2.0 + 4.0 * c, 2.0 + 4.0 * r) for r in range(4) for c in range(5)])
    expected = np.stack([(uv[:, 0] - 10.0) / 100.0 * 2.0,
                         (uv[:, 1] - 8.0) / 100.0 * 2.0,
                         np.full(len(uv), 2.0)], axis=-1)
    assert np.allclose(xyz.numpy(), expected, atol=1e-6)


def test_token_xyz_falls_back_to_image_median_then_zero():
    depth = np.full((8, 8), 3.0, dtype=np.float32)
    depth[0:4, 0:4] = np.nan  # one merged block (grid 1,4,4 -> 2x2 blocks of 4x4) has no valid depth
    K = np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    xyz = P.token_xyz(depth, K, (1, 4, 4))
    assert xyz.shape == (4, 3)
    assert np.allclose(xyz[0].numpy(), [-0.6, -0.6, 3.0])  # u=v=2; image-median depth, K centre 4
    empty = P.token_xyz(np.full((8, 8), np.nan, dtype=np.float32), K, (1, 4, 4))
    assert torch.equal(empty, torch.zeros(4, 3))


def test_gate_zero_is_identity():
    torch.manual_seed(0)
    emb = P.XYZEmbedding(hidden=8)
    feats = torch.randn(5, 8)
    xyz = torch.randn(5, 3)
    assert torch.equal(emb(xyz, feats), feats)
    with torch.no_grad():
        emb.gate.fill_(1.0)
    assert not torch.allclose(emb(xyz, feats), feats)


def test_pe_param_groups_use_50x_lr_without_weight_decay():
    emb = P.XYZEmbedding(hidden=4)
    (group,) = P.pe_param_groups(emb, 2e-4)
    assert group["lr"] == pytest.approx(50 * 2e-4) and group["weight_decay"] == 0.0
    assert {id(p) for p in group["params"]} == {id(p) for p in emb.parameters()}


def _tiny_model():
    vision = transformers.Qwen3VLVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64,
        num_position_embeddings=16, deepstack_visual_indexes=(0,),
    )
    text = transformers.Qwen3VLTextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(
        transformers.Qwen3VLConfig(vision_config=vision, text_config=text)
    )


def test_merger_hook_alignment_on_tiny_qwen():
    model = _tiny_model().eval()
    grid = torch.tensor([[1, 4, 6], [1, 4, 6]])  # 2 images -> 2 * 6 merged tokens
    pv = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16)
    with torch.no_grad():
        base = model.model.get_image_features(pv, grid)
    base = torch.cat(base.pooler_output, dim=0)
    assert base.shape[0] == 12

    seen = {}
    model.model.visual.merger.register_forward_hook(
        lambda _m, inp, out: seen.update(rows_in=inp[0].shape[0], rows_out=out.shape[0])
    )
    pe, _handle = P.install_pe(model)
    assert pe.hidden == base.shape[1]
    xyz = torch.randn(12, 3)

    model._physwm_pe_xyz = xyz
    with torch.no_grad():
        out = torch.cat(model.model.get_image_features(pv, grid).pooler_output, dim=0)
    assert seen["rows_in"] == 48 and seen["rows_out"] == 12  # hook aligns to merged tokens
    assert model._physwm_pe_xyz is None  # consumed
    assert torch.allclose(out, base)  # gate = 0 -> frozen features unchanged

    with torch.no_grad():
        pe.gate.fill_(0.5)
        model._physwm_pe_xyz = xyz
        out = torch.cat(model.model.get_image_features(pv, grid).pooler_output, dim=0)
    assert not torch.allclose(out, base)
    # a second call without xyz is a no-op, not a stale PE
    with torch.no_grad():
        again = torch.cat(model.model.get_image_features(pv, grid).pooler_output, dim=0)
    assert torch.allclose(again, base)

    model._physwm_pe_xyz = torch.randn(6, 3)
    with pytest.raises(ValueError, match="misalignment"):
        with torch.no_grad():
            model.model.get_image_features(pv, grid)


def test_save_and_load_pe_roundtrip(tmp_path):
    emb = P.XYZEmbedding(hidden=6)
    with torch.no_grad():
        for p in emb.parameters():
            p.add_(0.1)
    path = tmp_path / "pe.pt"
    P.save_pe(path, emb)
    again = P.load_pe(path)
    assert again.hidden == 6
    xyz, feats = torch.randn(4, 3), torch.randn(4, 6)
    assert torch.equal(again(xyz, feats), emb(xyz, feats))
