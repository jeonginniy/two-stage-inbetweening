"""
2단계 Context Transformer 테스트 — 가짜 데이터로 실행 (LaFAN1 불필요)

확인하는 것:
- 입력/출력 차원이 설정(스켈레톤 관절 수)에서 계산되는가 (135 를 하드코딩하지 않음)
- 입력 만들기: m_in, 어텐션 마스크(그림 5a), Keyframe PE 입력, 루트 선형보간, transition 합치기, 후처리(식 23)
- 상대 위치 어텐션이 직접 계산(반복문)과 같은가
- 어텐션 마스크가 실제로 동작하는가: 모르는 프레임 값을 바꿔도 다른 프레임 출력이 변하지 않음
- forward/backward: shape, 손실 유한, 모든 파라미터에 gradient
- 작은 모델이 고정 배치에 과적합되는가 (학습이 실제로 손실을 줄이는지)
- 학습 스크립트가 --fake-data 로 끝까지 도는가
"""

import math
import random
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_data_config, load_model_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.models.context_transformer import ContextTransformer, build_context_transformer, count_parameters
from src.models.layers import RelativeMultiHeadAttention, relative_index
from src.training.context import context_forward, infer_context, prepare_batch
from src.training.context_inputs import (
    TransitionSpec,
    build_context_input,
    data_mask,
    interpolate_root_placeholder,
    key_mask,
    keyframe_positions,
    merge_prediction,
    postprocess,
    sample_transition,
)
from src.training.losses import position_loss, rotation_loss, smooth_loss
from src.training.utils import noam_lambda
from tests.helpers import SKELETON_YAML, write_fake_lafan_bvh

TINY = dict(d_mask=1, n_layer=2, n_head=2, d_model=32, d_head=8, d_ff=32,
            d_encoder_h=32, d_decoder_h=32, dropout=0.0, attn_bias=False, attn_scale="d_head")


# -----------------------------------------------------------------------------
# 준비: 가짜 LaFAN1 폴더 → 배치, 정규화기
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fake_lafan1")
    for i, stem in enumerate(["walk1_subject1", "run1_subject2"]):
        write_fake_lafan_bvh(folder / f"{stem}.bvh", 120, seed=i)
    data_cfg = load_data_config()
    skel = Skeleton.from_yaml(SKELETON_YAML)
    layout = FeatureLayout(skel.num_joints)
    ds = LafanDataset(folder, ["subject1", "subject2"], 42, 5, skel, verbose=False)
    stats = RunningStats(layout.dim)
    for b in DataLoader(ds, batch_size=32):
        stats.update(motion_to_model_input(b["root_pos"], b["local_rot"], 10, skel.root_forward_local_axis).x_raw)
    norm = ZScoreNormalizer.from_stats(stats).to(dtype=torch.float32)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    return data_cfg, skel, layout, norm, batch


# -----------------------------------------------------------------------------
# 설정 / 차원
# -----------------------------------------------------------------------------
def test_config_dims_come_from_skeleton():
    cfg = load_model_config("context")
    assert "d_in" not in cfg["model"] and "d_out" not in cfg["model"]   # 차원은 설정에 적지 않음
    skel = Skeleton.from_yaml(load_data_config()["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    model = build_context_transformer(cfg["model"], layout)
    assert (model.d_in, model.d_out) == (layout.dim + 1, layout.dim) == (136, 135)
    # 공식 config 크기와 비슷한지 (약 1천만 파라미터)
    assert 5e6 < count_parameters(model) < 2e7
    for key in ("batch_size", "grad_accum_steps", "amp", "base_lr", "warmup_iters", "max_iters"):
        assert key in cfg["train"]


@pytest.mark.parametrize("J", [5, 22, 30])
def test_model_follows_joint_count(J):
    layout = FeatureLayout(J)
    model = build_context_transformer(TINY, layout)
    spec = TransitionSpec(4, 12)
    x = torch.randn(3, 16, layout.dim)
    out = model(build_context_input(x, spec, layout), keyframe_positions(16, spec), key_mask(16, spec))
    assert out.shape == (3, 16, J * 6 + 3)


def test_wrong_input_dim_raises():
    model = build_context_transformer(TINY, FeatureLayout(22))
    with pytest.raises(ValueError):
        model(torch.randn(1, 8, 135), torch.zeros(1, 8, 2))


# -----------------------------------------------------------------------------
# 입력 만들기 (논문 3.3)
# -----------------------------------------------------------------------------
def test_masks_and_keyframe_positions():
    spec = TransitionSpec(10, 40, midway=(25, 15))
    assert spec.midway == (15, 25) and spec.trans_len == 30 and spec.seq_slice == slice(10, 40)
    m = data_mask(42, spec)[:, 0]
    known = set(range(10)) | {15, 25, 40}
    assert all(m[t] == (1.0 if t in known else 0.0) for t in range(42))
    km = key_mask(42, spec)
    assert km.shape == (42, 42)
    assert torch.equal(km, m.bool()[None].expand(42, 42))        # 모든 query 가 같은 key 집합
    kf = keyframe_positions(42, spec)
    assert kf.shape == (1, 42, 2)
    assert kf[0, 9].tolist() == [0.0, 31.0] and kf[0, 40].tolist() == [31.0, 0.0]
    with pytest.raises(ValueError):
        TransitionSpec(10, 20, midway=(20,))                    # target 은 midway 가 될 수 없음


def test_sample_transition_ranges():
    rng = random.Random(0)
    seen = set()
    for _ in range(2000):
        s = sample_transition(10, 5, 30, 42, 0.1, 0.5, rng)
        assert 5 <= s.trans_len <= 30 and s.target_idx <= 40
        assert all(10 <= m < s.target_idx for m in s.midway)
        assert len(s.midway) <= int(0.1 * s.trans_len)
        seen.add(s.trans_len)
    assert seen == set(range(5, 31))
    with pytest.raises(ValueError):
        sample_transition(10, 5, 31, 41)


def test_root_interpolation_and_input():
    layout = FeatureLayout(3)
    spec = TransitionSpec(3, 9, midway=(6,))
    x = torch.randn(2, 11, layout.dim)
    xi = interpolate_root_placeholder(x, spec, layout.pos_slice)
    p = layout.pos_slice
    # 앵커 2 → 6 사이, 6 → 9 사이가 정확히 직선
    for a, b in [(2, 6), (6, 9)]:
        for t in range(a + 1, b):
            w = (t - a) / (b - a)
            assert torch.allclose(xi[:, t, p], (1 - w) * x[:, a, p] + w * x[:, b, p], atol=1e-6)
    assert torch.equal(xi[:, :, layout.rot_slice], x[:, :, layout.rot_slice])   # 회전은 그대로

    inp = build_context_input(x, spec, layout)
    assert inp.shape == (2, 11, layout.dim + 1)
    unknown = [3, 4, 5, 7, 8, 10]
    assert torch.all(inp[:, unknown, layout.rot_slice] == 0)                    # 회전 placeholder = 0
    assert torch.all(inp[:, unknown, -1] == 0) and torch.all(inp[:, [0, 1, 2, 6, 9], -1] == 1)
    assert torch.all(inp[:, 10, p] == 0)                                        # target 이후는 보간 안 함
    assert torch.equal(inp[:, [0, 1, 2, 6, 9], :-1], x[:, [0, 1, 2, 6, 9]])     # 아는 프레임은 정답


def test_merge_and_postprocess():
    spec = TransitionSpec(4, 10)
    gt = torch.randn(2, 12, 7)
    out = torch.randn(2, 12, 7, requires_grad=True)
    y = merge_prediction(gt, out, spec)
    assert torch.equal(y[:, :4], gt[:, :4]) and torch.equal(y[:, 10:], gt[:, 10:])
    assert torch.equal(y[:, 4:10], out[:, 4:10])
    y.sum().backward()
    assert torch.all(out.grad[:, 4:10] == 1) and torch.all(out.grad[:, :4] == 0) and torch.all(out.grad[:, 10:] == 0)

    # 등속 직선 운동: 정답 그대로면 δ = 0, 상수 c 를 더한 예측은 후처리로 정답에 돌아옴
    t = torch.arange(12, dtype=torch.float32)[None, :, None]
    lin = 3.0 + 0.5 * t.expand(2, 12, 7)
    assert torch.allclose(postprocess(lin, lin, spec), lin)
    shifted = lin.clone()
    shifted[:, 4:10] += 2.5
    assert torch.allclose(postprocess(shifted, lin, spec), lin, atol=1e-5)
    with pytest.raises(ValueError):
        postprocess(lin[:, :11], lin[:, :11], spec)          # target 다음 프레임이 없음


# -----------------------------------------------------------------------------
# 모델 내부
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_relative_attention_matches_loop():
    torch.manual_seed(0)
    B, T, d_model, H, d = 2, 5, 16, 2, 4
    attn = RelativeMultiHeadAttention(d_model, H, d, scale="d_head")
    x = torch.randn(B, T, d_model)
    rel = torch.randn(2 * T - 1, d)
    mask = torch.rand(T, T) > 0.3
    mask[:, 0] = True                                           # 모든 query 가 최소 1개는 봄
    out = attn(x, rel, mask)

    h = attn.norm(x)
    q = attn.q(h).view(B, T, H, d)
    k = attn.k(h).view(B, T, H, d)
    v = attn.v(h).view(B, T, H, d)
    ref = torch.zeros(B, T, H, d)
    for b in range(B):
        for hh in range(H):
            for i in range(T):
                s = torch.tensor([(q[b, i, hh] @ k[b, j, hh] + q[b, i, hh] @ rel[(j - i) + T - 1]) / math.sqrt(d)
                                  if mask[i, j] else -math.inf for j in range(T)])
                ref[b, i, hh] = s.softmax(0) @ v[b, :, hh]
    ref = x + attn.out(ref.reshape(B, T, H * d))
    assert torch.allclose(out, ref, atol=1e-5)
    assert relative_index(3).tolist() == [[2, 3, 4], [1, 2, 3], [0, 1, 2]]


def test_attention_mask_hides_unknown_frames():
    """모르는 프레임 j 의 입력을 바꾸면 j 자신의 출력만 바뀌어야 한다 (다른 query 는 j 를 key 로 못 봄)."""
    torch.manual_seed(0)
    layout = FeatureLayout(4)
    model = build_context_transformer(TINY, layout).eval()
    spec = TransitionSpec(3, 10)
    T = 12
    x_in = build_context_input(torch.randn(1, T, layout.dim), spec, layout)
    kf, km = keyframe_positions(T, spec), key_mask(T, spec)
    base = model(x_in, kf, km)
    for j in (5, 11):                                           # transition 프레임, target 이후 프레임
        x2 = x_in.clone()
        x2[:, j, :-1] += torch.randn(layout.dim) * 3
        out = model(x2, kf, km)
        others = [t for t in range(T) if t != j]
        assert torch.allclose(out[:, others], base[:, others], atol=1e-5)
        assert not torch.allclose(out[:, j], base[:, j])
    # 반대로 아는 프레임(context)을 바꾸면 transition 출력이 바뀐다
    x3 = x_in.clone()
    x3[:, 1, :-1] += 1.0
    assert not torch.allclose(model(x3, kf, km)[:, 5], base[:, 5])


# -----------------------------------------------------------------------------
# 손실 / forward-backward
# -----------------------------------------------------------------------------
def test_losses_zero_for_ground_truth(fake_setup):
    _, skel, layout, norm, batch = fake_setup
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 40)
    from src.training.context import global_positions
    gpos = global_positions(pb.x_gt_raw, pb.offsets, skel, layout)
    assert torch.allclose(gpos, pb.gpos_gt, atol=1e-3)                 # 6D 왕복 + FK 가 정답과 같음
    assert rotation_loss(pb.x_gt_raw, pb.x_gt_raw, spec, layout) == 0
    assert position_loss(gpos, pb.gpos_gt, spec) < 1e-3
    assert smooth_loss(gpos, spec) > 0                                 # 정답도 움직이므로 0 이 아님


def test_forward_backward_full_size(fake_setup):
    """설정 파일 그대로의 모델(약 1천만 파라미터)로 shape, 유한한 손실, 모든 파라미터 gradient 확인."""
    _, skel, layout, norm, batch = fake_setup
    cfg = load_model_config("context")
    torch.manual_seed(0)
    model = build_context_transformer(cfg["model"], layout)
    pb = prepare_batch(batch, skel, norm, 10)
    assert pb.x_gt.shape == (4, 42, layout.dim) and pb.gpos_gt.shape == (4, 42, skel.num_joints, 3)
    spec = TransitionSpec(10, 40, midway=(20,))
    losses = context_forward(model, pb, spec, skel, layout, norm, cfg["loss"])
    assert set(losses) == {"loss", "rot", "pos", "smooth"}
    assert all(torch.isfinite(v) for v in losses.values())
    expected = sum(cfg["loss"][f"w_{k}"] * losses[k] for k in ("rot", "pos", "smooth"))
    assert torch.allclose(losses["loss"], expected)
    losses["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_infer_context_shapes(fake_setup):
    _, skel, layout, norm, batch = fake_setup
    model = build_context_transformer(TINY, layout)
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 40)
    y = infer_context(model, pb.x_gt, spec, layout, use_postprocess=True)
    assert y.shape == pb.x_gt.shape and model.training                 # 원래 모드로 돌아옴
    assert torch.equal(y[:, :10], pb.x_gt[:, :10]) and torch.equal(y[:, 40:], pb.x_gt[:, 40:])


def test_tiny_model_overfits(fake_setup):
    """고정 배치 + 고정 transition 에서 손실이 크게 줄어야 한다 (학습 경로 전체가 연결돼 있다는 증거)."""
    _, skel, layout, norm, batch = fake_setup
    torch.manual_seed(0)
    model = build_context_transformer(dict(TINY, d_model=64, d_ff=64, d_encoder_h=64, d_decoder_h=64), layout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 30)
    w = load_model_config("context")["loss"]
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = context_forward(model, pb, spec, skel, layout, norm, w)["loss"]
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < 0.5 * first, (first, float(loss.detach()))


def test_noam_schedule():
    f = noam_lambda(8000)
    assert f(0) == f(1) and f(1) < f(4000) < f(8000)
    assert math.isclose(f(8000), 8000 ** -0.5) and f(32000) == pytest.approx(0.5 * f(8000))
    assert 0.0625 * f(8000) == pytest.approx(6.99e-4, rel=1e-2)


# -----------------------------------------------------------------------------
# 학습 스크립트 (가짜 데이터로 끝까지)
# -----------------------------------------------------------------------------
def test_train_script_fake_data(tmp_path):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_context.py"), "--env", "laptop",
           "--fake-data", "--max-iters", "6", "--batch-size", "4", "--run-name", f"pytest_{tmp_path.name}",
           "--num-workers", "0", "--set", "model.n_layer=1", "model.d_model=32", "model.d_ff=32",
           "model.d_encoder_h=32", "model.d_decoder_h=32", "model.d_head=8", "model.n_head=2",
           "train.log_interval=3", "train.val_interval=6", "train.ckpt_interval=6", "train.val_trans=[5,30]"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    run_dir = PROJECT_ROOT / "experiments" / f"pytest_{tmp_path.name}"
    try:
        assert res.returncode == 0, res.stdout + res.stderr
        assert "val30_loss" in res.stdout
        assert (run_dir / "last.pth").exists() and (run_dir / "best.pth").exists()
        assert (run_dir / "log.csv").exists() and (run_dir / "config.yaml").exists()
    finally:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
