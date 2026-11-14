"""
발 → 몸 모델 (feet_to_body/) 테스트: 가짜 데이터, LaFAN1 불필요
=============================================================

확인하는 것:
- 발 조건 = 정규화된 발 위치 12 ⊕ 접촉 라벨 4 (발 전용 모델 출력과 같은 형식)
- 모델 입력 152: 전신 부분·m_in = Context 입력과 같음, 발 조건은 모든 프레임에 그대로
- 어텐션 마스크: attend_all 이면 None, 아니면 Context 와 같은 마스크
- 모델 입출력 차원이 관절 수에서 계산됨
- body_forward: transition 밖은 정답, 손실 = 식 21 가중합, 발 조건이 transition 출력에 영향을 줌
  (attend_all=False 이면 한 프레임의 발 조건은 그 프레임 출력에만 영향)
- 정답 전신을 넣으면 body_foot_error = 0
- 배치 하나 과적합 (손실과 발 오차가 크게 줄어듦)
- validate 지표 키
- 학습 스크립트가 가짜 데이터로 끝까지 돌고, --resume 와 덮어쓰기 거부가 동작
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from feet_to_body.data import COND_DIM, attention_mask, build_body_input, foot_condition
from feet_to_body.model import build_body_model
from feet_to_body.train_utils import body_foot_error, body_forward, run_body_model, validate
from foot_only.data import FOOT_DIM, FOOT_JOINT_NAMES, compute_foot_stats, prepare_foot_batch
from src.config import load_data_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.training.context_inputs import TransitionSpec, build_context_input, key_mask
from tests.helpers import write_fake_lafan_bvh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = dict(d_mask=1, n_layer=1, n_head=2, d_model=32, d_head=8, d_ff=32,
            d_encoder_h=32, d_decoder_h=32, dropout=0.0, attn_bias=False, attn_scale="d_head")
LOSS_W = {"w_rot": 1.0, "w_pos": 0.01, "w_smooth": 0.005}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """가짜 BVH (subject1, subject5) → (data_cfg, skel, layout, ds, ctx_norm, foot_norm)."""
    tmp = tmp_path_factory.mktemp("body")
    for i, stem in enumerate(["walk1_subject1", "run1_subject5"]):
        write_fake_lafan_bvh(tmp / f"{stem}.bvh", 120, seed=i)
    data_cfg = load_data_config()
    skel = Skeleton.from_yaml(PROJECT_ROOT / data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    ds = LafanDataset(tmp, ["subject1", "subject5"], 42, 10, skel,
                      contact_threshold=data_cfg["contact_vel_threshold"], verbose=False)
    st = RunningStats(layout.dim)
    for b in DataLoader(ds, batch_size=32):
        st.update(motion_to_model_input(b["root_pos"], b["local_rot"], data_cfg["context_len"],
                                        skel.root_forward_local_axis).x_raw)
    ctx_norm = ZScoreNormalizer.from_stats(st)
    foot_norm = ZScoreNormalizer.from_stats(
        compute_foot_stats(DataLoader(ds, batch_size=32), skel, ctx_norm, data_cfg["context_len"]))
    return data_cfg, skel, layout, ds, ctx_norm, foot_norm


def _fb(fake, n=8):
    data_cfg, skel, _, ds, ctx_norm, foot_norm = fake
    batch = next(iter(DataLoader(ds, batch_size=n, shuffle=False)))
    return prepare_foot_batch(batch, skel, ctx_norm, foot_norm, data_cfg["context_len"])


def test_foot_condition(fake):
    fb = _fb(fake)
    cond = foot_condition(fb)
    assert cond.shape == (*fb.foot.shape[:2], COND_DIM) and COND_DIM == 16
    assert torch.equal(cond[..., :FOOT_DIM], fb.foot) and torch.equal(cond[..., FOOT_DIM:], fb.contact)


def test_build_body_input(fake):
    data_cfg, _, layout, *_ = fake
    fb = _fb(fake)
    ctx = data_cfg["context_len"]
    spec = TransitionSpec(ctx, ctx + 20, (ctx + 7,))
    cond = foot_condition(fb)
    x_in = build_body_input(fb.pb.x_gt, cond, spec, layout)
    ref = build_context_input(fb.pb.x_gt, spec, layout)                      # (B, T, D + 1)
    D = layout.dim
    assert x_in.shape == (*fb.pb.x_gt.shape[:2], D + COND_DIM + 1)
    assert torch.equal(x_in[..., :D], ref[..., :D])                           # 전신 부분 = Context 입력
    assert torch.equal(x_in[..., D:D + COND_DIM], cond)                       # 발 조건: 모든 프레임 그대로
    assert torch.equal(x_in[..., -1:], ref[..., -1:])                         # m_in
    with pytest.raises(ValueError):
        build_body_input(fb.pb.x_gt, cond[..., :FOOT_DIM], spec, layout)


def test_attention_mask():
    spec = TransitionSpec(10, 25)
    assert attention_mask(42, spec, True) is None
    assert torch.equal(attention_mask(42, spec, False), key_mask(42, spec))


def test_model_dims_follow_skeleton(fake):
    *_, layout, _, _, _ = fake
    m = build_body_model(TINY, layout)
    assert m.d_in == layout.dim + COND_DIM + 1 and m.d_out == layout.dim
    m7 = build_body_model(TINY, FeatureLayout(7))
    assert m7.d_in == 7 * 6 + 3 + COND_DIM + 1 and m7.d_out == 7 * 6 + 3


def test_body_forward_losses_and_condition(fake):
    data_cfg, skel, layout, _, ctx_norm, _ = fake
    fb = _fb(fake)
    ctx = data_cfg["context_len"]
    spec = TransitionSpec(ctx, ctx + 20)
    s = spec.seq_slice
    torch.manual_seed(0)
    m = build_body_model(TINY, layout).eval()
    cond = foot_condition(fb)
    losses, y, gpos = body_forward(m, fb, cond, spec, skel, layout, ctx_norm, LOSS_W)
    T = y.shape[1]
    outside = [t for t in range(T) if not (s.start <= t < s.stop)]
    assert torch.equal(y[:, outside], fb.pb.x_gt[:, outside])                # transition 밖은 정답
    assert gpos.shape == fb.pb.gpos_gt.shape
    want = sum(LOSS_W[f"w_{k}"] * losses[k] for k in ("rot", "pos", "smooth"))
    assert torch.allclose(losses["loss"], want)

    # 발 조건을 transition 프레임 하나에서 바꾸면 출력이 바뀐다
    j = ctx + 5
    cond2 = cond.clone()
    cond2[:, j] += 1.0
    with torch.no_grad():
        y_all = run_body_model(m, fb.pb.x_gt, cond2, spec, layout, attend_all=True)
        base_local = run_body_model(m, fb.pb.x_gt, cond, spec, layout, attend_all=False)
        y_local = run_body_model(m, fb.pb.x_gt, cond2, spec, layout, attend_all=False)
    diff_all = (y_all - y).abs().amax(dim=(0, 2))                             # (T,)
    assert diff_all[j] > 1e-4 and diff_all[j + 3] > 1e-6                      # 다른 프레임에도 영향 (모든 프레임을 봄)
    diff_local = (y_local - base_local).abs().amax(dim=(0, 2))
    assert diff_local[j] > 1e-4
    others = [t for t in range(T) if t != j]
    assert float(diff_local[others].max()) < 1e-6                             # key 마스크: 그 프레임에만 영향


def test_body_foot_error_zero_for_gt(fake):
    _, skel, *_ = fake
    fb = _fb(fake)
    err = body_foot_error(fb.pb.gpos_gt, fb.foot_raw, skel, slice(10, 30))
    assert err.shape == (fb.foot.shape[0], 20, 4) and float(err.max()) < 1e-4


def test_overfit_one_batch(fake):
    """배치 하나를 반복 학습하면 손실과 발 오차가 크게 줄어야 한다 (모델·손실 연결 확인)."""
    data_cfg, skel, layout, _, ctx_norm, _ = fake
    ctx = data_cfg["context_len"]
    torch.manual_seed(0)
    m = build_body_model(dict(TINY, n_layer=2, d_model=64, d_ff=128, d_encoder_h=64, d_decoder_h=64), layout)
    fb = _fb(fake, 4)
    spec = TransitionSpec(ctx, ctx + 15)
    cond = foot_condition(fb)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(600):
        opt.zero_grad()
        losses, _, gpos = body_forward(m, fb, cond, spec, skel, layout, ctx_norm, LOSS_W)
        losses["loss"].backward()
        opt.step()
        if first is None:
            first = {k: float(v.detach()) for k, v in losses.items()}
            first_foot = float(body_foot_error(gpos.detach(), fb.foot_raw, skel, spec.seq_slice).mean())
    last_foot = float(body_foot_error(gpos.detach(), fb.foot_raw, skel, spec.seq_slice).mean())
    assert float(losses["loss"].detach()) < 0.1 * first["loss"], (first, losses)
    assert last_foot < 0.1 * first_foot, (first_foot, last_foot)


def test_validate_keys(fake):
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    m = build_body_model(TINY, layout)
    res = validate(m, DataLoader(ds, batch_size=8), [5, 30], data_cfg, skel, layout, ctx_norm, foot_norm,
                   LOSS_W, True, "cpu")
    for t in (5, 30):
        for k in ("loss", "rot", "pos", "smooth", "gpos_l2", "gpos_l2_nopost", "foot_l2"):
            assert f"val{t}_{k}" in res and res[f"val{t}_{k}"] >= 0
        for j in FOOT_JOINT_NAMES:
            assert f"val{t}_foot_l2_{j}" in res
    assert m.training


def test_train_body_script_fake_data(tmp_path):
    """가짜 BVH 로 학습 스크립트가 끝까지 돌고, 덮어쓰기 거부·resume 가 동작하는지 (값은 의미 없음)."""
    run_name = f"pytest_body_{tmp_path.name}"
    base = [sys.executable, str(PROJECT_ROOT / "feet_to_body" / "train_body.py"), "--env", "laptop", "--fake-data",
            "--batch-size", "4", "--run-name", run_name, "--num-workers", "0", "--set", "model.n_layer=1",
            "model.d_model=32", "model.d_ff=32", "model.d_encoder_h=32", "model.d_decoder_h=32", "model.d_head=8",
            "model.n_head=2", "train.log_interval=3", "train.val_interval=6", "train.ckpt_interval=6",
            "train.val_trans=[5,30]"]
    run_dir = PROJECT_ROOT / "experiments" / run_name
    try:
        r = subprocess.run(base + ["--max-iters", "6"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "val30_loss" in r.stdout and "val30_foot_l2" in r.stdout and "입력 152 → 출력 135" in r.stdout
        for f in ("last.pth", "best.pth", "log.csv", "config.yaml"):
            assert (run_dir / f).exists(), f
        import yaml
        saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert saved["data"]["context_len"] == 10 and saved["cond"]["attend_all"] is True
        r = subprocess.run(base + ["--max-iters", "9"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "이미 있습니다" in (r.stdout + r.stderr)
        r = subprocess.run(base + ["--max-iters", "9", "--resume", "last"], cwd=PROJECT_ROOT, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0 and "iteration 6" in r.stdout and "끝: iteration 9" in r.stdout, r.stdout + r.stderr
        r = subprocess.run([*base[:2], "--env", "laptop", "--fake-data", "--run-name", "foot_only"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "기존 실험 폴더" in (r.stdout + r.stderr)   # 기존 실험 폴더 보호
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 평가 (feet_to_body/evaluate.py)
# -----------------------------------------------------------------------------
def test_predict_bodies(fake):
    """방법별 전신 예측: shape, transition 밖 = 정답, 정답 발 조건 = 정답 발, 예측한 발 조건은 transition 에서만 다름."""
    from feet_to_body.evaluate import BodyRun, predict_bodies
    from foot_only.evaluate import FootRun
    from foot_only.model import build_foot_model
    from src.evaluation.inference import LoadedRun
    from src.models.context_transformer import build_context_transformer
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    torch.manual_seed(0)
    run = LoadedRun({}, data_cfg, {"num_workers": 0}, torch.device("cpu"), skel, layout, ctx_norm,
                    build_context_transformer(TINY, layout).eval(), {}, Path("fake/ctx"))
    foot = FootRun({}, build_foot_model(TINY, layout).eval(), ctx_norm, foot_norm,
                   {"w_pos": 1.0, "w_contact": 1.0, "contact_pos_weight": None}, {}, Path("fake/foot"))
    body = BodyRun({}, build_body_model(TINY, layout).eval(), ctx_norm, foot_norm, True, {}, Path("fake/body"))
    methods = ["body_gt", "body_gt_post", "body_pred", "body_pred_post", "ctx_post", "ctx_nopost", "interp"]
    batch = next(iter(DataLoader(ds, batch_size=6, shuffle=False)))
    bp = predict_bodies(batch, 20, run, methods, body=body, foot=foot)
    L = data_cfg["context_len"] + 22
    s = bp.spec.seq_slice
    assert set(bp.preds) == set(methods)
    assert set(bp.cond_feet) == {"body_gt", "body_gt_post", "body_pred", "body_pred_post"}
    for m, x in bp.preds.items():
        assert x.shape == (6, L, layout.dim), m
        if not m.endswith("_post") and m != "ctx_post":                       # 후처리는 target 다음 프레임도 바꿈
            assert torch.allclose(x[:, :s.start], bp.pb.x_gt_raw[:, :s.start], atol=1e-4), m
            assert torch.allclose(x[:, s.stop:], bp.pb.x_gt_raw[:, s.stop:], atol=1e-4), m
    feet_gt = bp.pb.gpos_gt[:, :, skel.contact_joints].reshape(6, L, -1)
    assert torch.allclose(bp.cond_feet["body_gt"], feet_gt)
    d = (bp.cond_feet["body_pred"] - feet_gt).abs().amax(dim=(0, 2))          # (L,)
    assert float(d[:s.start].max()) < 1e-3 and float(d[s.stop:].max()) < 1e-3 and float(d[s].max()) > 1e-2
    assert not torch.allclose(bp.preds["body_gt"], bp.preds["body_pred"])     # 발 조건이 다르면 출력이 다름
