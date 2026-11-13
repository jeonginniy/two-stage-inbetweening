"""
발 전용 모델 (foot_only/) 테스트: 가짜 데이터, LaFAN1 불필요
==========================================================

확인하는 것:
- 발 관절 번호 = [LeftFoot, LeftToe, RightFoot, RightToe] = contact_joints
- FootBatch: 발 위치 = prepare_batch 의 gpos_gt 발 관절, 접촉 = 데이터셋 라벨
- 모델 입력 148: 전신 부분 = Context 입력과 같음, 발 위치는 아는 프레임만, m_in
- foot_to_world: 원래 월드 좌표 FK 와 일치
- 모델 입출력 차원이 관절 수에서 계산됨, 출력 분리
- foot_forward: transition 밖은 정답, 손실 값 직접 계산과 일치, gradient 가 transition 출력으로만
- 배치 하나 과적합 (손실이 거의 0 까지)
- validate 지표 키와 범위
- 학습 스크립트가 가짜 데이터로 끝까지 돌고, --resume 와 덮어쓰기 거부가 동작
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from foot_only.data import (
    FOOT_DIM, FOOT_JOINT_NAMES, build_foot_input, canonical_frame, compute_foot_stats, foot_indices, foot_to_world,
    prepare_foot_batch,
)
from foot_only.model import FOOT_OUT_DIM, build_foot_model, split_output
from foot_only.train_utils import foot_forward, validate
from src.config import load_data_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root
from src.training.context_inputs import TransitionSpec, build_context_input
from tests.helpers import write_fake_lafan_bvh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = dict(d_mask=1, n_layer=1, n_head=2, d_model=32, d_head=8, d_ff=32,
            d_encoder_h=32, d_decoder_h=32, dropout=0.0, attn_bias=False, attn_scale="d_head")
LOSS_W = {"w_pos": 1.0, "w_contact": 1.0, "contact_pos_weight": None}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """가짜 BVH (subject1, subject5) → (data_cfg, skel, layout, ds, ctx_norm, foot_norm)."""
    tmp = tmp_path_factory.mktemp("foot")
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


def _batch(ds, n=8):
    return next(iter(DataLoader(ds, batch_size=n, shuffle=False)))


def test_foot_indices(fake):
    _, skel, *_ = fake
    idx = foot_indices(skel)
    assert idx == list(skel.contact_joints) == [3, 4, 7, 8]
    assert tuple(skel.names[i] for i in idx) == FOOT_JOINT_NAMES


def test_foot_batch_matches_fk(fake):
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    b = _batch(ds)
    fb = prepare_foot_batch(b, skel, ctx_norm, foot_norm, data_cfg["context_len"])
    B, T = b["root_pos"].shape[:2]
    assert fb.foot_raw.shape == fb.foot.shape == (B, T, FOOT_DIM)
    assert fb.contact.shape == (B, T, 4)
    assert torch.allclose(fb.foot_raw.reshape(B, T, 4, 3), fb.pb.gpos_gt[:, :, [3, 4, 7, 8]])
    assert torch.equal(fb.contact, b["contacts"])
    assert torch.allclose(foot_norm.denormalize(fb.foot), fb.foot_raw, atol=1e-4)
    # 원래 월드 좌표로 되돌린 값 = 원본으로 바로 FK
    world = foot_to_world(fb.foot_raw, canonical_frame(b, skel, data_cfg["context_len"]))
    lp = local_positions_from_root(b["root_pos"], b["offsets"][:, None])
    ref = forward_kinematics(b["local_rot"], lp, skel.parents)[1][:, :, [3, 4, 7, 8]]
    assert (world - ref).abs().max() < 1e-3


def test_build_foot_input(fake):
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    ctx = data_cfg["context_len"]
    fb = prepare_foot_batch(_batch(ds), skel, ctx_norm, foot_norm, ctx)
    spec = TransitionSpec(ctx, ctx + 20, (15,))
    x = build_foot_input(fb.pb.x_gt, fb.foot, spec, layout)
    D = layout.dim
    assert x.shape[-1] == D + FOOT_DIM + 1 == 148
    ref = build_context_input(fb.pb.x_gt, spec, layout)
    assert torch.equal(x[..., :D], ref[..., :D]) and torch.equal(x[..., -1], ref[..., -1])
    known = spec.known_frames()
    unknown = [t for t in range(x.shape[1]) if t not in known]
    assert torch.equal(x[:, known, D:D + FOOT_DIM], fb.foot[:, known])
    assert x[:, unknown, D:D + FOOT_DIM].abs().max() == 0


def test_model_dims_follow_skeleton(fake):
    _, skel, layout, *_ = fake
    m = build_foot_model(TINY, layout)
    assert m.d_in == layout.dim + FOOT_DIM + 1 and m.d_out == FOOT_OUT_DIM == 16
    pos, logit = split_output(torch.zeros(2, 5, 16))
    assert pos.shape == (2, 5, 12) and logit.shape == (2, 5, 4)


def test_foot_forward_losses_and_grad(fake):
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    ctx = data_cfg["context_len"]
    torch.manual_seed(0)
    m = build_foot_model(TINY, layout)
    fb = prepare_foot_batch(_batch(ds), skel, ctx_norm, foot_norm, ctx)
    spec = TransitionSpec(ctx, ctx + 25, (20,))
    losses, foot_pred, prob = foot_forward(m, fb, spec, layout, foot_norm, LOSS_W)
    s = spec.seq_slice
    outside = [t for t in range(fb.foot.shape[1]) if not (s.start <= t < s.stop)]
    assert torch.allclose(foot_pred[:, outside], fb.foot_raw[:, outside], atol=1e-4)
    assert torch.equal(prob[:, outside], fb.contact[:, outside])
    assert torch.allclose(losses["pos"], (foot_pred[:, s] - fb.foot_raw[:, s]).abs().mean())
    logit = torch.logit(prob[:, s].clamp(1e-6, 1 - 1e-6))
    assert losses["contact"].item() == pytest.approx(
        F.binary_cross_entropy_with_logits(logit, fb.contact[:, s]).item(), rel=1e-3)
    assert torch.allclose(losses["loss"], losses["pos"] + losses["contact"])
    losses["loss"].backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_overfit_one_batch(fake):
    """배치 하나를 반복 학습하면 손실이 거의 0 까지 내려가야 한다 (모델·손실 연결 확인)."""
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    ctx = data_cfg["context_len"]
    torch.manual_seed(0)
    m = build_foot_model(dict(TINY, n_layer=2, d_model=64, d_ff=128, d_encoder_h=64, d_decoder_h=64), layout)
    fb = prepare_foot_batch(_batch(ds, 4), skel, ctx_norm, foot_norm, ctx)
    spec = TransitionSpec(ctx, ctx + 15)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(600):
        opt.zero_grad()
        losses, _, _ = foot_forward(m, fb, spec, layout, foot_norm, LOSS_W)
        losses["loss"].backward()
        opt.step()
        first = first or {k: float(v.detach()) for k, v in losses.items()}
    assert float(losses["pos"].detach()) < 0.05 * first["pos"], (first, losses)
    assert float(losses["contact"].detach()) < 0.05, (first, losses)


def test_validate_keys(fake):
    data_cfg, skel, layout, ds, ctx_norm, foot_norm = fake
    m = build_foot_model(TINY, layout)
    res = validate(m, DataLoader(ds, batch_size=8), [5, 30], data_cfg, skel, layout, ctx_norm, foot_norm,
                   LOSS_W, "cpu")
    for t in (5, 30):
        for k in ("loss", "pos", "contact", "foot_l2", "contact_acc", "poscontact_acc"):
            assert f"val{t}_{k}" in res
        for j in FOOT_JOINT_NAMES:
            for k in ("foot_l2", "contact_acc", "contact_f1", "poscontact_acc"):
                assert f"val{t}_{k}_{j}" in res
        assert 0 <= res[f"val{t}_contact_acc"] <= 1 and res[f"val{t}_foot_l2"] >= 0
    assert m.training


def test_train_foot_script_fake_data(tmp_path):
    """가짜 BVH 로 학습 스크립트가 끝까지 돌고, 덮어쓰기 거부·resume 가 동작하는지 (값은 의미 없음)."""
    run_name = f"pytest_foot_{tmp_path.name}"
    base = [sys.executable, str(PROJECT_ROOT / "foot_only" / "train_foot.py"), "--env", "laptop", "--fake-data",
            "--batch-size", "4", "--run-name", run_name, "--num-workers", "0", "--set", "model.n_layer=1",
            "model.d_model=32", "model.d_ff=32", "model.d_encoder_h=32", "model.d_decoder_h=32", "model.d_head=8",
            "model.n_head=2", "train.log_interval=3", "train.val_interval=6", "train.ckpt_interval=6",
            "train.val_trans=[5,30]"]
    run_dir = PROJECT_ROOT / "experiments" / run_name
    try:
        r = subprocess.run(base + ["--max-iters", "6"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "val30_loss" in r.stdout and "val30_poscontact_acc" in r.stdout
        for f in ("last.pth", "best.pth", "log.csv", "config.yaml"):
            assert (run_dir / f).exists(), f
        import yaml
        saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert saved["data"]["context_len"] == 10 and "foot_stats_file" in saved["foot_data"]   # 두 data 절 모두 저장
        r = subprocess.run(base + ["--max-iters", "9"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "이미 있습니다" in (r.stdout + r.stderr)
        r = subprocess.run(base + ["--max-iters", "9", "--resume", "last"], cwd=PROJECT_ROOT, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0 and "iteration 6" in r.stdout and "끝: iteration 9" in r.stdout, r.stdout + r.stderr
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
