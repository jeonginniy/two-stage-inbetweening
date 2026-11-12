"""
3단계 Detail Transformer 테스트 — 가짜 데이터로 실행 (LaFAN1 불필요)

확인하는 것:
- 입력/출력 차원이 스켈레톤에서 계산되는가 (입력 D+1, 포즈 D, 접촉 C = contact_joints 수)
- 어텐션 마스크 (그림 5b): target 까지 모두 보고, target 이후는 못 봄 → target 이후 값을 바꿔도 출력 불변
- Detail 입력: transition 은 Context 결과, 제약 (midway) 프레임은 정답, m_in 은 Context 와 같음
- 고정된 Context 에는 gradient 가 가지 않고, Detail 의 모든 파라미터에는 간다
- L_foot 는 ĉ 로 gradient 를 보내지 않는다 (논문 3.5.5), L_c 는 보낸다
- 손실 손계산 (L_c, L_foot)
- 작은 모델이 고정 배치에 과적합되는가
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.config import load_data_config, load_model_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.models.context_transformer import build_context_transformer
from src.models.detail_transformer import build_detail_transformer, check_frozen, freeze
from src.training.context import prepare_batch
from src.training.context_inputs import TransitionSpec, data_mask
from src.training.detail import (
    build_detail_input,
    detail_forward,
    detail_key_mask,
    infer_two_stage,
    run_context_frozen,
    run_detail,
)
from src.training.losses import contact_loss, foot_sliding_loss
from tests.helpers import SKELETON_YAML, write_fake_lafan_bvh

TINY = dict(d_mask=1, n_layer=2, n_head=2, d_model=32, d_head=8, d_ff=32,
            d_encoder_h=32, d_decoder_h=32, dropout=0.0, attn_bias=False, attn_scale="d_head")


@pytest.fixture(scope="module")
def fake_setup(tmp_path_factory):
    """가짜 LaFAN1 폴더 → 배치 (contacts 포함), 정규화기, 고정된 작은 Context 모델."""
    folder = tmp_path_factory.mktemp("fake_lafan1_detail")
    for i, stem in enumerate(["walk1_subject1", "run1_subject2"]):
        write_fake_lafan_bvh(folder / f"{stem}.bvh", 120, seed=i)
    skel = Skeleton.from_yaml(SKELETON_YAML)
    layout = FeatureLayout(skel.num_joints)
    ds = LafanDataset(folder, ["subject1", "subject2"], 42, 5, skel, verbose=False)
    stats = RunningStats(layout.dim)
    for b in DataLoader(ds, batch_size=32):
        stats.update(motion_to_model_input(b["root_pos"], b["local_rot"], 10, skel.root_forward_local_axis).x_raw)
    norm = ZScoreNormalizer.from_stats(stats).to(dtype=torch.float32)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    torch.manual_seed(0)
    ctx = freeze(build_context_transformer(TINY, layout))
    return skel, layout, norm, batch, ctx


def test_config_dims_come_from_skeleton():
    cfg = load_model_config("detail")
    assert "d_in" not in cfg["model"] and "d_out" not in cfg["model"]
    skel = Skeleton.from_yaml(load_data_config()["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    model = build_detail_transformer(cfg["model"], layout, len(skel.contact_joints))
    assert (model.d_in, model.d_pose, model.n_contact, model.d_out) == (136, 135, 4, 139)
    for key in ("w_rot", "w_pos", "w_contact", "w_foot"):
        assert key in cfg["loss"]
    assert "ckpt" in cfg["context"]


@pytest.mark.parametrize("J,C", [(5, 2), (22, 4)])
def test_model_shapes(J, C):
    layout = FeatureLayout(J)
    model = build_detail_transformer(TINY, layout, C)
    x = torch.randn(3, 20, layout.dim + 1)
    pose, logits = model(x, detail_key_mask(20, TransitionSpec(4, 12)))
    assert pose.shape == (3, 20, layout.dim) and logits.shape == (3, 20, C)


def test_detail_mask_and_input():
    spec = TransitionSpec(4, 12, midway=(7,))
    m = detail_key_mask(16, spec)
    assert m[:, :13].all() and not m[:, 13:].any()                  # key 0..target(12) 만
    x_gt = torch.randn(2, 16, 9)
    x_ctx = torch.randn(2, 16, 9)
    inp = build_detail_input(x_ctx, x_gt, spec)
    assert inp.shape == (2, 16, 10)
    assert torch.equal(inp[:, 7, :9], x_gt[:, 7])                    # 제약 프레임 = 정답
    assert torch.equal(inp[:, 8, :9], x_ctx[:, 8])                   # 나머지 transition = Context 결과
    assert torch.equal(inp[0, :, 9:], data_mask(16, spec))            # m_in 은 Context 와 같음


def test_target_after_frames_do_not_matter():
    """target 이후 프레임 값을 바꿔도 (key 마스크) 0..target 프레임 출력이 그대로여야 한다."""
    layout = FeatureLayout(5)
    torch.manual_seed(1)
    model = build_detail_transformer(TINY, layout, 2).eval()
    spec = TransitionSpec(4, 12)
    x_gt = torch.randn(1, 16, layout.dim)
    x_ctx = x_gt.clone()
    y1, c1 = run_detail(model, x_ctx, x_gt, spec)
    x_gt2, x_ctx2 = x_gt.clone(), x_ctx.clone()
    x_gt2[:, 13:] += 5.0
    x_ctx2[:, 13:] += 5.0
    y2, c2 = run_detail(model, x_ctx2, x_gt2, spec)
    assert torch.allclose(y1[:, spec.seq_slice], y2[:, spec.seq_slice], atol=1e-5)
    assert torch.allclose(c1[:, :13], c2[:, :13], atol=1e-5)


def test_contact_and_foot_losses_by_hand():
    spec = TransitionSpec(2, 4)                                       # transition 프레임 2, 3
    prob = torch.zeros(1, 5, 1)
    prob[:, 2:4] = torch.tensor([[[0.25], [1.0]]])
    gt = torch.ones(1, 5, 1)
    assert contact_loss(prob, gt, spec).item() == pytest.approx((0.75 + 0.0) / 2)
    g = torch.zeros(1, 5, 2, 3)
    g[:, :, 1, 0] = torch.tensor([0.0, 0.0, 2.0, 2.0, 2.0])           # 관절 1: 프레임 1→2 에 x 로 2 이동
    # 프레임 2: |0.25·(2,0,0)| → 원소 (0.5, 0, 0), 프레임 3: 속도 0 → 평균 = 0.5 / (2 프레임 × 1 관절 × 3)
    assert foot_sliding_loss(g, prob, [1], spec).item() == pytest.approx(0.5 / 6)


def test_foot_loss_does_not_push_contact_gradient():
    spec = TransitionSpec(2, 4)
    logits = torch.zeros(1, 5, 1, requires_grad=True)
    g = torch.randn(1, 5, 2, 3, requires_grad=True)
    foot_sliding_loss(g, torch.sigmoid(logits), [1], spec).backward()
    assert logits.grad is None or torch.all(logits.grad == 0)        # ĉ 로는 gradient 없음
    assert g.grad is not None and g.grad.abs().sum() > 0              # 위치로는 있음


def test_frozen_context_gets_no_gradient(fake_setup):
    skel, layout, norm, batch, ctx = fake_setup
    check_frozen(ctx)
    det = build_detail_transformer(TINY, layout, len(skel.contact_joints))
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 30, midway=(20,))
    w = load_model_config("detail")["loss"]
    losses = detail_forward(det, ctx, pb, batch["contacts"], spec, skel, layout, norm, w)
    assert set(losses) == {"loss", "rot", "pos", "contact", "foot"}
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    assert all(p.grad is None for p in ctx.parameters())
    assert all(p.grad is not None for p in det.parameters())


def test_infer_two_stage_keeps_known_frames(fake_setup):
    skel, layout, norm, batch, ctx = fake_setup
    det = build_detail_transformer(TINY, layout, len(skel.contact_joints))
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 25)
    y_ctx, y_det, prob = infer_two_stage(ctx, det, pb.x_gt, spec, layout)
    assert y_det.shape == pb.x_gt.shape and prob.shape == (4, 42, len(skel.contact_joints))
    known = [*range(10), 25]
    assert torch.allclose(y_det[:, known], pb.x_gt[:, known])
    assert torch.allclose(y_ctx, run_context_frozen(ctx, pb.x_gt, spec, layout))
    assert ((prob >= 0) & (prob <= 1)).all()


def test_tiny_detail_overfits(fake_setup):
    """고정 배치 + 고정 transition 에서 손실이 크게 줄어야 한다."""
    skel, layout, norm, batch, ctx = fake_setup
    torch.manual_seed(0)
    det = build_detail_transformer(dict(TINY, d_model=64, d_ff=64, d_encoder_h=64, d_decoder_h=64),
                                   layout, len(skel.contact_joints))
    opt = torch.optim.Adam(det.parameters(), lr=1e-3)
    pb = prepare_batch(batch, skel, norm, 10)
    spec = TransitionSpec(10, 30)
    w = load_model_config("detail")["loss"]
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = detail_forward(det, ctx, pb, batch["contacts"], spec, skel, layout, norm, w)["loss"]
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < 0.5 * first, (first, float(loss.detach()))
