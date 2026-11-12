"""
Detail Transformer 입력·학습·추론 (논문 3.4, 3.5, 3.6)
=====================================================

이 파일이 하는 일
-----------------
학습 스크립트 (scripts/train_detail.py), 평가 (scripts/eval_lengths.py), 테스트가 공통으로 쓰는 "배치 하나 처리" 함수들.

    run_context_frozen : 고정된 Context Transformer → transition 채우기 (+ 후처리 식 23)        (논문 3.6.1, 3.6.2)
    detail_key_mask    : (T, T) 어텐션 마스크 (그림 5b). 모든 프레임이 target 까지의 모든 프레임을 본다
    build_detail_input : x_det = Context 결과 (제약 프레임은 정답) ⊕ m_in → (B, T, D + 1)          (논문 3.4)
    run_detail         : Detail 모델 → 포즈 (transition 만 교체) + 접촉 확률 ĉ = sigmoid(·)     (식 14, 15)
    detail_forward     : 학습 한 스텝의 순전파 + 손실 (식 22)
    infer_two_stage    : 추론: Context → Detail (두 단계 전체)

프레임 번호 예 (context_len 10, trans 30, window 42):
    0..9 context | 10..39 transition | 40 target | 41 (target 다음; Context 후처리용, Detail 은 key 로 못 봄)

[공식코드] target 다음 프레임은 Detail 의 key 에서도 가린다 (논문 그림 5b 는 윈도우 끝까지 그려져 있지 않음). (G2)
[공식코드] Detail 입력을 만들 때 Context 출력은 후처리 (식 23) 한 값을 쓴다. Detail 출력 자체에는 기본으로 후처리를
          하지 않는다 (공식 README 의 Detail 수치도 후처리 없음). (G4)

shape 표기: B = 배치, T = 프레임, J = 관절, D = J*6+3, C = 접촉 관절 수 (LaFAN1: 4).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.training.context import PreparedBatch, global_positions, run_model
from src.training.context_inputs import TransitionSpec, data_mask, merge_prediction, postprocess
from src.training.losses import contact_loss, foot_sliding_loss, position_loss, rotation_loss, weighted_sum


# -----------------------------------------------------------------------------
# 1) 고정된 Context Transformer 로 transition 채우기
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_context_frozen(ctx_model: nn.Module, x_gt: Tensor, spec: TransitionSpec, layout: FeatureLayout,
                       use_postprocess: bool = True) -> Tensor:
    """고정된 Context Transformer 출력 (B, T, D) 정규화된 공간. transition 밖은 정답 (논문 3.6.2).

    use_postprocess=True 이면 식 (23) 후처리 (T ≥ target_idx + 2 필요). gradient 는 흐르지 않는다.
    """
    y = run_model(ctx_model, x_gt, spec, layout)
    return postprocess(y, x_gt, spec) if use_postprocess else y


# -----------------------------------------------------------------------------
# 2) Detail 입력과 마스크
# -----------------------------------------------------------------------------
def detail_key_mask(T: int, spec: TransitionSpec, device=None) -> Tensor:
    """어텐션 마스크 (논문 그림 5b). (T, T) bool, True = query i 가 key j 를 볼 수 있음.

    자리표시자가 없으므로 context·transition·target 프레임이 서로 모두 본다.
    [공식코드] target 이후 프레임 (후처리용 target+1 등) 은 key 에서 가린다. (G2)
    """
    visible = torch.zeros(T, dtype=torch.bool, device=device)
    visible[:spec.target_idx + 1] = True
    return visible[None, :].expand(T, T)


def build_detail_input(x_ctx: Tensor, x_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """Detail 입력 (B, T, D + 1) = x_det ⊕ m_in (논문 3.4).

    입력:
        x_ctx : (B, T, D) Context 결과 (정규화된 공간, transition 밖은 정답)
        x_gt  : (B, T, D) 정규화된 정답
    x_det = x_ctx 에서 제약 (midway) 프레임만 정답으로 되돌린 것 ("constrained values to ground truth").
    m_in 은 Context 와 같은 프레임 마스크 (context·제약·target = 1).
    """
    x_det = x_ctx.clone()
    if spec.midway:
        idx = list(spec.midway)
        x_det[:, idx] = x_gt[:, idx]
    B, T, _ = x_det.shape
    m = data_mask(T, spec, x_det.dtype, x_det.device)              # (T, 1)
    return torch.cat([x_det, m.expand(B, T, 1)], dim=-1)


# -----------------------------------------------------------------------------
# 3) Detail 모델 실행
# -----------------------------------------------------------------------------
def run_detail(det_model: nn.Module, x_ctx: Tensor, x_gt: Tensor, spec: TransitionSpec,
               contact_gt: Tensor | None = None, amp_dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
    """Detail 입력 만들기 → 모델 → transition 만 교체.

    입력:
        x_ctx, x_gt : (B, T, D) 정규화된 공간
        contact_gt  : (B, T, C) 정답 접촉. 주면 transition 밖 접촉은 정답으로 채운다 (공식 코드와 같음).
                      None 이면 모든 프레임이 예측값 (추론 때)
    출력:
        y_det        : (B, T, D) transition 은 Detail 출력, 나머지는 정답 (식 14, 논문 3.6.2 와 같은 방식)
        contact_prob : (B, T, C) ĉ = sigmoid(C(y_det)) (식 15)
    """
    T = x_gt.shape[1]
    x_in = build_detail_input(x_ctx, x_gt, spec)
    mask = detail_key_mask(T, spec, x_gt.device)
    with torch.autocast(device_type=x_gt.device.type, dtype=amp_dtype or torch.float16,
                        enabled=amp_dtype is not None):
        pose, logits = det_model(x_in, mask)
    y_det = merge_prediction(x_gt, pose.to(x_gt.dtype), spec)
    prob = torch.sigmoid(logits.to(x_gt.dtype))
    if contact_gt is not None:
        prob = merge_prediction(contact_gt.to(prob.dtype), prob, spec)
    return y_det, prob


# -----------------------------------------------------------------------------
# 4) 학습 한 스텝
# -----------------------------------------------------------------------------
def detail_forward(
    det_model: nn.Module,
    ctx_model: nn.Module,
    pb: PreparedBatch,
    contact_gt: Tensor,
    spec: TransitionSpec,
    skeleton: Skeleton,
    layout: FeatureLayout,
    normalizer: ZScoreNormalizer,
    loss_weights: dict[str, float],
    ctx_postprocess: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """학습 한 스텝의 순전파 + 손실 (식 22).

    순서: 고정된 Context → (후처리) → Detail 입력 → Detail → 역정규화 → FK → 손실
    입력:  pb (prepare_batch 결과), contact_gt (B, T, C) 정답 접촉 라벨 (LafanDataset 의 contacts)
    출력 dict: {"loss": 가중합, "rot": L_rot, "pos": L_pos, "contact": L_c, "foot": L_foot}
    """
    x_ctx = run_context_frozen(ctx_model, pb.x_gt, spec, layout, ctx_postprocess)
    y_det, prob = run_detail(det_model, x_ctx, pb.x_gt, spec, contact_gt, amp_dtype)
    y_raw = normalizer.denormalize(y_det)
    gpos = global_positions(y_raw, pb.offsets, skeleton, layout)
    losses = {
        "rot": rotation_loss(y_raw, pb.x_gt_raw, spec, layout),
        "pos": position_loss(gpos, pb.gpos_gt, spec),
        "contact": contact_loss(prob, contact_gt, spec),
        "foot": foot_sliding_loss(gpos, prob, skeleton.contact_joints, spec),
    }
    losses["loss"] = weighted_sum(losses, loss_weights)
    return losses


# -----------------------------------------------------------------------------
# 5) 추론 (두 단계 전체)
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_two_stage(
    ctx_model: nn.Module,
    det_model: nn.Module,
    x_gt: Tensor,
    spec: TransitionSpec,
    layout: FeatureLayout,
    ctx_postprocess: bool = True,
    det_postprocess: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """추론: 정규화된 x_gt (아는 프레임만 의미 있음) → Context → Detail.

    출력:  (y_ctx (B,T,D), y_det (B,T,D), contact_prob (B,T,C))  모두 정규화된 공간 / 확률
    det_postprocess=True 이면 Detail 출력에도 식 (23) 후처리 (공식 README 기본값은 끔). (G4)
    """
    was_training = det_model.training
    det_model.eval()
    ctx_model.eval()
    y_ctx = run_context_frozen(ctx_model, x_gt, spec, layout, ctx_postprocess)
    y_det, prob = run_detail(det_model, y_ctx, x_gt, spec)
    if det_postprocess:
        y_det = postprocess(y_det, x_gt, spec)
    det_model.train(was_training)
    return y_ctx, y_det, prob


def contact_accuracy(prob: Tensor, contact_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """transition 프레임에서 ĉ > 0.5 와 정답 라벨이 같은 비율 (검증 기록용). 입력 (B, T, C)."""
    s = spec.seq_slice
    return ((prob[:, s] > 0.5).to(contact_gt.dtype) == contact_gt[:, s]).float().mean()


def foot_indices(skeleton: Skeleton) -> Sequence[int]:
    """발 접촉 관절 번호 (ĉ 의 순서). 스켈레톤 설정의 contact_joints."""
    return skeleton.contact_joints
