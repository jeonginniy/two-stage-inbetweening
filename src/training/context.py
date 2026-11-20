"""
Context Transformer 학습/추론 한 스텝 (논문 3.3, 3.5, 3.6)
========================================================

이 파일이 하는 일
-----------------
학습 스크립트(scripts/train_context.py)와 테스트가 공통으로 쓰는 "배치 하나 처리" 함수들.

    prepare_batch      : LafanDataset 배치 → 정렬된 정답 (x_gt 정규화/원값, 전역 관절 위치 등)
    context_forward    : 입력 만들기 → 모델 → transition 합치기 → 손실 dict
    infer_context      : (학습 없이) 예측 + 선택적 후처리 (식 23). 3단계 Detail 학습/평가에서도 사용
    eval_transition    : 검증/시각화 공통 (자르기 → 정렬 → 추론 → FK)

손실 항은 loss_weights 에 가중치가 있는 것만 계산한다. 기본 (ctx_base) 은 rot/pos/smooth 세 개,
긴 transition 재학습 (docs/assumptions.md J 절) 은 w_vel (정답 속도 L1) 과 w_foot (정답 접촉 × 발 속도) 를 더한다.

차원 정보(관절 수, D)는 모두 Skeleton/FeatureLayout 에서 온다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, ZScoreNormalizer, decode_pose
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root
from src.training.context_inputs import (
    TransitionSpec,
    build_context_input,
    key_mask,
    keyframe_positions,
    merge_prediction,
    postprocess,
)
from src.training.losses import (
    foot_skate_loss,
    foot_sliding_loss,
    position_loss,
    rotation_loss,
    smooth_loss,
    velocity_loss,
    weighted_sum,
)


@dataclass
class PreparedBatch:
    """정렬(논문 3.6.1)까지 끝난 정답 배치.

    x_gt      : (B, T, D) 정규화된 정답
    x_gt_raw  : (B, T, D) 정규화 전 정답
    gpos_gt   : (B, T, J, 3) 정답 전역 관절 위치 (정렬된 좌표, L_pos 용)
    offsets   : (B, 1, J, 3) 뼈 offset (FK 용)
    contacts  : (B, T, C) 정답 발 접촉 라벨 0/1 (배치에 "contacts" 가 있을 때, L_foot 용). 없으면 None
    """

    x_gt: Tensor
    x_gt_raw: Tensor
    gpos_gt: Tensor
    offsets: Tensor
    contacts: Tensor | None = None


def global_positions(x_raw: Tensor, offsets: Tensor, skeleton: Skeleton, layout: FeatureLayout) -> Tensor:
    """정규화 전 벡터 (B, T, D) → 전역 관절 위치 (B, T, J, 3). 미분 가능."""
    root_pos, local_rot = decode_pose(x_raw, layout)
    local_pos = local_positions_from_root(root_pos, offsets)
    return forward_kinematics(local_rot, local_pos, skeleton.parents)[1]


def prepare_batch(
    batch: dict,
    skeleton: Skeleton,
    normalizer: ZScoreNormalizer,
    context_len: int,
    world_forward: str = "x",
    device: torch.device | str | None = None,
) -> PreparedBatch:
    """LafanDataset 배치 (root_pos (B,T,3), local_rot (B,T,J,3,3), offsets (B,J,3), [contacts (B,T,C)])
    → PreparedBatch. 접촉 라벨은 원래 월드 좌표에서 만든 값 (정렬과 무관한 0/1) 을 그대로 옮긴다."""
    root_pos = batch["root_pos"].to(device)
    local_rot = batch["local_rot"].to(device)
    offsets = batch["offsets"].to(device)[:, None]                         # (B, 1, J, 3)
    inp = motion_to_model_input(root_pos, local_rot, context_len, skeleton.root_forward_local_axis,
                                normalizer=normalizer, world_forward=world_forward)
    local_pos = local_positions_from_root(inp.root_pos, offsets)
    gpos_gt = forward_kinematics(inp.local_rot, local_pos, skeleton.parents)[1]
    contacts = batch.get("contacts")
    contacts = contacts.to(device=device, dtype=inp.x.dtype) if contacts is not None else None
    return PreparedBatch(x_gt=inp.x, x_gt_raw=inp.x_raw, gpos_gt=gpos_gt, offsets=offsets, contacts=contacts)


def run_model(
    model: nn.Module,
    x_gt: Tensor,
    spec: TransitionSpec,
    layout: FeatureLayout,
    amp_dtype: torch.dtype | None = None,
) -> Tensor:
    """입력 만들기 → 모델 → transition 합치기. 출력 (B, T, D) 정규화된 공간, x_gt 와 같은 dtype.

    amp_dtype (torch.float16 / bfloat16) 를 주면 모델 부분만 autocast 로 돌린다.
    손실의 FK 는 항상 원래 정밀도로 계산된다.
    """
    T = x_gt.shape[1]
    x_in = build_context_input(x_gt, spec, layout)
    kf = keyframe_positions(T, spec, x_gt.dtype, x_gt.device)
    mask = key_mask(T, spec, x_gt.device)
    with torch.autocast(device_type=x_gt.device.type, dtype=amp_dtype or torch.float16,
                        enabled=amp_dtype is not None):
        out = model(x_in, kf, mask)
    return merge_prediction(x_gt, out.to(x_gt.dtype), spec)


def context_forward(
    model: nn.Module,
    pb: PreparedBatch,
    spec: TransitionSpec,
    skeleton: Skeleton,
    layout: FeatureLayout,
    normalizer: ZScoreNormalizer,
    loss_weights: dict[str, float],
    amp_dtype: torch.dtype | None = None,
    post_mode: str | None = None,
) -> dict[str, Tensor]:
    """학습 한 스텝의 순전파 + 손실 (식 21).

    출력 dict: {"loss": 가중합, "rot": L_rot, "pos": L_pos, "smooth": L_smooth}
               + loss_weights 에 w_vel > 0 이면 "vel" (L_vel), w_foot > 0 이면 "foot" (L_foot, 정답 접촉 필요),
                 w_skate > 0 이면 "skate" (L_skate: 예측 발가락 높이 가중 × 수평 속도, J12)
    post_mode: None (기본, 논문·공식 코드와 같음) 이면 모델 출력 그대로 손실을 계산한다.
               "mean" | "ramp" 이면 평가 때와 같은 후처리 (식 23 / J5) 를 거친 출력에서 손실을 계산한다 [결정 J13].
               후처리는 선형 (미분 가능) 이라 gradient 가 그대로 흐르고, 후처리가 퍼뜨리는 보정 (루트 이동 →
               디딘 발이 미끄러짐) 도 L_foot / L_skate 가 본다. pb.x_gt 에 target 다음 프레임이 있어야 한다.
    """
    y = run_model(model, pb.x_gt, spec, layout, amp_dtype)
    if post_mode is not None:
        y = postprocess(y, pb.x_gt, spec, post_mode)
    y_raw = normalizer.denormalize(y)
    gpos = global_positions(y_raw, pb.offsets, skeleton, layout)
    losses = {
        "rot": rotation_loss(y_raw, pb.x_gt_raw, spec, layout),
        "pos": position_loss(gpos, pb.gpos_gt, spec),
        "smooth": smooth_loss(gpos, spec),
    }
    if float(loss_weights.get("w_vel", 0.0)) > 0:
        losses["vel"] = velocity_loss(gpos, pb.gpos_gt, spec)
    if float(loss_weights.get("w_foot", 0.0)) > 0:
        if pb.contacts is None:
            raise ValueError("w_foot > 0 이면 배치에 정답 발 접촉 라벨 (contacts) 이 있어야 합니다.")
        losses["foot"] = foot_sliding_loss(gpos, pb.contacts, skeleton.contact_joints, spec)
    if float(loss_weights.get("w_skate", 0.0)) > 0:
        # 발가락 = 좌/우 접촉 관절의 마지막 (LaFAN1: LeftToe, RightToe). 발목은 서 있어도 7–8 cm 라서 제외 (F4)
        toes = (skeleton.contact_left[-1], skeleton.contact_right[-1])
        losses["skate"] = foot_skate_loss(gpos, toes, spec, float(loss_weights.get("skate_height_cm", 3.0)))
    losses["loss"] = weighted_sum(losses, loss_weights)
    return losses


@torch.no_grad()
def infer_context(
    model: nn.Module,
    x_gt: Tensor,
    spec: TransitionSpec,
    layout: FeatureLayout,
    use_postprocess: bool = True,
    post_mode: str = "mean",
) -> Tensor:
    """추론: 정규화된 x_gt (아는 프레임만 의미 있음) → 예측 (B, T, D) 정규화된 공간.

    use_postprocess=True 이면 식 (23) 후처리 (T ≥ target_idx + 2 필요). post_mode: "mean" (식 23) | "ramp" (J5)
    """
    was_training = model.training
    model.eval()
    y = run_model(model, x_gt, spec, layout)
    if use_postprocess:
        y = postprocess(y, x_gt, spec, post_mode)
    model.train(was_training)
    return y


@torch.no_grad()
def eval_transition(
    model: nn.Module,
    batch: dict,
    trans: int,
    data_cfg: dict,
    skeleton: Skeleton,
    layout: FeatureLayout,
    normalizer: ZScoreNormalizer,
    device: torch.device | str | None = None,
    post_mode: str = "mean",
) -> tuple[PreparedBatch, TransitionSpec, Tensor, Tensor]:
    """검증/시각화 공통: LafanDataset 배치 → transition `trans` 프레임 생성 (후처리 포함).

    클립을 ctx + trans + 2 프레임 (target 다음 프레임까지, 후처리용) 으로 자른 뒤
    prepare_batch → infer_context → 역정규화 → FK. post_mode = 후처리 방식 (학습 설정 train.postprocess_mode)
    출력: (pb, spec, y 정규화된 예측 (B, L, D), gpos 예측 전역 관절 위치 (B, L, J, 3))
    """
    ctx = data_cfg["context_len"]
    L = ctx + trans + 2
    cut = {k: (v[:, :L] if k in ("root_pos", "local_rot", "contacts") else v) for k, v in batch.items()}
    pb = prepare_batch(cut, skeleton, normalizer, ctx, data_cfg["forward_world_axis"], device)
    spec = TransitionSpec(ctx, ctx + trans)
    y = infer_context(model, pb.x_gt, spec, layout, use_postprocess=True, post_mode=post_mode)
    gpos = global_positions(normalizer.denormalize(y), pb.offsets, skeleton, layout)
    return pb, spec, y, gpos


def gpos_l2(gpos: Tensor, gpos_gt: Tensor, frames: slice) -> Tensor:
    """전역 관절 위치 평균 L2 오차: 관절별 유클리드 거리를 (배치, 프레임, 관절) 전체에서 평균."""
    return (gpos[:, frames] - gpos_gt[:, frames]).norm(dim=-1).mean()
