"""
Context Transformer 학습/추론 한 스텝 (논문 3.3, 3.5, 3.6)
========================================================

이 파일이 하는 일
-----------------
학습 스크립트(scripts/train_context.py)와 테스트가 공통으로 쓰는 "배치 하나 처리" 함수들.

    prepare_batch      : LafanDataset 배치 → 정렬된 정답 (x_gt 정규화/원값, 전역 관절 위치 등)
    context_forward    : 입력 만들기 → 모델 → transition 합치기 → 손실 dict
    infer_context      : (학습 없이) 예측 + 선택적 후처리 (식 23). 3단계 Detail 학습/평가에서도 사용

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
from src.training.losses import position_loss, rotation_loss, smooth_loss, weighted_sum


@dataclass
class PreparedBatch:
    """정렬(논문 3.6.1)까지 끝난 정답 배치.

    x_gt      : (B, T, D) 정규화된 정답
    x_gt_raw  : (B, T, D) 정규화 전 정답
    gpos_gt   : (B, T, J, 3) 정답 전역 관절 위치 (정렬된 좌표, L_pos 용)
    offsets   : (B, 1, J, 3) 뼈 offset (FK 용)
    """

    x_gt: Tensor
    x_gt_raw: Tensor
    gpos_gt: Tensor
    offsets: Tensor


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
    """LafanDataset 배치 (root_pos (B,T,3), local_rot (B,T,J,3,3), offsets (B,J,3)) → PreparedBatch."""
    root_pos = batch["root_pos"].to(device)
    local_rot = batch["local_rot"].to(device)
    offsets = batch["offsets"].to(device)[:, None]                         # (B, 1, J, 3)
    inp = motion_to_model_input(root_pos, local_rot, context_len, skeleton.root_forward_local_axis,
                                normalizer=normalizer, world_forward=world_forward)
    local_pos = local_positions_from_root(inp.root_pos, offsets)
    gpos_gt = forward_kinematics(inp.local_rot, local_pos, skeleton.parents)[1]
    return PreparedBatch(x_gt=inp.x, x_gt_raw=inp.x_raw, gpos_gt=gpos_gt, offsets=offsets)


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
) -> dict[str, Tensor]:
    """학습 한 스텝의 순전파 + 손실 (식 21).

    출력 dict: {"loss": 가중합, "rot": L_rot, "pos": L_pos, "smooth": L_smooth}
    """
    y = run_model(model, pb.x_gt, spec, layout, amp_dtype)
    y_raw = normalizer.denormalize(y)
    gpos = global_positions(y_raw, pb.offsets, skeleton, layout)
    losses = {
        "rot": rotation_loss(y_raw, pb.x_gt_raw, spec, layout),
        "pos": position_loss(gpos, pb.gpos_gt, spec),
        "smooth": smooth_loss(gpos, spec),
    }
    losses["loss"] = weighted_sum(losses, loss_weights)
    return losses


@torch.no_grad()
def infer_context(
    model: nn.Module,
    x_gt: Tensor,
    spec: TransitionSpec,
    layout: FeatureLayout,
    use_postprocess: bool = True,
) -> Tensor:
    """추론: 정규화된 x_gt (아는 프레임만 의미 있음) → 예측 (B, T, D) 정규화된 공간.

    use_postprocess=True 이면 식 (23) 후처리 (T ≥ target_idx + 2 필요).
    """
    was_training = model.training
    model.eval()
    y = run_model(model, x_gt, spec, layout)
    if use_postprocess:
        y = postprocess(y, x_gt, spec)
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
) -> tuple[PreparedBatch, TransitionSpec, Tensor, Tensor]:
    """검증/시각화 공통: LafanDataset 배치 → transition `trans` 프레임 생성 (후처리 포함).

    클립을 ctx + trans + 2 프레임 (target 다음 프레임까지, 후처리용) 으로 자른 뒤
    prepare_batch → infer_context → 역정규화 → FK.
    출력: (pb, spec, y 정규화된 예측 (B, L, D), gpos 예측 전역 관절 위치 (B, L, J, 3))
    """
    ctx = data_cfg["context_len"]
    L = ctx + trans + 2
    cut = {k: (v[:, :L] if k in ("root_pos", "local_rot") else v) for k, v in batch.items()}
    pb = prepare_batch(cut, skeleton, normalizer, ctx, data_cfg["forward_world_axis"], device)
    spec = TransitionSpec(ctx, ctx + trans)
    y = infer_context(model, pb.x_gt, spec, layout, use_postprocess=True)
    gpos = global_positions(normalizer.denormalize(y), pb.offsets, skeleton, layout)
    return pb, spec, y, gpos


def gpos_l2(gpos: Tensor, gpos_gt: Tensor, frames: slice) -> Tensor:
    """전역 관절 위치 평균 L2 오차: 관절별 유클리드 거리를 (배치, 프레임, 관절) 전체에서 평균."""
    return (gpos[:, frames] - gpos_gt[:, frames]).norm(dim=-1).mean()
