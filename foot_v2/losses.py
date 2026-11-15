"""
발 전용 모델 v2 의 손실: foot_only 손실 + 발 속도 손실
====================================================

    foot_velocity_loss   예측 발 위치의 프레임 사이 변화량 vs 정답 변화량 (L1, cm/frame)
    foot_forward_v2      foot_only.train_utils.foot_forward 를 그대로 부르고 L_vel 을 더한다

손실 [결정 I12]
    L = w_pos·L_pos + w_contact·L_contact + w_vel·L_vel
    L_vel = mean | (p̂_t − p̂_{t−1}) − (p_t − p_{t−1}) |,   t = t_ctx … target (경계 포함)
        p = 정렬 좌표 발 위치 (cm). transition 밖의 p̂ 는 정답이므로 t = t_ctx 와 t = target 의 항은
        "마지막 context 프레임 / target 프레임과 매끄럽게 이어지는가" 를 본다.
        논문 식 21 의 L_smooth (전신 위치의 프레임 차이) 와 같은 생각을 발 위치에 적용한 것.
    접촉 구간에서는 정답 변화량이 거의 0 이라서 (접촉 정의 < 0.2 cm/frame) "딛은 발은 멈춰 있어라" 가 된다.
    접촉 라벨 정의는 바꾸지 않는다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from foot_only.data import FootBatch
from foot_only.train_utils import foot_forward
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.training.context_inputs import TransitionSpec


def foot_velocity_loss(foot_pred: Tensor, foot_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """발 속도 손실 L_vel.

    입력:  foot_pred, foot_gt : (B, T, 12) 정렬 좌표 발 위치 (cm). foot_pred 의 transition 밖은 정답이어야 함
           spec               : transition = [spec.seq_slice.start, spec.seq_slice.stop), stop = target 프레임
    출력:  스칼라 (cm/frame). 프레임 차이 n + 1 개 (t = start … stop) × 12 좌표의 평균
    """
    a, b = spec.seq_slice.start, spec.seq_slice.stop
    d_pred = foot_pred[:, a:b + 1] - foot_pred[:, a - 1:b]        # (B, n + 1, 12)
    d_gt = foot_gt[:, a:b + 1] - foot_gt[:, a - 1:b]
    return (d_pred - d_gt).abs().mean()


def foot_forward_v2(
    model: nn.Module,
    fb: FootBatch,
    spec: TransitionSpec,
    layout: FeatureLayout,
    foot_norm: ZScoreNormalizer,
    loss_w: dict,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """foot_forward + L_vel. 입력·출력은 foot_forward 와 같고 losses 에 "vel" 이 추가된다.

    losses["loss"] = foot_only 손실 + w_vel × L_vel   (w_vel = loss_w["w_vel"], 없으면 0 → foot_only 와 같음)
    """
    losses, foot_pred, prob = foot_forward(model, fb, spec, layout, foot_norm, loss_w, amp_dtype)
    losses["vel"] = foot_velocity_loss(foot_pred, fb.foot_raw, spec)
    losses["loss"] = losses["loss"] + float(loss_w.get("w_vel", 0.0)) * losses["vel"]
    return losses, foot_pred, prob
