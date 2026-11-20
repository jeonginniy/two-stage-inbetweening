"""
손실 함수 (논문 3.5)
===================

Context Transformer (식 21):  L_ctx = L_rot + w_pos·L_pos + w_smooth·L_smooth
모두 L1, transition 프레임에서만 계산한다 (= TransitionSpec.seq_slice).

    L_rot    : 6D 회전 (정규화를 되돌린 값) 의 L1              (식 18)
    L_pos    : FK 로 구한 전역 관절 위치의 L1 (cm)             (식 17)
    L_smooth : 전역 관절 위치의 프레임 간 속도 L1.             (식 20)
               마지막 context → 첫 transition, 마지막 transition → target 경계도 포함
               [공식코드] 예측 속도 자체를 작게 하는 항 (정답 속도와 비교하지 않음)

Context Transformer 추가 항 (긴 transition 재학습용, 기본 가중치 0 = 기존과 같음, docs/assumptions.md J4):
    L_vel    : 예측 속도와 정답 속도의 L1 (경계 포함)                     velocity_loss
    L_foot   : 정답 발 접촉 라벨 × 발 관절 전역 속도의 L1                 foot_sliding_loss 에 정답 라벨을 넣음
    L_skate  : 예측 발가락 높이 가중 (detach) × 발가락 수평 속도            foot_skate_loss (J12, 평가 지표 F4 와 같은 식)

Detail Transformer (식 22):   L_det = L_rot + w_pos·L_pos + w_contact·L_c + w_foot·L_foot   (3단계에서 추가)

    L_c      : 예측 발 접촉 확률 ĉ 과 정답 접촉 라벨 c 의 L1           (식 19)
    L_foot   : ĉ (gradient 끊음) × 발 관절 전역 속도 의 L1              (식 20)
               ĉ 의 gradient 를 끊지 않으면 ĉ 가 항상 0 으로 가서 손실을 없애 버린다 (논문 3.5.5)

평균은 모든 원소에 대한 평균 (공식 코드와 같음).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from src.data.representation import FeatureLayout
from src.training.context_inputs import TransitionSpec


def rotation_loss(y_raw: Tensor, x_gt_raw: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """L_rot. 입력 (B, T, D) 정규화를 되돌린 값."""
    s = spec.seq_slice
    return (y_raw[:, s, layout.rot_slice] - x_gt_raw[:, s, layout.rot_slice]).abs().mean()


def position_loss(gpos_pred: Tensor, gpos_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """L_pos. 입력 (B, T, J, 3) 전역 관절 위치."""
    s = spec.seq_slice
    return (gpos_pred[:, s] - gpos_gt[:, s]).abs().mean()


def smooth_loss(gpos_pred: Tensor, spec: TransitionSpec) -> Tensor:
    """L_smooth. 프레임 [t_ctx−1, target] 구간의 인접 프레임 차이."""
    seg = gpos_pred[:, spec.context_len - 1: spec.target_idx + 1]           # (B, n+2, J, 3)
    return (seg[:, 1:] - seg[:, :-1]).abs().mean()


def velocity_loss(gpos_pred: Tensor, gpos_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """L_vel [결정 J4]: 예측 속도와 **정답 속도** 의 L1 (cm/frame). 프레임 [t_ctx−1, target] 의 인접 차이.

    L_smooth (식 18) 는 예측 속도 자체를 0 쪽으로 당겨서 (L1 = 총변동) 긴 transition 일수록 "덜 움직이는"
    쪽으로 치우친다 (F10: trans 300 에서 루트 속도가 GT 의 56%). L_vel 은 정답 속도와 비교하므로 치우침이 없고,
    경계 (마지막 context → 첫 transition, 마지막 transition → target) 의 속도도 정답에 맞춘다
    (MotionBricks 의 "velocity loss" 와 같은 역할).
    입력:  gpos_pred, gpos_gt (B, T, J, 3). transition 밖은 정답이므로 경계 차이에만 gradient 가 간다.
    """
    a, b = spec.context_len - 1, spec.target_idx + 1
    vp = gpos_pred[:, a + 1:b] - gpos_pred[:, a:b - 1]                         # (B, n+1, J, 3)
    vg = gpos_gt[:, a + 1:b] - gpos_gt[:, a:b - 1]
    return (vp - vg).abs().mean()


def foot_skate_loss(gpos_pred: Tensor, foot_idx: Sequence[int], spec: TransitionSpec,
                    height_cm: float = 3.0) -> Tensor:
    """L_skate [결정 J12]: **예측 자신의** 발 높이로 정한 가중치 × 발 수평 속도 (cm/frame). 정답 접촉 라벨이 필요 없다.

    "발이 땅 근처 (h < H) 에 있으면 미끄러지면 안 된다". 평가 지표 발 미끄러짐 (F4) 과 같은 식:
        w_t = clip(2 − 2^(h_t / H), 0, 1)   (h = 0 이면 1, h ≥ H 면 0)
        L = mean_t,j  w_t · |XZ_t − XZ_{t−1}|
    w 는 gradient 를 끊는다 (detach). 끊지 않으면 발을 들어 올려 (h ≥ H) 손실을 없애는 쪽 (발이 떠 있음) 으로 배울 수 있다.
    L_foot (정답 접촉) 과 달리 예측한 걸음 박자가 정답과 달라도 "낮은 발은 멈춰라" 가 항상 성립한다 (긴 transition 에서 중요).
    입력:  gpos_pred (B, T, J, 3), foot_idx = 발가락 관절 번호들, spec
    출력:  스칼라
    """
    s = spec.seq_slice
    feet = gpos_pred[..., list(foot_idx), :]                                  # (B, T, F, 3)
    v = (feet[:, s.start:s.stop, :, [0, 2]] - feet[:, s.start - 1:s.stop - 1, :, [0, 2]]).norm(dim=-1)  # (B, n, F)
    w = (2.0 - torch.pow(2.0, feet[:, s, :, 1].detach() / height_cm)).clamp(0.0, 1.0)
    return (w * v).mean()


def contact_loss(contact_prob: Tensor, contact_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """L_c (식 19). 입력: contact_prob (B, T, C) sigmoid 뒤 확률, contact_gt (B, T, C) 0/1 라벨."""
    s = spec.seq_slice
    return (contact_prob[:, s] - contact_gt[:, s]).abs().mean()


def foot_sliding_loss(gpos_pred: Tensor, contact_prob: Tensor, foot_idx: Sequence[int],
                      spec: TransitionSpec) -> Tensor:
    """L_foot (식 20). 접촉 확률 × 발 관절 전역 속도의 L1.

    입력:
        gpos_pred    : (B, T, J, 3) 예측 전역 관절 위치 (transition 밖은 정답)
        contact_prob : (B, T, C) 예측 접촉 확률 ĉ. 여기서 detach (gradient 끊음, 논문 3.5.5)
        foot_idx     : 접촉 관절 번호 C 개 (skeleton.contact_joints, ĉ 와 같은 순서)
    속도 v_t = g_t − g_{t−1} (첫 transition 프레임은 마지막 context 프레임과의 차이). 단위 cm/frame.
    """
    s = spec.seq_slice
    feet = gpos_pred[..., list(foot_idx), :]                                  # (B, T, C, 3)
    vel = feet[:, s.start:s.stop] - feet[:, s.start - 1:s.stop - 1]          # (B, n, C, 3)
    return (contact_prob[:, s, :, None].detach() * vel).abs().mean()


def weighted_sum(losses: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
    """{"rot": ..., "pos": ..., "smooth": ...} 와 {"w_rot": ...} → 합. 가중치가 없는 항은 0 (기록만 됨)."""
    total = torch.zeros((), dtype=next(iter(losses.values())).dtype,
                        device=next(iter(losses.values())).device)
    for name, value in losses.items():
        total = total + float(weights.get(f"w_{name}", 0.0)) * value
    return total
