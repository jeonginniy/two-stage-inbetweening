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
