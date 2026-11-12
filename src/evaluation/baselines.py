"""
기준선 (baseline): 보간 (SLERP) 과 zero-velocity (4단계, 논문 4.1 / Harvey et al. 2020)
=====================================================================================

이 파일이 하는 일
-----------------
학습이 필요 없는 두 기준선으로 transition 을 채운다. 공식 README 의 inter_* / zerov_* 수치와 비교해서
"평가 지표 구현이 맞는지" 를 모델과 독립적으로 확인하는 데 쓴다 (scripts/eval_lengths.py).

    interpolation : 루트 위치 = 마지막 context ↔ target 선형보간, 로컬 회전 = 쿼터니언 SLERP (짧은 쪽 호)
    zero_velocity : transition 전체를 마지막 context 프레임 포즈로 (멈춰 있다가 target 에서 순간이동)

입력/출력은 모두 "정렬된 원값" 표현 (정규화 전, (B, T, D), D = J*6+3). context·target 및 그 이후는 정답 그대로.
루트가 아닌 관절의 로컬 위치는 항상 뼈 offset 이므로 (논문 3.2) 회전과 루트 위치만 채우면 된다.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.representation import FeatureLayout, decode_pose, encode_pose
from src.data.rotations import matrix_to_quat, quat_to_matrix
from src.training.context_inputs import TransitionSpec


def slerp(q0: Tensor, q1: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """쿼터니언 구면 선형보간 (짧은 쪽 호).

    입력:  q0, q1 : (..., 4) 단위 쿼터니언 (w, x, y, z),  w : q0 와 브로드캐스트되는 (..., 1) 가중치 (0 → q0, 1 → q1)
    출력:  (..., 4) 단위 쿼터니언
    """
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)                         # q 와 −q 는 같은 회전 → 짧은 쪽 호
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    near = s < eps                                             # 거의 같은 회전이면 선형보간
    a = torch.where(near, 1 - w, torch.sin((1 - w) * theta) / s.clamp_min(eps))
    b = torch.where(near, w, torch.sin(w * theta) / s.clamp_min(eps))
    q = a * q0 + b * q1
    return q / q.norm(dim=-1, keepdim=True)


def _splice(x_raw: Tensor, root: Tensor, rot: Tensor, spec: TransitionSpec) -> Tensor:
    """transition 구간만 (root (B,n,3), rot (B,n,J,3,3)) 로 바꾼 표현 (B, T, D)."""
    s = spec.seq_slice
    fill = encode_pose(root, rot)                              # (B, n, D)
    return torch.cat([x_raw[:, :s.start], fill, x_raw[:, s.stop:]], dim=1)


def interpolation(x_raw: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """보간 기준선. 입력/출력: (B, T, D) 정렬된 원값.

    transition 프레임 t 의 가중치 w = (t − (t_ctx − 1)) / (t_tgt − (t_ctx − 1))  (양 끝 0, 1 은 context/target 자체)
    """
    root, rot = decode_pose(x_raw, layout)                     # (B,T,3), (B,T,J,3,3)
    a, b = spec.context_len - 1, spec.target_idx
    t = torch.arange(spec.context_len, b, device=x_raw.device, dtype=x_raw.dtype)
    w = ((t - a) / (b - a))[None, :, None]                     # (1, n, 1)
    root_i = root[:, a:a + 1] + w * (root[:, b:b + 1] - root[:, a:a + 1])
    q0, q1 = matrix_to_quat(rot[:, a:a + 1]), matrix_to_quat(rot[:, b:b + 1])   # (B, 1, J, 4)
    rot_i = quat_to_matrix(slerp(q0, q1, w[..., None]))       # (B, n, J, 3, 3)
    return _splice(x_raw, root_i, rot_i, spec)


def zero_velocity(x_raw: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """zero-velocity 기준선. 입력/출력: (B, T, D). transition = 마지막 context 프레임 포즈 반복."""
    root, rot = decode_pose(x_raw, layout)
    a, n = spec.context_len - 1, spec.trans_len
    return _splice(x_raw, root[:, a:a + 1].expand(-1, n, -1),
                   rot[:, a:a + 1].expand(-1, n, -1, -1, -1), spec)
