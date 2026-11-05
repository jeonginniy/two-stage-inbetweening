"""
회전 표현 변환 함수 모음 (PyTorch)
================================

이 파일이 하는 일
-----------------
데이터 파이프라인에서 쓰는 모든 "회전 표현 ↔ 회전 표현" 변환을 모아 두었다.

    BVH 오일러각 (도)  ──euler_to_matrix──▶  3×3 회전행렬  ──matrix_to_6d──▶  6D (모델 입력)
          ▲                                       │  ▲
          └────────────matrix_to_euler────────────┘  └──sixd_to_matrix── 6D (모델 출력)
                                                  │
                                                  └──matrix_to_quat──▶ 쿼터니언 (4단계 L2Q 평가용)

사용 순서
---------
1. src/data/bvh.py       : BVH 의 오일러각 → euler_to_matrix 로 회전행렬
2. src/data/representation.py : 회전행렬 → matrix_to_6d 로 모델 입력 (논문 3.2)
3. 모델 출력 6D → sixd_to_matrix (Gram–Schmidt) 로 회전행렬 복원 (논문 3.2, 식 10)
4. 회전행렬 → matrix_to_euler 로 BVH 다시 쓰기 (결과 확인/시각화용)

모든 함수는 torch 텐서를 받아 torch 텐서를 돌려주며, 앞쪽 차원 "..." 은
임의 개수의 배치 차원이다 (예: (B, T, 22, 3, 3)). 미분 가능하므로 2단계 학습에서
손실 계산 (FK, 식 17) 에 그대로 쓸 수 있다.

용어/규약
---------
- 회전행렬 R 은 "열벡터" 규약: 회전된 벡터 v' = R @ v
- 오일러각 order="zyx" 는 R = Rz(a0) @ Ry(a1) @ Rx(a2) 를 뜻한다.
  BVH 의 "CHANNELS ... Zrotation Yrotation Xrotation" 이 바로 이 경우이며,
  채널 값도 (z각, y각, x각) 순서로 들어 있다. (벡터에는 X 회전이 가장 먼저 적용됨)
- 쿼터니언은 (w, x, y, z) 순서.
"""

from __future__ import annotations

import torch
from torch import Tensor

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# -----------------------------------------------------------------------------
# 오일러각 ↔ 회전행렬
# -----------------------------------------------------------------------------
def _check_order(order: str) -> str:
    """오일러 순서 문자열 검사. "ZYX" → "zyx". 세 축이 모두 달라야 한다 (Tait–Bryan)."""
    order = order.lower()
    if len(order) != 3 or set(order) != {"x", "y", "z"}:
        raise ValueError(f"오일러 순서는 x,y,z 를 한 번씩 포함해야 합니다: {order!r}")
    return order


def axis_rotation(axis: str, angle: Tensor) -> Tensor:
    """한 축에 대한 기본 회전행렬.

    입력:
        axis  : "x" | "y" | "z"
        angle : (...)  라디안
    출력:
        (..., 3, 3)  회전행렬
    """
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    if axis == "x":
        entries = [one, zero, zero,
                   zero, c, -s,
                   zero, s, c]
    elif axis == "y":
        entries = [c, zero, s,
                   zero, one, zero,
                   -s, zero, c]
    elif axis == "z":
        entries = [c, -s, zero,
                   s, c, zero,
                   zero, zero, one]
    else:
        raise ValueError(f"알 수 없는 축: {axis!r}")
    return torch.stack(entries, dim=-1).reshape(*angle.shape, 3, 3)


def euler_to_matrix(euler: Tensor, order: str, degrees: bool = True) -> Tensor:
    """오일러각 → 회전행렬.

    입력:
        euler   : (..., 3)  각도. euler[..., i] 는 order[i] 축에 대한 각도
        order   : 예) "zyx" → R = Rz(euler[...,0]) @ Ry(euler[...,1]) @ Rx(euler[...,2])
        degrees : True 면 euler 를 도(degree) 단위로 해석 (BVH 는 도 단위)
    출력:
        (..., 3, 3)  회전행렬
    """
    order = _check_order(order)
    angles = torch.deg2rad(euler) if degrees else euler
    R = axis_rotation(order[0], angles[..., 0])
    R = R @ axis_rotation(order[1], angles[..., 1])
    R = R @ axis_rotation(order[2], angles[..., 2])
    return R


def matrix_to_euler(R: Tensor, order: str, degrees: bool = True, eps: float = 1e-6) -> Tensor:
    """회전행렬 → 오일러각 (euler_to_matrix 의 역함수). BVH 로 다시 저장할 때 사용.

    입력:
        R     : (..., 3, 3)  회전행렬
        order : 예) "zyx"
    출력:
        (..., 3)  각도 [a0, a1, a2]. 중간 각 a1 은 [-90°, 90°] 범위.

    유도 (R = R_i(a0) R_j(a1) R_k(a2), i,j,k = order 의 축 번호):
        s = +1 (i,j,k 가 x→y→z 순환순서) / -1 (그 반대)
        sin(a1) = s * R[i,k]
        a0 = atan2(-s * R[j,k], R[k,k])
        a2 = atan2(-s * R[i,j], R[i,i])
    짐벌락(cos(a1)≈0) 일 때는 a2 = 0 으로 두고 R_i(a0) = R @ R_j(a1)^T 에서 a0 를 구한다.
    오일러각 자체는 유일하지 않으므로, 테스트는 "각도"가 아니라 "행렬"이 같은지 확인한다.
    """
    order = _check_order(order)
    i, j, k = (_AXIS_INDEX[a] for a in order)
    s = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0

    sin_b = (s * R[..., i, k]).clamp(-1.0, 1.0)
    a1 = torch.asin(sin_b)
    a0 = torch.atan2(-s * R[..., j, k], R[..., k, k])
    a2 = torch.atan2(-s * R[..., i, j], R[..., i, i])

    # 짐벌락 처리
    gimbal = torch.sqrt((1.0 - sin_b * sin_b).clamp_min(0.0)) < eps
    if bool(gimbal.any()):
        A = R @ axis_rotation(order[1], a1).transpose(-1, -2)  # = R_i(a0) 이어야 함
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        a0_g = torch.atan2(A[..., i2, i1], A[..., i1, i1])
        a0 = torch.where(gimbal, a0_g, a0)
        a2 = torch.where(gimbal, torch.zeros_like(a2), a2)

    out = torch.stack([a0, a1, a2], dim=-1)
    return torch.rad2deg(out) if degrees else out


# -----------------------------------------------------------------------------
# 회전행렬 ↔ 6D 표현  (논문 3.2, Zhou et al. 2019)
# -----------------------------------------------------------------------------
def matrix_to_6d(R: Tensor) -> Tensor:
    """회전행렬 → 6D 회전 표현 (논문 3.2).

    회전행렬의 "앞 두 열(column)" 을 취한다. 논문은 "first two rows or columns" 라고만
    적었는데, 여기서는 열을 쓴다 [가정: 공식 코드와 같은 선택].

    벡터 안의 배치 순서 (행 우선으로 펼침):
        [R00, R01, R10, R11, R20, R21]
         └ 1행 ┘  └ 2행 ┘  └ 3행 ┘    (각 행에서 0열, 1열)
    즉 첫 번째 열 = 원소 [0, 2, 4], 두 번째 열 = 원소 [1, 3, 5].
    [가정] 이 펼침 순서도 공식 코드와 같게 맞췄다 → 정규화 통계(μ, σ)를 공식 결과와
    차원별로 직접 비교할 수 있다 (scripts/compare_stats_with_official.py).

    입력:  R : (..., 3, 3)   예) (B, T, 22, 3, 3)
    출력:      (..., 6)      예) (B, T, 22, 6)
    """
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def sixd_to_matrix(d6: Tensor, eps: float = 1e-8) -> Tensor:
    """6D 표현 → 회전행렬 (Gram–Schmidt 직교화, 논문 3.2).

    네트워크 출력 6D 는 임의의 실수 6개이므로, 다음 과정으로 항상 올바른
    회전행렬(직교, det = +1)을 만든다:
        a1, a2 = 6D 를 (3,2) 로 본 두 열
        b1 = a1 / |a1|
        b2 = (a2 - (b1·a2) b1) / |...|     ← a2 에서 b1 성분을 뺀 뒤 정규화
        b3 = b1 × b2                       ← 오른손 좌표계 보장 (det = +1)
        R  = [b1 | b2 | b3]  (열로 쌓음)

    입력:  d6 : (..., 6)     예) (B, T, 22, 6)
    출력:       (..., 3, 3)  예) (B, T, 22, 3, 3)
    """
    m = d6.reshape(*d6.shape[:-1], 3, 2)
    a1, a2 = m[..., 0], m[..., 1]                                   # 각각 (..., 3)
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2_orth / a2_orth.norm(dim=-1, keepdim=True).clamp_min(eps)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                        # 열로 쌓기


# -----------------------------------------------------------------------------
# 회전행렬 ↔ 쿼터니언 (4단계 평가지표 L2Q 에서 사용, 논문 4.1)
# -----------------------------------------------------------------------------
def quat_to_matrix(q: Tensor) -> Tensor:
    """쿼터니언 (w, x, y, z) → 회전행렬. 입력은 내부에서 단위 길이로 정규화한다.

    입력:  q : (..., 4)
    출력:      (..., 3, 3)
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    entries = [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]
    return torch.stack(entries, dim=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: Tensor) -> Tensor:
    """회전행렬 → 쿼터니언 (w, x, y, z), w ≥ 0 이 되도록 부호를 맞춘다.

    방법: 4w², 4x², 4y², 4z² 를 대각 성분으로 계산한 뒤, 그중 가장 큰 성분을
    기준으로 나머지를 구한다 (작은 값으로 나누는 수치 불안정을 피하기 위함).

    입력:  R : (..., 3, 3)
    출력:      (..., 4)
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4w², 4x², 4y², 4z²
    sq = torch.stack([
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ], dim=-1)
    two_abs = torch.sqrt(sq.clamp_min(0.0))  # 2|w|, 2|x|, 2|y|, 2|z|

    # 후보 c: "c번째 성분이 가장 크다" 고 가정했을 때의 (4·q_c) * q 벡터
    cand = torch.stack([
        torch.stack([sq[..., 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, sq[..., 1], m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, sq[..., 2], m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, sq[..., 3]], dim=-1),
    ], dim=-2)                                                       # (..., 4, 4)
    cand = cand / (2.0 * two_abs[..., None]).clamp_min(1e-8)         # q 로 정규화

    best = two_abs.argmax(dim=-1)                                    # (...)
    q = torch.gather(cand, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)
