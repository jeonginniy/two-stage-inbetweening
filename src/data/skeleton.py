"""
스켈레톤 정의와 순운동학 (Forward Kinematics)
=============================================

이 파일이 하는 일
-----------------
1. Skeleton: 관절 이름, 부모 인덱스, 뼈 offset, 발 접촉 관절 등 "몸의 구조" 를 담는다.
   assets/skeleton_lafan1.yaml 에서 읽고, offset 은 실제 BVH 에서 채운다.
2. forward_kinematics: 각 관절의 로컬 회전/로컬 위치 → 월드 회전/월드 위치.
   논문 식 (17) 의 g_t = FK(p_t, r_t) 에 해당. 2단계 학습의 L_pos, L_smooth,
   L_foot 손실 계산에 그대로 쓰인다 (미분 가능).
3. local_positions_from_root: 논문 3.2 "각 관절의 부모 기준 로컬 위치는 상수(뼈 길이)"
   라는 가정에 따라, 루트 위치 p_t 와 offset 만으로 모든 관절의 로컬 위치를 만든다.

사용 순서
---------
Skeleton.from_yaml() → (BVH 로딩 후) skeleton.with_offsets(offsets)
→ local_positions_from_root(root_pos, offsets) → forward_kinematics(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from src.config import load_yaml


@dataclass(frozen=True)
class Skeleton:
    """관절 구조 정보.

    속성:
        names        : 관절 이름 리스트, 길이 J (LaFAN1 은 22)
        parents      : 부모 인덱스 리스트, 길이 J. 루트는 -1.
                       항상 parents[j] < j (부모가 자식보다 앞) 이어야 FK 가 한 번에 계산된다.
        contact_left : 왼발 접촉 관절 인덱스 (LaFAN1: [3, 4] = LeftFoot, LeftToe)
        contact_right: 오른발 접촉 관절 인덱스 (LaFAN1: [7, 8])
        root_forward_local_axis: 루트의 로컬 "앞" 축 (논문 3.6.1 정렬에 사용)
        offsets      : (J, 3) 부모 기준 뼈 offset. YAML 에는 없고 BVH 에서 채운다.
    """

    names: tuple[str, ...]
    parents: tuple[int, ...]
    contact_left: tuple[int, ...] = ()
    contact_right: tuple[int, ...] = ()
    root_forward_local_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    offsets: Tensor | None = field(default=None, compare=False)

    def __post_init__(self):
        if len(self.names) != len(self.parents):
            raise ValueError("names 와 parents 길이가 다릅니다.")
        if self.parents[0] != -1:
            raise ValueError("0번 관절은 루트(부모 -1)여야 합니다.")
        for j, p in enumerate(self.parents[1:], start=1):
            if not (0 <= p < j):
                raise ValueError(f"관절 {j}({self.names[j]})의 부모 {p} 가 자신보다 앞에 있지 않습니다.")

    # ----- 편의 속성 -----
    @property
    def num_joints(self) -> int:
        """관절 수 J."""
        return len(self.names)

    @property
    def contact_joints(self) -> tuple[int, ...]:
        """발 접촉 관절 인덱스 [왼발..., 오른발...]. 논문 식 (15) ĉ ∈ R^{T×4} 의 순서."""
        return self.contact_left + self.contact_right

    # ----- 생성/검증 -----
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Skeleton":
        """assets/skeleton_*.yaml 에서 스켈레톤을 만든다 (offsets 는 None)."""
        cfg = load_yaml(path)
        names = [j["name"] for j in cfg["joints"]]
        index = {n: i for i, n in enumerate(names)}
        parents = [-1 if j["parent"] is None else index[j["parent"]] for j in cfg["joints"]]
        contact = cfg.get("contact_joints", {})
        return cls(
            names=tuple(names),
            parents=tuple(parents),
            contact_left=tuple(index[n] for n in contact.get("left", [])),
            contact_right=tuple(index[n] for n in contact.get("right", [])),
            root_forward_local_axis=tuple(float(v) for v in cfg.get("root_forward_local_axis", (0, 1, 0))),
        )

    def with_offsets(self, offsets: Tensor) -> "Skeleton":
        """offset (J, 3) 을 채운 새 Skeleton 을 돌려준다."""
        if tuple(offsets.shape) != (self.num_joints, 3):
            raise ValueError(f"offsets shape 이 {(self.num_joints, 3)} 이어야 합니다: {tuple(offsets.shape)}")
        return replace(self, offsets=offsets)

    def check_matches(self, names: Sequence[str], parents: Sequence[int], source: str = "") -> None:
        """BVH 에서 읽은 관절 이름/부모가 이 스켈레톤 정의와 같은지 확인. 다르면 ValueError."""
        if tuple(names) != self.names or tuple(int(p) for p in parents) != self.parents:
            lines = [f"스켈레톤 정의와 BVH 계층이 다릅니다 {source}".rstrip()]
            for j in range(max(len(names), self.num_joints)):
                a = (self.names[j], self.parents[j]) if j < self.num_joints else None
                b = (names[j], int(parents[j])) if j < len(names) else None
                mark = "" if a == b else "   <-- 다름"
                lines.append(f"  [{j:2d}] 정의={a}  BVH={b}{mark}")
            raise ValueError("\n".join(lines))


def local_positions_from_root(root_pos: Tensor, offsets: Tensor) -> Tensor:
    """루트 위치 + 뼈 offset → 모든 관절의 로컬 위치 (논문 3.2).

    논문 3.2: "The local position of each joint relative to its parent is constant".
    그래서 루트(0번)만 매 프레임 값(p_t)을 쓰고, 나머지 관절은 offset 을 그대로 쓴다.

    입력:
        root_pos : (..., 3)        예) (B, T, 3)  루트 월드 위치 p_t
        offsets  : (J, 3) 또는 (..., J, 3) 와 브로드캐스트 가능한 shape
                   예) (J, 3) 또는 (B, 1, J, 3)
    출력:
        (..., J, 3)                예) (B, T, 22, 3)
    """
    offsets = offsets.to(dtype=root_pos.dtype, device=root_pos.device)
    shape = torch.broadcast_shapes(root_pos.shape[:-1] + (1, 3), offsets.shape)
    child = offsets[..., 1:, :].expand(*shape[:-2], shape[-2] - 1, 3)
    return torch.cat([root_pos[..., None, :].expand(*shape[:-2], 1, 3), child], dim=-2)


def forward_kinematics(local_rot: Tensor, local_pos: Tensor, parents: Sequence[int]) -> tuple[Tensor, Tensor]:
    """순운동학 FK (논문 식 17 의 FK(p_t, r_t)).

    부모부터 차례로 내려가며
        G_rot[j] = G_rot[parent] @ local_rot[j]
        G_pos[j] = G_pos[parent] + G_rot[parent] @ local_pos[j]
    루트(0번)는 로컬 = 월드.

    입력:
        local_rot : (..., J, 3, 3)  각 관절의 부모 기준 회전.   예) (B, T, 22, 3, 3)
        local_pos : (..., J, 3)     각 관절의 부모 기준 위치.   예) (B, T, 22, 3)
                                    (루트는 월드 위치)
        parents   : 길이 J, parents[j] < j
    출력:
        global_rot : (..., J, 3, 3)
        global_pos : (..., J, 3)    논문의 g_t ∈ R^{J×3}
    """
    J = len(parents)
    g_rot: list[Tensor] = [local_rot[..., 0, :, :]]
    g_pos: list[Tensor] = [local_pos[..., 0, :]]
    for j in range(1, J):
        p = parents[j]
        g_rot.append(g_rot[p] @ local_rot[..., j, :, :])
        g_pos.append(g_pos[p] + (g_rot[p] @ local_pos[..., j, :, None]).squeeze(-1))
    # 리스트에 모은 뒤 stack: in-place 대입을 피해서 autograd 가 안전하게 동작
    return torch.stack(g_rot, dim=-3), torch.stack(g_pos, dim=-2)


# -----------------------------------------------------------------------------
# 다리 IK (발 고정 후처리용, docs/assumptions.md J9)
# -----------------------------------------------------------------------------
def rotation_between(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """벡터 a 방향을 b 방향으로 돌리는 **최소 회전** (Rodrigues). 축 = a × b, 각 = a 와 b 사이 각.

    입력:  a, b (..., 3)  (길이는 상관없음)
    출력:  (..., 3, 3)    R @ â = b̂.  a ≈ b 이면 단위행렬. a ≈ −b (180°) 는 다리 IK 에서 생기지 않아 따로 다루지 않음
    """
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    v = torch.cross(a, b, dim=-1)                                  # sin θ · 축
    c = (a * b).sum(-1)[..., None, None]                           # cos θ
    zero = torch.zeros_like(v[..., 0])
    K = torch.stack([zero, -v[..., 2], v[..., 1],
                     v[..., 2], zero, -v[..., 0],
                     -v[..., 1], v[..., 0], zero], dim=-1).reshape(*v.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand_as(K)
    return eye + K + (K @ K) / (1 + c).clamp_min(eps)


def two_bone_ik(hip: Tensor, knee: Tensor, ankle: Tensor, target: Tensor,
                eps: float = 1e-3) -> tuple[Tensor, Tensor]:
    """해석적 2-뼈 IK: 엉덩이 hip 은 그대로, 뼈 길이를 지키면서 발목을 target 에 놓는 새 무릎·발목 위치.

    무릎이 굽는 평면은 지금 무릎 위치로 정한다 (hip → target 축에 수직인 무릎 방향 성분 = pole).
    target 이 닿지 않는 거리면 다리를 최대로 펴거나 접은 위치 (hip → target 방향) 로 제한한다.
    입력:  hip, knee, ankle, target (..., 3)  전역 위치 (cm)
    출력:  (knee_new, ankle_new) (..., 3)
    """
    a = (knee - hip).norm(dim=-1, keepdim=True)                    # 허벅지 길이
    b = (ankle - knee).norm(dim=-1, keepdim=True)                  # 정강이 길이
    dv = target - hip
    u = dv / dv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    d = dv.norm(dim=-1, keepdim=True).clamp(min=(a - b).abs() + eps, max=a + b - eps)
    pole = (knee - hip) - ((knee - hip) * u).sum(-1, keepdim=True) * u
    pole = pole / pole.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = (a ** 2 - b ** 2 + d ** 2) / (2 * d)                       # hip → 무릎을 u 축에 내린 길이
    h = (a ** 2 - x ** 2).clamp_min(0).sqrt()
    return hip + u * x + pole * h, hip + u * d
