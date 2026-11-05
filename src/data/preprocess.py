"""
클립 전처리: 루트 정렬(canonicalization)과 발 접촉 라벨
=====================================================

이 파일이 하는 일
-----------------
1. canonicalize_root / uncanonicalize_root  (논문 3.6.1)
   "we preprocess each clip so that the root joint faces the X+ axis and its XZ
    coordinates are zeroed out at the last context frame."
   → 클립마다 "마지막 context 프레임" 을 기준으로
       (a) 루트의 XZ 위치를 0 으로 옮기고 (Y=높이는 그대로)
       (b) 루트가 +X 방향을 보도록 Y축(위쪽 축) 기준으로 회전한다.
   이렇게 하면 모든 transition 이 같은 위치/방향에서 시작해서 학습이 쉬워진다.
   역변환(uncanonicalize)으로 원래 월드 좌표로 정확히 되돌릴 수 있다.

   전체 몸을 강체(rigid)로 옮기는 변환이지만, 다른 관절의 회전은 "부모 기준 로컬" 이므로
   루트의 위치/회전만 바꾸면 된다.

2. extract_foot_contacts  (논문 3.4, 식 15 의 정답 라벨 c)
   발/발가락 관절의 프레임 간 이동량이 임계값보다 작으면 "땅에 닿음(1)" 으로 본다.
   [가정] 논문에는 라벨 생성 방법이 없다. 공식 코드처럼 |dx|+|dy|+|dz| < 0.2 를 쓴다
   (임계값은 configs/data/lafan1.yaml 의 contact_vel_threshold).

사용 순서
---------
dataset.py 가 파일 단위로 extract_foot_contacts 를 미리 계산하고,
pipeline.py 가 배치마다 canonicalize_root → representation.encode_pose 순서로 호출한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from src.data.rotations import axis_rotation


@dataclass
class CanonicalFrame:
    """canonicalize 에 쓴 변환값 (역변환에 필요).

    속성:
        pos_offset : (..., 1, 3)     빼 준 루트 위치 [x, 0, z]
        rot_offset : (..., 1, 3, 3)  곱해 준 Y축 회전 R_y(θ)
    변환:  p_canon = R_y @ (p_world - pos_offset),   R_root_canon = R_y @ R_root_world
    """

    pos_offset: Tensor
    rot_offset: Tensor


def heading_rotation(root_rot: Tensor, forward_local: Sequence[float], world_forward: str = "x") -> Tensor:
    """루트가 world_forward 방향을 보게 만드는 Y축 회전 R_y(θ) 를 구한다.

    1) f = root_rot @ forward_local  (루트의 "앞" 방향, 월드 좌표)
    2) f 를 XZ 평면에 투영 (y 성분 0)
    3) R_y(θ) @ f 가 +X (또는 +Z) 를 향하도록 θ 계산
         R_y(θ) = [[c,0,s],[0,1,0],[-s,0,c]] 이므로
         +X 목표: θ = atan2(f_z, f_x)
         +Z 목표: θ = atan2(-f_x, f_z)

    입력:
        root_rot      : (..., 3, 3)  루트 회전 (월드)
        forward_local : 길이 3, 루트의 로컬 "앞" 축 (LaFAN1 가정: [0, 1, 0])
        world_forward : "x" | "z"
    출력:
        (..., 3, 3)  Y축 회전행렬

    주의: 루트의 앞 축이 거의 수직(누워 있는 자세 등)이면 XZ 투영이 매우 짧아져
    방향이 불안정해진다. atan2(0, 0) = 0 이라 NaN 은 나지 않는다.
    """
    fl = torch.as_tensor(forward_local, dtype=root_rot.dtype, device=root_rot.device)
    f = (root_rot @ fl)                                   # (..., 3)
    fx, fz = f[..., 0], f[..., 2]
    if world_forward == "x":
        theta = torch.atan2(fz, fx)
    elif world_forward == "z":
        theta = torch.atan2(-fx, fz)
    else:
        raise ValueError(f"world_forward 는 'x' 또는 'z': {world_forward!r}")
    return axis_rotation("y", theta)


def canonicalize_root(
    root_pos: Tensor,
    local_rot: Tensor,
    frame_idx: int,
    forward_local: Sequence[float],
    world_forward: str = "x",
) -> tuple[Tensor, Tensor, CanonicalFrame]:
    """논문 3.6.1 정렬: frame_idx 프레임에서 루트 XZ=0, 루트가 +X 를 보게 한다.

    frame_idx 에는 보통 context_len - 1 (마지막 context 프레임, 0-based) 을 준다.

    입력:
        root_pos  : (..., T, 3)        루트 월드 위치.        예) (B, T, 3)
        local_rot : (..., T, J, 3, 3)  로컬 회전 (0번=루트).  예) (B, T, 22, 3, 3)
        frame_idx : 기준 프레임 번호
    출력:
        root_pos_c  : (..., T, 3)        정렬된 루트 위치
        local_rot_c : (..., T, J, 3, 3)  루트 회전만 바뀐 로컬 회전
        frame       : CanonicalFrame (역변환용)
    """
    # (a) 위치 offset: 기준 프레임의 루트 X, Z (Y 는 0 으로 두어 높이 유지)
    ref = root_pos[..., frame_idx:frame_idx + 1, :]                         # (..., 1, 3)
    pos_offset = ref * torch.tensor([1.0, 0.0, 1.0], dtype=ref.dtype, device=ref.device)

    # (b) 회전 offset: 기준 프레임의 루트 방향을 +X 로
    root_rot_ref = local_rot[..., frame_idx:frame_idx + 1, 0, :, :]        # (..., 1, 3, 3)
    rot_offset = heading_rotation(root_rot_ref, forward_local, world_forward)  # (..., 1, 3, 3)

    root_pos_c = (rot_offset @ (root_pos - pos_offset)[..., None]).squeeze(-1)
    root_rot_c = rot_offset @ local_rot[..., 0, :, :]                       # (..., T, 3, 3)
    local_rot_c = torch.cat([root_rot_c[..., None, :, :], local_rot[..., 1:, :, :]], dim=-3)
    return root_pos_c, local_rot_c, CanonicalFrame(pos_offset, rot_offset)


def uncanonicalize_root(root_pos_c: Tensor, local_rot_c: Tensor, frame: CanonicalFrame) -> tuple[Tensor, Tensor]:
    """canonicalize_root 의 역변환 (정렬된 좌표 → 원래 월드 좌표).

        p_world      = R_y^T @ p_canon + pos_offset
        R_root_world = R_y^T @ R_root_canon

    입력/출력 shape 은 canonicalize_root 와 같다.
    """
    inv = frame.rot_offset.transpose(-1, -2)
    root_pos = (inv @ root_pos_c[..., None]).squeeze(-1) + frame.pos_offset
    root_rot = inv @ local_rot_c[..., 0, :, :]
    local_rot = torch.cat([root_rot[..., None, :, :], local_rot_c[..., 1:, :, :]], dim=-3)
    return root_pos, local_rot


def extract_foot_contacts(global_pos: Tensor, joint_indices: Sequence[int], threshold: float) -> Tensor:
    """발 접촉 라벨 c (논문 3.4, 식 15/19 의 정답값).

    프레임 t 와 t+1 사이의 이동량 |Δx|+|Δy|+|Δz| < threshold 이면 프레임 t 는 접촉(1).
    마지막 프레임은 다음 프레임이 없으므로 바로 앞 프레임 값을 복사한다.

    입력:
        global_pos    : (..., T, J, 3)  FK 로 구한 월드 관절 위치 (원본 월드 좌표에서 계산)
        joint_indices : 접촉을 볼 관절 번호들, 예) [3, 4, 7, 8]
        threshold     : 임계값 (LaFAN1 단위 cm/frame)
    출력:
        (..., T, len(joint_indices))  0.0 / 1.0 (float)
    """
    p = global_pos[..., list(joint_indices), :]                     # (..., T, n, 3)
    vel = (p[..., 1:, :, :] - p[..., :-1, :, :]).abs().sum(dim=-1)  # (..., T-1, n)
    contact = (vel < threshold).to(global_pos.dtype)
    return torch.cat([contact, contact[..., -1:, :]], dim=-2)
