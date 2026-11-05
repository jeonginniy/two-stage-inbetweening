"""
테스트용 합성(synthetic) 데이터 생성 도우미
==========================================

노트북에는 LaFAN1 BVH 가 없으므로, 데이터 없이 파이프라인을 검증하기 위해
"LaFAN1 과 같은 22관절 계층 + 임의의 뼈 길이 + 부드럽게 움직이는 회전" 을 만든다.

주의: 여기서 만드는 offset 과 동작은 실제 LaFAN1 값이 아니다 (형식만 같은 가짜 데이터).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from src.config import PROJECT_ROOT
from src.data.bvh import write_bvh
from src.data.rotations import euler_to_matrix
from src.data.skeleton import Skeleton

SKELETON_YAML = PROJECT_ROOT / "assets" / "skeleton_lafan1.yaml"


def lafan_skeleton_with_fake_offsets(seed: int = 0) -> Skeleton:
    """LaFAN1 계층 + 가짜 offset (J, 3). 크기는 5~30cm 정도로 임의 생성."""
    skel = Skeleton.from_yaml(SKELETON_YAML)
    g = torch.Generator().manual_seed(seed)
    offsets = (torch.rand(skel.num_joints, 3, generator=g, dtype=torch.float64) - 0.5) * 40.0
    offsets[0] = 0.0
    return skel.with_offsets(offsets)


def random_rotations(shape: tuple[int, ...], seed: int = 0, max_mid_deg: float = 80.0) -> Tensor:
    """임의 회전행렬 (*shape, 3, 3), float64. 중간 오일러각을 ±max_mid_deg 로 제한(짐벌락 회피)."""
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(*shape, 3, generator=g, dtype=torch.float64) * 2 - 1) * 180.0
    e[..., 1] = e[..., 1] / 180.0 * max_mid_deg
    return euler_to_matrix(e, "zyx")


def smooth_motion(num_frames: int, num_joints: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    """부드럽게 움직이는 가짜 동작.

    출력:
        root_pos  : (F, 3)        루트가 원을 그리며 이동, 높이 ~90cm
        local_rot : (F, J, 3, 3)  관절마다 다른 주파수의 사인파 오일러각
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(num_frames, dtype=torch.float64)[:, None] / 30.0       # (F, 1) 초
    amp = torch.rand(num_joints, 3, generator=g, dtype=torch.float64) * 60.0
    freq = 0.2 + torch.rand(num_joints, 3, generator=g, dtype=torch.float64) * 1.5
    phase = torch.rand(num_joints, 3, generator=g, dtype=torch.float64) * 2 * math.pi
    euler = amp * torch.sin(2 * math.pi * freq * t[..., None] + phase)    # (F, J, 3)
    euler[:, 0, 0] += 90.0 * t[:, 0]                                       # 루트가 천천히 회전
    local_rot = euler_to_matrix(euler, "zyx")
    root_pos = torch.stack([
        200.0 * torch.cos(0.3 * t[:, 0]),
        90.0 + 5.0 * torch.sin(3.0 * t[:, 0]),
        200.0 * torch.sin(0.3 * t[:, 0]),
    ], dim=-1)
    return root_pos, local_rot


def write_fake_lafan_bvh(path: Path, num_frames: int, seed: int = 0) -> tuple[Skeleton, Tensor, Tensor]:
    """가짜 LaFAN1 형식 BVH 파일을 쓴다. (skeleton, root_pos, local_rot) 를 돌려준다."""
    skel = lafan_skeleton_with_fake_offsets(seed=0)  # 모든 파일이 같은 뼈 길이
    root_pos, local_rot = smooth_motion(num_frames, skel.num_joints, seed=seed)
    write_bvh(path, skel.names, skel.parents, skel.offsets, local_rot, root_pos)
    return skel, root_pos, local_rot
