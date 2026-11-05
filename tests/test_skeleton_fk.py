"""
스켈레톤/FK 테스트 (src/data/skeleton.py) — 데이터 없이 실행 가능

확인하는 것:
- assets/skeleton_lafan1.yaml 이 22관절, 발 관절 [3,4,7,8] 로 읽히는가
- 손으로 계산한 3관절 사슬의 FK 결과와 같은가 (논문 식 17 의 FK)
- local_positions_from_root 가 루트=위치, 나머지=offset 으로 채우는가 (논문 3.2)
"""

import math

import pytest
import torch

from src.data.rotations import axis_rotation
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root
from tests.helpers import SKELETON_YAML


def test_lafan_skeleton_yaml():
    skel = Skeleton.from_yaml(SKELETON_YAML)
    assert skel.num_joints == 22
    assert skel.contact_joints == (3, 4, 7, 8)
    assert skel.names[0] == "Hips" and skel.parents[0] == -1
    assert skel.root_forward_local_axis == (0.0, 1.0, 0.0)


def test_skeleton_rejects_bad_parent_order():
    with pytest.raises(ValueError):
        Skeleton(names=("a", "b", "c"), parents=(-1, 2, 0))


def test_check_matches_reports_difference():
    skel = Skeleton.from_yaml(SKELETON_YAML)
    names = list(skel.names)
    names[5] = "WrongName"
    with pytest.raises(ValueError, match="WrongName"):
        skel.check_matches(names, skel.parents)


def test_fk_three_joint_chain_by_hand():
    # 루트(0) → 관절1 (offset (0,10,0)) → 관절2 (offset (0,5,0))
    parents = [-1, 0, 1]
    pi2 = torch.tensor(math.pi / 2, dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64)
    local_rot = torch.stack([axis_rotation("z", pi2), axis_rotation("x", pi2), I])  # (3, 3, 3)
    local_pos = torch.tensor([[1.0, 2.0, 3.0], [0.0, 10.0, 0.0], [0.0, 5.0, 0.0]], dtype=torch.float64)

    g_rot, g_pos = forward_kinematics(local_rot, local_pos, parents)
    # 관절1: 루트 + Rz90 @ (0,10,0) = (1,2,3) + (-10,0,0)
    # 관절2: 관절1 + (Rz90 @ Rx90) @ (0,5,0) = (-9,2,3) + Rz90 @ (0,0,5) = (-9,2,8)
    expected = torch.tensor([[1.0, 2.0, 3.0], [-9.0, 2.0, 3.0], [-9.0, 2.0, 8.0]], dtype=torch.float64)
    assert torch.allclose(g_pos, expected, atol=1e-12)
    assert torch.allclose(g_rot[2], axis_rotation("z", pi2) @ axis_rotation("x", pi2), atol=1e-12)


def test_fk_batch_shapes():
    skel = Skeleton.from_yaml(SKELETON_YAML)
    B, T, J = 2, 5, skel.num_joints
    local_rot = torch.eye(3).expand(B, T, J, 3, 3)
    local_pos = torch.randn(B, T, J, 3)
    g_rot, g_pos = forward_kinematics(local_rot, local_pos, skel.parents)
    assert g_rot.shape == (B, T, J, 3, 3) and g_pos.shape == (B, T, J, 3)


def test_local_positions_from_root():
    offsets = torch.arange(22 * 3, dtype=torch.float64).reshape(22, 3)
    root = torch.randn(2, 4, 3, dtype=torch.float64)           # (B, T, 3)
    lp = local_positions_from_root(root, offsets)
    assert lp.shape == (2, 4, 22, 3)
    assert torch.equal(lp[..., 0, :], root)
    assert torch.equal(lp[..., 1:, :], offsets[1:].expand(2, 4, 21, 3))
    # 배치마다 다른 offset (B, 1, J, 3) 도 지원
    lp2 = local_positions_from_root(root, offsets.expand(2, 1, 22, 3))
    assert torch.equal(lp, lp2)
