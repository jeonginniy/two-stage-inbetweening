"""
스켈레톤/FK 테스트 (src/data/skeleton.py) — 데이터 없이 실행 가능

확인하는 것:
- assets/skeleton_lafan1.yaml 이 22관절, 발 관절 [3,4,7,8] 로 읽히는가
- 손으로 계산한 3관절 사슬의 FK 결과와 같은가 (논문 식 17 의 FK)
- local_positions_from_root 가 루트=위치, 나머지=offset 으로 채우는가 (논문 3.2)
- 발 고정 후처리 (J9): 최소 회전, 2-뼈 IK (뼈 길이 유지·목표 도달), foot_lock (접촉 구간 발가락 고정, 다른 관절·프레임 불변)
"""

import math

import pytest
import torch

from src.data.rotations import axis_rotation
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root, rotation_between, two_bone_ik
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


# -----------------------------------------------------------------------------
# 발 고정 후처리 (J9)
# -----------------------------------------------------------------------------
def test_rotation_between_and_two_bone_ik():
    torch.manual_seed(0)
    a, b = torch.randn(50, 3, dtype=torch.float64), torch.randn(50, 3, dtype=torch.float64)
    R = rotation_between(a, b)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(50, 3, 3), atol=1e-9)
    assert torch.allclose((R @ (a / a.norm(dim=-1, keepdim=True))[..., None])[..., 0],
                          b / b.norm(dim=-1, keepdim=True), atol=1e-9)
    assert torch.allclose(rotation_between(a, 2 * a), torch.eye(3, dtype=torch.float64).expand(50, 3, 3), atol=1e-9)

    hip = torch.zeros(3, dtype=torch.float64)
    knee = torch.tensor([0.0, -40.0, 5.0], dtype=torch.float64)
    ankle = torch.tensor([0.0, -78.0, 0.0], dtype=torch.float64)
    la, lb = (knee - hip).norm(), (ankle - knee).norm()
    for target in ([4.0, -75.0, 3.0], [-6.0, -70.0, -2.0]):          # 닿는 거리
        t = torch.tensor(target, dtype=torch.float64)
        k2, a2 = two_bone_ik(hip, knee, ankle, t)
        assert torch.allclose(a2, t, atol=1e-6)
        assert (k2 - hip).norm() == pytest.approx(la.item()) and (a2 - k2).norm() == pytest.approx(lb.item())
    k2, a2 = two_bone_ik(hip, knee, ankle, torch.tensor([0.0, -200.0, 0.0], dtype=torch.float64))
    assert (a2 - hip).norm() < la + lb and (a2 - hip).norm() > la + lb - 0.01 and a2[1] < 0   # 못 닿으면 최대로 편다


def _standing_skeleton():
    """무릎을 굽히고 선 단순 인체 (LaFAN1 계층). 다리: 허벅지 40, 정강이 40, 발가락은 발목 앞 10 cm."""
    skel = Skeleton.from_yaml(SKELETON_YAML)
    off = torch.zeros(skel.num_joints, 3, dtype=torch.float64)
    for side, sx in (("Left", 10.0), ("Right", -10.0)):
        off[skel.names.index(f"{side}UpLeg")] = torch.tensor([sx, 0.0, 0.0])
        off[skel.names.index(f"{side}Leg")] = torch.tensor([0.0, -40.0, 0.0])
        off[skel.names.index(f"{side}Foot")] = torch.tensor([0.0, -40.0, 0.0])
        off[skel.names.index(f"{side}Toe")] = torch.tensor([0.0, 0.0, 10.0])
    for n in skel.names[9:]:
        off[skel.names.index(n)] = torch.tensor([0.0, 10.0, 0.0])
    return skel, off


def test_foot_lock_pins_sliding_toes():
    from src.data.representation import FeatureLayout, decode_pose, encode_pose
    from src.evaluation.inference import foot_lock
    from src.training.context_inputs import TransitionSpec

    skel, off = _standing_skeleton()
    J, T = skel.num_joints, 26
    spec = TransitionSpec(6, 22)                                      # transition 6..21, target 22
    rot = torch.eye(3, dtype=torch.float64).repeat(1, T, J, 1, 1)
    deg = math.radians(20.0)
    for side in ("Left", "Right"):                                    # 무릎 굽힘, 발은 평평
        rot[:, :, skel.names.index(f"{side}UpLeg")] = axis_rotation("x", torch.tensor(-deg, dtype=torch.float64))
        rot[:, :, skel.names.index(f"{side}Leg")] = axis_rotation("x", torch.tensor(2 * deg, dtype=torch.float64))
        rot[:, :, skel.names.index(f"{side}Foot")] = axis_rotation("x", torch.tensor(-deg, dtype=torch.float64))
    leg_h = 2 * 40 * math.cos(deg)
    root = torch.zeros(1, T, 3, dtype=torch.float64)
    root[..., 0] = torch.arange(T, dtype=torch.float64) * 1.0         # 1 cm/frame 로 미끄러지며 이동
    root[..., 1] = leg_h + 15.0                                       # 발가락 높이 15 cm (접촉 아님)
    root[:, 9:18, 1] = leg_h + 1.0                                    # 9..17 만 발가락 높이 1 cm (접촉)
    x_raw = encode_pose(root, rot)
    offsets = off[None, None]
    out = foot_lock(x_raw, offsets, skel, spec, height_cm=3.0, speed_cm=4.0, blend=2)

    layout = FeatureLayout(J)
    lp = lambda x: forward_kinematics(decode_pose(x, layout)[1], local_positions_from_root(decode_pose(x, layout)[0], offsets), skel.parents)[1]   # noqa: E731
    g0, g1 = lp(x_raw), lp(out)
    for toe in ("LeftToe", "RightToe"):
        j = skel.names.index(toe)
        xz = g1[0, 9:18, j][:, [0, 2]]
        assert (xz - xz.mean(0)).abs().max() < 1e-6                   # 접촉 구간 동안 고정
        assert (g0[0, 9:18, j][:, [0, 2]] - g0[0, 9:18, j][:, [0, 2]].mean(0)).abs().max() > 3   # 원래는 미끄러짐
        assert torch.allclose(g1[0, 9:18, j, 1], g0[0, 9:18, j, 1], atol=1e-6)    # 높이는 그대로
    legs = [skel.names.index(n) for n in ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
                                          "RightUpLeg", "RightLeg", "RightFoot", "RightToe")]
    others = [j for j in range(J) if j not in legs]
    assert torch.allclose(g1[..., others, :], g0[..., others, :], atol=1e-9)         # 루트·상체 그대로
    assert torch.allclose(g1[:, :spec.context_len], g0[:, :spec.context_len], atol=1e-9)   # context 그대로
    assert torch.allclose(g1[:, spec.target_idx:], g0[:, spec.target_idx:], atol=1e-9)     # target 이후 그대로
    for a, b in ((1, 2), (2, 3), (3, 4)):                             # 뼈 길이 그대로
        assert torch.allclose((g1[..., b, :] - g1[..., a, :]).norm(dim=-1), (g0[..., b, :] - g0[..., a, :]).norm(dim=-1))
