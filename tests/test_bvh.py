"""
BVH 읽기/쓰기 테스트 (src/data/bvh.py) — 데이터 없이 실행 가능

확인하는 것:
- 손으로 쓴 작은 BVH 를 정확히 파싱하는가 (관절, 부모, offset, 채널, End Site, 프레임 값)
- 파싱 결과로 FK 를 하면 손으로 계산한 위치가 나오는가
- write_bvh → load_bvh 왕복 시 회전/위치가 보존되는가 (LaFAN1 형식의 가짜 22관절 데이터)
"""

import numpy as np
import pytest
import torch

from src.data.bvh import load_bvh, parse_bvh, write_bvh
from src.data.rotations import axis_rotation
from src.data.skeleton import forward_kinematics
from tests.helpers import lafan_skeleton_with_fake_offsets, smooth_motion

MINI_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Spine
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0.0 5.0 0.0
    }
  }
  JOINT Leg
  {
    OFFSET 2.0 -10.0 0.0
    CHANNELS 3 Zrotation Yrotation Xrotation
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
1.0 2.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 90.0
"""


@pytest.fixture
def mini_path(tmp_path):
    p = tmp_path / "mini.bvh"
    p.write_text(MINI_BVH, encoding="utf-8")
    return p


def test_parse_hierarchy(mini_path):
    b = parse_bvh(mini_path)
    assert b.names == ["Hips", "Spine", "Leg"]
    assert b.parents.tolist() == [-1, 0, 0]
    assert np.allclose(b.offsets, [[0, 0, 0], [0, 10, 0], [2, -10, 0]])
    assert b.channels[0][:3] == ["Xposition", "Yposition", "Zposition"]
    assert len(b.channels[1]) == 3
    assert list(b.end_sites) == [1] and np.allclose(b.end_sites[1], [0, 5, 0])
    assert b.motion.shape == (2, 12)
    assert b.frame_time == pytest.approx(0.0333333)


def test_load_values_and_fk(mini_path):
    m = load_bvh(mini_path)
    assert m.local_rot.shape == (2, 3, 3, 3) and m.local_pos.shape == (2, 3, 3)
    assert m.rotation_orders == ["zyx", "zyx", "zyx"]
    # 위치 채널 없는 관절은 OFFSET 을 로컬 위치로 사용
    assert torch.allclose(m.local_pos[:, 1], torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64))
    assert torch.allclose(m.root_pos, torch.tensor([[1.0, 2.0, 3.0]] * 2, dtype=torch.float64))
    # 두 번째 프레임: Hips Zrot 90, Leg Xrot 90
    half_pi = torch.tensor(torch.pi / 2, dtype=torch.float64)
    assert torch.allclose(m.local_rot[1, 0], axis_rotation("z", half_pi), atol=1e-12)
    assert torch.allclose(m.local_rot[1, 2], axis_rotation("x", half_pi), atol=1e-12)

    _, g = forward_kinematics(m.local_rot, m.local_pos, m.parents.tolist())
    # Spine = (1,2,3) + Rz90 (0,10,0) = (-9,2,3) ;  Leg = (1,2,3) + Rz90 (2,-10,0) = (11,4,3)
    assert torch.allclose(g[1, 1], torch.tensor([-9.0, 2.0, 3.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(g[1, 2], torch.tensor([11.0, 4.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_bad_frame_count_raises(tmp_path):
    p = tmp_path / "bad.bvh"
    p.write_text(MINI_BVH.replace("Frames: 2", "Frames: 3"), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_bvh(p)


def test_write_then_load_roundtrip(tmp_path):
    skel = lafan_skeleton_with_fake_offsets()
    root_pos, local_rot = smooth_motion(60, skel.num_joints, seed=7)
    path = tmp_path / "walk1_subject1.bvh"
    write_bvh(path, skel.names, skel.parents, skel.offsets, local_rot, root_pos, precision=8)

    m = load_bvh(path)
    assert m.names == list(skel.names)
    assert m.parents.tolist() == list(skel.parents)
    assert np.allclose(m.offsets, skel.offsets.numpy())
    # 소수점 8자리로 저장했으므로 오차는 매우 작아야 함
    assert torch.allclose(m.local_rot, local_rot, atol=1e-6)
    assert torch.allclose(m.root_pos, root_pos, atol=1e-6)
