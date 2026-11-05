"""
회전 변환 테스트 (src/data/rotations.py) — 데이터 없이 노트북에서 실행 가능

확인하는 것:
- 오일러 → 행렬 이 BVH 규약(R = Rz Ry Rx)과 같은가
- 행렬 → 오일러 → 행렬 이 6가지 순서 모두에서 원래 행렬로 돌아오는가 (짐벌락 포함)
- 6D 표현 (논문 3.2): 행렬 → 6D → 행렬 이 정확히 복원되는가,
  아무 6D 값이나 넣어도 올바른 회전행렬(직교, det=+1)이 나오는가, 미분이 되는가
- 쿼터니언 왕복
"""

import itertools
import math

import pytest
import torch

from src.data.rotations import (
    axis_rotation,
    euler_to_matrix,
    matrix_to_6d,
    matrix_to_euler,
    matrix_to_quat,
    quat_to_matrix,
    sixd_to_matrix,
)
from tests.helpers import random_rotations

ATOL = 1e-10
ORDERS = ["".join(p) for p in itertools.permutations("xyz")]


def is_rotation(R, atol=1e-8):
    I = torch.eye(3, dtype=R.dtype).expand_as(R)
    return torch.allclose(R @ R.transpose(-1, -2), I, atol=atol) and torch.allclose(
        torch.linalg.det(R), torch.ones(R.shape[:-2], dtype=R.dtype), atol=atol
    )


def test_axis_rotation_z90_maps_x_to_y():
    R = axis_rotation("z", torch.tensor(math.pi / 2, dtype=torch.float64))
    v = R @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(v, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=ATOL)


def test_euler_zyx_is_Rz_Ry_Rx():
    e = torch.tensor([30.0, -20.0, 45.0], dtype=torch.float64)
    r = torch.deg2rad(e)
    expected = axis_rotation("z", r[0]) @ axis_rotation("y", r[1]) @ axis_rotation("x", r[2])
    assert torch.allclose(euler_to_matrix(e, "ZYX"), expected, atol=ATOL)


@pytest.mark.parametrize("order", ORDERS)
def test_matrix_euler_roundtrip(order):
    R = random_rotations((500,), seed=1)
    e = matrix_to_euler(R, order)
    assert e.shape == (500, 3)
    assert torch.allclose(euler_to_matrix(e, order), R, atol=1e-9)


@pytest.mark.parametrize("order", ORDERS)
def test_matrix_euler_gimbal_lock(order):
    # 중간 각도를 정확히 ±90도로 두면 짐벌락
    e = torch.tensor([[10.0, 90.0, 25.0], [-40.0, -90.0, 70.0]], dtype=torch.float64)
    R = euler_to_matrix(e, order)
    assert torch.allclose(euler_to_matrix(matrix_to_euler(R, order), order), R, atol=1e-6)


def test_6d_layout_is_first_two_columns():
    R = random_rotations((4,), seed=2)
    d6 = matrix_to_6d(R)
    assert d6.shape == (4, 6)
    assert torch.equal(d6[:, [0, 2, 4]], R[:, :, 0])  # 첫 번째 열
    assert torch.equal(d6[:, [1, 3, 5]], R[:, :, 1])  # 두 번째 열


def test_6d_roundtrip_with_model_shapes():
    # 모델에서 쓰는 shape (B, T, 22, 3, 3)
    R = random_rotations((2, 7, 22), seed=3)
    d6 = matrix_to_6d(R)
    assert d6.shape == (2, 7, 22, 6)
    R2 = sixd_to_matrix(d6)
    assert R2.shape == (2, 7, 22, 3, 3)
    assert torch.allclose(R2, R, atol=ATOL)


def test_arbitrary_6d_gives_valid_rotation():
    # 네트워크 출력처럼 아무 값이나 넣어도 회전행렬이 나와야 한다 (Gram–Schmidt)
    g = torch.Generator().manual_seed(4)
    d6 = torch.randn(1000, 6, generator=g, dtype=torch.float64) * 3
    assert is_rotation(sixd_to_matrix(d6))


def test_sixd_to_matrix_is_differentiable():
    d6 = torch.randn(3, 22, 6, dtype=torch.float64, requires_grad=True)
    sixd_to_matrix(d6).sum().backward()
    assert d6.grad is not None and torch.isfinite(d6.grad).all()


def test_quaternion_roundtrip():
    R = random_rotations((1000,), seed=5)
    q = matrix_to_quat(R)
    assert q.shape == (1000, 4)
    assert torch.allclose(q.norm(dim=-1), torch.ones(1000, dtype=torch.float64), atol=ATOL)
    assert (q[:, 0] >= 0).all()
    assert torch.allclose(quat_to_matrix(q), R, atol=1e-9)


def test_quaternion_known_value():
    # z축 90도 회전 = (cos45°, 0, 0, sin45°)
    R = axis_rotation("z", torch.tensor(math.pi / 2, dtype=torch.float64))
    s = math.sqrt(0.5)
    assert torch.allclose(matrix_to_quat(R), torch.tensor([s, 0, 0, s], dtype=torch.float64), atol=ATOL)
