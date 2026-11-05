"""
모델 입력 표현 / 정렬 / 전체 파이프라인 테스트 — 데이터 없이 실행 가능

확인하는 것 (논문 3.2, 3.6.1):
- encode_pose 결과가 (B, T, 135) 이고 [6D 회전 132개 | 루트 위치 3개] 배치인가
- decode(encode(x)) 가 원래 회전/위치로 정확히 돌아오는가
- z-score 정규화/역정규화, 저장/불러오기, RunningStats 가 numpy 계산과 같은가
- 정렬 후: 마지막 context 프레임에서 루트 XZ = 0, 루트가 +X 를 바라보는가, 높이(Y)는 그대로인가
- 정렬은 몸 전체의 강체 변환인가 (FK 월드 위치 = R_y (원래 위치 - offset))
- 전체 파이프라인: 클립 → 모델 입력 x → 복원 이 원래 클립과 같은가
"""

import numpy as np
import pytest
import torch

from src.data.pipeline import model_output_to_motion, motion_to_model_input
from src.data.preprocess import canonicalize_root, extract_foot_contacts, uncanonicalize_root
from src.data.representation import (
    FeatureLayout,
    RunningStats,
    ZScoreNormalizer,
    decode_pose,
    encode_pose,
    extract_root_position,
    extract_rotation_6d,
)
from src.data.rotations import matrix_to_6d
from src.data.skeleton import forward_kinematics, local_positions_from_root
from tests.helpers import lafan_skeleton_with_fake_offsets, random_rotations, smooth_motion

J = 22
LAYOUT = FeatureLayout(num_joints=J)
CTX = 10


@pytest.fixture
def batch():
    """(B=3, T=42) 크기의 가짜 배치 (원래 월드 좌표)."""
    roots, rots = [], []
    for s in range(3):
        r, q = smooth_motion(42 + 5 * s, J, seed=s)
        roots.append(r[5 * s:])
        rots.append(q[5 * s:])
    return torch.stack(roots), torch.stack(rots)  # (3, 42, 3), (3, 42, 22, 3, 3)


def test_layout_dims():
    assert LAYOUT.dim == 135 and LAYOUT.rot_dim == 132
    assert LAYOUT.pos_slice == slice(132, 135)


def test_encode_shape_and_layout(batch):
    root_pos, local_rot = batch
    x = encode_pose(root_pos, local_rot)
    assert x.shape == (3, 42, 135)
    assert torch.equal(x[..., 132:], root_pos)
    assert torch.equal(x[..., 6 * 5:6 * 6], matrix_to_6d(local_rot[..., 5, :, :]))  # 관절 5
    assert extract_rotation_6d(x, LAYOUT).shape == (3, 42, 22, 6)
    assert torch.equal(extract_root_position(x, LAYOUT), root_pos)


def test_decode_inverts_encode(batch):
    root_pos, local_rot = batch
    p, r = decode_pose(encode_pose(root_pos, local_rot), LAYOUT)
    assert torch.allclose(p, root_pos) and torch.allclose(r, local_rot, atol=1e-12)


def test_running_stats_matches_numpy():
    g = torch.Generator().manual_seed(0)
    data = torch.randn(10, 50, 135, generator=g, dtype=torch.float64) * 7 + 3
    rs = RunningStats(135)
    for chunk in data.split(3):           # 여러 배치로 나눠 넣어도 결과가 같아야 함
        rs.update(chunk)
    arr = data.numpy()
    assert np.allclose(rs.mean.numpy(), arr.mean(axis=(0, 1)))
    assert np.allclose(rs.std.numpy(), arr.std(axis=(0, 1)))


def test_normalizer_roundtrip_and_save(tmp_path):
    mean = torch.linspace(-1, 1, 135, dtype=torch.float64)
    std = torch.linspace(0.5, 2, 135, dtype=torch.float64)
    std[7] = 0.0                                   # 상수 차원 → σ=1 로 대체되어야 함
    norm = ZScoreNormalizer(mean, std)
    x = torch.randn(2, 5, 135, dtype=torch.float64)
    xn = norm.normalize(x)
    assert torch.isfinite(xn).all()
    assert torch.allclose(xn[..., 7], x[..., 7] - mean[7])
    assert torch.allclose(norm.denormalize(xn), x)

    path = tmp_path / "stats.npz"
    norm.save(path, context_len=10)
    loaded = ZScoreNormalizer.load(path)
    assert torch.equal(loaded.mean, norm.mean) and torch.equal(loaded.std, norm.std)
    assert loaded.raw_std[7] == 0.0


def test_canonicalize_properties(batch):
    root_pos, local_rot = batch
    fwd = (0.0, 1.0, 0.0)
    rp, lr, frame = canonicalize_root(root_pos, local_rot, CTX - 1, fwd)
    # (1) 마지막 context 프레임의 루트 XZ = 0, Y(높이)는 그대로
    assert torch.allclose(rp[:, CTX - 1, [0, 2]], torch.zeros(3, 2, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(rp[..., 1], root_pos[..., 1])
    # (2) 그 프레임에서 루트의 앞 방향(XZ 투영)이 +X
    f = lr[:, CTX - 1, 0] @ torch.tensor(fwd, dtype=torch.float64)
    f_xz = torch.nn.functional.normalize(f[:, [0, 2]], dim=-1)
    assert torch.allclose(f_xz, torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64), atol=1e-10)
    # (3) 루트가 아닌 관절의 로컬 회전은 그대로
    assert torch.equal(lr[..., 1:, :, :], local_rot[..., 1:, :, :])
    # (4) 역변환으로 정확히 복원
    p2, r2 = uncanonicalize_root(rp, lr, frame)
    assert torch.allclose(p2, root_pos, atol=1e-10) and torch.allclose(r2, local_rot, atol=1e-12)


def test_canonicalize_is_rigid_transform_of_whole_body(batch):
    root_pos, local_rot = batch
    skel = lafan_skeleton_with_fake_offsets()
    rp, lr, frame = canonicalize_root(root_pos, local_rot, CTX - 1, skel.root_forward_local_axis)
    _, g_world = forward_kinematics(local_rot, local_positions_from_root(root_pos, skel.offsets), skel.parents)
    _, g_canon = forward_kinematics(lr, local_positions_from_root(rp, skel.offsets), skel.parents)
    Ry = frame.rot_offset[:, :, None]                         # (B, 1, 1, 3, 3)
    expected = (Ry @ (g_world - frame.pos_offset[:, :, None])[..., None]).squeeze(-1)
    assert torch.allclose(g_canon, expected, atol=1e-9)


def test_world_forward_z_option():
    R = random_rotations((4, 12), seed=9)
    local_rot = R[..., None, :, :].expand(4, 12, 22, 3, 3)
    root = torch.randn(4, 12, 3, dtype=torch.float64)
    _, lr, _ = canonicalize_root(root, local_rot, 3, (0.0, 1.0, 0.0), world_forward="z")
    f = lr[:, 3, 0] @ torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    f_xz = torch.nn.functional.normalize(f[:, [0, 2]], dim=-1)
    assert torch.allclose(f_xz, torch.tensor([[0.0, 1.0]] * 4, dtype=torch.float64), atol=1e-10)


def test_full_pipeline_roundtrip(batch):
    """1단계 목표: 클립 → 모델 입력 → 원래 형태로 복원."""
    root_pos, local_rot = batch
    fwd = (0.0, 1.0, 0.0)
    # 통계 (실제로는 train_stats 분할에서 계산)
    stats = RunningStats(LAYOUT.dim)
    stats.update(motion_to_model_input(root_pos, local_rot, CTX, fwd).x_raw)
    norm = ZScoreNormalizer.from_stats(stats)

    inp = motion_to_model_input(root_pos, local_rot, CTX, fwd, normalizer=norm)
    assert inp.x.shape == (3, 42, 135)
    # 정규화된 입력의 평균≈0, 표준편차≈1 (σ≈0 차원 제외)
    valid = stats.std > 1e-6
    assert torch.allclose(inp.x.reshape(-1, 135).mean(0)[valid], torch.zeros(int(valid.sum()), dtype=torch.float64), atol=1e-8)

    p, r = model_output_to_motion(inp.x, LAYOUT, normalizer=norm, frame=inp.frame)
    assert torch.allclose(p, root_pos, atol=1e-9)
    assert torch.allclose(r, local_rot, atol=1e-9)
    # frame 을 안 주면 정렬된 좌표로 돌아온다
    p_c, r_c = model_output_to_motion(inp.x, LAYOUT, normalizer=norm)
    assert torch.allclose(p_c, inp.root_pos, atol=1e-9) and torch.allclose(r_c, inp.local_rot, atol=1e-9)


def test_pipeline_float32(batch):
    root_pos, local_rot = (t.float() for t in batch)
    inp = motion_to_model_input(root_pos, local_rot, CTX, (0.0, 1.0, 0.0))
    p, r = model_output_to_motion(inp.x, LAYOUT, frame=inp.frame)
    assert inp.x.dtype == torch.float32
    assert torch.allclose(p, root_pos, atol=1e-3) and torch.allclose(r, local_rot, atol=1e-5)


def test_foot_contacts():
    # 관절 3은 정지, 관절 4는 프레임마다 1cm 이동
    T = 6
    g = torch.zeros(T, 22, 3)
    g[:, 4, 0] = torch.arange(T, dtype=torch.float32)
    c = extract_foot_contacts(g, [3, 4], threshold=0.2)
    assert c.shape == (T, 2)
    assert torch.equal(c[:, 0], torch.ones(T)) and torch.equal(c[:, 1], torch.zeros(T))
