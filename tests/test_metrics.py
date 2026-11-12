"""
4단계 평가 지표·기준선 테스트 (src/evaluation/metrics.py, baselines.py) — 데이터 없이 실행

확인 내용
    - 예측 = 정답이면 L2P, L2Q, NPSS 가 0
    - L2P 는 두 좌표에 같은 평행이동을 해도 같음 (정렬 방식과 무관한 이유)
    - 쿼터니언 부호 뒤집힘 제거, NPSS 는 q → −q 에 불변
    - NPSS 손계산 (길이 2 신호) 과 비교
    - 발 미끄러짐: 정지 0, 바닥에서 미끄러지면 속도 그대로, H 이상 높이면 0
    - 평균 중심: 루트 XZ 평균 0, 높이 유지
    - 기준선: context·target 은 그대로, zero-velocity = 마지막 context 포즈, 보간은 양 끝 사이
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.data.representation import FeatureLayout, decode_pose, encode_pose
from src.data.rotations import axis_rotation, matrix_to_quat
from src.evaluation.baselines import interpolation, slerp, zero_velocity
from src.evaluation.metrics import (
    MetricAccumulator,
    foot_skating_per_clip,
    l2p_per_clip,
    l2q_per_clip,
    mean_center,
    npss_parts,
    npss_reduce,
    remove_quat_discontinuities,
)
from src.training.context_inputs import TransitionSpec

torch.manual_seed(0)
B, T, J = 3, 12, 5
FRAMES = slice(4, 9)


def random_quats(*shape) -> torch.Tensor:
    q = torch.randn(*shape, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_identical_prediction_gives_zero():
    gpos = torch.randn(B, T, J, 3, dtype=torch.float64) * 50
    q = remove_quat_discontinuities(random_quats(B, T, J))
    acc = MetricAccumulator(torch.zeros(J * 3), torch.ones(J * 3), foot_idx=[1, 2], height_cm=3.0)
    acc.add(gpos, gpos.clone(), q, q.clone(), FRAMES)
    r = acc.result()
    assert r["L2P"] == pytest.approx(0.0) and r["L2Q"] == pytest.approx(0.0) and r["NPSS"] == pytest.approx(0.0)
    assert r["n_clips"] == B


def test_l2p_translation_invariant_and_value():
    gt = torch.randn(B, T, J, 3, dtype=torch.float64)
    pred = gt + 0.1
    std = torch.full((J * 3,), 2.0, dtype=torch.float64)
    base = l2p_per_clip(pred, gt, torch.zeros(J * 3), std, FRAMES)
    # 프레임마다 (J*3) 차원 모두 0.1/2 → L2 = 0.05·sqrt(J*3)
    assert torch.allclose(base, torch.full((B,), 0.05 * math.sqrt(J * 3), dtype=torch.float64))
    shift = torch.tensor([30.0, 0.0, -12.0], dtype=torch.float64)
    assert torch.allclose(l2p_per_clip(pred + shift, gt + shift, torch.zeros(J * 3), std, FRAMES), base)


def test_remove_quat_discontinuities():
    q = random_quats(1, T, J)
    q = remove_quat_discontinuities(q)
    flipped = q.clone()
    flipped[:, 3::2] *= -1                                     # 일부 프레임 부호 뒤집기 (같은 회전)
    fixed = remove_quat_discontinuities(flipped)
    assert torch.allclose(fixed, q)
    assert (l2q_per_clip(fixed, q, FRAMES) < 1e-12).all()


def test_npss_sign_invariant_and_hand_example():
    q_gt, q_pr = random_quats(B, T, J), random_quats(B, T, J)
    e1, w1 = npss_parts(q_pr, q_gt, FRAMES)
    e2, w2 = npss_parts(-q_pr, -q_gt, FRAMES)
    assert npss_reduce(e1, w1) == pytest.approx(npss_reduce(e2, w2))
    # 손계산: 길이 2 신호. FFT 실수부: [a+b, a−b] → 파워 [(a+b)², (a−b)²]
    gt = torch.tensor([1.0, 1.0], dtype=torch.float64).view(1, 2, 1, 1).expand(1, 2, 1, 4)    # 파워 [4, 0]
    pr = torch.tensor([1.0, 0.0], dtype=torch.float64).view(1, 2, 1, 1).expand(1, 2, 1, 4)    # 파워 [1, 1]
    emd, w = npss_parts(pr, gt, slice(0, 2))
    # 정규화 누적: gt [1, 1], pr [0.5, 1] → EMD = 0.5 (차원마다), 가중치 4 씩 같음 → NPSS = 0.5
    assert np.allclose(emd, 0.5) and np.allclose(w, 4.0)
    assert npss_reduce(emd, w) == pytest.approx(0.5)


def test_foot_skating():
    H = 3.0
    g = torch.zeros(1, T, J, 3, dtype=torch.float64)
    assert foot_skating_per_clip(g, [1], H, FRAMES).item() == pytest.approx(0.0)       # 정지
    g[:, :, 1, 0] = torch.arange(T, dtype=torch.float64) * 2.0                          # 바닥 (h=0) 에서 2 cm/frame
    assert foot_skating_per_clip(g, [1], H, FRAMES).item() == pytest.approx(2.0)
    g[:, :, 1, 1] = H + 1.0                                                              # H 보다 높으면 가중치 0
    assert foot_skating_per_clip(g, [1], H, FRAMES).item() == pytest.approx(0.0)
    g[:, :, 1, 1] = H / 2                                                                # 가중치 2 − √2
    assert foot_skating_per_clip(g, [1], H, FRAMES).item() == pytest.approx(2.0 * (2 - math.sqrt(2)))


def test_mean_center():
    g = torch.randn(B, T, J, 3, dtype=torch.float64) * 100
    c = mean_center(g)
    assert torch.allclose(c[:, :, 0, [0, 2]].mean(dim=1), torch.zeros(B, 2, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(c[..., 1], g[..., 1])                                          # 높이 그대로
    assert torch.allclose(c - c[:, :, :1], g - g[:, :, :1])                              # 관절 사이 상대 위치 그대로


def _random_motion(T_: int = 16, J_: int = 4) -> torch.Tensor:
    angles = torch.randn(1, T_, J_, 3, dtype=torch.float64)
    rot = axis_rotation("x", angles[..., 0]) @ axis_rotation("y", angles[..., 1]) @ axis_rotation("z", angles[..., 2])
    root = torch.randn(1, T_, 3, dtype=torch.float64) * 30
    return encode_pose(root, rot)


def test_baselines_keep_known_frames():
    layout = FeatureLayout(4)
    x = _random_motion()
    spec = TransitionSpec(context_len=4, target_idx=11)
    for fn in (interpolation, zero_velocity):
        y = fn(x, spec, layout)
        assert torch.allclose(y[:, :spec.context_len], x[:, :spec.context_len])
        assert torch.allclose(y[:, spec.target_idx:], x[:, spec.target_idx:])
    # zero-velocity: transition = 마지막 context 프레임
    y = zero_velocity(x, spec, layout)
    assert torch.allclose(y[:, spec.seq_slice], x[:, spec.context_len - 1:spec.context_len].expand_as(y[:, spec.seq_slice]))
    # 보간: 루트는 직선 위, 회전은 SLERP 와 같음
    y = interpolation(x, spec, layout)
    root, rot = decode_pose(y, layout)
    root_x, rot_x = decode_pose(x, layout)
    a, b = spec.context_len - 1, spec.target_idx
    w = torch.tensor(1.0 / (b - a), dtype=torch.float64)
    assert torch.allclose(root[:, a + 1], root_x[:, a] + w * (root_x[:, b] - root_x[:, a]))
    q = slerp(matrix_to_quat(rot_x[:, a]), matrix_to_quat(rot_x[:, b]), w.view(1, 1, 1))
    q_y = matrix_to_quat(rot[:, a + 1])
    assert torch.allclose((q * q_y).sum(-1).abs(), torch.ones_like(q[..., 0]), atol=1e-8)
