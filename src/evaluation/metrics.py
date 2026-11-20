"""
평가 지표: L2P, L2Q, NPSS, 발 미끄러짐 (4단계, 논문 4.1)
======================================================

이 파일이 하는 일
-----------------
논문 4.1 (그리고 Harvey et al. 2020 "Robust Motion In-betweening") 의 LaFAN1 벤치마크 지표를 계산한다.
공식 README 수치와 비교할 수 있도록 공식 코드의 정의를 **읽고** 같은 식으로 새로 작성했다 (import/복사 없음).
정의가 논문에 없어 공식 코드를 따른 부분은 docs/assumptions.md F1~F4.

    L2P  : 전역 관절 위치를 z-score (μ, σ: bench_stats) 로 정규화 → 프레임마다 (J×3) 벡터의 L2
           → transition 프레임 평균 → 클립 평균                                    (낮을수록 좋음)
    L2Q  : 전역 회전 쿼터니언 (J×4) 의 L2 → transition 프레임 평균 → 클립 평균. 부호 뒤집힘은 시간 방향으로 제거
    NPSS : transition 구간 전역 쿼터니언의 시간축 파워 스펙트럼 → 누적분포의 EMD (차원별)
           → GT 전체 파워로 가중 평균. 가중치는 "데이터셋 전체" 에서 정규화한 뒤 합 (Gopalakrishnan et al. 2019)
    발 미끄러짐 (논문에 없음) : 발가락 높이 h < H 일 때 수평 속도 × (2 − 2^(h/H)), cm/frame
    자세·연속성 지표 (논문에 없음, docs/assumptions.md J8) :
        boundary_jump : 두 경계 (마지막 context → 첫 transition, 마지막 transition → target) 에서
                        예측 속도와 정답 속도의 차이 (관절 평균 L2, cm/frame). 점프 (순간이동) 가 있으면 커짐. GT = 0
        jitter        : transition 관절 가속도 크기 평균 (cm/frame²). 떨림. GT 값과 비교
        root_speed    : transition 루트 수평 속도 평균 (cm/frame). "덜 움직여서" 미끄러짐이 줄어든 것 구별 (F10)

쓰는 순서
---------
    mean, std = compute_bench_stats(...)          # 한 번 (캐시)
    acc = MetricAccumulator(mean, std, foot_idx, H)
    for 배치:  acc.add(gpos_pred, gpos_gt, gquat_pred, gquat_gt, spec.seq_slice)
    acc.result() → {"L2P", "L2Q", "NPSS", "foot_skate", "boundary_jump", "jitter", "root_speed", "n_clips"}

shape 표기: B = 배치, T = 프레임, J = 관절 (22), 좌표 단위 cm.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.representation import FeatureLayout, decode_pose
from src.data.rotations import matrix_to_quat
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root


# -----------------------------------------------------------------------------
# 표현 → 전역 위치/회전
# -----------------------------------------------------------------------------
def global_rot_pos(x_raw: Tensor, offsets: Tensor, skeleton: Skeleton,
                   layout: FeatureLayout) -> tuple[Tensor, Tensor]:
    """정규화 전 벡터 → 전역 회전/위치 (FK, 논문 식 17).

    입력:  x_raw (B, T, D), offsets (B, 1, J, 3)
    출력:  (global_rot (B, T, J, 3, 3), global_pos (B, T, J, 3))
    """
    root_pos, local_rot = decode_pose(x_raw, layout)
    local_pos = local_positions_from_root(root_pos, offsets)
    return forward_kinematics(local_rot, local_pos, skeleton.parents)


def mean_center(gpos: Tensor) -> Tensor:
    """학습용 정렬 좌표 (마지막 context 프레임 루트 XZ = 0) → "평균 중심" 좌표 (클립 전체 루트 XZ 평균 = 0).

    [공식코드] RMI 벤치마크 통계는 평균 중심 정렬로 계산한다. 두 정렬은 같은 yaw 회전을 쓰고 평행이동만 다르므로
    (회전이 선형이라 평균과 순서를 바꿀 수 있음) 루트 XZ 평균만 빼면 된다. 높이 Y 는 그대로.
    입력/출력:  (B, T, J, 3)
    """
    off = gpos[:, :, 0, :].mean(dim=1)                       # (B, 3) 루트 평균
    out = gpos.clone()
    out[..., 0] -= off[:, None, None, 0]
    out[..., 2] -= off[:, None, None, 2]
    return out


def remove_quat_discontinuities(q: Tensor) -> Tensor:
    """시간 방향 쿼터니언 부호 뒤집힘 제거. q 와 −q 는 같은 회전이므로, 앞 프레임과 내적이 음수면 부호를 바꾼다.

    입력/출력:  (B, T, J, 4). 프레임 0 의 부호를 기준으로 차례로 맞춘다.
    """
    q = q.clone()
    for t in range(1, q.shape[1]):
        flip = (q[:, t] * q[:, t - 1]).sum(-1, keepdim=True) < 0      # (B, J, 1)
        q[:, t] = torch.where(flip, -q[:, t], q[:, t])
    return q


def global_quats(global_rot: Tensor) -> Tensor:
    """전역 회전행렬 (B, T, J, 3, 3) → 부호 정리된 쿼터니언 (B, T, J, 4) (w, x, y, z)."""
    return remove_quat_discontinuities(matrix_to_quat(global_rot))


# -----------------------------------------------------------------------------
# 지표 (클립별)
# -----------------------------------------------------------------------------
def l2p_per_clip(gpos_pred: Tensor, gpos_gt: Tensor, mean: Tensor, std: Tensor, frames: slice) -> Tensor:
    """L2P (클립별). 입력: gpos (B, T, J, 3), mean/std (J*3,). 출력: (B,).

    z = (p − μ) / σ 의 차이는 (p_pred − p_gt) / σ 라서, 같은 평행이동을 한 두 좌표에서는 정렬 방식과 무관하다.
    """
    d = (gpos_pred[:, frames] - gpos_gt[:, frames]).flatten(-2) / std          # (B, n, J*3)
    return d.norm(dim=-1).mean(dim=1)


def l2q_per_clip(q_pred: Tensor, q_gt: Tensor, frames: slice) -> Tensor:
    """L2Q (클립별). 입력: q (B, T, J, 4) 부호 정리된 전역 쿼터니언. 출력: (B,)."""
    d = (q_pred[:, frames] - q_gt[:, frames]).flatten(-2)                      # (B, n, J*4)
    return d.norm(dim=-1).mean(dim=1)


def npss_parts(q_pred: Tensor, q_gt: Tensor, frames: slice, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """NPSS 의 클립·차원별 EMD 와 가중치 (GT 전체 파워). 최종 값은 npss_reduce 로 데이터셋 전체에서.

    [공식코드] 푸리에 계수의 실수부를 제곱해 파워로 쓴다 (원 논문은 크기). 공식 README 수치와 맞추기 위함 (F2).
    입력:  q (B, T, J, 4)
    출력:  (emd (B, J*4), weight (B, J*4))  numpy float64
    """
    gt = q_gt[:, frames].flatten(-2).double().cpu().numpy()                    # (B, n, J*4)
    pr = q_pred[:, frames].flatten(-2).double().cpu().numpy()
    p_gt = np.real(np.fft.fft(gt, axis=1)) ** 2                                # (B, n, D)
    p_pr = np.real(np.fft.fft(pr, axis=1)) ** 2
    tot_gt = p_gt.sum(axis=1)                                                  # (B, D)
    tot_pr = p_pr.sum(axis=1)
    norm_gt = p_gt / np.where(tot_gt < eps, np.inf, tot_gt)[:, None]           # 파워가 0 이면 정규화 값 0
    norm_pr = p_pr / np.where(tot_pr < eps, np.inf, tot_pr)[:, None]
    emd = np.abs(np.cumsum(norm_pr, axis=1) - np.cumsum(norm_gt, axis=1)).sum(axis=1)   # (B, D) L1 거리
    return emd, tot_gt


def npss_reduce(emd: np.ndarray, weight: np.ndarray) -> float:
    """NPSS = Σ_{클립, 차원} EMD × (가중치 / 전체 가중치 합). 입력: (N, D) 둘. 출력: 스칼라."""
    return float((emd * weight / weight.sum()).sum())


def foot_skating_per_clip(gpos: Tensor, foot_idx: Sequence[int], height_cm: float, frames: slice) -> Tensor:
    """발 미끄러짐 (클립별, cm/frame) [결정 F4].

    프레임 t 의 발 관절 속도 v_t = |XZ(t) − XZ(t−1)| 에 가중치 w = clip(2 − 2^(h_t / H), 0, 1) 을 곱한다
    (h_t = 높이 Y, h ≥ H 이면 0, h = 0 이면 1: 땅에 닿아 있을수록 미끄러지면 안 됨).
    transition 프레임 (frames) 과 발 관절 평균. 첫 transition 프레임의 속도는 마지막 context 프레임과의 차이.
    입력:  gpos (B, T, J, 3)
    출력:  (B,)
    """
    feet = gpos[..., list(foot_idx), :]                                        # (B, T, F, 3)
    s, e = frames.start, frames.stop
    v = (feet[:, s:e, :, [0, 2]] - feet[:, s - 1:e - 1, :, [0, 2]]).norm(dim=-1)   # (B, n, F)
    h = feet[:, s:e, :, 1]
    w = (2.0 - torch.pow(2.0, h / height_cm)).clamp(0.0, 1.0)
    return (v * w).mean(dim=(1, 2))


def boundary_jump_per_clip(gpos_pred: Tensor, gpos_gt: Tensor, frames: slice) -> Tensor:
    """경계 점프 (클립별, cm/frame) [결정 J8]. 두 경계 속도 오차의 평균.

    시작: v_s = g_s − g_{s−1}, 끝: v_e = g_e − g_{e−1} (e = target 프레임). 예측과 정답의 속도 차이를
    관절별 L2 → 관절 평균 → 두 경계 평균. transition 밖은 정답이라 경계 차이는 "생성한 첫/마지막 프레임" 이 결정한다.
    입력:  gpos (B, T, J, 3), frames = transition 구간 [s, e)
    출력:  (B,)
    """
    s, e = frames.start, frames.stop
    out = []
    for a, b in ((s - 1, s), (e - 1, e)):
        d = (gpos_pred[:, b] - gpos_pred[:, a]) - (gpos_gt[:, b] - gpos_gt[:, a])   # (B, J, 3)
        out.append(d.norm(dim=-1).mean(dim=-1))
    return (out[0] + out[1]) / 2


def jitter_per_clip(gpos: Tensor, frames: slice) -> Tensor:
    """떨림 (클립별, cm/frame²) [결정 J8]. transition 프레임 t 의 가속도 |g_{t+1} − 2g_t + g_{t−1}|, 관절·프레임 평균.

    입력:  gpos (B, T, J, 3). 첫/마지막 transition 프레임은 이웃한 context/target 프레임을 쓴다.
    출력:  (B,)
    """
    s, e = frames.start, frames.stop
    acc = gpos[:, s + 1:e + 1] - 2 * gpos[:, s:e] + gpos[:, s - 1:e - 1]          # (B, n, J, 3)
    return acc.norm(dim=-1).mean(dim=(1, 2))


def root_speed_per_clip(gpos: Tensor, frames: slice) -> Tensor:
    """루트 수평 속도 (클립별, cm/frame) [결정 J8]. transition 프레임과 그 앞 프레임의 XZ 차이 크기 평균.

    입력:  gpos (B, T, J, 3). 출력: (B,)
    """
    s, e = frames.start, frames.stop
    v = gpos[:, s:e, 0, [0, 2]] - gpos[:, s - 1:e - 1, 0, [0, 2]]                 # (B, n, 2)
    return v.norm(dim=-1).mean(dim=1)


# -----------------------------------------------------------------------------
# 데이터셋 전체 모으기
# -----------------------------------------------------------------------------
class MetricAccumulator:
    """배치마다 클립별 지표를 모았다가 데이터셋 전체 값을 낸다.

    생성 인자:
        mean, std  : (J*3,) L2P 통계 (compute_bench_stats)
        foot_idx   : 발 미끄러짐 관절 번호
        height_cm  : 발 미끄러짐 높이 임계값 H
    """

    def __init__(self, mean: Tensor, std: Tensor, foot_idx: Sequence[int], height_cm: float):
        self.mean, self.std = mean, std
        self.foot_idx, self.height_cm = list(foot_idx), float(height_cm)
        self.l2p: list[np.ndarray] = []
        self.l2q: list[np.ndarray] = []
        self.emd: list[np.ndarray] = []
        self.w: list[np.ndarray] = []
        self.skate: list[np.ndarray] = []
        self.jump: list[np.ndarray] = []
        self.jitter: list[np.ndarray] = []
        self.speed: list[np.ndarray] = []

    def add(self, gpos_pred: Tensor, gpos_gt: Tensor, q_pred: Tensor, q_gt: Tensor, frames: slice) -> None:
        """입력: gpos (B, T, J, 3), q (B, T, J, 4) 부호 정리된 전역 쿼터니언, frames = transition 구간."""
        std = self.std.to(gpos_pred)
        self.l2p.append(l2p_per_clip(gpos_pred, gpos_gt, self.mean.to(gpos_pred), std, frames).cpu().numpy())
        self.l2q.append(l2q_per_clip(q_pred, q_gt, frames).cpu().numpy())
        emd, w = npss_parts(q_pred, q_gt, frames)
        self.emd.append(emd)
        self.w.append(w)
        self.skate.append(foot_skating_per_clip(gpos_pred, self.foot_idx, self.height_cm, frames).cpu().numpy())
        self.jump.append(boundary_jump_per_clip(gpos_pred, gpos_gt, frames).cpu().numpy())
        self.jitter.append(jitter_per_clip(gpos_pred, frames).cpu().numpy())
        self.speed.append(root_speed_per_clip(gpos_pred, frames).cpu().numpy())

    def result(self) -> dict[str, float]:
        """{"L2P": 클립 평균, "L2Q": 클립 평균, "NPSS": 데이터셋 가중 합, "foot_skate": cm/frame,
        "boundary_jump": cm/frame, "jitter": cm/frame², "root_speed": cm/frame, "n_clips"}."""
        mean = lambda xs: float(np.concatenate(xs).mean())   # noqa: E731
        return {"L2P": mean(self.l2p), "L2Q": mean(self.l2q),
                "NPSS": npss_reduce(np.concatenate(self.emd), np.concatenate(self.w)),
                "foot_skate": mean(self.skate), "boundary_jump": mean(self.jump),
                "jitter": mean(self.jitter), "root_speed": mean(self.speed),
                "n_clips": int(sum(len(x) for x in self.l2p))}


@torch.no_grad()
def compute_bench_stats(loader, skeleton: Skeleton, context_len: int, world_forward: str,
                        device: torch.device) -> tuple[Tensor, Tensor]:
    """L2P 정규화 통계 μ, σ (J*3,) [공식코드 F1].

    각 클립을 마지막 context 프레임 기준으로 정렬 (논문 3.6.1, prepare_batch 와 같은 함수) → 평균 중심 →
    FK 전역 위치 → 모든 클립·프레임의 차원별 평균, 모집단 표준편차 (ddof=0). float64 로 누적.
    입력:  loader (LafanDataset DataLoader: root_pos, local_rot, offsets)
    출력:  (mean (J*3,), std (J*3,)) float64, cpu
    """
    from src.data.preprocess import canonicalize_root          # 여기서만 쓰므로 지역 import

    n, s1, s2 = 0, None, None
    for batch in loader:
        root = batch["root_pos"].to(device, torch.float64)
        rot = batch["local_rot"].to(device, torch.float64)
        offsets = batch["offsets"].to(device, torch.float64)[:, None]
        # 기준 프레임 = 마지막 context 프레임 (0-based context_len − 1, assumptions B4)
        root_c, rot_c, _ = canonicalize_root(root, rot, context_len - 1, skeleton.root_forward_local_axis,
                                             world_forward=world_forward)
        gpos = forward_kinematics(rot_c, local_positions_from_root(root_c, offsets), skeleton.parents)[1]
        g = mean_center(gpos).flatten(-2).reshape(-1, gpos.shape[-2] * 3)      # (B*T, J*3)
        n += g.shape[0]
        s1 = g.sum(0) if s1 is None else s1 + g.sum(0)
        s2 = (g ** 2).sum(0) if s2 is None else s2 + (g ** 2).sum(0)
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt()
    return mean.cpu(), std.cpu()
