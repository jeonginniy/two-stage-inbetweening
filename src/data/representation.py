"""
모델 입력 표현 (논문 3.2 Data Representation)
============================================

이 파일이 하는 일
-----------------
한 프레임의 포즈를 모델이 읽을 1D 벡터로 바꾸고(encode), 다시 되돌린다(decode).

논문 3.2 요약:
    - 프레임 t 의 포즈 = 모든 관절의 로컬 회전 r_t ∈ R^{J×6}  (6D 표현)
                       + 루트의 월드 위치 p_t ∈ R^3
    - r_t, p_t 를 펼쳐서(flatten) 이어 붙이면 x̃_t ∈ R^D,   D = J×6 + 3 = 22×6 + 3 = 135
    - z-score 정규화:  x_t = (x̃_t − μ_x) / σ_x
    - 클립 하나: x ∈ R^{T×D}

벡터 안의 배치 (FeatureLayout):
    인덱스  0 ~ 131 : 관절 0..21 의 6D 회전 (관절 j 는 [6j, 6j+6))
    인덱스 132 ~ 134 : 루트 위치 (x, y, z)

사용 순서
---------
(정렬된) root_pos, local_rot ──encode_pose──▶ x̃ ──ZScoreNormalizer.normalize──▶ x   (모델 입력)
모델 출력 y ──ZScoreNormalizer.denormalize──▶ ỹ ──decode_pose──▶ root_pos, local_rot  (식 10)

통계 μ_x, σ_x 는 RunningStats 로 train_stats 분할(window 50, offset 20)에서 구한다
(논문 3.6.1, scripts/compute_stats.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.data.rotations import matrix_to_6d, sixd_to_matrix


@dataclass(frozen=True)
class FeatureLayout:
    """x̃_t 벡터 안에서 각 정보가 놓인 위치.

    2단계에서 모델 출력 y 를 잘라 쓸 때 (논문 식 10 의 R(·), P(·)) 이 slice 를 쓴다.
    """

    num_joints: int

    @property
    def rot_dim(self) -> int:
        """회전 부분 길이 = J × 6 (LaFAN1: 132)."""
        return self.num_joints * 6

    @property
    def dim(self) -> int:
        """전체 길이 D = J × 6 + 3 (LaFAN1: 135)."""
        return self.rot_dim + 3

    @property
    def rot_slice(self) -> slice:
        return slice(0, self.rot_dim)

    @property
    def pos_slice(self) -> slice:
        return slice(self.rot_dim, self.rot_dim + 3)


def encode_pose(root_pos: Tensor, local_rot: Tensor) -> Tensor:
    """(루트 위치, 로컬 회전행렬) → x̃ (정규화 전 입력 벡터). 논문 3.2.

    입력:
        root_pos  : (..., 3)        예) (B, T, 3)
        local_rot : (..., J, 3, 3)  예) (B, T, 22, 3, 3)
    출력:
        x̃ : (..., J*6 + 3)          예) (B, T, 135)
    """
    rot6d = matrix_to_6d(local_rot)                     # (..., J, 6)
    rot_flat = rot6d.reshape(*rot6d.shape[:-2], -1)     # (..., J*6)
    return torch.cat([rot_flat, root_pos], dim=-1)


def extract_rotation_6d(x: Tensor, layout: FeatureLayout) -> Tensor:
    """논문 식 (10) 의 R(·): 벡터에서 6D 회전 부분만 꺼낸다.  (..., D) → (..., J, 6)."""
    r = x[..., layout.rot_slice]
    return r.reshape(*r.shape[:-1], layout.num_joints, 6)


def extract_root_position(x: Tensor, layout: FeatureLayout) -> Tensor:
    """논문 식 (10) 의 P(·): 벡터에서 루트 위치만 꺼낸다.  (..., D) → (..., 3)."""
    return x[..., layout.pos_slice]


def decode_pose(x: Tensor, layout: FeatureLayout) -> tuple[Tensor, Tensor]:
    """x̃ → (루트 위치, 로컬 회전행렬). encode_pose 의 역함수.

    6D → 회전행렬 은 Gram–Schmidt 를 쓰므로 모델 출력처럼 "회전행렬에서 나온 값이
    아닌" 6D 가 들어와도 항상 올바른 회전행렬이 나온다.

    입력:  x : (..., D)          예) (B, T, 135)   (정규화를 되돌린 값이어야 함)
    출력:
        root_pos  : (..., 3)        예) (B, T, 3)
        local_rot : (..., J, 3, 3)  예) (B, T, 22, 3, 3)
    """
    if x.shape[-1] != layout.dim:
        raise ValueError(f"마지막 차원이 {layout.dim} 이어야 합니다: {x.shape[-1]}")
    local_rot = sixd_to_matrix(extract_rotation_6d(x, layout))
    return extract_root_position(x, layout), local_rot


# -----------------------------------------------------------------------------
# z-score 정규화 (논문 3.2 마지막 문단, 3.6.1)
# -----------------------------------------------------------------------------
class RunningStats:
    """많은 배치를 차례로 넣으며 차원별 평균/표준편차를 계산한다 (메모리 절약용).

    공식 코드는 모든 데이터를 한 배열에 모은 뒤 np.mean/np.std 를 쓰지만, 결과는 같다:
        mean = Σx / N,   std = sqrt(Σx² / N − mean²)   (모집단 표준편차, ddof=0)
    수치 오차를 줄이기 위해 float64 로 누적한다.
    """

    def __init__(self, dim: int):
        self.n = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: Tensor) -> None:
        """x : (..., D). 앞쪽 모든 차원(배치, 시간)을 샘플로 보고 누적한다."""
        x = x.detach().to("cpu", torch.float64).reshape(-1, x.shape[-1])
        self.n += x.shape[0]
        self.sum += x.sum(dim=0)
        self.sum_sq += (x * x).sum(dim=0)

    @property
    def mean(self) -> Tensor:
        return self.sum / self.n

    @property
    def std(self) -> Tensor:
        var = self.sum_sq / self.n - self.mean ** 2
        return var.clamp_min(0.0).sqrt()


class ZScoreNormalizer:
    """x = (x̃ − μ) / σ  와 그 역변환.

    [가정/차이] 공식 코드는 σ 로 그대로 나누지만, 여기서는 σ < min_std 인 차원
    (값이 사실상 상수인 차원)은 σ = 1 로 바꿔서 0 나누기/폭주를 막는다.
    LaFAN1 에서 이런 차원이 실제로 있는지는 scripts/compute_stats.py 가 출력해 준다.
    """

    def __init__(self, mean: Tensor, std: Tensor, min_std: float = 1e-6):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        if mean.shape != std.shape or mean.ndim != 1:
            raise ValueError("mean, std 는 같은 shape 의 1D 텐서여야 합니다.")
        self.raw_std = std.clone()                                   # 저장/비교용 원래 σ
        self.mean = mean
        self.std = torch.where(std < min_std, torch.ones_like(std), std)
        self.min_std = min_std

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def to(self, device=None, dtype=None) -> "ZScoreNormalizer":
        """mean/std 를 지정한 device/dtype 으로 옮긴다 (자기 자신 반환)."""
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        return self

    def normalize(self, x_raw: Tensor) -> Tensor:
        """x̃ (..., D) → x (..., D)."""
        return (x_raw - self.mean.to(x_raw)) / self.std.to(x_raw)

    def denormalize(self, x: Tensor) -> Tensor:
        """x (..., D) → x̃ (..., D)."""
        return x * self.std.to(x) + self.mean.to(x)

    # ----- 저장/불러오기 (.npz) -----
    def save(self, path: str | Path, **meta) -> None:
        """mean, std(원래 값)와 부가정보(meta)를 .npz 로 저장."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean.cpu().double().numpy(),
            std=self.raw_std.cpu().double().numpy(),
            **{k: np.asarray(v) for k, v in meta.items()},
        )

    @classmethod
    def load(cls, path: str | Path, min_std: float = 1e-6) -> "ZScoreNormalizer":
        with np.load(path) as f:
            return cls(torch.from_numpy(f["mean"]), torch.from_numpy(f["std"]), min_std=min_std)

    @classmethod
    def from_stats(cls, stats: RunningStats, min_std: float = 1e-6) -> "ZScoreNormalizer":
        return cls(stats.mean, stats.std, min_std=min_std)
