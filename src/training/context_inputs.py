"""
Context Transformer 입력 만들기 (논문 3.3, 3.6.1, 3.6.2)
=====================================================

이 파일이 하는 일
-----------------
정규화된 정답 클립 x (B, T, D) 에서 "어디를 비울지" 를 정하고, 모델 입력과 마스크를 만든다.

    TransitionSpec   : context 길이, target 위치, 제약(midway) 프레임 번호   ← sample_transition
                       (길이 분포 uniform | log_uniform, 제약 없는 배치 확률·상한: docs/assumptions.md J2, J3)
    key_mask         : (T, T) 어텐션 마스크 (그림 5a). key 는 context·제약·target 프레임만
    data_mask        : (T, 1) m_in. 아는 프레임 1, 모르는 프레임 0
    keyframe_positions: (1, T, 2) Keyframe PE 입력 [t − (t_ctx−1), t_tgt − t]   (식 11)
    build_context_input: x ⊙ m_in, 루트 위치는 아는 프레임 사이 선형보간, 마지막에 m_in 이어붙임
                         → (B, T, D + 1)
    merge_prediction : transition 프레임만 모델 출력, 나머지는 정답 (3.6.2)
    postprocess      : 식 (23) 후처리 (근사). 추론 때만. mode "mean" (상수 δ, 식 23) | "ramp" (양 끝 δ 선형, J5)

프레임 번호 예 (context_len 10, trans 30, window 42):
    0..9 context | 10..39 transition | 40 target | 41 (target 다음; 후처리용, 모델은 안 봄)

[가정] 논문 3.3 의 "회전 = 0" 은 정규화된 공간의 0 (= 데이터 평균 회전) 으로 구현한다 (공식 코드와 같음).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.data.representation import FeatureLayout


@dataclass(frozen=True)
class TransitionSpec:
    """클립 하나(배치 전체 공통)의 빈칸 위치.

    속성:
        context_len : t_ctx (context 프레임 0 .. t_ctx−1)
        target_idx  : target 프레임 번호 = t_ctx + transition 길이
        midway      : transition 안의 제약 프레임 번호 (정렬됨, 중복 없음)
    """

    context_len: int
    target_idx: int
    midway: tuple[int, ...] = field(default=())

    def __post_init__(self):
        if self.target_idx <= self.context_len:
            raise ValueError("transition 길이는 1 이상이어야 합니다.")
        if any(not (self.context_len <= m < self.target_idx) for m in self.midway):
            raise ValueError(f"midway 프레임은 transition 안에 있어야 합니다: {self.midway}")
        object.__setattr__(self, "midway", tuple(sorted(set(self.midway))))

    @property
    def trans_len(self) -> int:
        return self.target_idx - self.context_len

    @property
    def seq_slice(self) -> slice:
        """transition 프레임 구간 [t_ctx, target)."""
        return slice(self.context_len, self.target_idx)

    @property
    def anchor_frames(self) -> list[int]:
        """선형보간 기준 프레임: 마지막 context, 제약 프레임들, target (오름차순)."""
        return [self.context_len - 1, *self.midway, self.target_idx]

    def known_frames(self) -> list[int]:
        """값을 아는 프레임 전체: context + 제약 + target."""
        return [*range(self.context_len), *self.midway, self.target_idx]


TRANS_DISTRIBUTIONS = ("uniform", "log_uniform")


def sample_trans_len(min_trans: int, max_trans: int, distribution: str = "uniform",
                     rng: random.Random | None = None) -> int:
    """transition 길이 하나를 뽑는다.

    uniform     : [min, max] 정수 균등 (논문 3.6.1, 기존 ctx_base)
    log_uniform : log(길이) 가 균등 → P(t) ∝ log((t+1)/t) ≈ 1/t  [결정 J2]
                  예) [5, 300] 이면 5–30 이 약 45%, 30–90 이 27%, 90–300 이 29%.
                  학습 범위를 길게 늘려도 짧은 길이를 충분히 본다 (균등이면 5–30 이 9% 뿐).
    """
    rng = rng or random
    if distribution == "uniform":
        return rng.randint(min_trans, max_trans)
    if distribution == "log_uniform":
        u = rng.uniform(math.log(min_trans), math.log(max_trans + 1))
        return min(max(int(math.exp(u)), min_trans), max_trans)
    raise ValueError(f"trans_dist 는 {TRANS_DISTRIBUTIONS} 중 하나: {distribution!r}")


def sample_transition(
    context_len: int,
    min_trans: int,
    max_trans: int,
    window: int,
    midway_amount: float = 0.0,
    midway_p: float = 0.0,
    rng: random.Random | None = None,
    *,
    distribution: str = "uniform",
    midway_none_p: float = 0.0,
    midway_max: int | None = None,
) -> TransitionSpec:
    """학습용 무작위 TransitionSpec (논문 3.6.1: transition 길이를 [min, max] 에서 균등하게).

    제약 프레임 [공식코드]: int(midway_amount × trans_len) 번 시도, 각각 확률 midway_p 로
    transition 안의 임의 프레임 하나를 제약 프레임으로 추가한다.

    추가 옵션 (기본값 = 기존 동작과 같음, 같은 rng 호출 순서) [결정 J3]:
        distribution  : 길이 분포 "uniform" | "log_uniform" (sample_trans_len)
        midway_none_p : 이 확률로 제약 프레임 없이 뽑는다 (평가는 항상 제약 없음)
        midway_max    : 시도 횟수 상한. 없으면 긴 transition 에서 제약이 너무 많아진다
                        (예: trans 300 → 30번 시도 → 평균 15개, 제약 없는 배치가 사실상 0)
    """
    rng = rng or random
    if context_len + max_trans >= window:
        raise ValueError(f"window({window}) 가 context_len + max_trans + 1 보다 작습니다.")
    trans = sample_trans_len(min_trans, max_trans, distribution, rng)
    target = context_len + trans
    midway = set()
    tries = int(midway_amount * trans)
    if midway_max is not None:
        tries = min(tries, int(midway_max))
    if midway_none_p > 0 and rng.random() < midway_none_p:
        tries = 0
    for _ in range(tries):
        if rng.random() < midway_p:
            midway.add(rng.randrange(context_len, target))
    return TransitionSpec(context_len, target, tuple(midway))


def key_mask(T: int, spec: TransitionSpec, device=None) -> Tensor:
    """어텐션 마스크 (논문 그림 5a). (T, T) bool, True = query i 가 key j 를 볼 수 있음.

    모든 query 가 context·제약·target 프레임만 key 로 본다 (열 단위 마스크).
    """
    visible = torch.zeros(T, dtype=torch.bool, device=device)
    visible[spec.known_frames()] = True
    return visible[None, :].expand(T, T)


def data_mask(T: int, spec: TransitionSpec, dtype=torch.float32, device=None) -> Tensor:
    """m_in (논문 3.3). (T, 1): 아는 프레임 1, 나머지 0."""
    m = torch.zeros(T, 1, dtype=dtype, device=device)
    m[spec.known_frames()] = 1.0
    return m


def keyframe_positions(T: int, spec: TransitionSpec, dtype=torch.float32, device=None) -> Tensor:
    """Keyframe PE 입력 (논문 식 11). (1, T, 2) = [t − (t_ctx − 1), t_tgt − t].

    마지막 context 프레임에서 첫 값이 0, target 프레임에서 둘째 값이 0.
    """
    t = torch.arange(T, dtype=dtype, device=device)
    return torch.stack([t - (spec.context_len - 1), spec.target_idx - t], dim=-1)[None]


def interpolate_root_placeholder(x: Tensor, spec: TransitionSpec, pos_slice: slice) -> Tensor:
    """아는 프레임(앵커) 사이의 transition 프레임 루트 위치를 선형보간으로 채운다 (논문 3.3).

    입력:  x (B, T, D) 정규화된 값. 앵커 프레임에는 정답이 있어야 한다.
    출력:  새 텐서 (입력은 바꾸지 않음)
    z-score 는 차원별 1차 함수라서, 정규화 공간에서 보간해도 원래 공간에서 보간한 것과 같다.
    """
    x = x.clone()
    anchors = spec.anchor_frames
    for a, b in zip(anchors[:-1], anchors[1:]):
        n = b - a - 1
        if n <= 0:
            continue
        w = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)[:, None] / (b - a)   # (n, 1)
        pa, pb = x[:, a:a + 1, pos_slice], x[:, b:b + 1, pos_slice]                      # (B, 1, 3)
        x[:, a + 1:b, pos_slice] = pa + w * (pb - pa)
    return x


def build_context_input(x: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """정규화된 정답 x (B, T, D) → Context Transformer 입력 (B, T, D + 1).

    1) 모르는 프레임은 0 (x ⊙ m_in)
    2) 루트 위치만 앵커 사이 선형보간
    3) m_in 을 마지막 차원에 이어붙임
    """
    B, T, D = x.shape
    if D != layout.dim:
        raise ValueError(f"x 마지막 차원 {D} ≠ layout.dim {layout.dim}")
    m = data_mask(T, spec, x.dtype, x.device)                     # (T, 1)
    x_in = interpolate_root_placeholder(x * m, spec, layout.pos_slice)
    return torch.cat([x_in, m.expand(B, T, 1)], dim=-1)


def merge_prediction(x_gt: Tensor, model_out: Tensor, spec: TransitionSpec) -> Tensor:
    """transition 프레임만 모델 출력으로, 나머지(context, target, 그 이후)는 정답으로 (논문 3.6.2).

    제약 프레임은 transition 안에 있으므로 모델 출력이 들어간다 (공식 코드와 같음).
    """
    y = x_gt.detach().clone()
    s = spec.seq_slice
    y = torch.cat([y[:, :s.start], model_out[:, s], y[:, s.stop:]], dim=1)
    return y


POSTPROCESS_MODES = ("mean", "ramp")


def postprocess(y: Tensor, x_gt: Tensor, spec: TransitionSpec, mode: str = "mean") -> Tensor:
    """식 (23) 후처리의 근사 [공식코드 방식].

    양 끝의 어긋남
        δ_s = (마지막 context 두 프레임으로 선형 외삽한 값 − 첫 transition 예측값)
        δ_e = (target 과 target+1 로 거꾸로 외삽한 값 − 마지막 transition 예측값)
    으로 transition 을 고친다. target + 1 프레임이 필요하다 (window = t_ctx + trans + 2 인 이유).

    mode:
        "mean" : transition 전체에 같은 δ = (δ_s + δ_e) / 2 를 더한다 (논문 식 23 = 상수 δ, 공식 코드).
                 모델이 target 에 못 미치면 어긋남이 반씩 나뉘어 양 경계에서 점프가 남는다 (F6).
        "ramp" : δ_t = (1 − w_t)·δ_s + w_t·δ_e, w_t = 0 (첫 transition 프레임) → 1 (마지막) 선형 [결정 J5].
                 양 경계의 어긋남이 각각 정확히 0 이 되고, 고친 양은 transition 전체에 고르게 퍼진다.
                 δ_s = δ_e 이면 "mean" 과 같다.

    입력:  y, x_gt : (B, T, D)  (같은 공간; 보통 정규화된 값), T ≥ target_idx + 2
    출력:  새 텐서
    """
    s, e = spec.context_len, spec.target_idx
    if x_gt.shape[1] < e + 2 or s < 2:
        raise ValueError("후처리에는 context 2프레임 이상과 target 다음 프레임이 필요합니다.")
    start_ext = 2 * x_gt[:, s - 1] - x_gt[:, s - 2]
    end_ext = 2 * x_gt[:, e] - x_gt[:, e + 1]
    d_s, d_e = start_ext - y[:, s], end_ext - y[:, e - 1]                    # (B, D) 각각
    out = y.clone()
    if mode == "mean":
        out[:, s:e] = y[:, s:e] + ((d_s + d_e) / 2)[:, None]
    elif mode == "ramp":
        n = e - s
        w = torch.arange(n, dtype=y.dtype, device=y.device) / max(n - 1, 1)  # (n,) 0 → 1
        if n == 1:
            w = w + 0.5                                                      # 한 프레임이면 평균
        out[:, s:e] = y[:, s:e] + (1 - w)[None, :, None] * d_s[:, None] + w[None, :, None] * d_e[:, None]
    else:
        raise ValueError(f"후처리 mode 는 {POSTPROCESS_MODES} 중 하나: {mode!r}")
    return out
