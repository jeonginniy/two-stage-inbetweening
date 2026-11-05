"""
Context Transformer 입력 만들기 (논문 3.3, 3.6.1, 3.6.2)
=====================================================

이 파일이 하는 일
-----------------
정규화된 정답 클립 x (B, T, D) 에서 "어디를 비울지" 를 정하고, 모델 입력과 마스크를 만든다.

    TransitionSpec   : context 길이, target 위치, 제약(midway) 프레임 번호   ← sample_transition
    key_mask         : (T, T) 어텐션 마스크 (그림 5a). key 는 context·제약·target 프레임만
    data_mask        : (T, 1) m_in. 아는 프레임 1, 모르는 프레임 0
    keyframe_positions: (1, T, 2) Keyframe PE 입력 [t − (t_ctx−1), t_tgt − t]   (식 11)
    build_context_input: x ⊙ m_in, 루트 위치는 아는 프레임 사이 선형보간, 마지막에 m_in 이어붙임
                         → (B, T, D + 1)
    merge_prediction : transition 프레임만 모델 출력, 나머지는 정답 (3.6.2)
    postprocess      : 식 (23) 후처리 (근사). 추론 때만

프레임 번호 예 (context_len 10, trans 30, window 42):
    0..9 context | 10..39 transition | 40 target | 41 (target 다음; 후처리용, 모델은 안 봄)

[가정] 논문 3.3 의 "회전 = 0" 은 정규화된 공간의 0 (= 데이터 평균 회전) 으로 구현한다 (공식 코드와 같음).
"""

from __future__ import annotations

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


def sample_transition(
    context_len: int,
    min_trans: int,
    max_trans: int,
    window: int,
    midway_amount: float = 0.0,
    midway_p: float = 0.0,
    rng: random.Random | None = None,
) -> TransitionSpec:
    """학습용 무작위 TransitionSpec (논문 3.6.1: transition 길이를 [min, max] 에서 균등하게).

    제약 프레임 [공식코드]: int(midway_amount × trans_len) 번 시도, 각각 확률 midway_p 로
    transition 안의 임의 프레임 하나를 제약 프레임으로 추가한다.
    """
    rng = rng or random
    if context_len + max_trans >= window:
        raise ValueError(f"window({window}) 가 context_len + max_trans + 1 보다 작습니다.")
    trans = rng.randint(min_trans, max_trans)
    target = context_len + trans
    midway = set()
    for _ in range(int(midway_amount * trans)):
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


def postprocess(y: Tensor, x_gt: Tensor, spec: TransitionSpec) -> Tensor:
    """식 (23) 후처리의 근사 [공식코드 방식].

    transition 전체에 같은 offset δ 를 더한다. δ 는 양 끝에서
        (마지막 context 두 프레임으로 선형 외삽한 값 − 첫 transition 예측값)
        (target 과 target+1 로 거꾸로 외삽한 값 − 마지막 transition 예측값)
    의 평균. target + 1 프레임이 필요하다 (window = t_ctx + trans + 2 인 이유).

    입력:  y, x_gt : (B, T, D)  (같은 공간; 보통 정규화된 값), T ≥ target_idx + 2
    출력:  새 텐서
    """
    s, e = spec.context_len, spec.target_idx
    if x_gt.shape[1] < e + 2 or s < 2:
        raise ValueError("후처리에는 context 2프레임 이상과 target 다음 프레임이 필요합니다.")
    start_ext = 2 * x_gt[:, s - 1] - x_gt[:, s - 2]
    end_ext = 2 * x_gt[:, e] - x_gt[:, e + 1]
    delta = ((start_ext - y[:, s]) + (end_ext - y[:, e - 1])) / 2           # (B, D)
    out = y.clone()
    out[:, s:e] = y[:, s:e] + delta[:, None]
    return out
