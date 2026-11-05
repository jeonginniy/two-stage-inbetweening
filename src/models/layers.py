"""
Transformer 구성 요소 (논문 3.3, 3.3.1)
=====================================

이 파일이 하는 일
-----------------
Context Transformer (그리고 3단계 Detail Transformer) 가 공유하는 층을 정의한다.

1. RelativePositionMLP      : 상대 거리 (j − i) → 벡터 r_{j−i} ∈ R^{d_head}   (논문 식 12–13, Learned Relative PE)
2. RelativeMultiHeadAttention: 점수 = (q_i · k_j + q_i · r_{j−i}) · scale, 마스크 적용 후 softmax
3. FeedForward              : 위치별 2층 MLP
4. KeyframePositionMLP      : [t − t_last_ctx, t_target − t] → R^{d_model}  (논문 식 11, Keyframe PE)

모든 블록은 pre-LayerNorm 방식:  x + Sublayer(LayerNorm(x)).

shape 표기: B = 배치, T = 프레임 수, H = 헤드 수, d = d_head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RelativePositionMLP(nn.Module):
    """상대 프레임 거리 → 상대 위치 임베딩 (논문 식 12–13).

    고정 길이 표를 쓰지 않고 거리 값 자체를 MLP 에 넣으므로, 학습 때보다 긴 클립에도
    값을 만들 수 있다 (논문 3.3.1 의 장점).

    forward(T) 출력: (2T−1, d_head). 행 k 는 상대 거리 k − (T−1) ∈ [−(T−1), T−1].
    """

    def __init__(self, d_head: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_head), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(d_head, d_head), nn.Dropout(dropout),
        )

    def forward(self, T: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        dist = torch.arange(-(T - 1), T, dtype=dtype, device=device)[:, None]   # (2T−1, 1)
        return self.net(dist)                                                   # (2T−1, d)


def relative_index(T: int, device: torch.device | None = None) -> Tensor:
    """(T, T) 정수 표. [i, j] = (j − i) + (T − 1) → RelativePositionMLP 출력의 행 번호.

    i = query 프레임, j = key 프레임. (공식 코드의 skew 트릭과 같은 방향: key − query)
    """
    t = torch.arange(T, device=device)
    return t[None, :] - t[:, None] + (T - 1)


class RelativeMultiHeadAttention(nn.Module):
    """상대 위치 임베딩을 쓰는 멀티헤드 셀프 어텐션 (pre-LN, residual 포함).

    점수_{h,i,j} = (q_{h,i} · k_{h,j} + q_{h,i} · r_{j−i}) × scale
    r 은 모든 헤드가 공유한다 (공식 코드와 같음).
    """

    def __init__(self, d_model: int, n_head: int, d_head: int, dropout: float = 0.0,
                 bias: bool = False, scale: str = "d_head"):
        super().__init__()
        self.n_head, self.d_head = n_head, d_head
        if scale == "d_head":
            self.scale = 1.0 / math.sqrt(d_head)
        elif scale == "d_model":
            self.scale = 1.0 / math.sqrt(d_model)
        else:
            raise ValueError(f"attn_scale 은 'd_head' 또는 'd_model': {scale!r}")
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out = nn.Linear(n_head * d_head, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, rel_emb: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """
        입력:
            x        : (B, T, d_model)
            rel_emb  : (2T−1, d_head)  RelativePositionMLP 출력
            key_mask : (T, T) 또는 (B, T, T) bool. True = 그 (query i, key j) 를 **볼 수 있음**.
                       None 이면 전부 볼 수 있음.
        출력:
            (B, T, d_model)  = x + Attention(LayerNorm(x))
        """
        B, T, _ = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)    # (B, H, T, d)
        k = self.k(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        content = q @ k.transpose(-1, -2)                                     # (B, H, T, T)
        qr = q @ rel_emb.transpose(0, 1)                                      # (B, H, T, 2T−1)
        idx = relative_index(T, x.device).expand(B, self.n_head, T, T)
        position = torch.gather(qr, -1, idx)                                  # (B, H, T, T): q_i·r_{j−i}
        score = (content + position) * self.scale

        if key_mask is not None:
            allowed = key_mask if key_mask.dim() == 3 else key_mask[None]
            score = score.masked_fill(~allowed[:, None], float("-inf"))
        attn = self.attn_drop(score.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, T, self.n_head * self.d_head)
        return x + self.out_drop(self.out(y))


class FeedForward(nn.Module):
    """위치별 FFN (pre-LN, residual 포함): x + W2·ReLU(W1·LN(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class KeyframePositionMLP(nn.Module):
    """Keyframe 위치 인코딩 (논문 식 11).

    입력 (..., T, 2): 각 프레임의 [마지막 context 프레임으로부터 거리, target 프레임까지 거리]
    출력 (..., T, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, keyframe_pos: Tensor) -> Tensor:
        return self.net(keyframe_pos)
