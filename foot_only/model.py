"""
발 전용 모델: Context Transformer 구조를 그대로 쓰고 입출력 차원만 바꾼다 [결정 H1, H2]
======================================================================================

    build_foot_model   설정 dict + FeatureLayout → ContextTransformer(d_in = D + 12 + d_mask, d_out = 12 + 4)
    split_output       모델 출력 (B, T, 16) → 발 위치 (정규화된 값, B, T, 12) + 접촉 logit (B, T, 4)

구조 (src/models/context_transformer.py, 논문 3.3): 입력 MLP → Keyframe PE 더하기 →
pre-LN Transformer 블록 6개 (상대 위치 어텐션) → LayerNorm → 출력 MLP. 바뀐 곳은 처음과 마지막 선형층의 차원뿐.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from foot_only.data import FOOT_DIM, N_FOOT
from src.data.representation import FeatureLayout
from src.models.context_transformer import ContextTransformer

FOOT_OUT_DIM = FOOT_DIM + N_FOOT        # 16 = 발 위치 12 + 접촉 logit 4


def build_foot_model(model_cfg: dict[str, Any], layout: FeatureLayout) -> ContextTransformer:
    """configs/foot_only.yaml 의 model 절 → 모델.

    입력 차원 = layout.dim (전신 D=135) + FOOT_DIM (12) + d_mask (1) = 148
    출력 차원 = FOOT_DIM (12) + N_FOOT (4) = 16
    """
    cfg = dict(model_cfg)
    d_mask = int(cfg.pop("d_mask"))
    return ContextTransformer(d_in=layout.dim + FOOT_DIM + d_mask, d_out=FOOT_OUT_DIM, **cfg)


def split_output(out: Tensor) -> tuple[Tensor, Tensor]:
    """(B, T, 16) → (발 위치 정규화된 값 (B, T, 12), 접촉 logit (B, T, 4))."""
    return out[..., :FOOT_DIM], out[..., FOOT_DIM:]
