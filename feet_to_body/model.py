"""
발 → 몸 모델: Context Transformer 구조를 그대로 쓰고 입력 차원만 바꾼다 [결정 I1]
================================================================================

    build_body_model   설정 dict + FeatureLayout → ContextTransformer(d_in = D + 16 + d_mask, d_out = D)

구조 (src/models/context_transformer.py, 논문 3.3): 입력 MLP → Keyframe PE 더하기 →
pre-LN Transformer 블록 6개 (상대 위치 어텐션) → LayerNorm → 출력 MLP.
ctx_base 와 다른 곳은 첫 선형층의 입력 차원 (136 → 152) 뿐이다. 출력은 ctx_base 와 같은 전신 135.
"""

from __future__ import annotations

from typing import Any

from feet_to_body.data import COND_DIM
from src.data.representation import FeatureLayout
from src.models.context_transformer import ContextTransformer


def build_body_model(model_cfg: dict[str, Any], layout: FeatureLayout) -> ContextTransformer:
    """configs/feet_to_body.yaml 의 model 절 → 모델.

    입력 차원 = layout.dim (전신 D = 135) + COND_DIM (16) + d_mask (1) = 152
    출력 차원 = layout.dim (135)
    """
    cfg = dict(model_cfg)
    d_mask = int(cfg.pop("d_mask"))
    return ContextTransformer(d_in=layout.dim + COND_DIM + d_mask, d_out=layout.dim, **cfg)
