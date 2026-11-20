"""
Context Transformer (논문 3.3)
=============================

이 파일이 하는 일
-----------------
"빈칸 채우기" 모델. context 프레임, (선택) 제약 프레임, target 프레임만 알고
transition 프레임을 예측한다.

    입력  x_ctx ⊕ m_in : (B, T, D + d_mask)        D = J*6+3 (LaFAN1: 135), d_mask = 1
      │ E  : 2층 MLP (PReLU)                  → (B, T, d_model)
      │ +  Keyframe PE (식 11)                 ← keyframe_pos (B or 1, T, 2)
      │ N × [ RelMHA (식 12–13, 마스크 그림 5a) → FFN ]   (pre-LN)
      │ LayerNorm → D : 2층 MLP (PReLU)
      ▼
    출력  y_ctx : (B, T, D)    (정규화된 공간)

입력/출력 차원은 생성자 인자로 받는다. 이 파일에 135, 22 같은 숫자는 없다.
보통은 build_context_transformer(model_cfg, layout) 로 만든다 (configs/model/context.yaml + 스켈레톤).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from src.data.representation import FeatureLayout
from src.models.layers import (
    FeedForward,
    KeyframePositionMLP,
    RelativeMultiHeadAttention,
    RelativePositionMLP,
)


class ContextTransformer(nn.Module):
    """논문 3.3 Context Transformer.

    생성 인자:
        d_in, d_out : 입력/출력 차원. 보통 d_in = D + d_mask, d_out = D
        나머지      : configs/model/context.yaml 의 model 절과 같은 이름
                      (pe_scale 은 설정에 없으면 1.0 = 예전 체크포인트와 같은 동작)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 512,
        n_layer: int = 6,
        n_head: int = 8,
        d_head: int = 64,
        d_ff: int = 512,
        d_encoder_h: int = 512,
        d_decoder_h: int = 512,
        dropout: float = 0.0,
        attn_bias: bool = False,
        attn_scale: str = "d_head",
        pe_scale: float = 1.0,
    ):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_encoder_h), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(d_encoder_h, d_model), nn.PReLU(), nn.Dropout(dropout),
        )
        # pe_scale: 두 위치 인코딩에 넣는 거리 (프레임) 에 곱하는 값. 1.0 = 기존 (ctx_base), 1/30 = 초 단위 (J6)
        self.keyframe_pe = KeyframePositionMLP(d_model, dropout, pe_scale)
        self.rel_pe = RelativePositionMLP(d_head, dropout, pe_scale)
        self.attn_layers = nn.ModuleList(
            RelativeMultiHeadAttention(d_model, n_head, d_head, dropout, attn_bias, attn_scale)
            for _ in range(n_layer)
        )
        self.ff_layers = nn.ModuleList(FeedForward(d_model, d_ff, dropout) for _ in range(n_layer))
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_decoder_h), nn.PReLU(),
            nn.Linear(d_decoder_h, d_out),
        )

    def forward(self, x: Tensor, keyframe_pos: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """
        입력:
            x            : (B, T, d_in)   x_ctx ⊕ m_in  (src.training.context_inputs.build_context_input)
            keyframe_pos : (B 또는 1, T, 2)
            key_mask     : (T, T) 또는 (B, T, T) bool, True = 볼 수 있음 (그림 5a)
        출력:
            (B, T, d_out)
        """
        if x.shape[-1] != self.d_in:
            raise ValueError(f"입력 마지막 차원이 {self.d_in} 이어야 합니다: {tuple(x.shape)}")
        h = self.encoder(x) + self.keyframe_pe(keyframe_pos.to(x.dtype))
        rel = self.rel_pe(x.shape[1], h.dtype, h.device)
        for attn, ff in zip(self.attn_layers, self.ff_layers):
            h = ff(attn(h, rel, key_mask))
        return self.decoder(self.final_norm(h))


def build_context_transformer(model_cfg: dict[str, Any], layout: FeatureLayout) -> ContextTransformer:
    """설정 dict + FeatureLayout → ContextTransformer.

    입력:
        model_cfg : configs/model/context.yaml 의 "model" 절
        layout    : FeatureLayout(skeleton.num_joints)  ← 관절 수는 스켈레톤 파일에서
    입력 차원 = layout.dim + d_mask, 출력 차원 = layout.dim.
    """
    cfg = dict(model_cfg)
    d_mask = int(cfg.pop("d_mask", 1))
    if d_mask != 1:
        raise NotImplementedError("지금은 프레임 단위 마스크(d_mask=1)만 지원합니다.")
    return ContextTransformer(d_in=layout.dim + d_mask, d_out=layout.dim, **cfg)


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
