"""
Detail Transformer (논문 3.4)
============================

이 파일이 하는 일
-----------------
"다듬기" 모델. Context Transformer 가 채운 transition 을 입력으로 받아 동작 디테일을 고치고
(발 미끄러짐 등), 발 접촉 확률도 함께 예측한다.

    입력  x_det ⊕ m_in : (B, T, D + d_mask)       D = J*6+3 (LaFAN1: 135), d_mask = 1
          x_det = 정답 클립에서 transition 구간만 Context 출력으로 바꾼 것 (src.training.detail.build_detail_input)
      │ E  : 2층 MLP (PReLU)                  → (B, T, d_model)
      │ (Keyframe PE 없음: 논문 3.4 "the position relative to context and target frames is no longer that relevant")
      │ N × [ RelMHA (식 12–13, 마스크 그림 5b) → FFN ]   (pre-LN, Context 와 같은 층: src/models/layers.py)
      │ LayerNorm → D : 2층 MLP (PReLU)
      ▼
    출력  (B, T, D + C)   앞 D = 포즈 y_det (정규화된 공간, 식 14 의 R(·), P(·) 로 자름)
                          뒤 C = 발 접촉 로짓 → sigmoid 하면 ĉ (식 15). C = 접촉 관절 수 (LaFAN1: 4)

Context Transformer 와 다른 점은 (1) Keyframe PE 가 없고 (2) 출력에 접촉 C 차원이 더 있는 것뿐이다.
기존 ContextTransformer 클래스와 체크포인트는 그대로 두고, 같은 층을 가져다 새 클래스로 만든다.
입력/출력 차원은 생성자 인자로 받는다. 이 파일에 135, 4, 22 같은 숫자는 없다.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.data.representation import FeatureLayout
from src.models.layers import FeedForward, RelativeMultiHeadAttention, RelativePositionMLP


class DetailTransformer(nn.Module):
    """논문 3.4 Detail Transformer.

    생성 인자:
        d_in      : 입력 차원 = D + d_mask
        d_pose    : 포즈 출력 차원 = D
        n_contact : 발 접촉 출력 수 C (스켈레톤의 contact_joints 수)
        나머지    : configs/model/detail.yaml 의 model 절과 같은 이름
    """

    def __init__(
        self,
        d_in: int,
        d_pose: int,
        n_contact: int,
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
    ):
        super().__init__()
        self.d_in, self.d_pose, self.n_contact = d_in, d_pose, n_contact
        self.d_out = d_pose + n_contact
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_encoder_h), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(d_encoder_h, d_model), nn.PReLU(), nn.Dropout(dropout),
        )
        self.rel_pe = RelativePositionMLP(d_head, dropout)
        self.attn_layers = nn.ModuleList(
            RelativeMultiHeadAttention(d_model, n_head, d_head, dropout, attn_bias, attn_scale)
            for _ in range(n_layer)
        )
        self.ff_layers = nn.ModuleList(FeedForward(d_model, d_ff, dropout) for _ in range(n_layer))
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_decoder_h), nn.PReLU(),
            nn.Linear(d_decoder_h, self.d_out),
        )

    def forward(self, x: Tensor, key_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        입력:
            x        : (B, T, d_in)   x_det ⊕ m_in
            key_mask : (T, T) 또는 (B, T, T) bool, True = 볼 수 있음 (그림 5b)
        출력:
            pose           : (B, T, D)  정규화된 공간의 포즈 (식 14)
            contact_logits : (B, T, C)  sigmoid 전 접촉 로짓 (식 15 의 C(y_det))
        """
        if x.shape[-1] != self.d_in:
            raise ValueError(f"입력 마지막 차원이 {self.d_in} 이어야 합니다: {tuple(x.shape)}")
        h = self.encoder(x)
        rel = self.rel_pe(x.shape[1], h.dtype, h.device)
        for attn, ff in zip(self.attn_layers, self.ff_layers):
            h = ff(attn(h, rel, key_mask))
        out = self.decoder(self.final_norm(h))
        return out[..., :self.d_pose], out[..., self.d_pose:]


def build_detail_transformer(model_cfg: dict[str, Any], layout: FeatureLayout, n_contact: int) -> DetailTransformer:
    """설정 dict + FeatureLayout + 접촉 관절 수 → DetailTransformer.

    입력:
        model_cfg : configs/model/detail.yaml 의 "model" 절
        layout    : FeatureLayout(skeleton.num_joints)
        n_contact : len(skeleton.contact_joints)
    입력 차원 = layout.dim + d_mask, 출력 = layout.dim + n_contact.
    """
    cfg = dict(model_cfg)
    d_mask = int(cfg.pop("d_mask", 1))
    if d_mask != 1:
        raise NotImplementedError("지금은 프레임 단위 마스크(d_mask=1)만 지원합니다.")
    if n_contact < 1:
        raise ValueError("스켈레톤 설정에 contact_joints 가 없습니다 (발 접촉 출력 수 0).")
    return DetailTransformer(d_in=layout.dim + d_mask, d_pose=layout.dim, n_contact=n_contact, **cfg)


def freeze(model: nn.Module) -> nn.Module:
    """파라미터를 고정 (requires_grad=False) 하고 eval 모드로. 논문 3.6.1: Context 를 학습 후 고정."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def check_frozen(model: nn.Module) -> None:
    """고정됐는지 확인 (학습 중 실수로 풀리면 에러)."""
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Context Transformer 파라미터가 고정되어 있지 않습니다.")
