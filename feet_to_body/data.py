"""
발 → 몸 모델의 데이터: 발 조건 (위치 + 접촉) 과 모델 입력
========================================================

이 파일이 하는 일 (사용 순서)
-----------------------------
    load_body_config     configs/feet_to_body.yaml 읽기
    foot_condition       FootBatch → 발 조건 (B, T, 16) = 정답 발 위치 12 (z-score) ⊕ 접촉 라벨 4
    build_body_input     전신 x ⊕ 발 조건 ⊕ m_in → 모델 입력 (B, T, D + 16 + 1 = 152)
    attention_mask       어텐션 key 마스크 (attend_all 이면 None = 모든 프레임을 본다)

배치 준비는 발 전용 모델의 prepare_foot_batch (foot_only/data.py) 를 그대로 쓴다:
정렬 (논문 3.6.1) + FK + 발 4관절 위치 (정렬 월드 좌표, cm) + 접촉 라벨.

발 조건 [결정 I1]
----------------
    cond (B, T, 16) = [LeftFoot xyz, LeftToe xyz, RightFoot xyz, RightToe xyz (z-score) | 접촉 4 (0~1)]
발 전용 모델의 출력 (위치 12 + 접촉 4) 과 같은 형식·같은 좌표·같은 통계다.
1차 실험에서는 정답 발을 넣는다 (foot_condition). 나중에 발 전용 모델의 예측
(정규화한 위치 12 ⊕ 접촉 확률 4) 을 같은 자리에 넣으면 된다.
context·target 프레임에서는 전신이 주어지므로 발 조건이 새 정보가 아니고, transition 프레임에서만 새 정보다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from foot_only.data import FOOT_DIM, N_FOOT, FootBatch
from src.config import CONFIG_DIR, resolve_path
from src.data.representation import FeatureLayout
from src.training.context_inputs import TransitionSpec, build_context_input, key_mask

COND_DIM = FOOT_DIM + N_FOOT            # 16 = 발 위치 12 + 접촉 4


def _merge(base: dict, over: dict) -> dict:
    """dict 를 깊게 합친다: over 의 값이 base 를 덮어쓴다 (dict 끼리는 안쪽까지 합침)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_body_config(path: str | Path | None = None) -> dict[str, Any]:
    """configs/feet_to_body.yaml (또는 주어진 yaml 경로) → dict.

    yaml 에 `base: <configs/ 안의 파일 이름>` 이 있으면 그 파일을 먼저 읽고, 이 파일의 값으로 덮어쓴다
    (예: configs/feet_to_body_pred.yaml 은 바뀐 값만 적는다).
    """
    p = resolve_path(path) if path else CONFIG_DIR / "feet_to_body.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    return _merge(load_body_config(CONFIG_DIR / base), cfg) if base else cfg


def foot_condition(fb: FootBatch) -> Tensor:
    """정답 발 조건 (B, T, 16) = fb.foot (B, T, 12, 정규화된 발 위치) ⊕ fb.contact (B, T, 4, 0/1)."""
    return torch.cat([fb.foot, fb.contact], dim=-1)


def build_body_input(x_gt: Tensor, cond: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """모델 입력 (B, T, D + 16 + 1) [결정 I1].

    입력:
        x_gt : (B, T, D)  정규화된 전신 정답 (D = 135). 아는 프레임 (context·제약·target) 만 쓰인다
        cond : (B, T, 16) 발 조건. **모든 프레임** 의 값이 그대로 들어간다 (가리지 않음)
    출력 = [전신 부분 (D) | 발 조건 (16) | m_in (1)]
        전신 부분 : Context Transformer 와 똑같이 만든다 (build_context_input: 모르는 프레임 0, 루트 선형보간)
        m_in      : 전신을 아는 프레임 1, 나머지 0 (논문 3.3)
    """
    if cond.shape[-1] != COND_DIM or cond.shape[:2] != x_gt.shape[:2]:
        raise ValueError(f"cond 는 (B, T, {COND_DIM}) 이어야 합니다: {tuple(cond.shape)} (x_gt {tuple(x_gt.shape)})")
    ctx_in = build_context_input(x_gt, spec, layout)               # (B, T, D + 1), 마지막 = m_in
    return torch.cat([ctx_in[..., :-1], cond.to(x_gt.dtype), ctx_in[..., -1:]], dim=-1)


def attention_mask(T: int, spec: TransitionSpec, attend_all: bool, device=None) -> Tensor | None:
    """어텐션 key 마스크 [결정 I2].

    attend_all=True  → None (모든 query 가 모든 프레임을 key 로 본다: transition 프레임에도 발 정보가 있음)
    attend_all=False → Context Transformer 와 같은 마스크 (T, T): context·제약·target 프레임만 key (논문 그림 5a)
    """
    return None if attend_all else key_mask(T, spec, device)
