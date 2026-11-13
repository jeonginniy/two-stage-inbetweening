"""
발 전용 모델의 데이터: 발 위치·접촉 타깃, 발 통계, 모델 입력
=============================================================

이 파일이 하는 일 (사용 순서)
-----------------------------
    load_foot_config      configs/foot_only.yaml 읽기
    foot_indices          발 관절 번호 [LeftFoot 3, LeftToe 4, RightFoot 7, RightToe 8] (스켈레톤 파일에서)
    prepare_foot_batch    LafanDataset 배치 → FootBatch
                            (기존 prepare_batch 로 정렬 + FK 를 그대로 하고, 거기서 발 4관절만 꺼낸다)
    build_foot_input      FootBatch → 모델 입력 (B, T, D + 12 + 1) = 전신 x ⊕ 발 위치 ⊕ m_in
    foot_to_world         정렬 좌표의 발 위치 → 원래 월드 좌표 (평가·시각화용)
    compute_foot_stats    train_stats 분할에서 발 위치 12차원의 평균/표준편차

좌표계 [결정 H1]
---------------
발 위치는 "정렬 월드" 좌표: 마지막 context 프레임에서 루트 XZ = 0, 루트가 +X 를 보게 돌린 좌표
(논문 3.6.1 의 정렬과 같음, 단위 cm, Y-up). 기존 PreparedBatch.gpos_gt 와 같은 좌표다.
정렬에 쓴 회전·이동 (CanonicalFrame) 을 되돌리면 원래 월드 좌표가 된다 (foot_to_world).
루트 기준 좌표와 달리 루트 궤적 없이도 월드 위치를 복원할 수 있고, 땅에 붙은 발은 값이 고정된다.

발 위치 벡터의 순서 (12차원)
---------------------------
    [LeftFoot xyz, LeftToe xyz, RightFoot xyz, RightToe xyz]   = foot_pos (B, T, 4, 3) 를 펼친 것
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import CONFIG_DIR, resolve_path
from src.data.preprocess import CanonicalFrame, canonicalize_root
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.training.context import PreparedBatch, prepare_batch
from src.training.context_inputs import TransitionSpec, build_context_input, data_mask

FOOT_JOINT_NAMES = ("LeftFoot", "LeftToe", "RightFoot", "RightToe")
N_FOOT = len(FOOT_JOINT_NAMES)          # 4
FOOT_DIM = N_FOOT * 3                   # 12 (발 위치 출력 차원)


def load_foot_config(path: str | Path | None = None) -> dict[str, Any]:
    """configs/foot_only.yaml (또는 주어진 yaml 경로) → dict."""
    p = resolve_path(path) if path else CONFIG_DIR / "foot_only.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def foot_indices(skeleton: Skeleton) -> list[int]:
    """발 관절 번호 4개 = 스켈레톤의 contact_joints (접촉 라벨과 같은 순서).

    이름이 [LeftFoot, LeftToe, RightFoot, RightToe] 인지 확인한다 (다르면 에러: 출력 순서가 바뀌면 안 됨).
    LaFAN1: [3, 4, 7, 8]
    """
    idx = list(skeleton.contact_joints)
    names = tuple(skeleton.names[i] for i in idx)
    if names != FOOT_JOINT_NAMES:
        raise ValueError(f"contact_joints 이름 {names} ≠ 기대 {FOOT_JOINT_NAMES}")
    return idx


@dataclass
class FootBatch:
    """발 전용 모델 배치 하나.

    pb         : PreparedBatch (x_gt (B,T,D) 정규화된 전신, x_gt_raw, gpos_gt (B,T,J,3) 정렬 좌표, offsets)
    foot_raw   : (B, T, 12) 정렬 좌표 발 위치 (cm)  = gpos_gt[..., [3,4,7,8], :] 를 펼친 것
    foot       : (B, T, 12) z-score 한 발 위치 (foot_norm 이 None 이면 foot_raw 와 같음)
    contact    : (B, T, 4)  발 접촉 라벨 0/1 (LafanDataset 이 만든 것, 속도 기준)
    """

    pb: PreparedBatch
    foot_raw: Tensor
    foot: Tensor
    contact: Tensor


def prepare_foot_batch(
    batch: dict,
    skeleton: Skeleton,
    ctx_norm: ZScoreNormalizer,
    foot_norm: ZScoreNormalizer | None,
    context_len: int,
    world_forward: str = "x",
    device: torch.device | str | None = None,
) -> FootBatch:
    """LafanDataset 배치 → FootBatch.

    정렬 (논문 3.6.1) 과 FK 는 기존 prepare_batch 를 그대로 쓴다 (Context Transformer 학습과 같은 정답).
    입력:  batch (root_pos (B,T,3), local_rot (B,T,J,3,3), offsets (B,J,3), contacts (B,T,4))
    """
    pb = prepare_batch(batch, skeleton, ctx_norm, context_len, world_forward, device)
    B, T = pb.gpos_gt.shape[:2]
    foot_raw = pb.gpos_gt[:, :, foot_indices(skeleton)].reshape(B, T, FOOT_DIM)
    foot = foot_norm.normalize(foot_raw) if foot_norm is not None else foot_raw
    contact = batch["contacts"].to(device=pb.x_gt.device, dtype=pb.x_gt.dtype)
    return FootBatch(pb=pb, foot_raw=foot_raw, foot=foot, contact=contact)


def build_foot_input(x_gt: Tensor, foot_gt: Tensor, spec: TransitionSpec, layout: FeatureLayout) -> Tensor:
    """모델 입력 (B, T, D + 12 + 1) [결정 H2].

    입력:
        x_gt    : (B, T, D)  정규화된 전신 정답 (D = 135)
        foot_gt : (B, T, 12) 정규화된 발 위치 정답
    출력 = [전신 부분 (D) | 발 위치 (12) | m_in (1)]
        전신 부분 : Context Transformer 와 똑같이 만든다 (build_context_input: 모르는 프레임 0, 루트 선형보간)
        발 위치   : 아는 프레임 (context·제약·target) 은 정답, 나머지는 0 (정규화 공간의 0 = 학습 데이터 평균)
        m_in      : 아는 프레임 1, 나머지 0 (논문 3.3)
    """
    T = x_gt.shape[1]
    ctx_in = build_context_input(x_gt, spec, layout)               # (B, T, D + 1), 마지막 = m_in
    m = data_mask(T, spec, x_gt.dtype, x_gt.device)                # (T, 1)
    return torch.cat([ctx_in[..., :-1], foot_gt * m, ctx_in[..., -1:]], dim=-1)


def canonical_frame(batch: dict, skeleton: Skeleton, context_len: int, world_forward: str = "x",
                    device: torch.device | str | None = None) -> CanonicalFrame:
    """배치를 정렬할 때 쓴 변환 (prepare_batch 안에서 쓰는 것과 같은 계산). 월드 복원용."""
    _, _, frame = canonicalize_root(batch["root_pos"].to(device), batch["local_rot"].to(device),
                                    context_len - 1, skeleton.root_forward_local_axis, world_forward)
    return frame


def foot_to_world(foot_raw: Tensor, frame: CanonicalFrame) -> Tensor:
    """정렬 좌표 발 위치 (B, T, 12) → 원래 월드 좌표 (B, T, 4, 3).

    p_world = R_y^T @ p_canon + pos_offset  (src/data/preprocess.uncanonicalize_root 와 같은 식)
    """
    B, T = foot_raw.shape[:2]
    p = foot_raw.reshape(B, T, N_FOOT, 3)
    inv = frame.rot_offset.transpose(-1, -2)                       # (B, 1, 3, 3)
    return (inv[:, :, None] @ p[..., None]).squeeze(-1) + frame.pos_offset[:, :, None]


@torch.no_grad()
def compute_foot_stats(loader: DataLoader, skeleton: Skeleton, ctx_norm: ZScoreNormalizer, context_len: int,
                       world_forward: str = "x", device: torch.device | str | None = None) -> RunningStats:
    """발 위치 12차원의 평균/표준편차 (모든 클립의 모든 프레임). train_stats 분할 loader 를 줄 것."""
    stats = RunningStats(FOOT_DIM)
    for batch in loader:
        fb = prepare_foot_batch(batch, skeleton, ctx_norm, None, context_len, world_forward, device)
        stats.update(fb.foot_raw)
    return stats


def foot_names_xyz() -> Sequence[str]:
    """12차원 각각의 이름 (출력·로그용). 예) LeftFoot.x"""
    return [f"{n}.{a}" for n in FOOT_JOINT_NAMES for a in "xyz"]
