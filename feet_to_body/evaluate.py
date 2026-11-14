"""
발 → 몸 모델 평가 계산: 체크포인트 읽기, 방법별 전신 예측
=========================================================

    load_body_run     체크포인트 + 같은 폴더 config.yaml → BodyRun
    predicted_cond    발 전용 모델의 예측 → 발 조건 (B, L, 16) (정규화한 위치 12 ⊕ 접촉 확률 4)
    predict_bodies    평가셋 배치 → 방법별 전신 예측 (정렬 좌표 원값 (B, L, D)) + 조건으로 준 발 위치

전신 지표 (L2P, L2Q, NPSS, 발 미끄러짐) 는 기존 src/evaluation/metrics.py 를 그대로 쓴다 (eval_body.py).
ctx_post / ctx_nopost / two_stage / interp 의 계산은 scripts/eval_lengths.py 의 model / model_nopost / detail / interp 와 같다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from feet_to_body.data import foot_condition
from feet_to_body.model import build_body_model
from feet_to_body.train_utils import run_body_model
from foot_only.data import FOOT_DIM, FootBatch, foot_indices
from foot_only.evaluate import FootRun
from foot_only.train_utils import foot_forward
from src.config import load_yaml, resolve_path
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.evaluation.baselines import interpolation
from src.evaluation.inference import DetailRun, LoadedRun, cut_batch, prepare_window
from src.training.context import PreparedBatch, run_model
from src.training.context_inputs import TransitionSpec, postprocess
from src.training.detail import run_detail
from src.training.utils import load_checkpoint

BODY_METHODS = ("body_gt", "body_gt_post", "body_pred", "body_pred_post")
METHODS = (*BODY_METHODS, "ctx_post", "ctx_nopost", "two_stage", "interp")
METHOD_LABEL = {"body_gt": "feet→body (GT feet)", "body_gt_post": "feet→body (GT feet) + post",
                "body_pred": "feet→body (predicted feet)", "body_pred_post": "feet→body (predicted feet) + post",
                "ctx_post": "Context + post", "ctx_nopost": "Context", "two_stage": "Two-stage", "interp": "SLERP"}


@dataclass
class BodyRun:
    """발 → 몸 모델 하나. cfg = experiments/<run>/config.yaml, ckpt = 체크포인트 dict."""

    cfg: dict[str, Any]
    model: torch.nn.Module
    ctx_norm: ZScoreNormalizer
    foot_norm: ZScoreNormalizer
    attend_all: bool
    ckpt: dict[str, Any]
    ckpt_path: Path


def load_body_run(ckpt: str | Path, layout: FeatureLayout, device: torch.device) -> BodyRun:
    """체크포인트 + 같은 폴더 config.yaml → BodyRun (모델 입력 152 라서 inference.load_run 을 쓸 수 없음)."""
    ckpt_path = resolve_path(ckpt)
    cfg_path = ckpt_path.parent / "config.yaml"
    for p in (ckpt_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 가 없습니다.")
    cfg = load_yaml(cfg_path)
    if cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 결과입니다 (평가 의미 없음).")
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = build_body_model(cfg["model"], layout).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ctx_norm = ZScoreNormalizer.load(resolve_path(cfg["stats"])).to(device=device, dtype=torch.float32)
    foot_norm = ZScoreNormalizer.load(resolve_path(cfg["foot_stats"])).to(device=device, dtype=torch.float32)
    return BodyRun(cfg, model, ctx_norm, foot_norm, bool(cfg["cond"]["attend_all"]), ck, ckpt_path)


def check_same_stats(body: BodyRun, ctx_run: LoadedRun, foot: FootRun | None = None) -> None:
    """전신 통계가 ctx_base 와 같고, 발 통계가 발 전용 모델과 같은지 (정답 창과 발 조건을 같이 쓰기 위한 조건)."""
    pairs = [("전신", body.ctx_norm, ctx_run.norm)]
    if foot is not None:
        pairs.append(("발", body.foot_norm, foot.foot_norm))
    for name, a, b in pairs:
        if not (torch.allclose(a.mean, b.mean) and torch.allclose(a.std, b.std)):
            raise ValueError(f"{name} 정규화 통계가 다릅니다.")


@torch.no_grad()
def predicted_cond(foot: FootRun, fb: FootBatch, spec: TransitionSpec, layout: FeatureLayout) -> tuple[Tensor, Tensor]:
    """발 전용 모델의 예측을 발 조건으로 바꾼다.

    출력:
        cond     : (B, L, 16) = 정규화한 발 위치 12 ⊕ 접촉 4. transition 은 예측 (접촉은 확률 0~1),
                   그 밖 (context·target 등) 은 정답 (foot_forward 가 합쳐 줌)
        foot_raw : (B, L, 12) 같은 발 위치 (cm, 정렬 좌표)
    """
    _, foot_raw, prob = foot_forward(foot.model, fb, spec, layout, foot.foot_norm, foot.loss_w)
    return torch.cat([foot.foot_norm.normalize(foot_raw), prob], dim=-1), foot_raw


@dataclass
class BodyPrediction:
    """배치 하나의 방법별 예측.

    pb, spec : 정답 창 (정렬 좌표) 과 TransitionSpec
    preds    : {방법: (B, L, D) 정렬 좌표 원값 (역정규화). transition 밖은 정답}
    cond_feet: {방법: (B, L, 12) 그 방법에 조건으로 준 발 위치 (cm)}  (발 → 몸 방법만)
    """

    pb: PreparedBatch
    spec: TransitionSpec
    preds: dict[str, Tensor]
    cond_feet: dict[str, Tensor]


@torch.no_grad()
def predict_bodies(batch: dict, trans: int, ctx_run: LoadedRun, methods: Sequence[str], body: BodyRun | None = None,
                   foot: FootRun | None = None, det: DetailRun | None = None) -> BodyPrediction:
    """평가셋 배치 → 방법별 전신 예측.

    입력:  batch (LafanDataset 배치, 길이 W ≥ ctx + trans + 2), trans, ctx_run (ctx_base),
           methods (METHODS 의 부분집합), body (body_* 에 필요), foot (body_pred* 에 필요), det (two_stage 에 필요)
    """
    pb, spec = prepare_window(ctx_run, batch, trans)
    B, L = pb.x_gt.shape[:2]
    norm, layout = ctx_run.norm, ctx_run.layout
    preds: dict[str, Tensor] = {}
    cond_feet: dict[str, Tensor] = {}

    if any(m in methods for m in BODY_METHODS):
        labels = cut_batch(batch, 0, L)["contacts"].to(device=pb.x_gt.device, dtype=pb.x_gt.dtype)
        foot_raw = pb.gpos_gt[:, :, foot_indices(ctx_run.skel)].reshape(B, L, FOOT_DIM)
        fb = FootBatch(pb=pb, foot_raw=foot_raw, foot=body.foot_norm.normalize(foot_raw), contact=labels)
        conds = {}
        if any(m in methods for m in ("body_gt", "body_gt_post")):
            conds["body_gt"] = (foot_condition(fb), foot_raw)
        if any(m in methods for m in ("body_pred", "body_pred_post")):
            conds["body_pred"] = predicted_cond(foot, fb, spec, layout)
        for name, (cond, feet) in conds.items():
            y = run_body_model(body.model, pb.x_gt, cond, spec, layout, body.attend_all)   # (B, L, D) 정규화
            if name in methods:
                preds[name], cond_feet[name] = norm.denormalize(y), feet
            if f"{name}_post" in methods:
                preds[f"{name}_post"], cond_feet[f"{name}_post"] = norm.denormalize(postprocess(y, pb.x_gt, spec)), feet

    if any(m in methods for m in ("ctx_post", "ctx_nopost", "two_stage")):
        y = run_model(ctx_run.model, pb.x_gt, spec, layout)
        y_post = postprocess(y, pb.x_gt, spec)                                             # 식 23
        if "ctx_nopost" in methods:
            preds["ctx_nopost"] = norm.denormalize(y)
        if "ctx_post" in methods:
            preds["ctx_post"] = norm.denormalize(y_post)
        if "two_stage" in methods:
            y_det, _ = run_detail(det.model, y_post if det.ctx_postprocess else y, pb.x_gt, spec)
            preds["two_stage"] = norm.denormalize(y_det)
    if "interp" in methods:
        preds["interp"] = interpolation(pb.x_gt_raw, spec, layout)
    return BodyPrediction(pb, spec, preds, cond_feet)
