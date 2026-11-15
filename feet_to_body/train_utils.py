"""
발 → 몸 모델 학습/검증 한 스텝
==============================

    run_body_model   입력 만들기 → 모델 → transition 만 교체. 출력 (B, T, D) 정규화된 전신
    body_forward     run_body_model + 손실 (논문 식 21, ctx_base 와 같은 식) → 손실 dict + 예측
    body_foot_error  출력 전신의 발 (FK) 이 조건으로 준 발과 얼마나 다른가 (cm)
    validate         val 분할에서 transition 길이별 손실·오차 (log.csv 에 쓰는 값)

손실 [결정 I3] = ctx_base 와 같음 (src/training/losses.py, 논문 식 21, 모두 L1, transition 프레임만)
    L = w_rot·L_rot (6D 회전) + w_pos·L_pos (FK 전역 위치, cm) + w_smooth·L_smooth (프레임 차이)
그래서 val{t}_loss / _rot / _pos / _smooth / _gpos_l2 는 experiments/ctx_base/log.csv 의 같은 이름 값과
바로 비교할 수 있다 (같은 val 분할, 같은 식).

발을 따르게 강제하는 장치 (IK 등) 는 없다. 따르는 정도는 val{t}_foot_l2 로 잰다.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor, nn

from feet_to_body.data import attention_mask, build_body_input, foot_condition
from foot_only.data import FOOT_JOINT_NAMES, N_FOOT, FootBatch, foot_indices, prepare_foot_batch
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.training.context import global_positions, gpos_l2
from src.training.context_inputs import TransitionSpec, keyframe_positions, merge_prediction, postprocess
from src.training.losses import position_loss, rotation_loss, smooth_loss, weighted_sum


def run_body_model(
    model: nn.Module,
    x_gt: Tensor,
    cond: Tensor,
    spec: TransitionSpec,
    layout: FeatureLayout,
    attend_all: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> Tensor:
    """입력 만들기 → 모델 → transition 합치기.

    입력:  x_gt (B, T, D) 정규화된 전신 (아는 프레임만 쓰임), cond (B, T, 16) 발 조건 (모든 프레임)
    출력:  (B, T, D) 정규화된 전신. transition 은 예측, 나머지는 정답
    amp_dtype 를 주면 모델 부분만 autocast (손실의 FK 는 항상 원래 정밀도).
    """
    T = x_gt.shape[1]
    x_in = build_body_input(x_gt, cond, spec, layout)                        # (B, T, 152)
    kf = keyframe_positions(T, spec, x_gt.dtype, x_gt.device)
    mask = attention_mask(T, spec, attend_all, x_gt.device)
    with torch.autocast(device_type=x_gt.device.type, dtype=amp_dtype or torch.float16,
                        enabled=amp_dtype is not None):
        out = model(x_in, kf, mask)                                           # (B, T, D)
    return merge_prediction(x_gt, out.to(x_gt.dtype), spec)


def body_forward(
    model: nn.Module,
    fb: FootBatch,
    cond: Tensor,
    spec: TransitionSpec,
    skeleton: Skeleton,
    layout: FeatureLayout,
    normalizer: ZScoreNormalizer,
    loss_w: dict,
    attend_all: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """순전파 + 손실 (논문 식 21).

    출력:
        losses : {"loss", "rot", "pos", "smooth"}  (context_forward 와 같은 계산)
        y      : (B, T, D) 정규화된 예측 (transition 만 예측)
        gpos   : (B, T, J, 3) 예측 전역 관절 위치 (정렬 좌표, cm)
    """
    y = run_body_model(model, fb.pb.x_gt, cond, spec, layout, attend_all, amp_dtype)
    y_raw = normalizer.denormalize(y)
    gpos = global_positions(y_raw, fb.pb.offsets, skeleton, layout)
    losses = {
        "rot": rotation_loss(y_raw, fb.pb.x_gt_raw, spec, layout),
        "pos": position_loss(gpos, fb.pb.gpos_gt, spec),
        "smooth": smooth_loss(gpos, spec),
    }
    losses["loss"] = weighted_sum(losses, loss_w)
    return losses, y, gpos


def body_foot_error(gpos: Tensor, foot_raw: Tensor, skeleton: Skeleton, frames: slice) -> Tensor:
    """출력 전신의 발 4관절 위치와 조건으로 준 발 위치의 거리.

    입력:  gpos (B, T, J, 3) 예측 전역 위치, foot_raw (B, T, 12) 조건 발 위치 (cm, 같은 정렬 좌표)
    출력:  (B, n, 4) 프레임·관절별 L2 거리 (cm), n = frames 의 프레임 수
    """
    B = gpos.shape[0]
    pred = gpos[:, frames][:, :, foot_indices(skeleton)]                      # (B, n, 4, 3)
    return (pred - foot_raw[:, frames].reshape(B, -1, N_FOOT, 3)).norm(dim=-1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    trans_list: list[int],
    data_cfg: dict,
    skel: Skeleton,
    layout: FeatureLayout,
    ctx_norm: ZScoreNormalizer,
    foot_norm: ZScoreNormalizer,
    loss_w: dict,
    attend_all: bool,
    device,
    cond_fn=None,
) -> dict[str, float]:
    """val 분할, transition 길이마다 (제약 프레임 없음).

    cond_fn(fb, spec) → (cond (B, L, 16), 조건 발 위치 (B, L, 12) cm). None 이면 정답 발 조건.
    (예측한 발로 학습할 때는 feet_to_body.evaluate.predicted_cond 를 감싼 함수를 준다.)

    val{t}_loss / _rot / _pos / _smooth   학습과 같은 손실 (후처리 없음, 배치 평균) — ctx_base 로그와 같은 이름·식
    val{t}_gpos_l2                         전역 관절 위치 평균 L2 (cm), **후처리 (식 23) 포함** — ctx_base 로그와 같음
    val{t}_gpos_l2_nopost                  같은 값, 후처리 없음
    val{t}_foot_l2, _foot_l2_<관절>        출력 전신의 발 vs **정답 발** (cm, 후처리 없음)
    val{t}_foot_follow                     출력 전신의 발 vs **조건으로 준 발** (cm). 정답 발 조건이면 foot_l2 와 같음
    val{t}_cond_l2                         조건으로 준 발 vs 정답 발 (cm). 정답 발 조건이면 0
    클립은 ctx + trans + 2 프레임으로 자른다 (ctx_base 검증과 같음; 후처리가 target 다음 프레임을 씀).
    """
    ctx, wf = data_cfg["context_len"], data_cfg["forward_world_axis"]
    sums: dict[str, float] = defaultdict(float)
    foot_sum = {t: torch.zeros(N_FOOT, dtype=torch.float64) for t in trans_list}
    foot_n = {t: 0.0 for t in trans_list}
    n_batches = 0
    model.eval()
    for batch in loader:
        n_batches += 1
        for trans in trans_list:
            L = ctx + trans + 2
            cut = {k: (v[:, :L] if k in ("root_pos", "local_rot", "contacts") else v) for k, v in batch.items()}
            fb = prepare_foot_batch(cut, skel, ctx_norm, foot_norm, ctx, wf, device)
            spec = TransitionSpec(ctx, ctx + trans)
            cond, cond_feet = cond_fn(fb, spec) if cond_fn is not None else (foot_condition(fb), fb.foot_raw)
            losses, y, gpos = body_forward(model, fb, cond, spec, skel, layout, ctx_norm, loss_w, attend_all)
            for k, v in losses.items():
                sums[f"val{trans}_{k}"] += float(v)
            s = spec.seq_slice
            y_post = postprocess(y, fb.pb.x_gt, spec)                         # 식 23 (ctx_base 검증과 같음)
            gpos_post = global_positions(ctx_norm.denormalize(y_post), fb.pb.offsets, skel, layout)
            sums[f"val{trans}_gpos_l2"] += float(gpos_l2(gpos_post, fb.pb.gpos_gt, s))
            sums[f"val{trans}_gpos_l2_nopost"] += float(gpos_l2(gpos, fb.pb.gpos_gt, s))
            err = body_foot_error(gpos, fb.foot_raw, skel, s)                 # (B, n, 4)
            foot_sum[trans] += err.sum(dim=(0, 1)).double().cpu()
            foot_n[trans] += float(err.shape[0] * err.shape[1])
            sums[f"val{trans}_foot_follow"] += float(body_foot_error(gpos, cond_feet, skel, s).mean())
            B = cond_feet.shape[0]
            sums[f"val{trans}_cond_l2"] += float(
                (cond_feet[:, s] - fb.foot_raw[:, s]).reshape(B, -1, N_FOOT, 3).norm(dim=-1).mean())
    model.train()

    res = {k: v / max(n_batches, 1) for k, v in sums.items()}
    for t in trans_list:
        l2 = foot_sum[t] / max(foot_n[t], 1.0)
        res[f"val{t}_foot_l2"] = float(l2.mean())
        for j, name in enumerate(FOOT_JOINT_NAMES):
            res[f"val{t}_foot_l2_{name}"] = float(l2[j])
    return res


