"""
발 전용 모델 학습/검증 한 스텝
==============================

    foot_forward    FootBatch + TransitionSpec → 모델 → transition 만 교체 → 손실 dict + 예측
    validate        val 분할에서 transition 길이별 손실·발 위치 오차·접촉 지표 (log.csv 에 쓰는 값)

손실 [결정 H7] (transition 프레임 [t_ctx, target) 만, 제약 프레임 포함 = ctx_base 와 같음)
--------------------------------------------------------------------------------
    L_pos     = mean |p̂ − p|            p = 정렬 좌표 발 위치 (cm), 12 좌표 평균 (L1)
    L_contact = BCE(sigmoid(logit), c)   c = 데이터셋 접촉 라벨 (0/1, 월드 좌표 속도 기준)
    L         = w_pos·L_pos + w_contact·L_contact

접촉 지표 두 가지 (H4: 접촉 정의는 원래 월드 좌표 기준)
-----------------------------------------------------
    contact      : 모델의 접촉 출력 sigmoid(logit) > 0.5 vs 라벨
    poscontact   : 예측 발 위치를 월드 좌표로 되돌려 (foot_to_world) 라벨과 같은 규칙
                   (|dx|+|dy|+|dz| (t→t+1) < 0.2 cm/frame) 으로 판정 vs 라벨.
                   마지막 transition 프레임은 target 프레임 (정답) 과의 차이로 판정.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from foot_only.data import (
    FOOT_JOINT_NAMES, N_FOOT, FootBatch, build_foot_input, canonical_frame, foot_to_world, prepare_foot_batch,
)
from foot_only.model import split_output
from src.data.preprocess import extract_foot_contacts
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.training.context_inputs import TransitionSpec, key_mask, keyframe_positions, merge_prediction


def foot_forward(
    model: nn.Module,
    fb: FootBatch,
    spec: TransitionSpec,
    layout: FeatureLayout,
    foot_norm: ZScoreNormalizer,
    loss_w: dict,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """순전파 + 손실.

    입력:  fb (prepare_foot_batch 결과, foot 은 정규화된 값), spec (transition 위치)
    출력:
        losses    : {"loss", "pos" (cm), "contact" (BCE)}
        foot_pred : (B, T, 12) 정렬 좌표 발 위치 (cm). transition 은 예측, 나머지는 정답
        prob      : (B, T, 4)  접촉 확률. transition 은 예측, 나머지는 정답 라벨
    """
    x_gt, T = fb.pb.x_gt, fb.pb.x_gt.shape[1]
    x_in = build_foot_input(x_gt, fb.foot, spec, layout)                     # (B, T, 148)
    kf = keyframe_positions(T, spec, x_gt.dtype, x_gt.device)
    mask = key_mask(T, spec, x_gt.device)
    with torch.autocast(device_type=x_gt.device.type, dtype=amp_dtype or torch.float16,
                        enabled=amp_dtype is not None):
        out = model(x_in, kf, mask)                                           # (B, T, 16)
    pos_n, logit = split_output(out.to(x_gt.dtype))
    foot_pred = foot_norm.denormalize(merge_prediction(fb.foot, pos_n, spec))
    s = spec.seq_slice
    pw = loss_w.get("contact_pos_weight")
    pw = torch.as_tensor(pw, dtype=logit.dtype, device=logit.device) if pw is not None else None
    losses = {
        "pos": (foot_pred[:, s] - fb.foot_raw[:, s]).abs().mean(),
        "contact": F.binary_cross_entropy_with_logits(logit[:, s], fb.contact[:, s], pos_weight=pw),
    }
    losses["loss"] = float(loss_w.get("w_pos", 1.0)) * losses["pos"] + \
        float(loss_w.get("w_contact", 1.0)) * losses["contact"]
    prob = merge_prediction(fb.contact, torch.sigmoid(logit), spec)
    return losses, foot_pred, prob


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
    device,
) -> dict[str, float]:
    """val 분할, transition 길이마다 (제약 프레임 없음, 후처리 없음):

    val{t}_loss / _pos / _contact          학습과 같은 손실 (배치 평균)
    val{t}_foot_l2, _foot_l2_<관절>        발 위치 L2 오차 (cm), transition 프레임 평균
    val{t}_contact_acc, _contact_acc_<관절>, _contact_f1_<관절>   접촉 출력 (> 0.5) vs 라벨
    val{t}_poscontact_acc, _poscontact_acc_<관절>                 예측 위치로 판정한 접촉 (월드 좌표) vs 라벨
    클립은 ctx + trans + 2 프레임으로 자른다 (ctx_base 검증과 같음).
    """
    ctx, wf, thr = data_cfg["context_len"], data_cfg["forward_world_axis"], data_cfg["contact_vel_threshold"]
    loss_sum: dict[str, float] = defaultdict(float)
    cnt: dict[str, Tensor] = defaultdict(lambda: torch.zeros(N_FOOT, dtype=torch.float64))
    n_batches = 0
    model.eval()
    for batch in loader:
        n_batches += 1
        for trans in trans_list:
            L = ctx + trans + 2
            cut = {k: (v[:, :L] if k in ("root_pos", "local_rot", "contacts") else v) for k, v in batch.items()}
            fb = prepare_foot_batch(cut, skel, ctx_norm, foot_norm, ctx, wf, device)
            spec = TransitionSpec(ctx, ctx + trans)
            losses, foot_pred, prob = foot_forward(model, fb, spec, layout, foot_norm, loss_w)
            for k, v in losses.items():
                loss_sum[f"val{trans}_{k}"] += float(v)
            s = spec.seq_slice
            B = foot_pred.shape[0]
            err = (foot_pred[:, s] - fb.foot_raw[:, s]).reshape(B, -1, N_FOOT, 3).norm(dim=-1)   # (B, n, 4)
            cnt[f"{trans}_l2"] += err.sum(dim=(0, 1)).double().cpu()
            cnt[f"{trans}_n"] += float(err.shape[0] * err.shape[1])
            lab = fb.contact[:, s] > 0.5                                                          # (B, n, 4)
            pred = prob[:, s] > 0.5
            world = foot_to_world(foot_pred, canonical_frame(cut, skel, ctx, wf, device))         # (B, L, 4, 3)
            pos_c = extract_foot_contacts(world, range(N_FOOT), thr)[:, s] > 0.5
            for name, p in (("c", pred), ("pc", pos_c)):
                cnt[f"{trans}_{name}_ok"] += (p == lab).sum(dim=(0, 1)).double().cpu()
                cnt[f"{trans}_{name}_tp"] += (p & lab).sum(dim=(0, 1)).double().cpu()
                cnt[f"{trans}_{name}_fp"] += (p & ~lab).sum(dim=(0, 1)).double().cpu()
                cnt[f"{trans}_{name}_fn"] += (~p & lab).sum(dim=(0, 1)).double().cpu()
    model.train()

    res = {k: v / max(n_batches, 1) for k, v in loss_sum.items()}
    for t in trans_list:
        n = cnt[f"{t}_n"]
        l2 = cnt[f"{t}_l2"] / n
        res[f"val{t}_foot_l2"] = float(l2.mean())
        for j, name in enumerate(FOOT_JOINT_NAMES):
            res[f"val{t}_foot_l2_{name}"] = float(l2[j])
        for tag, key in (("contact", "c"), ("poscontact", "pc")):
            acc = cnt[f"{t}_{key}_ok"] / n
            res[f"val{t}_{tag}_acc"] = float(acc.mean())
            for j, name in enumerate(FOOT_JOINT_NAMES):
                res[f"val{t}_{tag}_acc_{name}"] = float(acc[j])
            if key == "c":
                tp, fp, fn = cnt[f"{t}_c_tp"], cnt[f"{t}_c_fp"], cnt[f"{t}_c_fn"]
                f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
                for j, name in enumerate(FOOT_JOINT_NAMES):
                    res[f"val{t}_contact_f1_{name}"] = float(f1[j])
    return res
