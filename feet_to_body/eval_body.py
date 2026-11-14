"""
[연구실] 발 → 몸 모델 평가: 정답 발 / 예측한 발 조건 vs Context / Two-stage / SLERP
================================================================================

계산 함수는 feet_to_body/evaluate.py, 지표는 기존 src/evaluation/metrics.py (공식 RMI 정의) 를 그대로 쓴다.

지표 (transition 프레임만)
    L2P, L2Q, NPSS   : 기존 평가 (scripts/eval_lengths.py) 와 같은 정의·같은 L2P 통계
    foot_skate       : 발끝 2개 수평 속도 × 높이 가중 (cm/frame). GT 값도 함께 기록
    gpos_l2          : 전역 관절 위치 평균 L2 (cm)
    foot_l2          : 출력 전신의 발 4관절 vs 정답 발 (cm)
    foot_follow      : 출력 전신의 발 4관절 vs **조건으로 준 발** (cm, 발 → 몸 방법만).
                       정답 발 조건이면 foot_l2 와 같고, 예측한 발 조건이면 "준 발을 얼마나 따랐는가"

설정: configs/eval/feet_to_body.yaml + configs/eval/lengths.yaml + configs/eval/foot_only.yaml (foot_ckpt)
실행: python feet_to_body/eval_body.py   → outputs/eval/feet_to_body/{metrics.csv, metrics.png, run_info.json}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from eval_lengths import load_bench_stats  # noqa: E402  (L2P 통계: 기존 평가와 같은 캐시)
from feet_to_body.evaluate import (  # noqa: E402
    BODY_METHODS, METHOD_LABEL, METHODS, check_same_stats, load_body_run, predict_bodies,
)
from feet_to_body.train_utils import body_foot_error  # noqa: E402
from foot_only.evaluate import load_foot_run  # noqa: E402
from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    build_eval_dataset, describe, load_detail, load_run, max_trans_for_window, train_range_label,
)
from src.evaluation.metrics import MetricAccumulator, foot_skating_per_clip, global_quats, global_rot_pos  # noqa: E402

COLS = ("L2P", "L2Q", "NPSS", "foot_skate", "gpos_l2", "foot_l2", "foot_follow")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sets", nargs="+", default=["official", "long"], choices=["official", "long"])
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS, help="기본: 설정 파일의 methods")
    ap.add_argument("--body-ckpt", default=None, help="기본: 설정 파일의 body_ckpt")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: 설정 파일의 out_dir)")
    return ap.parse_args()


@torch.no_grad()
def eval_one_length(ds, trans, methods, batch_size, ctx_run, body, foot, det, mean, std, foot_idx, height_cm) -> dict:
    """평가셋 ds 전체에서 transition `trans` 의 방법별 지표.

    출력: {방법: {L2P, L2Q, NPSS, foot_skate, gpos_l2, foot_l2, foot_follow (없으면 nan), n_clips}, "_gt_skate": 값}
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=ctx_run.env["num_workers"])
    accs = {m: MetricAccumulator(mean, std, foot_idx, height_cm) for m in methods}
    extra = {m: {"gpos_l2": [], "foot_l2": [], "foot_follow": []} for m in methods}
    gt_skate = []
    for batch in loader:
        bp = predict_bodies(batch, trans, ctx_run, methods, body, foot, det)
        pb, s = bp.pb, bp.spec.seq_slice
        B, L = pb.x_gt.shape[:2]
        grot_gt, gpos_gt = global_rot_pos(pb.x_gt_raw, pb.offsets, ctx_run.skel, ctx_run.layout)
        q_gt = global_quats(grot_gt)
        gt_skate.append(foot_skating_per_clip(gpos_gt, foot_idx, height_cm, s).cpu().numpy())
        feet_gt = gpos_gt[:, :, ctx_run.skel.contact_joints].reshape(B, L, -1)            # (B, L, 12)
        for m, x_raw in bp.preds.items():
            grot, gpos = global_rot_pos(x_raw, pb.offsets, ctx_run.skel, ctx_run.layout)
            accs[m].add(gpos, gpos_gt, global_quats(grot), q_gt, s)
            extra[m]["gpos_l2"].append((gpos[:, s] - gpos_gt[:, s]).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy())
            extra[m]["foot_l2"].append(body_foot_error(gpos, feet_gt, ctx_run.skel, s).mean(dim=(1, 2)).cpu().numpy())
            if m in bp.cond_feet:
                extra[m]["foot_follow"].append(
                    body_foot_error(gpos, bp.cond_feet[m], ctx_run.skel, s).mean(dim=(1, 2)).cpu().numpy())
    out = {}
    for m in methods:
        out[m] = accs[m].result()
        for k, v in extra[m].items():
            out[m][k] = float(np.concatenate(v).mean()) if v else float("nan")
    out["_gt_skate"] = float(np.concatenate(gt_skate).mean())
    return out


def eval_with_retry(*args, batch_size: int, **kw) -> tuple[dict, int]:
    """eval_one_length + GPU 메모리 부족 시 배치를 절반으로 줄여 다시. 출력: (결과, 실제 배치 크기)."""
    bs = batch_size
    while True:
        try:
            return eval_one_length(*args[:2], args[2], bs, *args[3:], **kw), bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            print(f"  [메모리 부족] batch {bs} → {bs // 2} 로 줄여 다시 계산")
            bs //= 2


def print_table(rows: list[dict], set_name: str, methods: list[str]) -> None:
    print(f"\n[{set_name}]  (GT 발 미끄러짐: " +
          ", ".join(f"trans {t} = {v:.3f}" for t, v in sorted({r['trans']: r['gt_foot_skate'] for r in rows
                                                               if r['set'] == set_name}.items())) + ")")
    print(f"{'trans':>5} {'method':<34}" + "".join(f"{c:>12}" for c in COLS))
    for r in rows:
        if r["set"] == set_name:
            print(f"{r['trans']:>5} {METHOD_LABEL[r['method']]:<34}" + "".join(f"{r[c]:>12.4f}" for c in COLS))


def plot(rows: list[dict], methods: list[str], train_range, path: Path) -> None:
    """2×2 그래프 (long 평가셋 우선, 없으면 official): L2P, L2Q, 발 미끄러짐, 전역 위치 L2. 영어 라벨."""
    sets = sorted({r["set"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title) in zip(axes.ravel(), (("L2P", "L2P"), ("L2Q", "L2Q"), ("foot_skate", "foot skating (cm/frame)"),
                                               ("gpos_l2", "global position L2 (cm)"))):
        for m in methods:
            for st, ls in zip(sets, ("-", "--")):
                rr = sorted((r for r in rows if r["method"] == m and r["set"] == st), key=lambda r: r["trans"])
                if rr:
                    ax.plot([r["trans"] for r in rr], [r[col] for r in rr], ls, marker="o", ms=4,
                            color=f"C{methods.index(m)}", label=f"{METHOD_LABEL[m]} [{st}]")
        if col == "foot_skate":
            for st, ls in zip(sets, ("-", "--")):
                rr = sorted({(r["trans"], r["gt_foot_skate"]) for r in rows if r["set"] == st})
                ax.plot([t for t, _ in rr], [v for _, v in rr], ls, color="black", label=f"GT [{st}]")
        ax.axvspan(train_range[0], train_range[1], color="gray", alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("transition length (frames); gray = training range")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("feet -> body vs Context / Two-stage / SLERP")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ecfg = load_named_config("eval", "lengths")
    bcfg = load_named_config("eval", "feet_to_body")
    fcfg = load_named_config("eval", "foot_only")
    methods = list(args.methods or bcfg["methods"])
    batch_size = int(args.batch_size or ecfg["batch_size"])

    ctx_run = load_run(ecfg["ckpt"], env_name=args.env, device=args.device)
    info = {"ctx": describe(ctx_run)}
    det = foot = body = None
    if "two_stage" in methods:
        det = load_detail(ecfg["detail_ckpt"], ctx_run)
        info["detail"] = {"ckpt": str(det.ckpt_path), "iteration": int(det.ckpt["iteration"])}
    if any(m in methods for m in ("body_pred", "body_pred_post")):
        foot = load_foot_run(fcfg["foot_ckpt"], ctx_run.layout, ctx_run.device)
        info["foot_only"] = {"ckpt": str(foot.ckpt_path), "iteration": int(foot.ckpt["iteration"])}
    if any(m in methods for m in BODY_METHODS):
        body = load_body_run(args.body_ckpt or bcfg["body_ckpt"], ctx_run.layout, ctx_run.device)
        check_same_stats(body, ctx_run, foot)
        if int(body.cfg["data"]["context_len"]) != ctx_run.ctx:
            raise ValueError("발 → 몸 모델과 Context 의 context_len 이 다릅니다.")
        info["body"] = {"ckpt": str(body.ckpt_path), "iteration": int(body.ckpt["iteration"]),
                        "best_val": float(body.ckpt["best_val"]), "attend_all": body.attend_all}
    for k, v in info.items():
        print(f"{k}: {v.get('ckpt')} (iteration {v.get('iteration')})")

    mean, std = load_bench_stats(ctx_run, ecfg)
    fs = ecfg["foot_skating"]
    foot_idx = [ctx_run.skel.names.index(n) for n in fs["joints"]]
    height = float(fs["height_threshold_cm"])
    out_dir = resolve_path(args.out or bcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for set_name in args.sets:
        ds, sp = build_eval_dataset(ctx_run, ecfg, set_name)
        max_t = max_trans_for_window(sp["window"], ctx_run.ctx)
        print(f"\n[평가셋 {set_name}] {len(ds)} 클립 (window {sp['window']}, offset {sp['offset']}) → trans ≤ {max_t}")
        for trans in bcfg["lengths"][set_name]:
            if trans > max_t:
                raise ValueError(f"{set_name} 평가셋 window {sp['window']} 에 trans {trans} 가 들어가지 않습니다.")
            tic = time.time()
            res, bs = eval_with_retry(ds, trans, methods, ctx_run, body, foot, det, mean, std, foot_idx, height,
                                      batch_size=batch_size)
            label = train_range_label(trans, ctx_run.train_range)
            for m in methods:
                rows.append(dict(set=set_name, trans=trans, train_range=label, method=m, batch_size=bs,
                                 gt_foot_skate=res["_gt_skate"], **res[m]))
            print(f"  trans {trans:3d} ({label}, batch {bs}): L2P " +
                  ", ".join(f"{m} {res[m]['L2P']:.3f}" for m in methods) + f"  ({time.time() - tic:.0f}초)")

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, methods, ctx_run.train_range, out_dir / "metrics.png")
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({**info, "sets": args.sets, "lengths": bcfg["lengths"], "methods": methods, "foot_skating": fs,
                   "time": datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, indent=2, default=str)
    for set_name in args.sets:
        print_table(rows, set_name, methods)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
