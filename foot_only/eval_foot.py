"""
[연구실] 발 전용 모델 평가: foot-only vs Context (±후처리) vs Two-stage vs SLERP
==============================================================================

docs/foot_only_eval_plan.md 의 평가. 계산 함수는 foot_only/evaluate.py, 기존 평가 코드는 import 만 한다.

이 스크립트가 하는 일
--------------------
1. 설정: configs/eval/foot_only.yaml (발 전용 체크포인트, 평가셋별 길이, 방법, 출력 폴더)
         + configs/eval/lengths.yaml (ctx_base / det_cont 체크포인트, 평가셋 정의, 발 미끄러짐 설정, 배치 크기)
2. 체크포인트: ctx_base (inference.load_run), det_cont (inference.load_detail: 같은 Context 위에서 학습했는지 확인),
   foot_only (evaluate.load_foot_run). 세 모델의 입력 정규화 통계가 같은지 확인.
3. 평가셋 × 길이마다 모든 방법의 발 4관절 위치를 같은 정답 창에서 구해 지표 계산
     official (공식 벤치마크 2232 클립): 5, 15, 30, 45
     long     (2151 클립, 모든 길이에서 같은 시작 프레임): 30, 60, 90
4. 저장 (outputs/eval/foot_only/)
     metrics_long.csv : set, trans, train_range, method, source (pos|out|-), motion (all|동작), joint (all|관절), metric, value
     summary.csv      : 전체 클립 (motion = all) 의 방법별 주요 값 한 줄씩
     metrics.png      : 길이 vs 발 L2 / 발 미끄러짐 / 접촉 F1 (위치 판정) / 접촉 F1 (출력)
     run_info.json    : 체크포인트, 설정, 평가 시각
   GT 행 (method = gt) 은 검산용: 위치로 판정한 접촉 acc 가 1 이어야 하고 발 미끄러짐은 GT 값.

실행 예:
    python foot_only/eval_foot.py                                  # 설정의 평가셋·길이·방법 전부
    python foot_only/eval_foot.py --sets official --methods foot_only ctx_post
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from foot_only.data import FOOT_JOINT_NAMES  # noqa: E402
from foot_only.evaluate import (  # noqa: E402
    METHOD_LABEL, METHODS, FootMetricAccumulator, check_same_input_stats, load_foot_run,
    motion_name, predict_feet,
)
from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    build_eval_dataset, describe, load_detail, load_run, max_trans_for_window, train_range_label,
)

ANKLES, TOES = ("LeftFoot", "RightFoot"), ("LeftToe", "RightToe")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", choices=["official", "long"], default=["official", "long"])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="기본: foot_only.yaml 의 methods")
    ap.add_argument("--batch-size", type=int, default=None, help="기본: lengths.yaml 의 batch_size")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: foot_only.yaml 의 out_dir)")
    ap.add_argument("--env", default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


@torch.no_grad()
def eval_one_length(ds, trans, methods, batch_size, ctx_run, foot, det, fs) -> list[dict]:
    """평가셋 ds 전체, transition `trans` → FootMetricAccumulator.rows() (long 형식 행)."""
    toe_idx = [FOOT_JOINT_NAMES.index(n) for n in fs["joints"]]           # 발 4관절 배열 안 번호 (발끝 → [1, 3])
    acc = FootMetricAccumulator(toe_idx, float(fs["height_threshold_cm"]), ctx_run.data_cfg["contact_vel_threshold"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=ctx_run.env["num_workers"])
    for batch in loader:
        motions = [motion_name(ds.files[int(i)].stem) for i in batch["file_index"]]
        acc.add(motions, predict_feet(batch, trans, ctx_run, methods, foot=foot, det=det))
    return acc.rows()


def eval_with_retry(*args, batch_size: int) -> tuple[list[dict], int]:
    """GPU 메모리가 부족하면 배치를 절반으로 줄여 그 길이를 처음부터 다시 (eval_lengths.py 와 같은 방식)."""
    ds, trans, methods, *rest = args
    bs = batch_size
    while True:
        try:
            return eval_one_length(ds, trans, methods, bs, *rest), bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            print(f"  [메모리 부족] trans {trans}: batch {bs} → {bs // 2}")
            bs //= 2


def pick(rows, **kw) -> float:
    """행 목록에서 조건에 맞는 값 하나 (없으면 nan)."""
    for r in rows:
        if all(r[k] == v for k, v in kw.items()):
            return r["value"]
    return float("nan")


def mean2(rows, joints, **kw) -> float:
    """두 관절 값의 평균 (예: 발목 L/R F1)."""
    return sum(pick(rows, joint=j, **kw) for j in joints) / len(joints)


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    """motion = all 행 → (set, trans, method) 마다 한 줄 요약."""
    keys = sorted({(r["set"], r["trans"]) for r in rows}, key=lambda k: (k[0] != "official", k[1]))
    out = []
    for set_name, trans in keys:
        sel = [r for r in rows if r["set"] == set_name and r["trans"] == trans and r["motion"] == "all"]
        for m in [*methods, "gt"]:
            q = dict(method=m)
            row = {"set": set_name, "trans": trans, "train_range": sel[0]["train_range"], "method": m,
                   "n_clips": pick(sel, metric="n_clips", **q),
                   "foot_l2": pick(sel, metric="foot_l2", joint="all", **q),
                   "foot_skate": pick(sel, metric="foot_skate", **q)}
            for src in ("pos", "out"):
                row[f"f1_{src}"] = pick(sel, metric="f1", source=src, joint="all", **q)
                row[f"f1_{src}_ankle"] = mean2(sel, ANKLES, metric="f1", source=src, **q)
                row[f"f1_{src}_toe"] = mean2(sel, TOES, metric="f1", source=src, **q)
                row[f"acc_{src}"] = pick(sel, metric="acc", source=src, joint="all", **q)
            out.append(row)
    return out


def fmt(v: float, w: int = 6, p: int = 3) -> str:
    return f"{'-':>{w}}" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{w}.{p}f}"


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'set':<8} {'trans':>5} {'범위':<4} {'method':<11} {'L2 cm':>6} {'skate':>6} "
          f"{'F1pos':>6} {'발목':>6} {'발끝':>6} {'F1out':>6} {'발목':>6} {'발끝':>6} {'accpos':>6}")
    for r in summary:
        print(f"{r['set']:<8} {r['trans']:5d} {r['train_range']:<4} {r['method']:<11} {fmt(r['foot_l2'], 6, 2)} "
              f"{fmt(r['foot_skate'])} {fmt(r['f1_pos'])} {fmt(r['f1_pos_ankle'])} {fmt(r['f1_pos_toe'])} "
              f"{fmt(r['f1_out'])} {fmt(r['f1_out_ankle'])} {fmt(r['f1_out_toe'])} {fmt(r['acc_pos'])}")


def print_motions(rows: list[dict], methods: list[str], motions: list[str]) -> None:
    """동작별 접촉 F1 (관절별, 위치 판정 / 출력) 과 라벨 접촉 프레임 수."""
    for (set_name, trans) in sorted({(r["set"], r["trans"]) for r in rows}, key=lambda k: (k[0] != "official", k[1])):
        sel = [r for r in rows if r["set"] == set_name and r["trans"] == trans]
        print(f"\n[{set_name} trans {trans}] 동작별 접촉 F1 (관절 순서 {', '.join(FOOT_JOINT_NAMES)})")
        for g in motions:
            n = pick(sel, method="gt", motion=g, metric="n_clips")
            if math.isnan(n):
                continue
            npos = [int(pick(sel, method="gt", source="pos", motion=g, joint=j, metric="n_pos")) for j in FOOT_JOINT_NAMES]
            print(f"  {g} ({int(n)} 클립, 라벨 접촉 프레임 {npos})")
            for m in methods:
                for src in ("pos", "out"):
                    f1 = [pick(sel, method=m, source=src, motion=g, joint=j, metric="f1") for j in FOOT_JOINT_NAMES]
                    if all(math.isnan(v) for v in f1) and src == "out":
                        continue
                    l2 = pick(sel, method=m, source="-", motion=g, joint="all", metric="foot_l2")
                    print(f"    {m:<11} {src:<3} F1 {' '.join(fmt(v, 5, 2) for v in f1)}   L2 {l2:6.2f} cm")


def plot(summary: list[dict], methods: list[str], train_range, path: Path) -> None:
    """길이 vs 지표 4개. official 은 실선·원, long 은 점선·네모 (trans 30 은 두 평가셋 모두)."""
    colors = {"foot_only": "tab:green", "ctx_post": "tab:orange", "ctx_nopost": "tab:red", "two_stage": "tab:purple",
              "interp": "tab:blue", "gt": "black"}
    panels = [("foot_l2", "foot position L2 (cm)"), ("foot_skate", "foot skating (cm/frame, toes)"),
              ("f1_pos", "contact F1 from predicted positions (world)"), ("f1_out", "contact F1 from contact output")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        ax.axvspan(train_range[0], train_range[1], color="tab:green", alpha=0.08, lw=0)
        for m in [*methods, "gt"]:
            if m == "gt" and key in ("foot_l2", "f1_out"):
                continue
            for set_name, style in (("official", "-o"), ("long", "--s")):
                pts = sorted((r["trans"], r[key]) for r in summary
                             if r["method"] == m and r["set"] == set_name and not math.isnan(r[key]))
                if pts:
                    ax.plot(*zip(*pts), style, ms=4, color=colors[m], lw=1.2 if m != "gt" else 0.8,
                            label=f"{METHOD_LABEL[m]} ({set_name})")
        ticks = sorted({r["trans"] for r in summary})
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("transition length (frames, 30 fps)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    axes.flat[3].legend(fontsize=7, loc="lower left")
    fig.suptitle("Foot-only vs Context / Two-stage / SLERP (green band = training range, black = GT)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ecfg = load_named_config("eval", "lengths")
    fcfg = load_named_config("eval", "foot_only")
    methods = list(args.methods or fcfg["methods"])
    batch_size = int(args.batch_size or ecfg["batch_size"])

    ctx_run = load_run(ecfg["ckpt"], env_name=args.env, device=args.device)
    info = {"ctx": describe(ctx_run)}
    det = None
    if "two_stage" in methods:
        det = load_detail(ecfg["detail_ckpt"], ctx_run)
        info["detail"] = {"ckpt": str(det.ckpt_path), "iteration": int(det.ckpt["iteration"]),
                          "ctx_postprocess": det.ctx_postprocess}
    foot = None
    if "foot_only" in methods:
        foot = load_foot_run(fcfg["foot_ckpt"], ctx_run.layout, ctx_run.device)
        check_same_input_stats(foot, ctx_run)
        if int(foot.cfg["data"]["context_len"]) != ctx_run.ctx:
            raise ValueError("발 전용 모델과 Context 의 context_len 이 다릅니다.")
        info["foot_only"] = {"ckpt": str(foot.ckpt_path), "iteration": int(foot.ckpt["iteration"]),
                             "best_val": float(foot.ckpt["best_val"])}
    for k, v in info.items():
        print(f"{k}: {v.get('ckpt')} (iteration {v.get('iteration')})")

    fs = ecfg["foot_skating"]
    out_dir = resolve_path(args.out or fcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    used_bs = defaultdict(dict)
    for set_name in args.sets:
        ds, sp = build_eval_dataset(ctx_run, ecfg, set_name)
        max_t = max_trans_for_window(sp["window"], ctx_run.ctx)
        print(f"\n[평가셋 {set_name}] {len(ds)} 클립 (window {sp['window']}, offset {sp['offset']}) → trans ≤ {max_t}")
        for trans in fcfg["lengths"][set_name]:
            if trans > max_t:
                raise ValueError(f"{set_name} 평가셋 window {sp['window']} 에 trans {trans} 가 들어가지 않습니다.")
            tic = time.time()
            res, bs = eval_with_retry(ds, trans, methods, ctx_run, foot, det, fs, batch_size=batch_size)
            label = train_range_label(trans, ctx_run.train_range)
            used_bs[set_name][trans] = bs
            rows += [dict(set=set_name, trans=trans, train_range=label, **r) for r in res]
            l2 = ", ".join(f"{m} {pick(res, method=m, motion='all', joint='all', metric='foot_l2'):.2f}" for m in methods)
            print(f"  trans {trans:3d} ({label}, batch {bs}): 발 L2 cm {l2}  ({time.time() - tic:.0f}초)")

    with open(out_dir / "metrics_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, methods)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    plot(summary, methods, ctx_run.train_range, out_dir / "metrics.png")
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({**info, "sets": args.sets, "lengths": fcfg["lengths"], "methods": methods,
                   "batch_size": batch_size, "used_batch_size": used_bs, "foot_skating": fs,
                   "time": datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, indent=2)
    print_summary(summary)
    print_motions(rows, methods, list(fcfg["print_motions"]))
    print(f"\n저장: {out_dir}/{{metrics_long.csv, summary.csv, metrics.png, run_info.json}}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
