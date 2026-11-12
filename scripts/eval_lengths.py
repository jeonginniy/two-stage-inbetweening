"""
[연구실] transition 길이별 평가: L2P / L2Q / NPSS / 발 미끄러짐 (4단계, 교수님 미팅 피드백 [3])
==========================================================================================

이 스크립트가 하는 일
--------------------
1. 설정 읽기: configs/eval/lengths.yaml (길이 목록, 평가셋, 방법, 지표 설정, README 비교 수치)
   + <ckpt 폴더>/config.yaml (학습 때 설정: context_len 10, 학습 transition 범위 [5, 30])
2. L2P 정규화 통계 μ, σ (bench_stats: Subject 1–4, window 50, offset 20, 평균 중심 정렬) 를 계산하거나 캐시에서 읽음
   → data/cache/stats/bench_stats_rmi_*.npz
3. 평가셋 × 길이 × 방법마다 지표 계산 (src/evaluation/metrics.py)
     평가셋 official : 공식 벤치마크 (Subject 5, window 65, offset 40, 2232 클립). trans ≤ 53 만 (README 비교: 5/15/30/45)
     평가셋 long     : Subject 5, window 10 + max(lengths) + 2, offset 40 (2151 클립). 모든 길이에서 같은 시작 프레임
     방법: model (Context Transformer + 후처리 식 23), model_nopost (후처리 없음), interp (SLERP), zerov (zero-velocity)
   Detail Transformer 는 아직 구현·학습되지 않아 Context Transformer 만 평가한다.
4. 저장 (outputs/eval/<run>/)
     metrics_by_length.csv : 평가셋, 클립 수, 길이 (프레임/초), 학습 범위 안/밖, 방법, L2P, L2Q, NPSS,
                             발 미끄러짐 (예측/GT), README 수치 (official 평가셋), 실제 배치 크기
     metrics_by_length.png : long 평가셋의 길이 vs 각 지표 (학습 범위 [5, 30] 음영, README 수치 점)
     run_info.json         : 체크포인트, 설정, 평가 시각
5. GPU 메모리가 부족하면 배치를 절반으로 줄여 그 길이를 처음부터 다시 계산하고, 줄였다고 출력·CSV 에 남긴다.
6. --render 를 주면 길이마다 scripts/render_samples.py 와 같은 샘플 (video.yaml seed) 의 영상도 만든다.

모델 구조상 최대 길이는 없다 (상대 위치 PE 식 12–13, Keyframe PE 식 11 이 거리 "값" 을 MLP 에 넣음).
학습 범위 [min_trans, max_trans] 를 넘는 길이는 외삽이며 결과표에 "학습 범위 밖" 으로 표시한다.

실행 예:
    python scripts/eval_lengths.py                                          # 설정의 길이·평가셋·방법 전부
    python scripts/eval_lengths.py --sets official --methods interp zerov  # 기준선만 (지표 구현 확인용)
    python scripts/eval_lengths.py --lengths 30 60 --sets long --batch-size 32
    python scripts/eval_lengths.py --render                                 # + 길이별 샘플 영상
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # render_samples (--render)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_named_config, resolve_path  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.evaluation.baselines import interpolation, zero_velocity  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    LoadedRun,
    build_eval_dataset,
    describe,
    load_run,
    max_trans_for_window,
    prepare_window,
    train_range_label,
)
from src.evaluation.metrics import (  # noqa: E402
    MetricAccumulator,
    compute_bench_stats,
    foot_skating_per_clip,
    global_quats,
    global_rot_pos,
)
from src.training.context import run_model  # noqa: E402
from src.training.context_inputs import postprocess  # noqa: E402

METHODS = ("model", "model_nopost", "interp", "zerov")
METHOD_LABEL = {"model": "Context Transformer (+post)", "model_nopost": "Context Transformer (no post)",
                "interp": "Interpolation (SLERP)", "zerov": "Zero-velocity"}
METRICS = ("L2P", "L2Q", "NPSS", "foot_skate")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=None, help="기본: lengths.yaml 의 lengths")
    ap.add_argument("--sets", nargs="+", choices=["official", "long"], default=["official", "long"])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="기본: lengths.yaml 의 methods")
    ap.add_argument("--ckpt", type=Path, default=None, help="기본: lengths.yaml 의 ckpt")
    ap.add_argument("--eval-config", default="lengths", help="configs/eval/<이름>.yaml 또는 .yaml 경로")
    ap.add_argument("--batch-size", type=int, default=None, help="기본: lengths.yaml 의 batch_size")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: lengths.yaml 의 out_dir)")
    ap.add_argument("--render", action="store_true", help="길이마다 render_samples.py 와 같은 샘플 영상도 저장")
    ap.add_argument("--env", default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# L2P 통계 (캐시)
# -----------------------------------------------------------------------------
def load_bench_stats(run: LoadedRun, ecfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """bench_stats μ, σ (J*3,) 를 캐시에서 읽거나 계산해서 저장한다 [공식코드 F1]."""
    sp = ecfg["bench_stats"]
    cache = (run.env["paths"]["cache_dir"] / "stats" /
             f"bench_stats_rmi_ctx{run.ctx}_w{sp['window']}_o{sp['offset']}_s{sp['start_frame']}.npz")
    if cache.exists():
        with np.load(cache) as f:
            print(f"L2P 통계 캐시: {cache}")
            return torch.from_numpy(f["mean"]), torch.from_numpy(f["std"])
    print("L2P 통계 계산 중 (Subject 1–4, 평균 중심 정렬) ...")
    ds = LafanDataset(run.env["paths"]["lafan1_bvh_dir"], sp["actors"], sp["window"], sp["offset"], run.skel,
                      start_frame=sp["start_frame"], contact_threshold=run.data_cfg["contact_vel_threshold"],
                      cache_dir=run.env["paths"]["cache_dir"], dtype=torch.float64, verbose=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=run.env["num_workers"])
    mean, std = compute_bench_stats(loader, run.skel, run.ctx, run.data_cfg["forward_world_axis"], run.device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mean=mean.numpy(), std=std.numpy(), n_clips=len(ds))
    print(f"  {len(ds)} 클립 → {cache}")
    return mean, std


# -----------------------------------------------------------------------------
# 길이 하나 평가
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_one_length(run: LoadedRun, ds, trans: int, methods: list[str], batch_size: int,
                    mean: torch.Tensor, std: torch.Tensor, foot_idx: list[int], height_cm: float) -> dict:
    """평가셋 ds 전체에서 transition `trans` 의 방법별 지표.

    배치마다: 정답 창 (prepare_window) → 방법별 예측 (정렬된 원값 (B, L, D)) → FK → 전역 쿼터니언 → 지표 누적.
    모델은 한 번만 돌리고, 후처리 (식 23) 는 그 출력에 따로 적용해서 model / model_nopost 를 같이 얻는다.
    출력:  {방법: {"L2P", "L2Q", "NPSS", "foot_skate", "n_clips"}, "_gt_skate": GT 발 미끄러짐}
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=run.env["num_workers"])
    accs = {m: MetricAccumulator(mean, std, foot_idx, height_cm) for m in methods}
    gt_skate = []
    for batch in loader:
        pb, spec = prepare_window(run, batch, trans)
        s = spec.seq_slice
        grot_gt, gpos_gt = global_rot_pos(pb.x_gt_raw, pb.offsets, run.skel, run.layout)
        q_gt = global_quats(grot_gt)
        gt_skate.append(foot_skating_per_clip(gpos_gt, foot_idx, height_cm, s).cpu().numpy())

        preds = {}
        if "model" in methods or "model_nopost" in methods:
            y = run_model(run.model, pb.x_gt, spec, run.layout)          # (B, L, D) 정규화, transition 만 예측
            if "model_nopost" in methods:
                preds["model_nopost"] = run.norm.denormalize(y)
            if "model" in methods:
                preds["model"] = run.norm.denormalize(postprocess(y, pb.x_gt, spec))
        if "interp" in methods:
            preds["interp"] = interpolation(pb.x_gt_raw, spec, run.layout)
        if "zerov" in methods:
            preds["zerov"] = zero_velocity(pb.x_gt_raw, spec, run.layout)

        for m, x_raw in preds.items():
            grot, gpos = global_rot_pos(x_raw, pb.offsets, run.skel, run.layout)
            accs[m].add(gpos, gpos_gt, global_quats(grot), q_gt, s)
    out = {m: a.result() for m, a in accs.items()}
    out["_gt_skate"] = float(np.concatenate(gt_skate).mean())
    return out


def eval_with_retry(run, ds, trans, methods, batch_size, *stat_args) -> tuple[dict, int]:
    """eval_one_length + GPU 메모리 부족 시 배치를 절반으로 줄여 처음부터 다시. 출력: (결과, 실제 배치 크기)."""
    bs = batch_size
    while True:
        try:
            return eval_one_length(run, ds, trans, methods, bs, *stat_args), bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            print(f"  [메모리 부족] trans {trans}: batch {bs} → {bs // 2} 로 줄여 다시 계산")
            bs //= 2


# -----------------------------------------------------------------------------
# 저장 / 그래프
# -----------------------------------------------------------------------------
def readme_value(ecfg: dict, method: str, trans: int) -> list[float | None]:
    """README 수치 [L2P, L2Q, NPSS] (없으면 None 3개)."""
    ref = ecfg.get("readme_reference", {}).get(method, {})
    return list(ref.get(trans, ref.get(str(trans), [None, None, None])))


def plot_lengths(rows: list[dict], train_range: tuple[int, int], path: Path, run_name: str) -> None:
    """long 평가셋: 길이 vs L2P / L2Q / NPSS / 발 미끄러짐. README 수치 (official) 는 점으로."""
    long_rows = [r for r in rows if r["set"] == "long"]
    if not long_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"model": "tab:orange", "model_nopost": "tab:red", "interp": "tab:blue", "zerov": "tab:gray"}
    for ax, metric in zip(axes.flat, METRICS):
        ax.axvspan(train_range[0], train_range[1], color="tab:green", alpha=0.10, lw=0,
                   label=f"training range {train_range[0]}–{train_range[1]}")
        for m in METHODS:
            pts = sorted((r["trans"], r[metric]) for r in long_rows if r["method"] == m)
            if pts:
                ax.plot(*zip(*pts), "-o", ms=4, color=colors[m], label=METHOD_LABEL[m])
        if metric == "foot_skate":
            pts = sorted({(r["trans"], r["foot_skate_gt"]) for r in long_rows})
            ax.plot(*zip(*pts), "--", color="black", lw=1.2, label="GT")
        else:
            k = METRICS.index(metric)
            for m in ("model", "interp"):
                ref = sorted({(r["trans"], r[f"readme_{metric}"]) for r in rows
                              if r["set"] == "official" and r["method"] == m and r[f"readme_{metric}"] is not None})
                if ref:
                    ax.plot(*zip(*ref), "x", ms=8, mew=2, color=colors[m],
                            label=f"README {METHOD_LABEL[m]}" if k == 0 else None)
        ax.set_xscale("log")
        ticks = sorted({r["trans"] for r in long_rows})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("transition length (frames, 30 fps)")
        ax.set_title({"L2P": "L2P (global position, z-scored)", "L2Q": "L2Q (global quaternion)",
                      "NPSS": "NPSS", "foot_skate": "foot skating (cm/frame)"}[metric])
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    axes.flat[3].legend(fontsize=8, loc="upper left")
    n = long_rows[0]["n_clips"]
    fig.suptitle(f"{run_name}: error vs transition length (long eval set, {n} clips, same start frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_table(rows: list[dict], set_name: str) -> None:
    """콘솔 표: 평가셋 하나, 길이 × 방법, README 수치를 옆에."""
    sel = [r for r in rows if r["set"] == set_name]
    if not sel:
        return
    print(f"\n=== 평가셋 {set_name} ({sel[0]['n_clips']} 클립) ===")
    print(f"{'trans':>5} {'범위':<7} {'method':<13} {'L2P':>8} {'(README)':>9} {'L2Q':>8} {'(README)':>9} "
          f"{'NPSS':>8} {'(README)':>9} {'skate':>7} {'GT':>6} {'bs':>4}")
    for r in sel:
        ref = lambda k: f"{r[k]:9.4f}" if r[k] is not None else f"{'-':>9}"   # noqa: E731
        print(f"{r['trans']:5d} {r['train_range']:<7} {r['method']:<13} {r['L2P']:8.4f} {ref('readme_L2P')} "
              f"{r['L2Q']:8.4f} {ref('readme_L2Q')} {r['NPSS']:8.4f} {ref('readme_NPSS')} "
              f"{r['foot_skate']:7.3f} {r['foot_skate_gt']:6.3f} {r['batch_size']:4d}")


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ecfg = load_named_config("eval", args.eval_config)
    lengths = sorted(args.lengths or ecfg["lengths"])
    methods = list(args.methods or ecfg["methods"])
    batch_size = int(args.batch_size or ecfg["batch_size"])
    if args.lengths and max(args.lengths) > max(ecfg["lengths"]):
        ecfg = dict(ecfg, lengths=sorted(set(ecfg["lengths"]) | set(args.lengths)))   # long window 가 커짐
        print(f"[알림] --lengths 가 설정보다 길어 long 평가셋 window 를 늘립니다 (다른 길이와 시작 프레임이 달라질 수 있음).")
    run = load_run(args.ckpt or ecfg["ckpt"], env_name=args.env, device=args.device)
    info = describe(run)
    print(f"체크포인트 {info['ckpt']} (iteration {info['iteration']}), device={run.device}, "
          f"context {run.ctx}, 학습 transition 범위 {run.train_range}")
    print("Detail Transformer 는 구현·학습되지 않아 Context Transformer 만 평가합니다.")

    mean, std = load_bench_stats(run, ecfg)
    fs = ecfg["foot_skating"]
    foot_idx = [run.skel.names.index(n) for n in fs["joints"]]
    out_dir = resolve_path((args.out or ecfg["out_dir"]).format(run=run.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for set_name in args.sets:
        ds, sp = build_eval_dataset(run, ecfg, set_name)
        max_t = max_trans_for_window(sp["window"], run.ctx)
        print(f"\n[평가셋 {set_name}] {len(ds)} 클립 (actors {sp['actors']}, window {sp['window']}, "
              f"offset {sp['offset']}) → trans ≤ {max_t}")
        for trans in lengths:
            if trans > max_t:
                print(f"  trans {trans}: window {sp['window']} 에 들어가지 않아 건너뜀 (long 평가셋에서 계산)")
                continue
            tic = time.time()
            res, bs = eval_with_retry(run, ds, trans, methods, batch_size, mean, std, foot_idx,
                                      float(fs["height_threshold_cm"]))
            label = train_range_label(trans, run.train_range)
            msg = []
            for m in methods:
                r = res[m]
                ref = readme_value(ecfg, m, trans) if set_name == "official" else [None, None, None]
                rows.append({"set": set_name, "n_clips": r["n_clips"], "trans": trans,
                             "seconds": round(trans / run.data_cfg["fps"], 3), "train_range": label,
                             "method": m, "L2P": r["L2P"], "L2Q": r["L2Q"], "NPSS": r["NPSS"],
                             "foot_skate": r["foot_skate"], "foot_skate_gt": res["_gt_skate"],
                             "readme_L2P": ref[0], "readme_L2Q": ref[1], "readme_NPSS": ref[2],
                             "batch_size": bs})
                msg.append(f"{m} L2P {r['L2P']:.4f}")
            print(f"  trans {trans:3d} ({label}, batch {bs}{' ← 줄임' if bs != batch_size else ''}): "
                  f"{', '.join(msg)}  ({time.time() - tic:.0f}초)")

    csv_path = out_dir / "metrics_by_length.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    png_path = out_dir / "metrics_by_length.png"
    plot_lengths(rows, run.train_range, png_path, run.name)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({**info, "lengths": lengths, "sets": args.sets, "methods": methods,
                   "batch_size": batch_size, "foot_skating": fs, "bench_stats": ecfg["bench_stats"],
                   "time": datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, indent=2)
    for set_name in args.sets:
        print_table(rows, set_name)
    print(f"\n저장: {csv_path}\n      {png_path if png_path.exists() else '(그래프 없음: long 평가셋 미계산)'}")

    if args.render:
        from render_samples import render_clips, select_samples, write_manifest
        from src.visualization.skeleton_video import VideoStyle
        vcfg = load_named_config("viz", "video")
        style = VideoStyle.from_cfg(vcfg)
        ds, _ = build_eval_dataset(run, ecfg, "long")
        idx = select_samples(ds, int(vcfg["n_samples"]), int(vcfg["seed"]), int(vcfg["before_frames"]))
        for trans in lengths:
            res = render_clips(run, ds, idx, trans, vcfg, style, use_postprocess=bool(vcfg["postprocess"]))
            write_manifest(res, Path(res[0]["path"]).parent / "samples.csv")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
