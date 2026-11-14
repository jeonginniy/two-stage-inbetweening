"""
[연구실] 발 전용 모델 영상: 발끝 궤적 · 높이 · 속도 · 접촉 (GT vs foot-only vs Two-stage)
=======================================================================================

발 전용 모델은 발 4관절 위치만 내므로 전신 스켈레톤 영상 (src/visualization/skeleton_video.py) 대신
"발이 땅에 제대로 서 있는지" 를 보는 영상을 만든다. 계산은 평가와 같은 함수 (foot_only/evaluate.predict_feet).

화면 (1280x720)
    왼쪽     : 위에서 본 발끝 2개 (LeftToe ●, RightToe ■) 궤적 (정렬 좌표 X = 진행 방향, Z = 옆).
               접촉 중인 발은 한 점에 머물러야 하고, 미끄러지면 선이 늘어진다.
    오른쪽 위: plot_joint 높이 (cm) — GT 접촉 구간은 회색 띠
    오른쪽 중: plot_joint 수평 속도 (cm/frame) — 발 미끄러짐. 0.2 선 = 접촉 기준 (참고)
    오른쪽 아래: plot_joint 접촉 띠: GT 라벨 / 방법별 접촉 출력 (out) / 예측 위치로 판정 (pos, 월드 좌표)
    제목: 클립, 길이, 프레임, 구간 (context 파랑 / generated 주황 / target 빨강), 클립 지표 (발 L2, 발 미끄러짐)

설정: configs/viz/foot_only.yaml (클립, 길이, 색, 출력 경로)
실행: python foot_only/render_feet.py        → outputs/videos/foot_only/<clip>/trans<길이>.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from foot_only.data import FOOT_JOINT_NAMES  # noqa: E402
from foot_only.evaluate import (  # noqa: E402
    METHOD_LABEL, check_same_input_stats, load_foot_run, position_contacts, predict_feet,
)
from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import build_eval_dataset, collate, load_detail, load_run  # noqa: E402
from src.evaluation.metrics import foot_skating_per_clip  # noqa: E402

TOES = [FOOT_JOINT_NAMES.index("LeftToe"), FOOT_JOINT_NAMES.index("RightToe")]


def find_clip(ds, name: str) -> int:
    """'<BVH 이름>_<시작 프레임>' → 평가셋 클립 번호."""
    stem, start = name.rsplit("_", 1)
    for fi, f in enumerate(ds.files):
        if f.stem == stem:
            idx = int(ds._cum[fi]) + int(start) // ds.offset
            if ds.locate(idx) != (fi, int(start)):
                raise ValueError(f"{name}: 시작 프레임 {start} 가 offset {ds.offset} 의 배수가 아닙니다.")
            return idx
    raise ValueError(f"{name}: 평가셋에 {stem} 이 없습니다.")


def render_one(path: Path, data: dict, vcfg: dict, title: str) -> None:
    """data (numpy, 아래 참고) → mp4.

    data:
        feet    {방법: (L, 4, 3)}  정렬 좌표 발 위치 ("gt" 포함)
        out     {방법: (L, 4)}     접촉 출력 0/1 (없으면 빠짐)
        pos     {방법: (L, 4)}     예측 위치로 판정한 접촉 0/1
        label   (L, 4)             GT 접촉 라벨
        ctx, trans, metrics {방법: (발 L2, 발 미끄러짐)}
    """
    ctx, trans = data["ctx"], data["trans"]
    tgt = ctx + trans
    names = ["gt", *vcfg["methods"]]
    col = vcfg["colors"]
    seg_col = vcfg["segment_colors"]
    j = FOOT_JOINT_NAMES.index(vcfg["plot_joint"])
    frames = list(range(tgt + 1)) + [tgt] * (int(vcfg["target_hold_frames"]) - 1)
    t_axis = np.arange(tgt + 1)

    fig = plt.figure(figsize=vcfg["figsize"], dpi=vcfg["dpi"])
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], height_ratios=[1, 1, 0.8], hspace=0.45, wspace=0.18,
                          left=0.05, right=0.98, top=0.86, bottom=0.07)
    ax_top = fig.add_subplot(gs[:, 0])
    ax_h, ax_v, ax_c = (fig.add_subplot(gs[r, 1]) for r in range(3))

    # --- 왼쪽: 위에서 본 궤적 (X 진행 방향 → 가로, Z → 세로)
    allp = np.concatenate([data["feet"][m][: tgt + 1, TOES] for m in names])
    lo, hi = allp[..., [0, 2]].min(axis=(0, 1)), allp[..., [0, 2]].max(axis=(0, 1))
    c, half = (lo + hi) / 2, max((hi - lo).max() / 2 + 15, 40)
    ax_top.set_xlim(c[0] - half, c[0] + half)
    ax_top.set_ylim(c[1] - half, c[1] + half)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (cm, forward)")
    ax_top.set_ylabel("Z (cm)")
    ax_top.set_title("toes from above (● LeftToe, ■ RightToe; filled = contact output / GT label)", fontsize=9)
    ax_top.grid(alpha=0.3)
    tg = data["feet"]["gt"][tgt, TOES]
    ax_top.scatter(tg[:, 0], tg[:, 2], s=160, facecolors="none", edgecolors=seg_col["target"], lw=1.5, zorder=1,
                   label="target toes")
    lines, dots = {}, {}
    for m in names:
        for k, (ti, mk) in enumerate(zip(TOES, ("o", "s"))):
            (lines[m, k],) = ax_top.plot([], [], "-", color=col[m], lw=1.6 if m != "gt" else 2.2,
                                         alpha=0.9, label=("GT" if m == "gt" else METHOD_LABEL[m]) if k == 0 else None)
            dots[m, k] = ax_top.scatter([], [], s=55, marker=mk, color=col[m], zorder=3)
    ax_top.legend(fontsize=8, loc="upper left")

    # --- 오른쪽: 높이, 속도, 접촉 띠
    def shade(ax):
        lab = data["label"][: tgt + 1, j]
        for t in np.flatnonzero(lab > 0.5):
            ax.axvspan(t - 0.5, t + 0.5, color="0.85", lw=0, zorder=0)
        ax.axvspan(-0.5, ctx - 0.5, color=seg_col["context"], alpha=0.07, lw=0)
        ax.axvline(tgt, color=seg_col["target"], lw=1, alpha=0.6)
        ax.set_xlim(-0.5, tgt + 0.5)
        ax.grid(alpha=0.3)

    h_lines, v_lines = {}, {}
    heights = {m: data["feet"][m][: tgt + 1, j, 1] for m in names}
    speeds = {m: np.r_[np.nan, np.linalg.norm(np.diff(data["feet"][m][: tgt + 1, j][:, [0, 2]], axis=0), axis=1)]
              for m in names}
    for ax, series, store, ttl in ((ax_h, heights, h_lines, f"{vcfg['plot_joint']} height (cm)  — gray = GT contact"),
                                   (ax_v, speeds, v_lines, f"{vcfg['plot_joint']} horizontal speed (cm/frame)")):
        shade(ax)
        for m in names:
            (store[m],) = ax.plot([], [], color=col[m], lw=1.5 if m != "gt" else 2.0)
        vals = np.concatenate([series[m][np.isfinite(series[m])] for m in names])
        ax.set_ylim(min(0, vals.min()) - 1, np.percentile(vals, 99) * 1.15 + 1)
        ax.set_title(ttl, fontsize=9)
    ax_v.axhline(0.2, color="k", lw=0.8, ls=":")
    cursors = [ax.axvline(0, color="k", lw=0.8) for ax in (ax_h, ax_v, ax_c)]

    rows = [("GT label", data["label"][:, j], col["gt"])]
    for m in vcfg["methods"]:
        if m in data["out"]:
            rows.append((f"{m} out", data["out"][m][:, j], col[m]))
        rows.append((f"{m} pos", data["pos"][m][:, j], col[m]))
    for r, (_, v, cc) in enumerate(rows):
        on = np.flatnonzero(v[: tgt + 1] > 0.5)
        ax_c.scatter(on, np.full(len(on), r), marker="s", s=14, color=cc)
    ax_c.set_yticks(range(len(rows)))
    ax_c.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax_c.set_ylim(len(rows) - 0.5, -0.5)
    shade(ax_c)
    ax_c.set_title(f"{vcfg['plot_joint']} contact (out = model output, pos = judged from predicted positions)", fontsize=9)
    ax_c.set_xlabel("frame (context | generated | target)")

    met = "   ".join(f"{METHOD_LABEL[m]}: L2 {data['metrics'][m][0]:.1f} cm, skate {data['metrics'][m][1]:.2f}"
                     for m in vcfg["methods"]) + f"   (GT skate {data['metrics']['gt'][1]:.2f} cm/frame)"
    sup = fig.suptitle("", fontsize=12, y=0.975)
    fig.text(0.5, 0.91, met, ha="center", fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=int(vcfg["fps"]))
    with writer.saving(fig, str(path), dpi=vcfg["dpi"]):
        for n, t in enumerate(frames):
            seg = "CONTEXT" if t < ctx else ("GENERATED" if t < tgt else "TARGET")
            sup.set_text(f"{title}   frame {t}/{tgt}   {seg}")
            sup.set_color(seg_col[seg.lower()])
            for m in names:
                for k, ti in enumerate(TOES):
                    p = data["feet"][m][: t + 1, ti]
                    lines[m, k].set_data(p[:, 0], p[:, 2])
                    dots[m, k].set_offsets(p[-1:, [0, 2]])
                    filled = (data["label"] if m == "gt" else data["out"].get(m, data["pos"][m]))[t, ti] > 0.5
                    dots[m, k].set_facecolor(col[m] if filled else "white")
                    dots[m, k].set_edgecolor(col[m])
                h_lines[m].set_data(t_axis[: t + 1], heights[m][: t + 1])
                v_lines[m].set_data(t_axis[: t + 1], speeds[m][: t + 1])
            for cur in cursors:
                cur.set_xdata([t, t])
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--env", default=None)
    args = ap.parse_args()
    vcfg = load_named_config("viz", "foot_only")
    ecfg = load_named_config("eval", "lengths")
    fcfg = load_named_config("eval", "foot_only")
    ctx_run = load_run(ecfg["ckpt"], env_name=args.env)
    det = load_detail(ecfg["detail_ckpt"], ctx_run)
    foot = load_foot_run(fcfg["foot_ckpt"], ctx_run.layout, ctx_run.device)
    check_same_input_stats(foot, ctx_run)
    ds, _ = build_eval_dataset(ctx_run, ecfg, "long")
    thr = ctx_run.data_cfg["contact_vel_threshold"]
    fs = ecfg["foot_skating"]
    toe_idx = [FOOT_JOINT_NAMES.index(n) for n in fs["joints"]]

    for clip in args.clips or vcfg["clips"]:
        batch = collate(ds, [find_clip(ds, clip)])
        for trans in args.lengths or vcfg["lengths"]:
            fp = predict_feet(batch, trans, ctx_run, list(vcfg["methods"]), foot=foot, det=det)
            s = fp.spec.seq_slice
            np_ = lambda x: x[0].detach().cpu().numpy()                                   # noqa: E731
            data = {"ctx": ctx_run.ctx, "trans": trans, "label": np_(fp.labels), "feet": {}, "out": {}, "pos": {},
                    "metrics": {}}
            for m, (feet, prob) in fp.preds.items():
                if m not in ("gt", *vcfg["methods"]):
                    continue
                data["feet"][m] = np_(feet)
                data["pos"][m] = np_(position_contacts(feet, fp.frame, thr))
                if prob is not None:
                    data["out"][m] = np_((prob > 0.5).float())
                l2 = float((feet[:, s] - fp.feet_gt[:, s]).norm(dim=-1).mean())
                sk = float(foot_skating_per_clip(feet, toe_idx, float(fs["height_threshold_cm"]), s))
                data["metrics"][m] = (l2, sk)
            path = resolve_path(vcfg["out_path"].format(clip=clip, trans=trans))
            render_one(path, data, vcfg, f"{clip}  trans {trans}")
            print(f"{path}  " + ", ".join(f"{m} L2 {v[0]:.1f} skate {v[1]:.2f}" for m, v in data["metrics"].items()))


if __name__ == "__main__":
    with torch.no_grad():
        main()
