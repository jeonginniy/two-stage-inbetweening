"""
[연구실] 발 → 몸 모델 영상: GT / Two-stage / 발 → 몸 (예측한 발) / 발 → 몸 (정답 발) 전신 스켈레톤 2×2
====================================================================================================

계산은 평가와 같은 함수 (feet_to_body/evaluate.predict_bodies) 를 쓴다. 발 → 몸 출력은 평가의 주 수치와 같이
후처리 (식 23) 를 적용한 것 (body_*_post).

화면 (1280x720)
    네 칸 모두 같은 카메라·같은 축 범위. 스켈레톤 색 = 구간 (context 파랑 / generated 주황 / target 빨강).
    GT 가 아닌 칸에는 GT 를 얇은 회색으로 겹쳐 그려서 어긋남이 보이게 한다.
    발 → 몸 칸의 초록 점 = 모델에 조건으로 준 발 4관절 (예측한 발 또는 정답 발).
    칸 제목: 방법, 그 클립의 전신 위치 오차 (전역 관절 위치 평균 L2, cm), 발 미끄러짐 (cm/frame)

설정: configs/viz/feet_to_body.yaml (클립, 길이, 칸 구성, 체크포인트, 색, 출력 경로)
실행: python feet_to_body/render_body.py      → outputs/videos/feet_to_body/<clip>/trans<길이>.mp4
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

from feet_to_body.evaluate import BODY_METHODS, check_same_stats, load_body_run, predict_bodies  # noqa: E402
from foot_only.evaluate import load_foot_run  # noqa: E402
from foot_only.render_feet import find_clip  # noqa: E402
from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import build_eval_dataset, collate, load_detail, load_run  # noqa: E402
from src.evaluation.metrics import foot_skating_per_clip, global_rot_pos  # noqa: E402
# 같은 프로젝트의 그리기 도우미 (바닥 격자가 있는 3D 축, 스켈레톤 선 묶음) 를 그대로 쓴다
from src.visualization.skeleton_video import _scene_box, _setup_axis, _Skeleton3D, to_plot  # noqa: E402

# 2×2 칸의 화면 영역 (x0, y0, x1, y1), figure 비율
BANDS = [(0.01, 0.47, 0.495, 0.86), (0.505, 0.47, 0.99, 0.86), (0.01, 0.02, 0.495, 0.41), (0.505, 0.02, 0.99, 0.41)]


def render_one(path: Path, panels: list[dict], ctx: int, trans: int, parents, vcfg: dict, title: str) -> None:
    """칸 목록 → mp4.

    panels: [{"label": 칸 제목, "gpos": (L, J, 3) 정렬 좌표 (cm), "given": (L, 4, 3) 준 발 또는 None,
              "is_gt": bool}], 첫 칸이 GT 가 아니어도 되지만 "is_gt" 칸의 gpos 를 회색 겹침에 쓴다.
    """
    tgt = ctx + trans
    seg_col = vcfg["segment_colors"]
    frames = list(range(tgt + 1)) + [tgt] * (int(vcfg["target_hold_frames"]) - 1)
    gt = next(p for p in panels if p["is_gt"])["gpos"]
    pts = [to_plot(p["gpos"][: tgt + 1]) for p in panels]
    lo, hi, floor_h = _scene_box(pts)                                   # 네 칸 공통 카메라 범위

    fig = plt.figure(figsize=vcfg["figsize"], dpi=vcfg["dpi"])
    sup = fig.suptitle("", fontsize=13, y=0.975)
    items = []
    for p, P, band in zip(panels, pts, BANDS):
        ax = _setup_axis(fig, lo, hi, floor_h, band)
        ghost = None if p["is_gt"] else _Skeleton3D(ax, parents, lw=1.0, alpha=0.7, zorder=2)
        skel = _Skeleton3D(ax, parents, lw=2.2, alpha=1.0, zorder=3)
        given = None
        if p["given"] is not None:
            (given,) = ax.plot([], [], [], "o", ms=4, color=vcfg["given_feet_color"], zorder=4)
        fig.text((band[0] + band[2]) / 2, band[3] + 0.012, p["label"], ha="center", fontsize=10)
        items.append((P, ghost, skel, given, to_plot(p["given"][: tgt + 1]) if p["given"] is not None else None))
    G = to_plot(gt[: tgt + 1])

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=int(vcfg["fps"]))
    with writer.saving(fig, str(path), dpi=vcfg["dpi"]):
        for t in frames:
            seg = "context" if t < ctx else ("generated" if t < tgt else "target")
            sup.set_text(f"{title}   frame {t}/{tgt}   {seg.upper()}   (gray = GT, green dots = feet given to the model)")
            sup.set_color(seg_col[seg])
            for P, ghost, skel, given, gp in items:
                if ghost is not None:
                    ghost.update(G[t], vcfg["ghost_color"])
                skel.update(P[t], seg_col[seg])
                if given is not None:
                    given.set_data(gp[t, :, 0], gp[t, :, 1])
                    given.set_3d_properties(gp[t, :, 2])
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--env", default=None)
    args = ap.parse_args()
    vcfg = load_named_config("viz", "feet_to_body")
    ecfg = load_named_config("eval", "lengths")
    ctx_run = load_run(ecfg["ckpt"], env_name=args.env)
    det = load_detail(ecfg["detail_ckpt"], ctx_run)
    foot = load_foot_run(load_named_config("eval", "foot_only")["foot_ckpt"], ctx_run.layout, ctx_run.device)
    bodies = {}
    for p in vcfg["panels"]:
        if p["method"] in BODY_METHODS and p["body_ckpt"] not in bodies:
            bodies[p["body_ckpt"]] = load_body_run(p["body_ckpt"], ctx_run.layout, ctx_run.device)
            check_same_stats(bodies[p["body_ckpt"]], ctx_run, foot)
    ds, _ = build_eval_dataset(ctx_run, ecfg, "long")
    fs = ecfg["foot_skating"]
    toe_idx = [ctx_run.skel.names.index(n) for n in fs["joints"]]
    skel, layout = ctx_run.skel, ctx_run.layout

    for clip in args.clips or vcfg["clips"]:
        batch = collate(ds, [find_clip(ds, clip)])
        for trans in args.lengths or vcfg["lengths"]:
            panels, log = [], []
            for p in vcfg["panels"]:
                m = p["method"]
                bp = predict_bodies(batch, trans, ctx_run, [] if m == "gt" else [m], bodies.get(p.get("body_ckpt")), foot, det)
                x_raw = bp.pb.x_gt_raw if m == "gt" else bp.preds[m]
                _, gpos = global_rot_pos(x_raw, bp.pb.offsets, skel, layout)              # (1, L, J, 3)
                _, gpos_gt = global_rot_pos(bp.pb.x_gt_raw, bp.pb.offsets, skel, layout)
                s = bp.spec.seq_slice
                l2 = float((gpos[:, s] - gpos_gt[:, s]).norm(dim=-1).mean())
                sk = float(foot_skating_per_clip(gpos, toe_idx, float(fs["height_threshold_cm"]), s))
                given = bp.cond_feet.get(m)
                label = f"{p['label']}   skate {sk:.2f} cm/frame" + ("" if m == "gt" else f",  position error {l2:.1f} cm")
                panels.append({"label": label, "gpos": gpos[0].cpu().numpy(), "is_gt": m == "gt",
                               "given": given[0].reshape(-1, 4, 3).cpu().numpy() if given is not None else None})
                log.append(f"{m} L2 {l2:.1f} skate {sk:.2f}")
            path = resolve_path(vcfg["out_path"].format(clip=clip, trans=trans))
            render_one(path, panels, ctx_run.ctx, trans, skel.parents, vcfg, f"{clip}  trans {trans}")
            print(f"{path}  " + ", ".join(log))


if __name__ == "__main__":
    with torch.no_grad():
        main()
