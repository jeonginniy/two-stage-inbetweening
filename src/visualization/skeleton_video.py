"""
스켈레톤 비교 영상: 구간별 색 구분 (4단계, 교수님 미팅 피드백 [1])
================================================================

이 파일이 하는 일
-----------------
정답(GT)과 생성(PRED) 스켈레톤을 나란히 그린 mp4 를 만든다. 스켈레톤 색은 "지금 프레임이 어느 구간인지" 를 따른다.

    BEFORE (회색)  : context 앞의 원본 프레임 (모델 입력 아님. 앞뒤 흐름을 보려고 붙임)
    CONTEXT (파랑) : 모델이 받은 과거 입력 (논문 3.3 의 context 프레임 0 .. t_ctx−1)
    GENERATED (주황): transition 구간. GT 쪽은 원본 동작, PRED 쪽은 모델이 생성한 동작
    TARGET (빨강)  : target 프레임 (모델 입력). 1초 동안 멈춰서 보여준다
    + GENERATED 동안 target 포즈를 반투명(빨강)하게 겹쳐 그려서 "어디로 가야 하는지" 보이게 한다.

화면 구성
    위: 프레임 번호 / 구간 이름 / 범례,  가운데: 3D 스켈레톤,  아래: 프레임별 위치 오차 (cm) 그래프
배치 (VideoStyle.layout)
    side   : 같은 3D 축에 GT 와 PRED 를 진행 방향(+X)에 수직(Z)으로 side_offset 만큼 띄워 나란히
    panels : 좌우 두 패널, 두 패널의 축 범위를 똑같이 맞춤
    auto   : side 로 그렸을 때 두 캐릭터가 어느 프레임에서든 가까우면(겹치면) panels

쓰는 순서
---------
    style = VideoStyle.from_cfg(load_named_config("viz", "video"))
    segs  = segment_labels(before=30, ctx=10, trans=30, hold=30)          # 프레임마다 구간 이름
    render_comparison_video(path, gt, pred, segs, target_pose, parents, head, style, title, frame_info, err)

모든 좌표는 데이터 좌표 (Y 위, cm, 학습용 정렬 좌표) 로 받는다. 그릴 때만 matplotlib 3D (Z 위) 로 바꾼다.
필요한 것: matplotlib, 시스템 ffmpeg.

scripts/render_compare.py (이전 발표용 영상) 도 이 파일의 pick_korean_font / to_plot / fit_box 를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

UNITS_PER_METER = 100.0                      # LaFAN1 BVH 는 cm
SEGMENTS = ("before", "context", "generated", "target")
SEGMENT_NAMES = {"before": "BEFORE", "context": "CONTEXT", "generated": "GENERATED", "target": "TARGET"}


# -----------------------------------------------------------------------------
# 공용 그리기 도우미 (render_compare.py 와 공유)
# -----------------------------------------------------------------------------
def pick_korean_font() -> bool:
    """한글 글꼴이 있으면 matplotlib 기본 글꼴로 설정하고 True. 없으면 False (영어 라벨 사용)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Malgun Gothic", "AppleGothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def to_plot(p: np.ndarray) -> np.ndarray:
    """데이터 좌표 (..., 3) (Y 위, 오른손) → matplotlib 3D (Z 위, 오른손): (x, y, z) → (x, −z, y)."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def fit_box(fig, ax, lo: np.ndarray, hi: np.ndarray, band: tuple[float, float, float, float]) -> None:
    """3D 상자 (lo~hi, matplotlib 좌표 (3,)) 가 화면 영역 band = (x0, y0, x1, y1) (figure 비율) 를 최대한 채우게 한다.

    matplotlib 3D 축은 항상 정사각형 영역에 그려서, 길고 납작한 장면은 넓은 화면을 다 못 쓴다.
    그래서 축을 그림보다 큰 정사각형으로 두고, 상자 꼭짓점 8개의 화면 위치를 재서 zoom 과 축 위치를 맞춘다.
    """
    W, H = fig.get_size_inches() * fig.dpi
    side = max(W, H) * 1.5
    cx, cy = (band[0] + band[2]) / 2, (band[1] + band[3]) / 2
    ax.set_position([cx - side / W / 2, cy - side / H / 2, side / W, side / H])
    zoom = 1.0
    ax.set_box_aspect(tuple(hi - lo), zoom=zoom)
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1)
    bx0, by0, bx1, by1 = band
    for _ in range(2):                                   # 1) 크기 맞추기 2) 위치 맞추기
        fig.canvas.draw()
        xs, ys, _ = proj3d.proj_transform(*corners, ax.M)
        d = ax.transData.transform(np.c_[xs, ys]) / [W, H]          # figure 비율
        w, h = np.ptp(d[:, 0]), np.ptp(d[:, 1])
        k = min((bx1 - bx0) / w, (by1 - by0) / h)
        pos = ax.get_position()
        dx = (bx0 + bx1) / 2 - (d[:, 0].min() + d[:, 0].max()) / 2 * k - (1 - k) * (pos.x0 + pos.width / 2)
        dy = (by0 + by1) / 2 - (d[:, 1].min() + d[:, 1].max()) / 2 * k - (1 - k) * (pos.y0 + pos.height / 2)
        zoom *= k
        ax.set_box_aspect(tuple(hi - lo), zoom=zoom)
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])


# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
@dataclass
class VideoStyle:
    """configs/viz/video.yaml 의 그리기 관련 값."""

    colors: dict[str, str]
    ghost_enabled: bool = True
    ghost_alpha: float = 0.25
    layout: str = "auto"                 # auto | side | panels
    side_offset_cm: float = 150.0
    overlap_margin_cm: float = 20.0
    fps: int = 30
    dpi: int = 100
    figsize: tuple[float, float] = field(default=(12.8, 7.2))

    @classmethod
    def from_cfg(cls, cfg: dict) -> "VideoStyle":
        """configs/viz/video.yaml dict → VideoStyle. 색 4개 (before/context/generated/target) 가 모두 있어야 한다."""
        missing = [s for s in SEGMENTS if s not in cfg["colors"]]
        if missing:
            raise ValueError(f"configs/viz/video.yaml 의 colors 에 {missing} 가 없습니다.")
        if cfg["layout"] not in ("auto", "side", "panels"):
            raise ValueError(f"layout 은 auto/side/panels: {cfg['layout']!r}")
        return cls(colors=dict(cfg["colors"]), ghost_enabled=bool(cfg["target_ghost"]["enabled"]),
                   ghost_alpha=float(cfg["target_ghost"]["alpha"]), layout=cfg["layout"],
                   side_offset_cm=float(cfg["side_offset_cm"]), overlap_margin_cm=float(cfg["overlap_margin_cm"]),
                   fps=int(cfg["fps"]), dpi=int(cfg["dpi"]), figsize=tuple(cfg["figsize"]))


# -----------------------------------------------------------------------------
# 구간 / 배치 정하기
# -----------------------------------------------------------------------------
def segment_labels(before: int, ctx: int, trans: int, hold: int) -> list[str]:
    """영상 프레임마다 구간 이름.

    [before 개 "before"] [ctx 개 "context"] [trans 개 "generated"] [hold 개 "target"]
    hold 는 target 프레임 1개 + 같은 포즈를 멈춰서 보여주는 (hold − 1) 개.
    출력:  길이 before + ctx + trans + hold 인 list
    """
    if hold < 1:
        raise ValueError("target_hold_frames 는 1 이상이어야 합니다 (target 프레임 자체).")
    return ["before"] * before + ["context"] * ctx + ["generated"] * trans + ["target"] * hold


def video_frame_index(before: int, ctx: int, trans: int, hold: int) -> np.ndarray:
    """영상 프레임 → 늘린 정답 시퀀스 ([before | 모델 창]) 의 프레임 번호.

    target 유지 구간은 같은 target 프레임 (before + ctx + trans) 을 반복한다.
    출력:  (before + ctx + trans + hold,) int
    """
    n_play = before + ctx + trans + 1                  # 마지막이 target 프레임
    return np.concatenate([np.arange(n_play), np.full(hold - 1, n_play - 1)])


def min_horizontal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """두 스켈레톤 궤적 사이 최소 수평 거리 (cm). a, b : (F, J, 3) 데이터 좌표 (XZ 가 수평).

    같은 프레임끼리 모든 관절 쌍의 XZ 거리 중 가장 작은 값.
    """
    d = a[:, :, None, [0, 2]] - b[:, None, :, [0, 2]]      # (F, J, J, 2)
    return float(np.sqrt((d ** 2).sum(-1)).min())


def choose_layout(gt: np.ndarray, pred: np.ndarray, style: VideoStyle) -> str:
    """"side" 또는 "panels". layout=auto 이면 side 배치에서 두 캐릭터가 overlap_margin 보다 가까워지는지 본다.

    입력:  gt, pred : (F, J, 3) 데이터 좌표 (띄우기 전)
    """
    if style.layout != "auto":
        return style.layout
    shifted = pred.copy()
    shifted[..., 2] += style.side_offset_cm
    return "panels" if min_horizontal_distance(gt, shifted) < style.overlap_margin_cm else "side"


# -----------------------------------------------------------------------------
# 그리기
# -----------------------------------------------------------------------------
def _scene_box(points: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """matplotlib 좌표 점들 → 카메라 범위 (lo (3,), hi (3,), 바닥 높이). 축 단위 비율은 fit_box 가 맞춘다."""
    allp = np.concatenate([p.reshape(-1, 3) for p in points])
    lo, hi = allp.min(0), allp.max(0)
    floor_h = min(0.0, float(lo[2]))
    margin = 0.15 * UNITS_PER_METER
    lo, hi = lo - margin, hi + margin
    lo[2], hi[2] = floor_h, hi[2] + 0.25 * UNITS_PER_METER   # 머리 위 이름표 공간
    return lo, hi, floor_h


def _setup_axis(fig, lo: np.ndarray, hi: np.ndarray, floor_h: float, band) -> "plt.Axes":
    """바닥 평면 + 1 m 격자가 있는 3D 축 하나 (고정 카메라)."""
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_proj_type("ortho")
    ax.computed_zorder = False                                 # 바닥이 뼈를 덮지 않게 직접 순서 지정
    ax.view_init(elev=24, azim=-50)                            # 비스듬히: 진행(+X)·옆(Z) 둘 다 보이게
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    fit_box(fig, ax, lo, hi, band)
    gx, gy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
    ax.plot_surface(gx, gy, np.full_like(gx, floor_h), color="0.94", shade=False, zorder=0)
    step = UNITS_PER_METER
    for v in np.arange(np.ceil(lo[0] / step) * step, hi[0], step):
        ax.plot([v, v], [lo[1], hi[1]], [floor_h, floor_h], color="0.80", lw=0.6, zorder=1)
    for v in np.arange(np.ceil(lo[1] / step) * step, hi[1], step):
        ax.plot([lo[0], hi[0]], [v, v], [floor_h, floor_h], color="0.80", lw=0.6, zorder=1)
    return ax


class _Skeleton3D:
    """3D 축 위의 스켈레톤 선 묶음. 프레임마다 위치와 색을 바꾼다."""

    def __init__(self, ax, parents: Sequence[int], lw: float, alpha: float, zorder: int):
        self.bones = [(j, p) for j, p in enumerate(parents) if p >= 0]
        self.lines = [ax.plot([0, 0], [0, 0], [0, 0], lw=lw, alpha=alpha, solid_capstyle="round",
                              zorder=zorder)[0] for _ in self.bones]

    def update(self, P: np.ndarray, color: str | None = None, visible: bool = True) -> None:
        """P : (J, 3) matplotlib 좌표."""
        for ln, (j, p) in zip(self.lines, self.bones):
            ln.set_data([P[p, 0], P[j, 0]], [P[p, 1], P[j, 1]])
            ln.set_3d_properties([P[p, 2], P[j, 2]])
            if color is not None:
                ln.set_color(color)
            ln.set_visible(visible)


def render_comparison_video(
    path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    segments: Sequence[str],
    target_pose: np.ndarray,
    parents: Sequence[int],
    head: int,
    style: VideoStyle,
    title: str,
    frame_info: Sequence[str],
    error_cm: np.ndarray,
) -> str:
    """GT | PRED 비교 영상 저장 (피드백 [1][2]).

    입력:
        path        : 저장할 mp4 경로
        gt, pred    : (F, J, 3) 영상 프레임마다의 전역 관절 위치 (데이터 좌표, cm). F = len(segments)
        segments    : (F,) 구간 이름 ("before" / "context" / "generated" / "target")
        target_pose : (J, 3) target 프레임 정답 포즈 (반투명 겹치기용)
        parents     : 관절 부모 번호 (J,)
        head        : 머리 관절 번호 (이름표 위치)
        style       : VideoStyle
        title       : 맨 위 제목
        frame_info  : (F,) 프레임마다 오른쪽 위에 쓸 문자열 (예: "frame 052/099  BVH #1234")
        error_cm    : (F,) 프레임별 PRED vs GT 평균 관절 위치 오차 (cm)
    출력:  실제로 쓴 배치 ("side" 또는 "panels")
    """
    F, J, _ = gt.shape
    if pred.shape != gt.shape or len(segments) != F or len(frame_info) != F or error_cm.shape != (F,):
        raise ValueError("gt, pred, segments, frame_info, error_cm 의 프레임 수가 같아야 합니다.")
    layout = choose_layout(gt, pred, style)
    off = style.side_offset_cm if layout == "side" else 0.0

    # 데이터 좌표 → matplotlib 좌표. side 이면 PRED 를 Z (진행 방향 +X 에 수직) 로 띄운다.
    gt_p = to_plot(gt)
    pred_shift = pred.copy()
    pred_shift[..., 2] += off
    pred_p = to_plot(pred_shift)
    ghost_gt = to_plot(target_pose)
    tp = target_pose.copy()
    tp[..., 2] += off
    ghost_pred = to_plot(tp)

    fig = plt.figure(figsize=style.figsize)
    band_top, band_bottom = 0.86, 0.27
    if layout == "side":
        lo, hi, floor_h = _scene_box([gt_p, pred_p, ghost_gt, ghost_pred])
        ax = _setup_axis(fig, lo, hi, floor_h, (0.02, band_bottom, 0.98, band_top))
        axes = {"gt": ax, "pred": ax}
    else:
        # 두 패널 모두 같은 범위 (두 캐릭터의 합집합) → 위치·크기를 직접 비교할 수 있게
        lo, hi, floor_h = _scene_box([gt_p, pred_p, ghost_gt])
        axes = {"gt": _setup_axis(fig, lo, hi, floor_h, (0.01, band_bottom, 0.49, band_top)),
                "pred": _setup_axis(fig, lo, hi, floor_h, (0.51, band_bottom, 0.99, band_top))}
        fig.add_artist(Line2D([0.5, 0.5], [band_bottom, band_top + 0.02], color="0.75", lw=1))

    pts = {"gt": gt_p, "pred": pred_p}
    ghosts = {"gt": ghost_gt, "pred": ghost_pred}
    names = {"gt": "GT", "pred": "PRED"}
    skel = {k: _Skeleton3D(axes[k], parents, lw=4.0, alpha=0.95, zorder=4) for k in pts}
    ghost = {k: _Skeleton3D(axes[k], parents, lw=4.0, alpha=style.ghost_alpha, zorder=3) for k in pts}
    for k in pts:
        ghost[k].update(ghosts[k], style.colors["target"], visible=False)
    tags = {k: axes[k].text(0, 0, 0, names[k], fontsize=13, weight="bold", ha="center", va="bottom", zorder=10)
            for k in pts}
    if layout == "panels":
        for k, x in (("gt", 0.25), ("pred", 0.75)):
            fig.text(x, band_top + 0.005, {"gt": "GT (ground truth)", "pred": "PRED (Context Transformer)"}[k],
                     ha="center", fontsize=12, weight="bold", color="0.25")

    # 위: 제목, 구간 이름, 프레임 번호, 범례
    fig.text(0.5, 0.972, title, ha="center", fontsize=10, color="0.3")
    seg_txt = fig.text(0.02, 0.925, "", fontsize=20, weight="bold")
    info_txt = fig.text(0.98, 0.935, "", fontsize=11, ha="right", family="monospace")
    handles = [Patch(color=style.colors[s], label=SEGMENT_NAMES[s]) for s in SEGMENTS]
    if style.ghost_enabled:
        handles.append(Patch(color=style.colors["target"], alpha=style.ghost_alpha, label="target pose (ghost)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.925), ncol=len(handles),
               fontsize=9, frameon=False, handlelength=1.2, columnspacing=1.0)

    # 아래: 프레임별 위치 오차 + 구간 배경색
    ex = fig.add_axes([0.06, 0.07, 0.91, 0.16], zorder=2)
    seg_arr = np.asarray(segments)
    start = 0
    for f in range(1, F + 1):                                   # 같은 구간이 이어지는 범위마다 배경색
        if f == F or seg_arr[f] != seg_arr[start]:
            ex.axvspan(start - 0.5, f - 0.5, color=style.colors[seg_arr[start]], alpha=0.12, lw=0)
            start = f
    ex.plot(np.arange(F), error_cm, color="black", lw=1.6, label="PRED vs GT mean joint error")
    ex.set_xlim(-0.5, F - 0.5)
    ex.set_ylim(0, max(1.0, float(error_cm.max())) * 1.15)
    ex.set_ylabel("pos. error (cm)")
    ex.set_xlabel("video frame (30 fps, 1x)")
    ex.legend(loc="upper left", fontsize=9, frameon=False)
    ex.grid(alpha=0.3)
    cursor = ex.axvline(0, color="black", lw=1.2)

    writer = FFMpegWriter(fps=style.fps, bitrate=4000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), style.dpi):
        for f in range(F):
            seg = segments[f]
            color = style.colors[seg]
            seg_txt.set_text(SEGMENT_NAMES[seg])
            seg_txt.set_color(color)
            info_txt.set_text(frame_info[f])
            show_ghost = style.ghost_enabled and seg == "generated"
            for k in pts:
                P = pts[k][f]
                skel[k].update(P, color)
                ghost[k].update(ghosts[k], visible=show_ghost)
                h = P[head]
                tags[k].set_position_3d((h[0], h[1], h[2] + 0.15 * UNITS_PER_METER))
                tags[k].set_color(color)
            cursor.set_xdata([f, f])
            writer.grab_frame()
    plt.close(fig)
    return layout
