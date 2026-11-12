"""
[연구실] Context Transformer 결과 영상 (발표용): 원본 vs 생성 vs SLERP
==================================================================

이 스크립트가 하는 일
--------------------
1. <ckpt 폴더>/config.yaml (기본 experiments/ctx_base/config.yaml) 에서 학습 때 쓴 설정을 읽는다.
2. val 분할에서 클립 --n 개를 무작위로 고른다 (--seed). 영상 앞뒤를 채울 원본 프레임이
   같은 BVH 안에 충분히 있는 클립만 고른다.
3. scripts/train_context.py 의 validate() 와 "같은 함수" (src.training.context.eval_transition, gpos_l2)
   로 transition --trans 프레임을 생성한다: 자르기 → 정렬 → 정규화 → 모델 → 후처리 → 역정규화 → FK.
4. 클립마다 검증 값 출력
     - transition 구간 평균 위치 오차 (validate 의 val{trans}_gpos_l2 와 같은 식, 단위 cm)
     - context 구간 원본 vs 생성 최대 오차 (≈ 0 이어야 함. 모델 출력은 transition 에만 들어감)
     - 앞뒤로 늘린 원본이 모델 창의 원본과 같은 좌표인지 (정렬 변환 일치, ≈ 0)
     - --slerp 이면 SLERP 기준선의 같은 오차
5. --total_frames (기본 300 = 10초, 30fps) 길이 mp4 저장.
     [원본 재생] [context (입력)] [transition (생성)] [target] [원본 재생]
   transition 이 영상 가운데 오도록 앞뒤를 같은 BVH 의 원본 프레임으로 채운다.
   앞뒤 프레임에도 모델 창과 같은 정렬 변환 (마지막 context 프레임 기준 루트 XZ=0, +X 방향) 을 적용한다.
   생성/SLERP 스켈레톤은 transition 밖에서는 원본과 같은 포즈.
   --layout side   : 원본(파랑) 가운데, 생성(빨강) +offset, SLERP(회색) −offset (진행 방향 +X 에 수직인 Z 축)
   --layout overlay: 세 개를 같은 자리에 겹쳐 그림
   아래 패널: 프레임별 위치 오차 (생성/SLERP vs 원본), transition 배경 칠하기, 현재 프레임 커서.

필요한 것: matplotlib, ffmpeg (sudo apt install ffmpeg). 한글 글꼴이 없으면 영어 라벨로 대신한다
(sudo apt install fonts-nanum 또는 fonts-noto-cjk).

실행 예:
    python scripts/render_compare.py --slerp                          # ctx_base best.pth, trans 30, 3개, 10초
    python scripts/render_compare.py --trans 15 --n 5 --seed 1 --slerp --layout overlay
    python scripts/render_compare.py --ckpt experiments/ctx_base/last.pth --total_frames 450 --offset 2.0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter, writers  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from src.config import load_env_config, load_yaml, require_dir, resolve_device, resolve_path  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer, decode_pose, encode_pose  # noqa: E402
from src.data.rotations import matrix_to_quat, quat_to_matrix  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402
from src.models.context_transformer import build_context_transformer  # noqa: E402
from src.training.context import (  # noqa: E402
    PreparedBatch,
    eval_transition,
    global_positions,
    gpos_l2,
    prepare_batch,
)
from src.training.context_inputs import TransitionSpec  # noqa: E402
from src.training.utils import load_checkpoint  # noqa: E402
# 그리기 도우미는 4단계 시각화 모듈과 공유 (scripts/render_samples.py 도 사용)
from src.visualization.skeleton_video import fit_box, pick_korean_font, to_plot  # noqa: E402
from src.evaluation.baselines import slerp  # noqa: E402  (4단계 기준선과 같은 SLERP)

FPS = 30
UNITS_PER_METER = 100.0                     # LaFAN1 BVH 는 cm
COLORS = {"gt": "tab:blue", "pred": "tab:red", "slerp": "0.5"}
TRACK_ORDER = ("slerp", "gt", "pred")       # 나중에 그린 것이 위


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/ctx_base/best.pth"))
    ap.add_argument("--config", type=Path, default=None,
                    help="학습 때 저장된 config.yaml (기본: <ckpt 폴더>/config.yaml)")
    ap.add_argument("--trans", type=int, default=30,
                    help="transition 길이 (val window 65 안: context + trans + 2 ≤ 65)")
    ap.add_argument("--n", type=int, default=3, help="만들 영상 수")
    ap.add_argument("--seed", type=int, default=0, help="val 클립 고르기 시드")
    ap.add_argument("--out", type=Path, default=Path("experiments/ctx_base/videos"))
    ap.add_argument("--slerp", action="store_true", help="SLERP 기준선(회색)도 함께 그림")
    ap.add_argument("--total_frames", "--total-frames", dest="total_frames", type=int, default=300,
                    help="영상 전체 프레임 수 (30fps; 300 = 10초)")
    ap.add_argument("--layout", choices=["side", "overlay"], default="side",
                    help="side: 나란히 (원본 가운데), overlay: 겹쳐 그림")
    ap.add_argument("--offset", type=float, default=1.5, help="side 배치 간격 (m)")
    ap.add_argument("--env", default=None, help="configs/env/<env>.yaml (기본: config.yaml 의 env)")
    ap.add_argument("--device", default=None, help="예) cpu, cuda:0 (기본: env 설정)")
    ap.add_argument("--dpi", type=int, default=100)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# SLERP 기준선 (회전: 쿼터니언 구면보간, 루트 위치: 선형보간)
# -----------------------------------------------------------------------------
def slerp_pose(pb: PreparedBatch, spec: TransitionSpec, layout: FeatureLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """마지막 context 프레임과 target 사이를 SLERP/선형보간 → (루트 위치 (B,L,3), 로컬 회전 (B,L,J,3,3)), 정렬된 좌표.

    정답(정렬된 원값) 에서 시작한다. transition 밖은 정답 그대로.
    """
    root, rot = decode_pose(pb.x_gt_raw, layout)               # (B,L,3), (B,L,J,3,3)
    a, b = spec.context_len - 1, spec.target_idx
    t = torch.arange(spec.context_len, b, device=root.device, dtype=root.dtype)
    w = ((t - a) / (b - a))[None, :, None]                     # (1, n, 1)
    root_i = root[:, a:a + 1] + w * (root[:, b:b + 1] - root[:, a:a + 1])
    q0, q1 = matrix_to_quat(rot[:, a:a + 1]), matrix_to_quat(rot[:, b:b + 1])   # (B,1,J,4)
    rot_i = quat_to_matrix(slerp(q0, q1, w[..., None]))       # (B,n,J,3,3)
    s = spec.seq_slice
    root = torch.cat([root[:, :s.start], root_i, root[:, s.stop:]], dim=1)
    rot = torch.cat([rot[:, :s.start], rot_i, rot[:, s.stop:]], dim=1)
    return root, rot


def slerp_baseline(pb: PreparedBatch, spec: TransitionSpec, skel: Skeleton, layout: FeatureLayout) -> torch.Tensor:
    """SLERP 기준선 → 전역 관절 위치 (B, L, J, 3). 같은 FK (global_positions) 사용."""
    root, rot = slerp_pose(pb, spec, layout)
    return global_positions(encode_pose(root, rot), pb.offsets, skel, layout)


# -----------------------------------------------------------------------------
# 앞뒤로 늘린 원본 (같은 정렬 변환)
# -----------------------------------------------------------------------------
def padding_frames(total: int, ctx: int, trans: int) -> tuple[int, int]:
    """(pre, post): 모델 창 [context | transition | target] 앞/뒤에 붙일 원본 프레임 수.

    transition 의 가운데가 영상 가운데 (total // 2) 에 오도록 pre 를 정한다.
    """
    shown = ctx + trans + 1
    pre = total // 2 - ctx - trans // 2
    post = total - pre - shown
    if pre < 0 or post < 1:                    # post ≥ 1: 후처리에 쓰는 target 다음 프레임
        raise ValueError(f"--total_frames {total} 가 너무 짧습니다 (context {ctx} + trans {trans} + 2 이상, "
                         f"가운데 정렬 위해 {2 * (ctx + trans // 2) + 2} 이상 권장).")
    return pre, post


def extended_batch(ds: LafanDataset, clips: list[tuple[int, int]], pre: int, total: int) -> dict:
    """clips [(파일 번호, 클립 시작)] → 시작 pre 프레임 전부터 total 프레임 (LafanDataset 배치와 같은 key)."""
    items = []
    for fi, start in clips:
        sl = slice(start - pre, start - pre + total)
        items.append({"root_pos": ds.root_pos[fi][sl], "local_rot": ds.local_rot[fi][sl],
                      "offsets": ds.offsets[fi]})
    return default_collate(items)


def splice(full: torch.Tensor, window: torch.Tensor, pre: int, frames: slice) -> torch.Tensor:
    """full (B, total, J, 3) 원본에 window (B, L, J, 3) 의 frames 구간만 덮어쓴다."""
    out = full.clone()
    out[:, pre + frames.start:pre + frames.stop] = window[:, frames]
    return out


# -----------------------------------------------------------------------------
# 그리기
# -----------------------------------------------------------------------------
def render_video(path: Path, tracks: dict[str, np.ndarray], errors: dict[str, np.ndarray], parents,
                 head: int, phases: list[tuple[str, int, int]], trans_range: tuple[int, int],
                 title: str, korean: bool, layout: str, offset: float, dpi: int) -> None:
    """tracks: {"gt"|"pred"|"slerp": (F, J, 3) 데이터 좌표, 같은 정렬}. errors: {"pred"|"slerp": (F,) cm}.
    phases: [(이름, 시작, 끝(포함 안 함))], trans_range: transition [시작, 끝).
    """
    names = ({"gt": "원본", "pred": "생성", "slerp": "SLERP"} if korean
             else {"gt": "original", "pred": "generated", "slerp": "SLERP"})
    long_names = ({"gt": "원본", "pred": "생성 (Context Transformer)", "slerp": "SLERP 기준선"} if korean
                  else {"gt": "original", "pred": "generated (Context Transformer)", "slerp": "SLERP baseline"})
    n_frames = next(iter(tracks.values())).shape[0]
    shift = {"gt": 0.0, "pred": offset, "slerp": -offset} if layout == "side" else {}
    pts = {}
    for k, v in tracks.items():
        v = v.copy()
        v[..., 2] += shift.get(k, 0.0)                         # 진행 방향(+X) 에 수직인 Z 축으로 띄움
        pts[k] = to_plot(v)
    order = [k for k in TRACK_ORDER if k in pts]

    # 고정 카메라 범위: 모든 스켈레톤·모든 프레임이 들어가게, 축 단위 비율 같게
    allp = np.concatenate([p.reshape(-1, 3) for p in pts.values()])
    lo, hi = allp.min(0), allp.max(0)
    floor_h = min(0.0, float(lo[2]))
    margin = 0.15 * UNITS_PER_METER
    lo, hi = lo - margin, hi + margin
    lo[2], hi[2] = floor_h, hi[2] + 0.2 * UNITS_PER_METER      # 이름표 공간
    span = hi - lo

    fig = plt.figure(figsize=(12.8, 9.0))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_proj_type("ortho")
    ax.computed_zorder = False                                 # 바닥이 뼈를 덮지 않게 직접 순서 지정
    ax.view_init(elev=24, azim=-50)                            # 비스듬히: 진행(+X)·옆(Z) 둘 다 보이게
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    fit_box(fig, ax, lo, hi, band=(0.02, 0.27, 0.98, 0.90))

    # 바닥 평면 + 1 m 격자 (한 번만)
    gx, gy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
    ax.plot_surface(gx, gy, np.full_like(gx, floor_h), color="0.92", shade=False, zorder=0)
    step = UNITS_PER_METER
    for v in np.arange(np.ceil(lo[0] / step) * step, hi[0], step):
        ax.plot([v, v], [lo[1], hi[1]], [floor_h, floor_h], color="0.78", lw=0.6, zorder=1)
    for v in np.arange(np.ceil(lo[1] / step) * step, hi[1], step):
        ax.plot([lo[0], hi[0]], [v, v], [floor_h, floor_h], color="0.78", lw=0.6, zorder=1)

    # 뼈 선 / 이름표: 처음에 한 번만 만들고 매 프레임 갱신
    bones = [(j, p) for j, p in enumerate(parents) if p >= 0]
    lines = {k: [ax.plot([0, 0], [0, 0], [0, 0], color=COLORS[k], lw=4.0 if k != "slerp" else 3.5,
                         alpha=0.95 if k != "pred" or layout == "side" else 0.8,
                         solid_capstyle="round", zorder=3 + order.index(k))[0] for _ in bones]
             for k in order}
    tags = {}
    if layout == "side":
        for k in order:
            tags[k] = ax.text(0, 0, 0, names[k], color=COLORS[k], fontsize=13, weight="bold",
                              ha="center", va="bottom", zorder=10)
    else:
        fig.legend(handles=[Line2D([], [], color=COLORS[k], lw=3, label=long_names[k])   # 3D 축은 그림보다 커서 fig 기준
                            for k in ("gt", "pred", "slerp") if k in pts],
                   loc="upper right", bbox_to_anchor=(0.97, 0.90), fontsize=12, frameon=False)

    fig.text(0.5, 0.965, title, ha="center", fontsize=10, color="0.3")
    phase_txt = fig.text(0.06, 0.925, "", fontsize=18, weight="bold")
    frame_txt = fig.text(0.97, 0.925, "", fontsize=12, ha="right", family="monospace")

    # 아래 패널: 프레임별 위치 오차
    ex = fig.add_axes([0.06, 0.07, 0.91, 0.17], zorder=2)
    t0, t1 = trans_range
    ex.axvspan(t0 - 0.5, t1 - 0.5, color=COLORS["pred"], alpha=0.08, lw=0)
    for _, a, _b in phases[1:]:
        ex.axvline(a - 0.5, color="0.8", lw=0.8, ls=":")
    ymax = 1.0
    for k in ("slerp", "pred"):
        if k in errors:
            ex.plot(np.arange(n_frames), errors[k], color=COLORS[k], lw=2, label=long_names[k])
            ymax = max(ymax, float(errors[k].max()))
    ex.set_xlim(0, n_frames - 1)
    ex.set_ylim(0, ymax * 1.15)
    ex.set_ylabel("위치 오차 (cm)" if korean else "pos. error (cm)")
    ex.set_xlabel("frame")
    ex.legend(loc="upper right", fontsize=9, frameon=False)
    ex.grid(alpha=0.3)
    cursor = ex.axvline(0, color="black", lw=1.2)

    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    with writer.saving(fig, str(path), dpi):
        for f in range(n_frames):
            ph = next(name for name, a, b in phases if a <= f < b)
            phase_txt.set_text(ph)
            phase_txt.set_color(COLORS["pred"] if t0 <= f < t1 else "black")
            frame_txt.set_text(f"frame {f:3d} / {n_frames - 1}   {f / FPS:5.2f} s")
            for k in order:
                P = pts[k][f]
                for ln, (j, p) in zip(lines[k], bones):
                    ln.set_data([P[p, 0], P[j, 0]], [P[p, 1], P[j, 1]])
                    ln.set_3d_properties([P[p, 2], P[j, 2]])
                if k in tags:
                    h = P[head]
                    tags[k].set_position_3d((h[0], h[1], h[2] + 0.18 * UNITS_PER_METER))
            cursor.set_xdata([f, f])
            writer.grab_frame()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 공용: 설정·모델 읽기 → 클립 고르기 → 생성  (scripts/export_bvh.py 도 사용)
# -----------------------------------------------------------------------------
@dataclass
class Run:
    """학습 결과 하나 (설정, 모델, val 데이터)."""

    cfg: dict
    data_cfg: dict
    device: torch.device
    skel: Skeleton
    layout: FeatureLayout
    norm: ZScoreNormalizer
    model: torch.nn.Module
    ckpt: dict
    ckpt_path: Path
    val_ds: LafanDataset
    split: str

    @property
    def ctx(self) -> int:
        return self.data_cfg["context_len"]


def load_run(ckpt: Path, config: Path | None = None, env_name: str | None = None,
             device: str | None = None) -> Run:
    """<ckpt 폴더>/config.yaml (학습 때 저장된 그대로) → 모델, 정규화, val 데이터셋."""
    ckpt_path = resolve_path(ckpt)
    cfg_path = resolve_path(config) if config else ckpt_path.parent / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 {ckpt_path} 가 없습니다.")
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일 {cfg_path} 가 없습니다 (train_context.py 가 run 폴더에 저장).")
    cfg = load_yaml(cfg_path)
    data_cfg = cfg["data"]
    if cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 결과입니다.")
    env = load_env_config(env_name or cfg["env"])
    dev = torch.device(device) if device else resolve_device(env)

    skel = Skeleton.from_yaml(resolve_path(data_cfg["skeleton"]))
    layout = FeatureLayout(skel.num_joints)
    norm = ZScoreNormalizer.load(resolve_path(cfg["stats"])).to(device=dev, dtype=torch.float32)
    split = cfg["train"].get("val_split", "val")
    sp = data_cfg["splits"][split]
    val_ds = LafanDataset(require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더"), sp["actors"],
                          sp["window"], sp["offset"], skel, start_frame=sp["start_frame"],
                          contact_threshold=data_cfg["contact_vel_threshold"],
                          cache_dir=env["paths"]["cache_dir"], dtype=torch.float32)
    ck = load_checkpoint(ckpt_path, map_location=dev)
    model = build_context_transformer(cfg["model"], layout).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"체크포인트 {ckpt_path} (iteration {ck['iteration']}, best_val {ck['best_val']:.4f}), "
          f"device={dev}, {split} 클립 {len(val_ds)}")
    return Run(cfg, data_cfg, dev, skel, layout, norm, model, ck, ckpt_path, val_ds, split)


def select_clips(run: Run, trans: int, total: int, n: int, seed: int) -> tuple[int, int, list[int]]:
    """앞뒤 원본 여유가 있는 val 클립 n 개를 고른다. 출력 (pre, post, 클립 번호 목록)."""
    ds, ctx = run.val_ds, run.ctx
    window = run.data_cfg["splits"][run.split]["window"]
    if ctx + trans + 2 > window:
        raise ValueError(f"trans {trans}: {run.split} window {window} < context {ctx} + trans + 2")
    pre, post = padding_frames(total, ctx, trans)
    ok = []
    for i in range(len(ds)):                     # 앞 pre / 뒤 (total - pre) 프레임이 같은 BVH 안에 있는 클립만
        fi, start = ds.locate(i)
        if start - pre >= 0 and start - pre + total <= ds.frames[fi]:
            ok.append(i)
    if not ok:
        raise ValueError(f"앞뒤 여유 (앞 {pre}, 뒤 {total - pre} 프레임) 가 있는 {run.split} 클립이 없습니다. "
                         f"--total_frames 를 줄이세요.")
    idx = sorted(random.Random(seed).sample(ok, min(n, len(ok))))
    print(f"영상 {total} 프레임 ({total / FPS:.1f}초) = 원본 {pre} + context {ctx} + transition {trans} + "
          f"target 1 + 원본 {post}  (앞뒤 여유 있는 클립 {len(ok)}/{len(ds)} 중 {len(idx)}개)")
    return pre, post, idx


@dataclass
class Inference:
    """클립 묶음 하나의 생성 결과. 좌표는 모두 학습용 정렬 좌표.

    pb, spec, y, gpos : eval_transition 결과 (모델 창, 길이 L = ctx + trans + 2)
    gpos_slerp        : SLERP 기준선 전역 위치 (모델 창) 또는 None
    ext_batch         : 앞뒤로 늘린 원본 (월드 좌표, LafanDataset 배치 형식, total 프레임)
    pb_full           : ext_batch 를 같은 기준 프레임으로 정렬한 PreparedBatch
    """

    idx: list[int]
    clips: list[tuple[int, int]]
    pre: int
    pb: PreparedBatch
    spec: TransitionSpec
    y: torch.Tensor
    gpos: torch.Tensor
    gpos_slerp: torch.Tensor | None
    ext_batch: dict
    pb_full: PreparedBatch


def infer_clips(run: Run, idx: list[int], trans: int, pre: int, total: int, with_slerp: bool) -> Inference:
    """validate() 와 같은 함수 (eval_transition) 로 생성 + 앞뒤로 늘린 원본을 같은 정렬로."""
    ds, ctx = run.val_ds, run.ctx
    clips = [ds.locate(i) for i in idx]
    batch = default_collate([ds[i] for i in idx])
    pb, spec, y, gpos = eval_transition(run.model, batch, trans, run.data_cfg, run.skel, run.layout,
                                        run.norm, run.device)
    gpos_slerp = slerp_baseline(pb, spec, run.skel, run.layout) if with_slerp else None
    # 정렬 기준 프레임 = 늘린 시퀀스의 pre + ctx − 1 = 모델 창의 ctx − 1
    ext = extended_batch(ds, clips, pre, total)
    pb_full = prepare_batch(ext, run.skel, run.norm, pre + ctx, run.data_cfg["forward_world_axis"], run.device)
    return Inference(idx, clips, pre, pb, spec, y, gpos, gpos_slerp, ext, pb_full)


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if not writers.is_available("ffmpeg"):
        raise FileNotFoundError("ffmpeg 가 없습니다. sudo apt install ffmpeg")
    run = load_run(args.ckpt, args.config, args.env, args.device)
    skel, val_ds, ckpt, ckpt_path = run.skel, run.val_ds, run.ckpt, run.ckpt_path
    head = skel.names.index("Head") if "Head" in skel.names else skel.num_joints - 1
    ctx, trans, total = run.ctx, args.trans, args.total_frames
    pre, post, idx = select_clips(run, trans, total, args.n, args.seed)
    inf = infer_clips(run, idx, trans, pre, total, args.slerp)
    clips, pb, spec, gpos, gpos_slerp = inf.clips, inf.pb, inf.spec, inf.gpos, inf.gpos_slerp

    gt_full = inf.pb_full.gpos_gt                                # (B, total, J, 3)
    L = pb.gpos_gt.shape[1]
    align_err = float((gt_full[:, pre:pre + L] - pb.gpos_gt).norm(dim=-1).max())
    pred_full = splice(gt_full, gpos, pre, spec.seq_slice)      # transition 밖은 원본 포즈
    slerp_full = splice(gt_full, gpos_slerp, pre, spec.seq_slice) if gpos_slerp is not None else None

    korean = pick_korean_font()
    if not korean:
        print("[알림] 한글 글꼴이 없어 영상 라벨을 영어로 씁니다 (sudo apt install fonts-nanum).")
    ph = (("원본 재생", "context (입력)", "transition (생성)", "target", "원본 재생") if korean
          else ("original", "context (input)", "transition (generated)", "target", "original"))
    t0, t1 = pre + ctx, pre + spec.target_idx
    phases = [(ph[0], 0, pre), (ph[1], pre, t0), (ph[2], t0, t1), (ph[3], t1, t1 + 1), (ph[4], t1 + 1, total)]
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_frames = slice(0, spec.context_len)

    l2_all, l2_slerp_all = [], []
    for b, (fi, start) in enumerate(clips):
        name = val_ds.files[fi].stem
        frame0 = start + val_ds.start_frame                      # 모델 창 첫 프레임의 원본 BVH 번호
        one = slice(b, b + 1)
        gt, pred = pb.gpos_gt[one], gpos[one]
        l2 = float(gpos_l2(pred, gt, spec.seq_slice))
        ctx_err = float((pred[:, ctx_frames] - gt[:, ctx_frames]).norm(dim=-1).max())
        tgt_err = float((pred[:, spec.target_idx] - gt[:, spec.target_idx]).norm(dim=-1).max())
        a_err = float((gt_full[one, pre:pre + L] - gt).norm(dim=-1).max())
        l2_all.append(l2)
        print(f"\n[{b + 1}/{len(clips)}] {name} 모델 창 프레임 {frame0}–{frame0 + spec.target_idx}, "
              f"영상 프레임 {frame0 - pre}–{frame0 - pre + total - 1} (val 클립 #{idx[b]})")
        print(f"  transition 평균 위치 오차 (gpos_l2): {l2:.4f}")
        print(f"  context 구간 원본 vs 생성 최대 오차: {ctx_err:.2e} → "
              f"{'OK (완전히 같음)' if ctx_err < 1e-4 else '경고: context 가 원본과 다름!'}"
              f"   (target 프레임 최대 오차 {tgt_err:.2e})")
        print(f"  앞뒤 원본 정렬 일치 (늘린 원본 vs 모델 창 원본) 최대 오차: {a_err:.2e} → "
              f"{'OK' if a_err < 1e-3 else '경고: 정렬 변환이 다름!'}")

        tracks = {"gt": gt_full[b].cpu().numpy(), "pred": pred_full[b].cpu().numpy()}
        errors = {"pred": (pred_full[b] - gt_full[b]).norm(dim=-1).mean(-1).cpu().numpy()}
        title = f"{name}  frame {frame0 - pre}–{frame0 - pre + total - 1}  |  trans {trans}  |  gpos_l2 {l2:.2f} cm"
        if slerp_full is not None:
            l2s = float(gpos_l2(gpos_slerp[one], gt, spec.seq_slice))
            l2_slerp_all.append(l2s)
            print(f"  SLERP 기준선 gpos_l2: {l2s:.4f}")
            tracks["slerp"] = slerp_full[b].cpu().numpy()
            errors["slerp"] = (slerp_full[b] - gt_full[b]).norm(dim=-1).mean(-1).cpu().numpy()
            title += f" (SLERP {l2s:.2f})"
        title += f"  |  {ckpt_path.parent.name}/{ckpt_path.name} it {ckpt['iteration']}"

        path = out_dir / (f"{b:02d}_{name}_f{frame0}_trans{trans}_{args.layout}"
                          f"{'_slerp' if args.slerp else ''}.mp4")
        tic = time.time()
        render_video(path, tracks, errors, skel.parents, head, phases, (t0, t1), title, korean,
                     args.layout, args.offset * UNITS_PER_METER, args.dpi)
        print(f"  저장: {path}  ({time.time() - tic:.0f}초)")

    print(f"\n{len(clips)}개 평균 gpos_l2 (trans {trans}): {np.mean(l2_all):.4f}"
          + (f", SLERP {np.mean(l2_slerp_all):.4f}" if l2_slerp_all else "")
          + "  ※ 클립 몇 개의 평균이라 log.csv 의 val 전체 평균과는 다를 수 있음")
    if align_err >= 1e-3:
        print(f"[경고] 앞뒤 원본 정렬 최대 오차 {align_err:.2e}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
