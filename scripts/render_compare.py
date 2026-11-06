"""
[연구실] Context Transformer 결과 영상: 원본 vs 생성 스켈레톤 겹쳐 그리기
=====================================================================

이 스크립트가 하는 일
--------------------
1. <ckpt 폴더>/config.yaml (기본 experiments/ctx_base/config.yaml) 에서 학습 때 쓴 설정을 읽는다.
2. val 분할에서 클립 --n 개를 무작위로 고른다 (--seed).
3. scripts/train_context.py 의 validate() 와 "같은 함수" (src.training.context.eval_transition, gpos_l2)
   로 transition --trans 프레임을 생성한다: 자르기 → 정렬 → 정규화 → 모델 → 후처리 → 역정규화 → FK.
4. 클립마다 검증 값 출력
     - transition 구간 평균 위치 오차 (validate 의 val{trans}_gpos_l2 와 같은 식, 단위는 데이터 단위 cm)
     - context 구간 원본 vs 생성 최대 오차 (≈ 0 이어야 함. 모델 출력은 transition 에만 들어감)
     - --slerp 이면 SLERP 기준선의 같은 오차
5. 원본(파랑)·생성(빨강)·SLERP(회색, --slerp) 를 한 3D 화면에 겹친 30fps mp4 저장.
   화면 위에 구간 이름 (context (입력) / transition (생성) / target) 과 프레임 번호.

좌표는 학습과 같은 "정렬된" 좌표 (마지막 context 프레임 기준으로 루트가 +X 를 보게 회전·이동, Y 가 위).

필요한 것: matplotlib, ffmpeg (sudo apt install ffmpeg). 한글 글꼴이 없으면 영어 라벨로 대신한다
(sudo apt install fonts-nanum 또는 fonts-noto-cjk).

실행 예:
    python scripts/render_compare.py                                  # ctx_base best.pth, trans 30, 3개
    python scripts/render_compare.py --trans 15 --n 5 --seed 1 --slerp
    python scripts/render_compare.py --ckpt experiments/ctx_base/last.pth --out /tmp/videos
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter, writers  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from src.config import load_env_config, load_yaml, require_dir, resolve_device, resolve_path  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer, decode_pose, encode_pose  # noqa: E402
from src.data.rotations import matrix_to_quat, quat_to_matrix  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402
from src.models.context_transformer import build_context_transformer  # noqa: E402
from src.training.context import PreparedBatch, eval_transition, global_positions, gpos_l2  # noqa: E402
from src.training.context_inputs import TransitionSpec  # noqa: E402
from src.training.utils import load_checkpoint  # noqa: E402

FPS = 30
COLORS = {"gt": "tab:blue", "pred": "tab:red", "slerp": "0.55"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/ctx_base/best.pth"))
    ap.add_argument("--config", type=Path, default=None,
                    help="학습 때 저장된 config.yaml (기본: <ckpt 폴더>/config.yaml)")
    ap.add_argument("--trans", type=int, default=30, choices=[5, 15, 30], help="transition 길이")
    ap.add_argument("--n", type=int, default=3, help="만들 영상 수")
    ap.add_argument("--seed", type=int, default=0, help="val 클립 고르기 시드")
    ap.add_argument("--out", type=Path, default=Path("experiments/ctx_base/videos"))
    ap.add_argument("--slerp", action="store_true", help="SLERP 기준선(회색)도 함께 그림")
    ap.add_argument("--env", default=None, help="configs/env/<env>.yaml (기본: config.yaml 의 env)")
    ap.add_argument("--device", default=None, help="예) cpu, cuda:0 (기본: env 설정)")
    ap.add_argument("--dpi", type=int, default=100)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# SLERP 기준선 (회전: 쿼터니언 구면보간, 루트 위치: 선형보간)
# -----------------------------------------------------------------------------
def slerp(q0: torch.Tensor, q1: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """q0, q1 : (..., 4) 단위 쿼터니언 (w,x,y,z), w : q0 과 브로드캐스트 가능한 (..., 1) 가중치."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)                         # 짧은 쪽 호
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    near = s < eps                                             # 거의 같은 회전 → 선형보간
    a = torch.where(near, 1 - w, torch.sin((1 - w) * theta) / s.clamp_min(eps))
    b = torch.where(near, w, torch.sin(w * theta) / s.clamp_min(eps))
    q = a * q0 + b * q1
    return q / q.norm(dim=-1, keepdim=True)


def slerp_baseline(pb: PreparedBatch, spec: TransitionSpec, skel: Skeleton, layout: FeatureLayout) -> torch.Tensor:
    """마지막 context 프레임과 target 사이를 SLERP/선형보간 → 전역 관절 위치 (B, L, J, 3).

    정답(정렬된 원값) 에서 시작해 같은 FK (global_positions) 로 위치를 구한다.
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
    return global_positions(encode_pose(root, rot), pb.offsets, skel, layout)


# -----------------------------------------------------------------------------
# 그리기
# -----------------------------------------------------------------------------
def pick_korean_font() -> bool:
    """한글 글꼴이 있으면 설정하고 True. 없으면 False (영어 라벨 사용)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Malgun Gothic", "AppleGothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def to_plot(p: np.ndarray) -> np.ndarray:
    """데이터 좌표 (Y 위, 오른손) → matplotlib 3D (Z 위, 오른손): (x, y, z) → (x, -z, y)."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def render_video(path: Path, tracks: dict[str, np.ndarray], parents, spec: TransitionSpec,
                 title: str, korean: bool, dpi: int) -> None:
    """tracks: {"gt"|"pred"|"slerp": (F, J, 3) 데이터 좌표}. 프레임 0..target 을 그린다."""
    labels = ({"gt": "원본", "pred": "생성 (Context Transformer)", "slerp": "SLERP"} if korean
              else {"gt": "original", "pred": "generated (Context Transformer)", "slerp": "SLERP"})
    phase_names = (("context (입력)", "transition (생성)", "target") if korean
                   else ("context (input)", "transition (generated)", "target"))
    n_frames = spec.target_idx + 1
    pts = {k: to_plot(v[:n_frames]) for k, v in tracks.items()}

    # 고정 카메라 범위: 모든 스켈레톤·모든 프레임이 들어가게, 축 비율 1:1:1
    allp = np.concatenate([p.reshape(-1, 3) for p in pts.values()])
    lo, hi = allp.min(0), allp.max(0)
    floor_h = min(0.0, float(lo[2]))
    lo[2] = floor_h
    center, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1
    lo, hi = center - half, center + half
    lo[2], hi[2] = floor_h, floor_h + 2 * half

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-60)
    ax.set_xlabel("X")
    ax.set_ylabel("-Z")
    ax.set_zlabel("Y (up)")

    # 바닥 평면 (한 번만)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 2), np.linspace(lo[1], hi[1], 2))
    ax.plot_surface(gx, gy, np.full_like(gx, floor_h), color="0.85", alpha=0.5, shade=False, zorder=0)
    for v in np.linspace(lo[0], hi[0], 11):
        ax.plot([v, v], [lo[1], hi[1]], [floor_h, floor_h], color="0.7", lw=0.5)
    for v in np.linspace(lo[1], hi[1], 11):
        ax.plot([lo[0], hi[0]], [v, v], [floor_h, floor_h], color="0.7", lw=0.5)

    # 뼈 선: 처음에 한 번만 만들고 매 프레임 set_data / set_3d_properties 로 갱신
    bones = [(j, p) for j, p in enumerate(parents) if p >= 0]
    order = [k for k in ("slerp", "gt", "pred") if k in pts]    # 나중에 그린 것이 위
    lines = {k: [ax.plot([0, 0], [0, 0], [0, 0], color=COLORS[k], lw=2.5 if k != "slerp" else 2.0,
                         alpha=0.9 if k != "pred" else 0.85, solid_capstyle="round")[0] for _ in bones]
             for k in order}
    ax.legend(handles=[Line2D([], [], color=COLORS[k], lw=3, label=labels[k]) for k in ("gt", "pred", "slerp")
                       if k in pts], loc="upper right")

    fig.suptitle(title, fontsize=10)
    phase_txt = fig.text(0.05, 0.93, "", fontsize=15, weight="bold")
    frame_txt = fig.text(0.05, 0.895, "", fontsize=11)

    writer = FFMpegWriter(fps=FPS, bitrate=2400)
    with writer.saving(fig, str(path), dpi):
        for f in range(n_frames):
            if f < spec.context_len:
                ph, color = phase_names[0], "black"
            elif f < spec.target_idx:
                ph, color = phase_names[1], COLORS["pred"]
            else:
                ph, color = phase_names[2], "black"
            phase_txt.set_text(ph)
            phase_txt.set_color(color)
            frame_txt.set_text(f"frame {f:3d} / {n_frames - 1}   "
                               f"(context 0–{spec.context_len - 1}, transition {spec.context_len}–"
                               f"{spec.target_idx - 1}, target {spec.target_idx})")
            for k in order:
                P = pts[k][f]
                for ln, (j, p) in zip(lines[k], bones):
                    ln.set_data([P[p, 0], P[j, 0]], [P[p, 1], P[j, 1]])
                    ln.set_3d_properties([P[p, 2], P[j, 2]])
            writer.grab_frame()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt)
    cfg_path = resolve_path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 {ckpt_path} 가 없습니다.")
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일 {cfg_path} 가 없습니다 (train_context.py 가 run 폴더에 저장).")
    if not writers.is_available("ffmpeg"):
        raise FileNotFoundError("ffmpeg 가 없습니다. sudo apt install ffmpeg")

    # ----- 설정 (학습 때 저장된 그대로) -----
    cfg = load_yaml(cfg_path)
    data_cfg = cfg["data"]
    if cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 결과입니다.")
    env = load_env_config(args.env or cfg["env"])
    device = torch.device(args.device) if args.device else resolve_device(env)

    skel = Skeleton.from_yaml(resolve_path(data_cfg["skeleton"]))
    layout = FeatureLayout(skel.num_joints)
    norm = ZScoreNormalizer.load(resolve_path(cfg["stats"])).to(device=device, dtype=torch.float32)

    split = cfg["train"].get("val_split", "val")
    sp = data_cfg["splits"][split]
    ctx = data_cfg["context_len"]
    if ctx + args.trans + 2 > sp["window"]:
        raise ValueError(f"trans {args.trans}: {split} window {sp['window']} < context {ctx} + trans + 2")
    val_ds = LafanDataset(require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더"), sp["actors"],
                          sp["window"], sp["offset"], skel, start_frame=sp["start_frame"],
                          contact_threshold=data_cfg["contact_vel_threshold"],
                          cache_dir=env["paths"]["cache_dir"], dtype=torch.float32)

    # ----- 모델 -----
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = build_context_transformer(cfg["model"], layout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"체크포인트 {ckpt_path} (iteration {ckpt['iteration']}, best_val {ckpt['best_val']:.4f}), "
          f"device={device}, {split} 클립 {len(val_ds)}, trans {args.trans}")

    # ----- 클립 고르기 → 생성 (validate() 와 같은 함수) -----
    idx = sorted(random.Random(args.seed).sample(range(len(val_ds)), min(args.n, len(val_ds))))
    batch = default_collate([val_ds[i] for i in idx])
    pb, spec, _, gpos = eval_transition(model, batch, args.trans, data_cfg, skel, layout, norm, device)
    gpos_slerp = slerp_baseline(pb, spec, skel, layout) if args.slerp else None
    ctx_frames = slice(0, spec.context_len)

    korean = pick_korean_font()
    if not korean:
        print("[알림] 한글 글꼴이 없어 영상 라벨을 영어로 씁니다 (sudo apt install fonts-nanum).")
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    l2_all, l2_slerp_all = [], []
    for b, i in enumerate(idx):
        fi, start = val_ds.locate(i)
        name = val_ds.files[fi].stem
        frame0 = start + val_ds.start_frame                      # 원본 BVH 프레임 번호
        one = slice(b, b + 1)
        gt, pred = pb.gpos_gt[one], gpos[one]
        l2 = float(gpos_l2(pred, gt, spec.seq_slice))
        ctx_err = float((pred[:, ctx_frames] - gt[:, ctx_frames]).norm(dim=-1).max())
        tgt_err = float((pred[:, spec.target_idx] - gt[:, spec.target_idx]).norm(dim=-1).max())
        l2_all.append(l2)
        print(f"\n[{b + 1}/{len(idx)}] {name} 프레임 {frame0}–{frame0 + spec.target_idx} (val 클립 #{i})")
        print(f"  transition 평균 위치 오차 (gpos_l2): {l2:.4f}")
        print(f"  context 구간 원본 vs 생성 최대 오차: {ctx_err:.2e} → "
              f"{'OK (완전히 같음)' if ctx_err < 1e-4 else '경고: context 가 원본과 다름!'}"
              f"   (target 프레임 최대 오차 {tgt_err:.2e})")
        tracks = {"gt": gt[0].cpu().numpy(), "pred": pred[0].cpu().numpy()}
        if gpos_slerp is not None:
            l2s = float(gpos_l2(gpos_slerp[one], gt, spec.seq_slice))
            l2_slerp_all.append(l2s)
            print(f"  SLERP 기준선 gpos_l2: {l2s:.4f}")
            tracks["slerp"] = gpos_slerp[b].cpu().numpy()

        path = out_dir / f"{b:02d}_{name}_f{frame0}_trans{args.trans}{'_slerp' if args.slerp else ''}.mp4"
        title = (f"{name}  frame {frame0}  |  trans {args.trans}  |  gpos_l2 {l2:.2f}"
                 + (f"  (SLERP {l2_slerp_all[-1]:.2f})" if gpos_slerp is not None else "")
                 + f"  |  {ckpt_path.parent.name}/{ckpt_path.name} it {ckpt['iteration']}")
        render_video(path, tracks, skel.parents, spec, title, korean, args.dpi)
        print(f"  저장: {path}")

    print(f"\n{len(idx)}개 평균 gpos_l2 (trans {args.trans}): {np.mean(l2_all):.4f}"
          + (f", SLERP {np.mean(l2_slerp_all):.4f}" if l2_slerp_all else "")
          + "  ※ 클립 몇 개의 평균이라 log.csv 의 val 전체 평균과는 다를 수 있음")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
