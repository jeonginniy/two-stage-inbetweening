"""
[연구실 전용] 발 전용 모델 데이터 검증 (STEP 2)
==============================================

확인하는 것 (출력 숫자를 보고 판단)
---------------------------------
1. shape          : FootBatch 와 모델 입력 텐서의 shape (train 분할 배치, window 42)
2. FK 일치        : 발 위치 타깃 (정렬 좌표) 을 원래 월드로 되돌린 값 vs 두 가지 독립 경로의 발 관절 월드 위치
                      (a) 원본 BVH 값 (root_pos, local_rot) 으로 바로 FK
                      (b) 기존 파이프라인 역변환 (model_output_to_motion: 역정규화 → 6D 복원 → 정렬 되돌리기) 후 FK
                    최대 오차가 ~1e-4 cm 이하면 OK (float32 기준)
3. round-trip     : 발 위치 정규화 → 역정규화 최대 오차
4. 접촉 라벨      : (a) 발 위치로 라벨을 다시 계산해 데이터셋 라벨과 비교 (일치율)
                        - 월드 좌표로 되돌려 계산: 100% 일치해야 함 (window 마지막 프레임 제외: 데이터셋은 파일의
                          다음 프레임을 보고, 다시 계산할 때는 window 밖을 모르므로 앞 프레임 복사)
                        - 정렬 좌표로 계산: |dx|+|dy|+|dz| (L1) 는 Y축 회전에 따라 값이 변해서 임계값 근처
                          프레임이 조금 달라진다 (참고용)
                    (b) 접촉 프레임 / 비접촉 프레임의 발 높이·속도 분포
                    (c) 샘플 클립 그래프: 관절별 높이 (Y) 와 속도 (|dx|+|dy|+|dz|), 접촉 구간 음영
                        → outputs/foot_only/check_data/contact_<클립>.png

실행:
    python foot_only/check_data.py --env lab
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
from torch.utils.data import DataLoader  # noqa: E402

from foot_only.data import (  # noqa: E402
    FOOT_DIM, FOOT_JOINT_NAMES, build_foot_input, canonical_frame, foot_indices, foot_to_world,
    load_foot_config, prepare_foot_batch,
)
from src.config import PROJECT_ROOT, load_data_config, load_env_config, require_dir, resolve_device  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.pipeline import model_output_to_motion  # noqa: E402
from src.data.preprocess import extract_foot_contacts  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root  # noqa: E402
from src.training.context_inputs import TransitionSpec  # noqa: E402


def world_feet_fk(root_pos, local_rot, offsets, skel, idx):
    """원래 월드 좌표 FK → 발 관절 (B, T, 4, 3)."""
    lp = local_positions_from_root(root_pos, offsets[:, None])
    return forward_kinematics(local_rot, lp, skel.parents)[1][:, :, idx]


def plot_contacts(path: Path, title: str, feet: np.ndarray, contact: np.ndarray, thr: float) -> None:
    """feet (T, 4, 3) 정렬 좌표, contact (T, 4) → 관절 4개 × (높이, 속도) 그래프."""
    T = feet.shape[0]
    vel = np.abs(np.diff(feet, axis=0)).sum(-1)                      # (T-1, 4) |dx|+|dy|+|dz|
    vel = np.concatenate([vel, vel[-1:]], 0)                          # 라벨과 같은 규칙 (마지막 프레임 복사)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    t = np.arange(T)
    for j, ax in enumerate(axes):
        ax.fill_between(t, 0, 1, where=contact[:, j] > 0.5, color="tab:green", alpha=0.18,
                        transform=ax.get_xaxis_transform(), label="contact = 1")
        ax.plot(t, feet[:, j, 1], color="tab:blue", lw=1.4, label="height Y (cm)")
        ax.set_ylabel(f"{FOOT_JOINT_NAMES[j]}\nheight (cm)", color="tab:blue")
        ax.set_ylim(bottom=min(0.0, feet[:, j, 1].min() - 1))
        ax2 = ax.twinx()
        ax2.plot(t, vel[:, j], color="tab:red", lw=1.0, label="|dx|+|dy|+|dz| (cm/frame)")
        ax2.axhline(thr, color="tab:red", ls="--", lw=0.8)
        ax2.set_yscale("log")
        ax2.set_ylim(1e-3, max(vel[:, j].max() * 1.5, 1.0))
        ax2.set_ylabel("speed (cm/frame, log)", color="tab:red")
        if j == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2 + [], loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("frame (30 fps)")
    fig.suptitle(f"{title}   (green = contact label, red dashed = threshold {thr} cm/frame)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--config", default=None, help="발 전용 설정 yaml (기본 configs/foot_only.yaml)")
    ap.add_argument("--n-batches", type=int, default=50, help="FK·라벨 검사에 쓸 train 배치 수 (배치 64)")
    ap.add_argument("--plot-seqs", nargs="+", default=["walk1_subject5", "run1_subject5", "jumps1_subject5"],
                    help="접촉 그래프를 그릴 BVH 이름 (Subject 5, 각 파일의 600–839 프레임)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "foot_only" / "check_data")
    args = ap.parse_args()

    env = load_env_config(args.env)
    data_cfg = load_data_config()
    fcfg = load_foot_config(args.config)["data"]
    device = resolve_device(env)
    bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
    cache_dir = env["paths"]["cache_dir"]
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    ctx_len, wf, thr = data_cfg["context_len"], data_cfg["forward_world_axis"], data_cfg["contact_vel_threshold"]
    idx = foot_indices(skel)
    print(f"발 관절 번호 (contact_joints): {dict(zip(FOOT_JOINT_NAMES, idx))}")

    ctx_norm = ZScoreNormalizer.load(cache_dir / "stats" / fcfg["context_stats_file"]).to(device, torch.float32)
    foot_norm = ZScoreNormalizer.load(cache_dir / "stats" / fcfg["foot_stats_file"]).to(device, torch.float32)

    sp = data_cfg["splits"]["train"]
    ds = LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel, start_frame=sp["start_frame"],
                      contact_threshold=thr, cache_dir=cache_dir, verbose=False)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))

    # ---------------- 1. shape ----------------
    batch = next(iter(loader))
    fb = prepare_foot_batch(batch, skel, ctx_norm, foot_norm, ctx_len, wf, device)
    spec = TransitionSpec(ctx_len, ctx_len + 30, (20,))
    x_in = build_foot_input(fb.pb.x_gt, fb.foot, spec, layout)
    print("\n[1] shape")
    print(f"  x_gt     {tuple(fb.pb.x_gt.shape)}   (B, T, D={layout.dim})")
    print(f"  foot_raw {tuple(fb.foot_raw.shape)}   (B, T, {FOOT_DIM})")
    print(f"  foot     {tuple(fb.foot.shape)}")
    print(f"  contact  {tuple(fb.contact.shape)}    (B, T, 4)")
    print(f"  모델 입력 {tuple(x_in.shape)}   (B, T, D + 12 + 1 = {layout.dim + FOOT_DIM + 1})")
    m = x_in[..., -1]
    tr = [t for t in range(spec.context_len, spec.target_idx) if t not in spec.midway]
    print(f"  입력 발 위치: 모르는 프레임 최대 |값| {x_in[:, tr, layout.dim:layout.dim + FOOT_DIM].abs().max():.1e} "
          f"(0 이어야 함), 아는 프레임 = 정답 오차 "
          f"{(x_in[:, spec.known_frames(), layout.dim:-1] - fb.foot[:, spec.known_frames()]).abs().max():.1e}, "
          f"m_in 합 {int(m[0].sum())} (= context {ctx_len} + 제약 1 + target 1)")

    # ---------------- 2. FK 일치, 3. round-trip, 4a. 라벨 재계산 ----------------
    err_a = err_b = rt = 0.0
    agree_w = agree_c = total = 0
    h_c, h_n, v_c, v_n = [], [], [], []
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        fb = prepare_foot_batch(batch, skel, ctx_norm, foot_norm, ctx_len, wf, device)
        frame = canonical_frame(batch, skel, ctx_len, wf, device)
        world = foot_to_world(fb.foot_raw, frame)                                       # (B, T, 4, 3)
        ref_a = world_feet_fk(batch["root_pos"].to(device), batch["local_rot"].to(device),
                              batch["offsets"].to(device), skel, idx)
        root_w, rot_w = model_output_to_motion(fb.pb.x_gt, layout, ctx_norm, frame)
        ref_b = world_feet_fk(root_w, rot_w, batch["offsets"].to(device), skel, idx)
        err_a = max(err_a, float((world - ref_a).norm(dim=-1).max()))
        err_b = max(err_b, float((world - ref_b).norm(dim=-1).max()))
        rt = max(rt, float((foot_norm.denormalize(foot_norm.normalize(fb.foot_raw)) - fb.foot_raw).abs().max()))
        B, T = fb.foot_raw.shape[:2]
        feet = fb.foot_raw.reshape(B, T, 4, 3)
        lab = fb.contact[:, :-1]                                                         # 마지막 프레임 제외
        agree_w += int((extract_foot_contacts(world, range(4), thr)[:, :-1] == lab).sum())
        agree_c += int((extract_foot_contacts(feet, range(4), thr)[:, :-1] == lab).sum())
        total += lab.numel()
        vel = (feet[:, 1:] - feet[:, :-1]).abs().sum(-1)                                 # (B, T-1, 4)
        c = fb.contact[:, :-1] > 0.5
        h = feet[:, :-1, :, 1]
        h_c.append(h[c].cpu()), h_n.append(h[~c].cpu()), v_c.append(vel[c].cpu()), v_n.append(vel[~c].cpu())
    nclip = min(args.n_batches, len(loader)) * 64
    print(f"\n[2] FK 일치 (train 클립 약 {nclip}개, 발 4관절 × 42프레임, 원래 월드 좌표)")
    print(f"  (a) 원본 BVH 로 바로 FK            최대 오차 {err_a:.2e} cm")
    print(f"  (b) 파이프라인 역변환 후 FK        최대 오차 {err_b:.2e} cm")
    print(f"\n[3] 정규화 → 역정규화 round-trip 최대 오차 {rt:.2e} cm")
    print(f"\n[4a] 다시 계산한 접촉 라벨 vs 데이터셋 라벨 (window 마지막 프레임 제외, {total} 개)")
    print(f"  월드 좌표로 계산: 일치 {agree_w / total * 100:.4f}% ({total - agree_w} 개 다름)  ← 100% 이어야 함")
    print(f"  정렬 좌표로 계산: 일치 {agree_c / total * 100:.4f}% ({total - agree_c} 개 다름)  ← L1 속도가 회전에 따라 변함 (참고)")
    h_c, h_n, v_c, v_n = (torch.cat(z).numpy() for z in (h_c, h_n, v_c, v_n))
    q = lambda a: " / ".join(f"{np.percentile(a, p):6.2f}" for p in (5, 50, 95))  # noqa: E731
    print("[4b] 발 높이·속도 분포 (5% / 50% / 95% 분위, 관절 4개 합침)")
    print(f"  접촉 프레임   ({len(h_c) / (len(h_c) + len(h_n)) * 100:4.1f}%)  높이 {q(h_c)} cm   속도 {q(v_c)} cm/frame")
    print(f"  비접촉 프레임 ({len(h_n) / (len(h_c) + len(h_n)) * 100:4.1f}%)  높이 {q(h_n)} cm   속도 {q(v_n)} cm/frame")

    # ---------------- 4c. 샘플 클립 그래프 ----------------
    plot_start = 600                                                      # 각 파일 앞 600 프레임 (20초) 건너뜀
    plot_ds = LafanDataset(bvh_dir, ["subject5"], 240, 10_000, skel, start_frame=plot_start,
                           contact_threshold=thr, cache_dir=cache_dir, verbose=False)
    names = [p.stem for p in plot_ds.files]
    print("\n[4c] 접촉 그래프")
    for seq in args.plot_seqs:
        if seq not in names:
            print(f"  {seq}: 파일 없음, 건너뜀")
            continue
        item_idx = int(plot_ds._cum[names.index(seq)])
        item = plot_ds[item_idx]
        one = {k: (v[None] if torch.is_tensor(v) else v) for k, v in item.items()}
        fb = prepare_foot_batch(one, skel, ctx_norm, None, ctx_len, wf, device)
        feet = fb.foot_raw[0].reshape(-1, 4, 3).cpu().numpy()
        f0 = plot_start + item["start"]                                   # 실제 BVH 프레임 번호
        path = args.out / f"contact_{seq}_{f0}.png"
        plot_contacts(path, f"{seq}  BVH frames {f0}-{f0 + 239}", feet,
                      fb.contact[0].cpu().numpy(), thr)
        print(f"  저장: {path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(f"[오류] {e}")
