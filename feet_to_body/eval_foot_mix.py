"""
[연구실] 발 오차 민감도: 발 조건의 오차를 줄이면 전신이 얼마나 좋아지는가 (학습 없음)
==============================================================================

질문: 발 전용 모델의 발 오차를 몇 cm 까지 줄여야 발 → 몸 방식이 Two-stage 를 확실히 이기는가?

방법: 발 조건을 정답 발과 예측한 발 사이에서 섞어 준다.
        cond(α) = (1 − α) × 정답 발 조건 + α × 발 전용 모델이 예측한 발 조건      (B, L, 16)
      α = 0 은 정답 발 (body_gt), α = 1 은 예측한 발 그대로 (body_pred). 위치는 정규화 공간에서 섞지만
      정규화가 선형이라 cm 에서 섞는 것과 같다. 접촉 (라벨 0/1 과 확률) 도 같은 α 로 섞는다.
      출력에는 평가와 같은 후처리 (식 23) 를 적용한다 (body_*_post 와 같은 계산).

주의: 이렇게 만든 발 오차는 "지금 발 전용 모델의 오차와 방향이 같고 크기만 작은" 오차다.
      더 좋은 발 모델의 오차가 이런 모양이라는 보장은 없다 (I11).

지표: L2P, L2Q, NPSS, foot_skate (기존 metrics.py), cond_l2 (준 발 vs 정답 발, cm), foot_l2 (출력 발 vs 정답 발, cm)
설정: configs/eval/feet_to_body.yaml 의 foot_mix 절
실행: python feet_to_body/eval_foot_mix.py   → outputs/eval/feet_to_body_mix/{metrics.csv, metrics.png}
"""

from __future__ import annotations

import argparse
import csv
import sys
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

from eval_lengths import load_bench_stats  # noqa: E402
from feet_to_body.data import foot_condition  # noqa: E402
from feet_to_body.evaluate import check_same_stats, load_body_run, predicted_cond  # noqa: E402
from feet_to_body.train_utils import body_foot_error, run_body_model  # noqa: E402
from foot_only.data import FOOT_DIM, FootBatch, foot_indices  # noqa: E402
from foot_only.evaluate import load_foot_run  # noqa: E402
from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import build_eval_dataset, cut_batch, load_run, prepare_window  # noqa: E402
from src.evaluation.metrics import MetricAccumulator, global_quats, global_rot_pos  # noqa: E402
from src.training.context_inputs import postprocess  # noqa: E402


def mix_condition(cond_gt: torch.Tensor, cond_pred: torch.Tensor, alpha: float) -> torch.Tensor:
    """발 조건 섞기. 입력 (B, L, 16) 두 개, alpha ∈ [0, 1] → (B, L, 16). transition 밖은 둘이 같아서 그대로."""
    return (1.0 - alpha) * cond_gt + alpha * cond_pred


@torch.no_grad()
def eval_length(ds, trans, alphas, ctx_run, bodies, foot, mean, std, foot_idx, height, batch_size) -> list[dict]:
    """평가셋 ds, transition `trans` 에서 (체크포인트 × α) 별 지표. 출력: 행 dict 의 리스트."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=ctx_run.env["num_workers"])
    keys = [(n, a) for n in bodies for a in alphas]
    accs = {k: MetricAccumulator(mean, std, foot_idx, height) for k in keys}
    extra = {k: {"cond_l2": [], "foot_l2": []} for k in keys}
    skel, layout, norm = ctx_run.skel, ctx_run.layout, ctx_run.norm
    for batch in loader:
        pb, spec = prepare_window(ctx_run, batch, trans)
        B, L = pb.x_gt.shape[:2]
        s = spec.seq_slice
        labels = cut_batch(batch, 0, L)["contacts"].to(device=pb.x_gt.device, dtype=pb.x_gt.dtype)
        foot_raw = pb.gpos_gt[:, :, foot_indices(skel)].reshape(B, L, FOOT_DIM)            # (B, L, 12) 정답 발 (cm)
        fb = FootBatch(pb=pb, foot_raw=foot_raw, foot=foot.foot_norm.normalize(foot_raw), contact=labels)
        cond_gt = foot_condition(fb)                                                       # (B, L, 16)
        cond_pred, pred_raw = predicted_cond(foot, fb, spec, layout)                       # (B, L, 16), (B, L, 12)
        grot_gt, gpos_gt = global_rot_pos(pb.x_gt_raw, pb.offsets, skel, layout)
        q_gt = global_quats(grot_gt)
        for a in alphas:
            cond = mix_condition(cond_gt, cond_pred, a)
            given = (1.0 - a) * foot_raw + a * pred_raw                                    # 준 발 위치 (cm)
            cond_err = (given - foot_raw)[:, s].reshape(B, -1, 4, 3).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy()
            for name, body in bodies.items():
                y = postprocess(run_body_model(body.model, pb.x_gt, cond, spec, layout, body.attend_all), pb.x_gt, spec)
                grot, gpos = global_rot_pos(norm.denormalize(y), pb.offsets, skel, layout)
                accs[(name, a)].add(gpos, gpos_gt, global_quats(grot), q_gt, s)
                extra[(name, a)]["cond_l2"].append(cond_err)
                extra[(name, a)]["foot_l2"].append(body_foot_error(gpos, foot_raw, skel, s).mean(dim=(1, 2)).cpu().numpy())
    rows = []
    for name, a in keys:
        r = accs[(name, a)].result()
        rows.append(dict(trans=trans, body=name, alpha=a, **r,
                         **{k: float(np.concatenate(v).mean()) for k, v in extra[(name, a)].items()}))
    return rows


def plot(rows: list[dict], refs: dict, path: Path) -> None:
    """길이별 그래프: 가로 = 준 발의 오차 (cm), 세로 = L2P. 점선 = Two-stage / Context + post (기존 평가 값)."""
    lengths = sorted({r["trans"] for r in rows})
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.2 * len(lengths), 4))
    for ax, t in zip(np.atleast_1d(axes), lengths):
        for i, name in enumerate(sorted({r["body"] for r in rows})):
            rr = sorted((r for r in rows if r["trans"] == t and r["body"] == name), key=lambda r: r["alpha"])
            ax.plot([r["cond_l2"] for r in rr], [r["L2P"] for r in rr], marker="o", color=f"C{i}", label=name)
        for (m, ls) in (("two_stage", "--"), ("ctx_post", ":")):
            if (t, m) in refs:
                ax.axhline(refs[(t, m)], color="black", ls=ls, lw=1, label=m)
        ax.set_title(f"trans {t}")
        ax.set_xlabel("error of given feet (cm)")
        ax.set_ylabel("L2P")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("feet -> body: full-body error vs error of the given feet (GT/predicted mix)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    ecfg = load_named_config("eval", "lengths")
    bcfg = load_named_config("eval", "feet_to_body")
    mcfg = bcfg["foot_mix"]
    ctx_run = load_run(ecfg["ckpt"], env_name=args.env, device=args.device)
    foot = load_foot_run(load_named_config("eval", "foot_only")["foot_ckpt"], ctx_run.layout, ctx_run.device)
    bodies = {n: load_body_run(p, ctx_run.layout, ctx_run.device) for n, p in mcfg["body_ckpts"].items()}
    for n, b in bodies.items():
        check_same_stats(b, ctx_run, foot)
        print(f"{n}: {b.ckpt_path} (iteration {b.ckpt['iteration']})")
    mean, std = load_bench_stats(ctx_run, ecfg)
    fs = ecfg["foot_skating"]
    foot_idx = [ctx_run.skel.names.index(n) for n in fs["joints"]]
    ds, _ = build_eval_dataset(ctx_run, ecfg, mcfg["set"])

    rows: list[dict] = []
    for trans in mcfg["lengths"]:
        rows += eval_length(ds, trans, [float(a) for a in mcfg["alphas"]], ctx_run, bodies, foot, mean, std, foot_idx,
                            float(fs["height_threshold_cm"]), int(ecfg["batch_size"]))
    # 비교 기준 (Two-stage, Context + 후처리): 기존 평가 결과 파일에서 읽는다 (없으면 생략)
    refs = {}
    ref_csv = resolve_path(bcfg["out_dir"]) / "metrics.csv"
    if ref_csv.exists():
        for r in csv.DictReader(open(ref_csv, encoding="utf-8")):
            if r["set"] == mcfg["set"] and r["method"] in ("two_stage", "ctx_post"):
                refs[(int(r["trans"]), r["method"])] = float(r["L2P"])
    out_dir = resolve_path(mcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, refs, out_dir / "metrics.png")
    cols = ("cond_l2", "L2P", "L2Q", "NPSS", "foot_skate", "foot_l2")
    print(f"\n{'trans':>5} {'body':<16}{'alpha':>6}" + "".join(f"{c:>11}" for c in cols))
    for r in rows:
        print(f"{r['trans']:>5} {r['body']:<16}{r['alpha']:>6.2f}" + "".join(f"{r[c]:>11.4f}" for c in cols))
    for (t, m), v in sorted(refs.items()):
        print(f"기준 trans {t} {m}: L2P {v:.4f}")
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
