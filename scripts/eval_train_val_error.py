"""
[연구실] Context 체크포인트별 train / val 위치 오차 (G15 인과 확인, docs/assumptions.md G20 · G21)
=====================================================================================

이 스크립트가 하는 일
--------------------
Context Transformer 체크포인트 여러 개 (보통 한 run 의 중간 체크포인트 snap_it*.pth) 에 대해
"train 분할에서 얼마나 정확한가" 와 "val 분할에서 얼마나 정확한가" 를 같은 클립·같은 식으로 잰다.
G15 가설 = Context 가 train 에서 너무 정확하면 (val 보다 훨씬 쉬운 입력) Detail 이 평가 때보다 쉬운 입력으로 학습된다.
G20 에서 공식 Context (iteration 181000) 는 같은 측정으로 train trans 30 = 4.297 cm 였다.

    클립 고르기 (G15 · G20 과 같음):
        train 분할 (Subject 1–4, window 42, offset 1) 에서 random.Random(seed=0).sample 로 2000 클립
        val   분할 (Subject 5, window 65, offset 160, start_frame 35) 560 클립 전부
    클립마다:
        입력 데이터 (앞 L = 10 + t + 2 프레임) → prepare_batch (정렬 + z-score + 정답 FK, 논문 3.2)
        → Context (모델 입력 → 모델 출력, 논문 3.3) → [후처리 식 23 / 없음]
        → 역정규화 → FK 전역 위치 (복원) → gpos_l2 = mean ‖g_pred − g_gt‖₂ (cm, 클립 × transition 프레임 × 관절)
    --select-train30 X 를 주면 "train, 후처리 있음, trans 30" 오차가 X 에 가장 가까운 체크포인트 경로를
    --select-file 에 적는다 (G21 의 미리 정한 기준; 그다음 scripts/train_detail.py --context-ckpt 에 넘긴다).

실행 순서 (G21)
--------------
    1) python scripts/train_context.py --run-name ctx_g15 --max-iters 300000 --set train.snapshot_interval=10000
    2) python scripts/eval_train_val_error.py --ckpts experiments/ctx_g15/snap_it*.pth --select-train30 4.297
       → outputs/g15/ctx_g15/train_val_error.csv, selected_ctx.txt
    3) python scripts/train_detail.py --run-name det_g15 --context-ckpt $(cat outputs/g15/ctx_g15/selected_ctx.txt)

체크포인트는 읽기만 한다. 결과는 outputs/ (git 제외) 에만 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.evaluation.inference import load_run  # noqa: E402
from src.training.context import global_positions, gpos_l2, prepare_batch  # noqa: E402
from src.training.context_inputs import TransitionSpec  # noqa: E402
from src.training.detail import run_context_frozen  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="Context 체크포인트들 (같은 run 폴더, config.yaml 이 옆에 있어야 함)")
    ap.add_argument("--env", default=None, help="configs/env/<env>.yaml 이름 (기본: 체크포인트 config.yaml 에 적힌 env)")
    ap.add_argument("--n-train", type=int, default=2000, help="train 분할에서 뽑을 클립 수 (G15 · G20 과 같게 2000)")
    ap.add_argument("--seed", type=int, default=0, help="train 클립 표본 seed (G15 · G20 과 같게 0)")
    ap.add_argument("--trans", type=int, nargs="+", default=[5, 15, 30], help="transition 길이들")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-csv", default=None, help="기본: outputs/g15/<run>/train_val_error.csv")
    ap.add_argument("--select-train30", type=float, default=None,
                    help="train·후처리·trans 30 오차가 이 값 (cm) 에 가장 가까운 체크포인트를 고른다 (G21: 4.297)")
    ap.add_argument("--select-file", default=None, help="고른 체크포인트 경로를 적을 파일 (기본: <out-csv 폴더>/selected_ctx.txt)")
    return ap.parse_args()


def make_split(run, split: str, n: int | None, seed: int) -> Subset:
    """분할 이름 → 클립 묶음 (Subset). n 이 있으면 random.Random(seed).sample 로 n 개 (G15 와 같은 표본).

    각 항목: root_pos (window, 3), local_rot (window, J, 3, 3), contacts (window, 4), offsets (J, 3), ...
    """
    dc = run.data_cfg
    sp = dc["splits"][split]
    ds = LafanDataset(run.env["paths"]["lafan1_bvh_dir"], sp["actors"], sp["window"], sp["offset"], run.skel,
                      start_frame=sp["start_frame"], contact_threshold=dc["contact_vel_threshold"],
                      cache_dir=run.env["paths"]["cache_dir"], dtype=torch.float32, verbose=False)
    idx = random.Random(seed).sample(range(len(ds)), n) if n else list(range(len(ds)))
    return Subset(ds, idx)


@torch.no_grad()
def measure(run, clips: Subset, t: int, use_post: bool, batch_size: int) -> float:
    """transition 길이 t 의 평균 전역 위치 오차 (cm).

    입력: run (inference.LoadedRun: 모델·정규화·스켈레톤), clips (Subset), t (int), use_post (식 23 후처리 여부)
    처리: 배치 (B, window, ...) → 앞 L = ctx + t + 2 프레임 → prepare_batch → x_gt (B, L, D)
          → run_context_frozen → y (B, L, D) 정규화 공간 → 역정규화 → global_positions (B, L, J, 3) cm
          → gpos_l2 (transition 프레임 ctx .. ctx+t-1)
    출력: 클립 수로 가중 평균한 오차 (float, cm)
    """
    dc = run.data_cfg
    spec = TransitionSpec(run.ctx, run.ctx + t)
    L = run.ctx + t + 2                              # target 다음 프레임까지 (후처리 외삽에 필요)
    tot = cnt = 0.0
    for b in DataLoader(clips, batch_size=batch_size):
        b = {k: (v[:, :L] if k in ("root_pos", "local_rot", "contacts") else v) for k, v in b.items()}
        pb = prepare_batch(b, run.skel, run.norm, run.ctx, dc["forward_world_axis"], run.device)
        y = run_context_frozen(run.model, pb.x_gt, spec, run.layout, use_post)
        g = global_positions(run.norm.denormalize(y), pb.offsets, run.skel, run.layout)
        tot += float(gpos_l2(g, pb.gpos_gt, spec.seq_slice)) * len(g)
        cnt += len(g)
    return tot / cnt


def main() -> None:
    args = parse_args()
    ckpts = sorted(resolve_path(c) for c in args.ckpts)
    run0 = load_run(ckpts[0], env_name=args.env)
    out_csv = resolve_path(args.out_csv) if args.out_csv else resolve_path("outputs/g15") / ckpts[0].parent.name / "train_val_error.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    splits = {"train": make_split(run0, "train", args.n_train, args.seed), "val": make_split(run0, "val", None, args.seed)}
    print(f"클립: train {len(splits['train'])} (seed {args.seed}), val {len(splits['val'])}, 체크포인트 {len(ckpts)} 개")

    rows = []
    for ck in ckpts:
        run = run0 if ck == ckpts[0] else load_run(ck, env_name=args.env)
        it = int(torch.load(ck, map_location="cpu", weights_only=False)["iteration"])
        res = {}
        for split, clips in splits.items():
            for use_post in (True, False):
                for t in args.trans:
                    e = measure(run, clips, t, use_post, args.batch_size)
                    res[(split, use_post, t)] = e
                    rows.append({"ckpt": str(ck), "iteration": it, "split": split, "post": use_post, "trans": t,
                                 "n_clips": len(clips), "gpos_l2_cm": round(e, 5)})
        print(f"{ck.name:22s} it {it:6d} | 후처리: train " + "/".join(f"{res[('train', True, t)]:.3f}" for t in args.trans)
              + " val " + "/".join(f"{res[('val', True, t)]:.3f}" for t in args.trans) + " cm", flush=True)
        del run
        torch.cuda.empty_cache()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {out_csv}")

    if args.select_train30 is not None:
        # [결정 G21] 미리 정한 기준: train · 후처리 있음 · trans 30 오차가 목표값에 가장 가까운 체크포인트
        cand = [r for r in rows if r["split"] == "train" and r["post"] and r["trans"] == 30]
        best = min(cand, key=lambda r: abs(r["gpos_l2_cm"] - args.select_train30))
        sel = resolve_path(args.select_file) if args.select_file else out_csv.parent / "selected_ctx.txt"
        sel.write_text(best["ckpt"] + "\n", encoding="utf-8")
        print(f"선택 (train30 목표 {args.select_train30} cm): {best['ckpt']} (iteration {best['iteration']}, "
              f"train30 {best['gpos_l2_cm']:.3f} cm) → {sel}")


if __name__ == "__main__":
    main()
