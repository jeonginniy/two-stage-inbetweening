"""
발 → 몸 모델 학습 (정답 발 조건)
===============================

순서
----
1. (한 번만) python scripts/compute_stats.py           → 전신 통계 (ctx_base 와 같은 파일, 이미 있으면 생략)
2. (한 번만) python foot_only/compute_foot_stats.py    → 발 위치 통계 (발 전용 모델과 같은 파일, 이미 있으면 생략)
3. python feet_to_body/train_body.py --run-name <이름>   → experiments/<이름>/{config.yaml, log.csv, last.pth, best.pth, best_mean.pth}
   이어서: 같은 명령 + --resume last
   1차 (정답 발, 끝남)  : experiments/feet_to_body        (configs/feet_to_body.yaml)
   2차 (예측한 발) [I8] : python feet_to_body/train_body.py --config configs/feet_to_body_pred.yaml --run-name feet_to_body_pred

학습 설정·손실은 ctx_base 와 같다 (configs/feet_to_body.yaml, [ctx_base] 표기):
batch 32, Noam (0.0625, warmup 8000), 500k iteration, train window 42, transition [5, 30] (배치마다 하나),
제약 프레임 (0.1, 0.5) 은 학습 때만, 손실 = 논문 식 21. best = val trans 30 의 val30_loss 최소.
다른 점은 입력에 모든 프레임의 발 조건 (위치 12 + 접촉 4) 이 들어가고 [결정 I1], 어텐션이 모든 프레임을 본다는 것 [결정 I2].
발 조건은 cond.source 에 따라 정답 발 (gt) 또는 고정한 발 전용 모델의 예측 (foot_model) [결정 I8].
best_mean.pth = 모든 val 길이의 손실 평균이 최소인 시점 (train.save_best_mean) [결정 I9].

log.csv (iteration, epoch, kind, name, value):
    train : loss, rot, pos, smooth, lr                                (log_interval 마다, 구간 평균)
    val   : feet_to_body/train_utils.validate 의 모든 값                (val_interval 마다)
    → loss / rot / pos / smooth / gpos_l2 는 experiments/ctx_base/log.csv 와 같은 이름·같은 식 (바로 비교 가능)

데이터 없이 동작 확인:
    python feet_to_body/train_body.py --env laptop --fake-data --run-name body_smoke --max-iters 20 --batch-size 8 \
        --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from feet_to_body.data import foot_condition, load_body_config  # noqa: E402
from feet_to_body.evaluate import predicted_cond  # noqa: E402
from feet_to_body.model import build_body_model  # noqa: E402
from feet_to_body.train_utils import body_forward, validate  # noqa: E402
from foot_only.data import FOOT_DIM, compute_foot_stats, prepare_foot_batch  # noqa: E402
from foot_only.evaluate import load_foot_run  # noqa: E402
from src.config import load_data_config, load_env_config, require_dir, resolve_device, resolve_path  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402
from src.models.context_transformer import count_parameters  # noqa: E402
from src.training.context_inputs import sample_transition  # noqa: E402
from src.training.utils import (  # noqa: E402
    amp_dtype_from_str, infinite_batches, load_checkpoint, noam_lambda, save_checkpoint, set_seed,
)
from train_context import apply_overrides, compute_fake_stats, make_fake_lafan  # noqa: E402

PROTECTED_RUNS = ("ctx_base", "ctx_smoke", "det_base", "det_cont", "foot_only", "foot_check_1k", "feet_to_body")   # 기존 실험 폴더에는 절대 쓰지 않음


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--config", default=None, help="설정 yaml (기본 configs/feet_to_body.yaml)")
    ap.add_argument("--run-name", required=True, help="결과 폴더 이름 → experiments/<이름> (기존 실험 폴더 이름은 거부)")
    ap.add_argument("--resume", default=None, help="'last' / 'best' (run 폴더 안) 또는 체크포인트 경로")
    ap.add_argument("--batch-size", type=int, default=None, help="train.batch_size")
    ap.add_argument("--max-iters", type=int, default=None, help="train.max_iters")
    ap.add_argument("--amp", default=None, help="train.amp (none | fp16 | bf16)")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader 워커 수 (기본: env 설정)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기. 예) cond.attend_all=false")
    ap.add_argument("--fake-data", action="store_true", help="가짜 BVH 로 동작만 확인 (결과 의미 없음)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env_config(args.env)
    data_cfg = load_data_config()
    cfg = load_body_config(args.config)
    for k, v in {"batch_size": args.batch_size, "max_iters": args.max_iters, "amp": args.amp}.items():
        if v is not None:
            cfg["train"][k] = None if v in ("none", "null") else v
    apply_overrides(cfg, args.set)
    tcfg, loss_w = cfg["train"], cfg["loss"]
    attend_all = bool(cfg["cond"]["attend_all"])
    source = cfg["cond"].get("source", "gt")
    if source not in ("gt", "foot_model"):
        raise ValueError(f"cond.source 는 gt 또는 foot_model 이어야 합니다: {source}")
    if source == "foot_model" and args.fake_data:
        raise ValueError("--fake-data 에서는 cond.source=foot_model 을 쓸 수 없습니다 (발 전용 체크포인트가 필요).")

    device = resolve_device(env)
    num_workers = env["num_workers"] if args.num_workers is None else args.num_workers
    set_seed(int(tcfg.get("seed", 0)))
    rng = random.Random(int(tcfg.get("seed", 0)))

    # ----- 결과 폴더 (기존 실험 보호, 덮어쓰기 거부) -----
    if args.run_name in PROTECTED_RUNS:
        raise ValueError(f"{args.run_name} 은 기존 실험 폴더입니다. 다른 --run-name 을 쓰세요.")
    exp_root = env["paths"].get("experiments_dir") or resolve_path("experiments")
    run_dir = exp_root / args.run_name
    if (run_dir / "last.pth").exists() and not args.resume:
        raise ValueError(f"{run_dir}/last.pth 가 이미 있습니다. 이어서 하려면 --resume last, 새로 하려면 다른 --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- 데이터·통계 -----
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    ctx, wf = data_cfg["context_len"], data_cfg["forward_world_axis"]
    cache_dir = env["paths"]["cache_dir"]
    fdat = cfg["data"]
    if args.fake_data:
        bvh_dir = make_fake_lafan(run_dir / "fake_lafan1")
        ctx_stats = compute_fake_stats(bvh_dir, data_cfg, skel, run_dir / "fake_stats.npz")
        sp = data_cfg["splits"]["train_stats"]
        sds = LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel,
                           contact_threshold=data_cfg["contact_vel_threshold"], dtype=torch.float64, verbose=False)
        st = compute_foot_stats(DataLoader(sds, batch_size=64), skel,
                                ZScoreNormalizer.load(ctx_stats).to(dtype=torch.float64), ctx, wf)
        foot_stats = run_dir / "fake_foot_stats.npz"
        ZScoreNormalizer.from_stats(st).save(foot_stats)
        cache_dir = None
        print("[주의] --fake-data: 가짜 BVH 로 동작만 확인합니다. 학습 결과는 의미가 없습니다.")
    else:
        bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
        ctx_stats = cache_dir / "stats" / fdat["context_stats_file"]
        foot_stats = cache_dir / "stats" / fdat["foot_stats_file"]
        for p, how in ((ctx_stats, "scripts/compute_stats.py"), (foot_stats, "foot_only/compute_foot_stats.py")):
            if not p.exists():
                raise FileNotFoundError(f"{p} 가 없습니다. 먼저 {how} 를 실행하세요.")
    ctx_norm = ZScoreNormalizer.load(ctx_stats).to(device=device, dtype=torch.float32)
    foot_norm = ZScoreNormalizer.load(foot_stats).to(device=device, dtype=torch.float32)
    if ctx_norm.dim != layout.dim or foot_norm.dim != FOOT_DIM:
        raise ValueError(f"통계 차원 (입력 {ctx_norm.dim}, 발 {foot_norm.dim}) ≠ ({layout.dim}, {FOOT_DIM})")

    def make_ds(split: str) -> LafanDataset:
        sp = data_cfg["splits"][split]
        return LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel, start_frame=sp["start_frame"],
                            contact_threshold=data_cfg["contact_vel_threshold"], cache_dir=cache_dir,
                            dtype=torch.float32, verbose=not args.fake_data)

    val_split = tcfg.get("val_split", "val")
    train_ds, val_ds = make_ds("train"), make_ds(val_split)
    window = data_cfg["splits"]["train"]["window"]
    min_trans, max_trans = data_cfg["min_trans"], data_cfg["max_trans"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=int(tcfg.get("val_batch_size", 64)), shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    if len(train_loader) == 0:
        raise ValueError(f"train 클립 수({len(train_ds)})가 batch_size({tcfg['batch_size']})보다 적습니다.")
    val_trans = [t for t in tcfg.get("val_trans", [max_trans]) if ctx + t + 2 <= data_cfg["splits"][val_split]["window"]]

    # ----- 발 조건 [결정 I8]: 정답 발, 또는 고정한 발 전용 모델의 예측 -----
    foot_run = None
    if source == "foot_model":
        foot_run = load_foot_run(cfg["cond"]["foot_ckpt"], layout, device)     # eval 모드, 학습하지 않음
        for name, a, b in (("전신", foot_run.ctx_norm, ctx_norm), ("발", foot_run.foot_norm, foot_norm)):
            if not (torch.allclose(a.mean, b.mean) and torch.allclose(a.std, b.std)):
                raise ValueError(f"발 전용 모델의 {name} 정규화 통계가 이 학습과 다릅니다.")
        for prm in foot_run.model.parameters():
            prm.requires_grad_(False)

    def make_cond(fb, spec):
        """(발 조건 (B, T, 16), 조건 발 위치 (B, T, 12) cm). foot_model 이면 transition 만 예측, 나머지는 정답."""
        if foot_run is None:
            return foot_condition(fb), fb.foot_raw
        return predicted_cond(foot_run, fb, spec, layout)

    # ----- 모델·최적화 -----
    model = build_body_model(cfg["model"], layout).to(device)
    base_lr = float(tcfg["base_lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
                                 weight_decay=float(tcfg.get("weight_decay", 0.0)))
    lr_factor = noam_lambda(int(tcfg["warmup_iters"]))
    amp_dtype = amp_dtype_from_str(tcfg.get("amp"), device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    accum = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = tcfg.get("grad_clip")
    max_iters = int(tcfg["max_iters"])

    iteration, epoch, best_val, best_mean = 0, 0, math.inf, math.inf
    save_mean = bool(tcfg.get("save_best_mean", False))
    if args.resume:
        ckpt_path = run_dir / f"{args.resume}.pth" if args.resume in ("last", "best") else resolve_path(args.resume)
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        iteration, epoch, best_val = ckpt["iteration"], ckpt["epoch"], ckpt["best_val"]
        if save_mean and (run_dir / "best_mean.pth").exists():                 # 평균 규칙의 지금까지 최솟값
            best_mean = float(load_checkpoint(run_dir / "best_mean.pth", map_location="cpu")["best_val"])
        print(f"이어서 학습: {ckpt_path} (iteration {iteration}, best {best_val:.4f})")

    # data = 데이터셋 설정 (configs/data/lafan1.yaml), foot_data = configs/feet_to_body.yaml 의 data 절 (이름이 겹쳐서 분리)
    resolved = {"env": env["name"], "data": {k: str(v) if isinstance(v, Path) else v for k, v in data_cfg.items()},
                "foot_data": copy.deepcopy(fdat), **{k: copy.deepcopy(v) for k, v in cfg.items() if k != "data"},
                "fake_data": args.fake_data, "stats": str(ctx_stats), "foot_stats": str(foot_stats)}
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)

    print(f"device={device}, amp={amp_dtype}, 입력 {model.d_in} → 출력 {model.d_out}, "
          f"파라미터 {count_parameters(model):,}")
    print(f"train 클립 {len(train_ds)} (window {window}), val 클립 {len(val_ds)} (trans {val_trans}), "
          f"batch {tcfg['batch_size']} × accum {accum}, max_iters {max_iters}, loss {loss_w}, attend_all {attend_all}, "
          f"발 조건 {source}" + (f" ({foot_run.ckpt_path}, iteration {foot_run.ckpt['iteration']})" if foot_run else ""))
    print(f"결과 폴더: {run_dir}")

    log_path = run_dir / "log.csv"
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if log_path.stat().st_size == 0:
        log.writerow(["iteration", "epoch", "kind", "name", "value"])

    # ----- 학습 루프 -----
    batches = infinite_batches(train_loader)
    running: dict[str, float] = {}
    t0 = time.time()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while iteration < max_iters:
        iteration += 1
        lr = base_lr * lr_factor(iteration)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            epoch, batch = next(batches)
            spec = sample_transition(ctx, min_trans, max_trans, window, float(tcfg.get("midway_targets_amount", 0.0)),
                                     float(tcfg.get("midway_targets_p", 0.0)), rng)
            fb = prepare_foot_batch(batch, skel, ctx_norm, foot_norm, ctx, wf, device)
            cond, _ = make_cond(fb, spec)
            losses, _, _ = body_forward(model, fb, cond, spec, skel, layout, ctx_norm, loss_w, attend_all, amp_dtype)
            loss = losses["loss"] / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach()) / accum
        if not math.isfinite(running["loss"]):
            raise FloatingPointError(f"iteration {iteration}: 손실이 {running['loss']} 입니다.")
        if scaler is not None:
            scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        log_int = int(tcfg.get("log_interval", 100))
        if iteration % log_int == 0 or iteration == max_iters:
            n = log_int if iteration % log_int == 0 else (iteration % log_int)
            avg = {k: v / n for k, v in running.items()}
            dt = (time.time() - t0) / n
            mem = f", VRAM 최대 {torch.cuda.max_memory_allocated(device) / 2**20:.0f}MB" if device.type == "cuda" else ""
            print(f"it {iteration:7d} ep {epoch:3d} lr {lr:.2e} | loss {avg['loss']:.4f} rot {avg['rot']:.4f} "
                  f"pos {avg['pos']:.3f}cm smooth {avg['smooth']:.3f} | {dt * 1000:.0f}ms/it{mem}")
            for k, v in avg.items():
                log.writerow([iteration, epoch, "train", k, f"{v:.6g}"])
            log.writerow([iteration, epoch, "train", "lr", f"{lr:.6g}"])
            log_file.flush()
            running, t0 = {}, time.time()

        if iteration % int(tcfg.get("val_interval", 2000)) == 0 or iteration == max_iters:
            res = validate(model, val_loader, val_trans, data_cfg, skel, layout, ctx_norm, foot_norm, loss_w, attend_all,
                           device, make_cond)
            print("  [val] " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()
                                         if k.endswith(("_loss", "_gpos_l2", "_foot_l2"))))
            for k, v in res.items():
                log.writerow([iteration, epoch, "val", k, f"{v:.6g}"])
            log_file.flush()
            key = f"val{val_trans[-1]}_loss" if val_trans else None
            if key and res[key] < best_val:
                best_val = res[key]
                save_checkpoint(run_dir / "best.pth", model, optimizer, scaler, iteration, epoch, best_val, resolved)
                print(f"  best 갱신 ({key} = {best_val:.4f}) → best.pth")
            if save_mean and val_trans:                                        # [결정 I9] 모든 val 길이의 손실 평균
                mean_loss = sum(res[f"val{t}_loss"] for t in val_trans) / len(val_trans)
                log.writerow([iteration, epoch, "val", "val_mean_loss", f"{mean_loss:.6g}"])
                log_file.flush()
                if mean_loss < best_mean:
                    best_mean = mean_loss
                    save_checkpoint(run_dir / "best_mean.pth", model, optimizer, scaler, iteration, epoch, best_mean,
                                    resolved)
                    print(f"  best_mean 갱신 (val 손실 평균 = {best_mean:.4f}) → best_mean.pth")
            t0 = time.time()

        if iteration % int(tcfg.get("ckpt_interval", 5000)) == 0 or iteration == max_iters:
            save_checkpoint(run_dir / "last.pth", model, optimizer, scaler, iteration, epoch, best_val, resolved)

    log_file.close()
    print(f"끝: iteration {iteration}, best val {best_val:.4f}. 결과: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
