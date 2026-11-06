"""
[연구실] Context Transformer 학습 (2단계, 논문 3.3 · 3.5 · 3.6)
=============================================================

이 스크립트가 하는 일
--------------------
1. 설정 읽기: configs/env/<env>.yaml (경로, device), configs/data/lafan1.yaml (window, context_len 등),
   configs/model/context.yaml (모델 크기, batch size, 학습률, 손실 가중치)
2. LaFAN1 train 분할 (Subject 1–4, window 42, offset 1) 과 val 분할 (Subject 5) 준비
3. 매 iteration:
     transition 길이 [min_trans, max_trans] 무작위 (+ 제약 프레임) → 입력/마스크 → 모델 →
     L_rot + 0.01·L_pos + 0.005·L_smooth (식 21) → Adam (Noam 학습률)
4. val_interval 마다 val 손실, ckpt_interval 마다 체크포인트 저장
   결과 폴더: <experiments_dir>/<run_name>/  (config.yaml, log.csv, last.pth, best.pth)

정규화 통계는 scripts/compute_stats.py 가 만든 파일을 쓴다 (먼저 실행해야 함).

실행 예:
    python scripts/train_context.py --env lab
    python scripts/train_context.py --env lab --batch-size 64 --amp bf16 --run-name ctx_bs64
    python scripts/train_context.py --env lab --resume last              # 이어서 학습
    python scripts/train_context.py --env lab --set model.n_layer=4 train.max_iters=100000

    # 데이터 없이 스크립트가 끝까지 도는지만 확인 (가짜 LaFAN1 형식 BVH 생성, 작은 모델):
    python scripts/train_context.py --env laptop --fake-data --max-iters 20 \
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import (  # noqa: E402
    load_data_config,
    load_env_config,
    load_model_config,
    require_dir,
    resolve_device,
    resolve_path,
)
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402
from src.models.context_transformer import build_context_transformer, count_parameters  # noqa: E402
from src.training.context import context_forward, eval_transition, gpos_l2, prepare_batch  # noqa: E402
from src.training.context_inputs import sample_transition  # noqa: E402
from src.training.utils import (  # noqa: E402
    amp_dtype_from_str,
    infinite_batches,
    load_checkpoint,
    noam_lambda,
    save_checkpoint,
    set_seed,
)


# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
def apply_overrides(cfg: dict, items: list[str]) -> None:
    """--set a.b=값 들을 cfg 에 반영. 값은 YAML 로 해석 (숫자, null, [1,2] 등)."""
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set 형식은 key.sub=value: {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--config", default="context", help="configs/model/<이름>.yaml 또는 yaml 경로 (기본 context)")
    ap.add_argument("--run-name", default=None, help="결과 폴더 이름 (기본: context_<날짜시간>)")
    ap.add_argument("--stats", type=Path, default=None,
                    help="정규화 통계 .npz (기본: <cache_dir>/stats/lafan1_context_stats.npz)")
    ap.add_argument("--resume", default=None, help="체크포인트 경로, 또는 'last' / 'best' (run 폴더 안)")
    # 자주 바꾸는 값은 바로가기 인자로 (설정 파일 값을 덮어씀)
    ap.add_argument("--batch-size", type=int, default=None, help="train.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="train.grad_accum_steps")
    ap.add_argument("--max-iters", type=int, default=None, help="train.max_iters")
    ap.add_argument("--amp", default=None, help="train.amp (none | fp16 | bf16)")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader 워커 수 (기본: env 설정)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="임의 설정 덮어쓰기. 예) model.n_layer=4 train.base_lr=0.03")
    ap.add_argument("--fake-data", action="store_true",
                    help="LaFAN1 없이 가짜 BVH 로 스크립트 동작만 확인 (결과는 의미 없음)")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# 가짜 데이터 (스크립트 동작 확인용)
# -----------------------------------------------------------------------------
def make_fake_lafan(folder: Path, frames: int = 160) -> Path:
    """subject1~5 에 가짜 BVH 를 2개씩 만든다. 실제 LaFAN1 값이 아님."""
    from tests.helpers import write_fake_lafan_bvh

    folder.mkdir(parents=True, exist_ok=True)
    seed = 0
    for subj in range(1, 6):
        for motion in ("walk1", "run1"):
            path = folder / f"{motion}_subject{subj}.bvh"
            if not path.exists():
                write_fake_lafan_bvh(path, frames, seed=seed)
            seed += 1
    return folder


def compute_fake_stats(bvh_dir: Path, data_cfg: dict, skel: Skeleton, out: Path) -> Path:
    from src.data.pipeline import motion_to_model_input
    from src.data.representation import RunningStats

    sp = data_cfg["splits"]["train_stats"]
    ds = LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel,
                      contact_threshold=data_cfg["contact_vel_threshold"], dtype=torch.float64, verbose=False)
    stats = RunningStats(FeatureLayout(skel.num_joints).dim)
    for b in DataLoader(ds, batch_size=64):
        stats.update(motion_to_model_input(b["root_pos"], b["local_rot"], data_cfg["context_len"],
                                           skel.root_forward_local_axis).x_raw)
    ZScoreNormalizer.from_stats(stats).save(out)
    return out


# -----------------------------------------------------------------------------
# 검증
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, trans_list, data_cfg, skel, layout, norm, loss_w, device) -> dict[str, float]:
    """val 분할에서 transition 길이별 손실 (학습과 같은 식, 후처리 없음) 과
    전역 관절 위치 평균 L2 오차 (cm, 후처리 포함) 를 계산한다.
    """
    sums: dict[str, float] = {}
    n_batches = 0
    model.eval()
    for batch in loader:
        n_batches += 1
        for trans in trans_list:
            # 자르기 → 정렬 → 추론(후처리) → FK. scripts/render_compare.py 도 같은 함수를 쓴다.
            pb, spec, _, gpos = eval_transition(model, batch, trans, data_cfg, skel, layout, norm, device)
            losses = context_forward(model, pb, spec, skel, layout, norm, loss_w)
            l2 = gpos_l2(gpos, pb.gpos_gt, spec.seq_slice)
            for k, v in losses.items():
                sums[f"val{trans}_{k}"] = sums.get(f"val{trans}_{k}", 0.0) + float(v)
            sums[f"val{trans}_gpos_l2"] = sums.get(f"val{trans}_gpos_l2", 0.0) + float(l2)
    model.train()
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    env = load_env_config(args.env)
    data_cfg = load_data_config()
    cfg = load_model_config(args.config)

    # ----- 설정 덮어쓰기 -----
    shortcuts = {"batch_size": args.batch_size, "grad_accum_steps": args.grad_accum,
                 "max_iters": args.max_iters, "amp": args.amp}
    for k, v in shortcuts.items():
        if v is not None:
            cfg["train"][k] = None if v in ("none", "null") else v
    apply_overrides(cfg, args.set)
    tcfg, loss_w = cfg["train"], cfg["loss"]

    device = resolve_device(env)
    num_workers = env["num_workers"] if args.num_workers is None else args.num_workers
    set_seed(int(tcfg.get("seed", 0)))
    rng = random.Random(int(tcfg.get("seed", 0)))

    # ----- 결과 폴더 -----
    if args.resume in ("last", "best") and not args.run_name:
        raise ValueError("--resume last/best 는 --run-name 과 함께 써야 합니다 (어느 폴더인지).")
    exp_root = env["paths"].get("experiments_dir") or resolve_path("experiments")
    run_name = args.run_name or (Path(args.resume).parent.name if args.resume and args.resume not in ("last", "best")
                                 else None) or time.strftime("context_%Y%m%d_%H%M%S")
    run_dir = exp_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- 데이터 -----
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)                       # D = J*6+3 (스켈레톤 파일에서)
    cache_dir = env["paths"]["cache_dir"]
    if args.fake_data:
        bvh_dir = make_fake_lafan(run_dir / "fake_lafan1")
        stats_path = compute_fake_stats(bvh_dir, data_cfg, skel, run_dir / "fake_stats.npz")
        cache_dir = None
        print("[주의] --fake-data: 가짜 BVH 로 동작만 확인합니다. 학습 결과는 의미가 없습니다.")
    else:
        bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
        stats_path = args.stats or (cache_dir / "stats" / "lafan1_context_stats.npz")
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"정규화 통계 {stats_path} 가 없습니다. 먼저 scripts/compute_stats.py 를 실행하세요.")
    norm = ZScoreNormalizer.load(stats_path).to(device=device, dtype=torch.float32)
    if norm.dim != layout.dim:
        raise ValueError(f"통계 차원 {norm.dim} 이 스켈레톤으로 계산한 D={layout.dim} 과 다릅니다.")

    def make_ds(split: str) -> LafanDataset:
        sp = data_cfg["splits"][split]
        return LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel,
                            start_frame=sp["start_frame"], contact_threshold=data_cfg["contact_vel_threshold"],
                            cache_dir=cache_dir, dtype=torch.float32, verbose=not args.fake_data)

    train_ds, val_ds = make_ds("train"), make_ds(tcfg.get("val_split", "val"))
    window = data_cfg["splits"]["train"]["window"]
    ctx, min_trans, max_trans = data_cfg["context_len"], data_cfg["min_trans"], data_cfg["max_trans"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=int(tcfg.get("val_batch_size", 64)), shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    if len(train_loader) == 0:
        raise ValueError(f"train 클립 수({len(train_ds)})가 batch_size({tcfg['batch_size']})보다 적습니다.")
    val_trans = [t for t in tcfg.get("val_trans", [max_trans]) if ctx + t + 2 <= data_cfg["splits"][tcfg.get("val_split", "val")]["window"]]

    # ----- 모델/최적화 -----
    model = build_context_transformer(cfg["model"], layout).to(device)
    base_lr = float(tcfg["base_lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
                                 weight_decay=float(tcfg.get("weight_decay", 0.0)))
    lr_factor = noam_lambda(int(tcfg["warmup_iters"]))
    amp_dtype = amp_dtype_from_str(tcfg.get("amp"), device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    accum = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = tcfg.get("grad_clip")
    max_iters = int(tcfg["max_iters"])

    iteration, epoch, best_val = 0, 0, math.inf
    if args.resume:
        ckpt_path = run_dir / f"{args.resume}.pth" if args.resume in ("last", "best") else Path(args.resume)
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        iteration, epoch, best_val = ckpt["iteration"], ckpt["epoch"], ckpt["best_val"]
        print(f"이어서 학습: {ckpt_path} (iteration {iteration})")

    resolved = {"env": env["name"], "data": {k: str(v) if isinstance(v, Path) else v for k, v in data_cfg.items()},
                **copy.deepcopy(cfg), "fake_data": args.fake_data, "stats": str(stats_path)}
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)

    print(f"device={device}, amp={amp_dtype}, 관절 {skel.num_joints}, D={layout.dim}, "
          f"입력 {model.d_in} → 출력 {model.d_out}, 파라미터 {count_parameters(model):,}")
    print(f"train 클립 {len(train_ds)} (window {window}), val 클립 {len(val_ds)}, "
          f"batch {tcfg['batch_size']} × accum {accum}, max_iters {max_iters}")
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
            # 배치마다 transition 길이/제약 프레임을 새로 뽑는다 (배치 안에서는 공통)
            spec = sample_transition(ctx, min_trans, max_trans, window,
                                     float(tcfg.get("midway_targets_amount", 0.0)),
                                     float(tcfg.get("midway_targets_p", 0.0)), rng)
            pb = prepare_batch(batch, skel, norm, ctx, data_cfg["forward_world_axis"], device)
            losses = context_forward(model, pb, spec, skel, layout, norm, loss_w, amp_dtype)
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
            print(f"it {iteration:7d} ep {epoch:3d} lr {lr:.2e} | loss {avg['loss']:.4f} "
                  f"rot {avg['rot']:.4f} pos {avg['pos']:.3f} smooth {avg['smooth']:.3f} | {dt * 1000:.0f}ms/it{mem}")
            for k, v in avg.items():
                log.writerow([iteration, epoch, "train", k, f"{v:.6g}"])
            log.writerow([iteration, epoch, "train", "lr", f"{lr:.6g}"])
            log_file.flush()
            running, t0 = {}, time.time()

        if iteration % int(tcfg.get("val_interval", 2000)) == 0 or iteration == max_iters:
            res = validate(model, val_loader, val_trans, data_cfg, skel, layout, norm, loss_w, device)
            print("  [val] " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if k.endswith(("_loss", "_gpos_l2"))))
            for k, v in res.items():
                log.writerow([iteration, epoch, "val", k, f"{v:.6g}"])
            log_file.flush()
            key = f"val{val_trans[-1]}_loss" if val_trans else None
            if key and res[key] < best_val:
                best_val = res[key]
                save_checkpoint(run_dir / "best.pth", model, optimizer, scaler, iteration, epoch, best_val, resolved)
                print(f"  best 갱신 ({key} = {best_val:.4f}) → best.pth")
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
