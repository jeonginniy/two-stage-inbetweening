"""
[연구실] Detail Transformer 학습 (3단계, 논문 3.4 · 3.5 · 3.6)
============================================================

이 스크립트가 하는 일
--------------------
1. 설정 읽기: configs/env/<env>.yaml (경로, device), configs/data/lafan1.yaml (window, context_len 등),
   configs/model/detail.yaml (모델 크기, batch size, 학습률 0.025, 손실 가중치, 고정할 Context 체크포인트)
2. 고정할 Context Transformer 불러오기: <ctx 폴더>/config.yaml + best.pth (읽기만 함, 논문 3.6.1 "freeze")
   정규화 통계도 Context 와 같은 파일을 쓴다 (docs/assumptions.md G8)
3. LaFAN1 train 분할 (Subject 1–4, window 42, offset 1) 과 val 분할 (Subject 5)
4. 매 iteration:
     transition 길이 [min_trans, max_trans] 무작위 (+ 제약 프레임)
     → 고정된 Context (+ 후처리 식 23) → Detail 입력 (제약 프레임 = 정답) ⊕ m_in → Detail
     → L_rot + 0.02·L_pos + 0.05·L_c + 0.05·L_foot (식 22) → Adam (Noam 학습률)
5. val_interval 마다 val: trans 마다 손실, 두 단계 결과의 위치 오차 (cm), 같은 클립의 Context 만 결과 오차 (비교용),
   발 접촉 정확도. best 체크포인트 = val trans (마지막 값) 손실 최소
   결과 폴더: <experiments_dir>/<run_name>/  (config.yaml, log.csv, last.pth, best.pth)

실행 예:
    python scripts/train_detail.py --run-name det_base
    python scripts/train_detail.py --run-name det_smoke --max-iters 300 --set train.log_interval=50 train.val_interval=300
    python scripts/train_detail.py --run-name det_base --resume last          # 이어서 학습
    python scripts/train_detail.py --run-name det_cont --resume experiments/det_base/last.pth --max-iters 1000000
                                    # 다른 폴더에서 이어서 (det_base 는 읽기만, 결과는 det_cont)
    python scripts/train_detail.py --context-ckpt experiments/ctx_base/last.pth --run-name det_from_last

    # 데이터 없이 스크립트가 끝까지 도는지만 확인 (가짜 BVH + 무작위 초기화 Context, 작은 모델, 결과는 의미 없음):
    python scripts/train_detail.py --env laptop --fake-data --max-iters 20 --batch-size 8 \\
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
sys.path.insert(0, str(Path(__file__).resolve().parent))       # train_context (가짜 데이터 도우미, --set 처리)

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import (  # noqa: E402
    load_data_config,
    load_env_config,
    load_model_config,
    load_yaml,
    require_dir,
    resolve_device,
    resolve_path,
)
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import FeatureLayout, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402
from src.models.context_transformer import build_context_transformer, count_parameters  # noqa: E402
from src.models.detail_transformer import build_detail_transformer, check_frozen, freeze  # noqa: E402
from src.training.context import global_positions, gpos_l2, prepare_batch  # noqa: E402
from src.training.context_inputs import TransitionSpec, sample_transition  # noqa: E402
from src.training.detail import contact_accuracy, detail_forward, infer_two_stage  # noqa: E402
from src.training.utils import (  # noqa: E402
    amp_dtype_from_str,
    infinite_batches,
    load_checkpoint,
    noam_lambda,
    save_checkpoint,
    set_seed,
)
from train_context import apply_overrides, compute_fake_stats, make_fake_lafan  # noqa: E402

TIME_KEYS = ("root_pos", "local_rot", "contacts")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--config", default="detail", help="configs/model/<이름>.yaml 또는 yaml 경로 (기본 detail)")
    ap.add_argument("--run-name", default=None, help="결과 폴더 이름 (기본: detail_<날짜시간>)")
    ap.add_argument("--context-ckpt", type=Path, default=None,
                    help="고정할 Context 체크포인트 (기본: detail.yaml 의 context.ckpt)")
    ap.add_argument("--resume", default=None, help="체크포인트 경로, 또는 'last' / 'best' (run 폴더 안)")
    ap.add_argument("--batch-size", type=int, default=None, help="train.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="train.grad_accum_steps")
    ap.add_argument("--max-iters", type=int, default=None, help="train.max_iters")
    ap.add_argument("--amp", default=None, help="train.amp (none | fp16 | bf16)")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader 워커 수 (기본: env 설정)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="임의 설정 덮어쓰기. 예) model.n_layer=4 train.base_lr=0.01")
    ap.add_argument("--fake-data", action="store_true",
                    help="LaFAN1 없이 가짜 BVH + 무작위 Context 로 스크립트 동작만 확인 (결과는 의미 없음)")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# 고정할 Context Transformer
# -----------------------------------------------------------------------------
def load_frozen_context(ckpt: Path, data_cfg: dict, layout: FeatureLayout, device) -> tuple[torch.nn.Module, dict]:
    """Context 체크포인트 + 같은 폴더의 config.yaml → 고정된 모델, 그 설정.

    data_cfg 의 context_len / forward_world_axis / 스켈레톤이 Context 학습 때와 같은지 확인한다
    (다르면 Context 가 학습 때와 다른 입력을 받게 됨).
    """
    ckpt_path = resolve_path(ckpt)
    cfg_path = ckpt_path.parent / "config.yaml"
    if not ckpt_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(f"Context 체크포인트 {ckpt_path} 또는 {cfg_path} 가 없습니다.")
    ctx_cfg = load_yaml(cfg_path)
    if ctx_cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 Context 입니다.")
    for key in ("context_len", "forward_world_axis"):
        if ctx_cfg["data"][key] != data_cfg[key]:
            raise ValueError(f"{key}: Context 학습 때 {ctx_cfg['data'][key]} ≠ 지금 {data_cfg[key]}")
    if Path(ctx_cfg["data"]["skeleton"]).name != Path(data_cfg["skeleton"]).name:
        raise ValueError("Context 학습 때와 스켈레톤 파일이 다릅니다.")
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = build_context_transformer(ctx_cfg["model"], layout).to(device)
    model.load_state_dict(ck["model"])
    print(f"고정된 Context: {ckpt_path} (iteration {ck['iteration']}, best_val {ck['best_val']:.4f})")
    return freeze(model), {**ctx_cfg, "ckpt": str(ckpt_path), "iteration": int(ck["iteration"])}


# -----------------------------------------------------------------------------
# 검증
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(det, ctx, loader, trans_list, data_cfg, skel, layout, norm, loss_w, ctx_post, device) -> dict[str, float]:
    """val 분할에서 transition 길이별
        val{t}_loss/rot/pos/contact/foot : 학습과 같은 손실
        val{t}_gpos_l2      : 두 단계 결과 (Detail, 후처리 없음) 의 전역 위치 평균 L2 (cm)
        val{t}_ctx_gpos_l2  : 같은 클립의 Context 만 결과 (+후처리) 오차 (cm). Detail 이 얼마나 고치는지 비교용
        val{t}_contact_acc  : 발 접촉 정확도 (ĉ > 0.5 vs 정답)
    """
    sums: dict[str, float] = {}
    n = 0
    det.eval()
    ctx_len = data_cfg["context_len"]
    for batch in loader:
        n += 1
        for t in trans_list:
            L = ctx_len + t + 2
            cut = {k: (v[:, :L] if k in TIME_KEYS else v) for k, v in batch.items()}
            pb = prepare_batch(cut, skel, norm, ctx_len, data_cfg["forward_world_axis"], device)
            contacts = cut["contacts"].to(device)
            spec = TransitionSpec(ctx_len, ctx_len + t)
            losses = detail_forward(det, ctx, pb, contacts, spec, skel, layout, norm, loss_w, ctx_post)
            y_ctx, y_det, prob = infer_two_stage(ctx, det, pb.x_gt, spec, layout, ctx_post, det_postprocess=False)
            g_det = global_positions(norm.denormalize(y_det), pb.offsets, skel, layout)
            g_ctx = global_positions(norm.denormalize(y_ctx), pb.offsets, skel, layout)
            vals = {**{k: float(v) for k, v in losses.items()},
                    "gpos_l2": float(gpos_l2(g_det, pb.gpos_gt, spec.seq_slice)),
                    "ctx_gpos_l2": float(gpos_l2(g_ctx, pb.gpos_gt, spec.seq_slice)),
                    "contact_acc": float(contact_accuracy(prob, contacts, spec))}
            for k, v in vals.items():
                sums[f"val{t}_{k}"] = sums.get(f"val{t}_{k}", 0.0) + v
    det.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    env = load_env_config(args.env)
    data_cfg = load_data_config()
    cfg = load_model_config(args.config)

    shortcuts = {"batch_size": args.batch_size, "grad_accum_steps": args.grad_accum,
                 "max_iters": args.max_iters, "amp": args.amp}
    for k, v in shortcuts.items():
        if v is not None:
            cfg["train"][k] = None if v in ("none", "null") else v
    apply_overrides(cfg, args.set)
    if args.context_ckpt:
        cfg["context"]["ckpt"] = str(args.context_ckpt)
    tcfg, loss_w = cfg["train"], cfg["loss"]
    ctx_post = bool(cfg["context"].get("postprocess", True))

    device = resolve_device(env)
    num_workers = env["num_workers"] if args.num_workers is None else args.num_workers
    set_seed(int(tcfg.get("seed", 0)))
    rng = random.Random(int(tcfg.get("seed", 0)))

    # ----- 결과 폴더 -----
    if args.resume in ("last", "best") and not args.run_name:
        raise ValueError("--resume last/best 는 --run-name 과 함께 써야 합니다 (어느 폴더인지).")
    exp_root = env["paths"].get("experiments_dir") or resolve_path("experiments")
    run_name = args.run_name or (Path(args.resume).parent.name if args.resume and args.resume not in ("last", "best")
                                 else None) or time.strftime("detail_%Y%m%d_%H%M%S")
    run_dir = exp_root / run_name
    ctx_dir = resolve_path(cfg["context"]["ckpt"]).parent
    if not args.fake_data and run_dir.resolve() == ctx_dir.resolve():
        raise ValueError(f"결과 폴더 {run_dir} 가 Context 폴더와 같습니다. --run-name 을 바꾸세요 (Context 결과 보호).")
    if not args.resume and (run_dir / "last.pth").exists():
        raise ValueError(f"{run_dir} 에 이미 학습 결과가 있습니다. --resume last 로 이어서 하거나 --run-name 을 바꾸세요.")
    # 다른 폴더의 체크포인트에서 이어가기 (예: det_base/last.pth → det_cont). 원래 폴더는 읽기만 한다.
    resume_external = bool(args.resume) and args.resume not in ("last", "best") and \
        resolve_path(args.resume).parent.resolve() != run_dir.resolve()
    if resume_external and (run_dir / "last.pth").exists():
        raise ValueError(f"{run_dir} 에 이미 학습 결과가 있습니다. 다른 폴더에서 이어가려면 새 --run-name 을 쓰세요.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- 데이터 + 고정된 Context -----
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    n_contact = len(skel.contact_joints)
    cache_dir = env["paths"]["cache_dir"]
    if args.fake_data:
        bvh_dir = make_fake_lafan(run_dir / "fake_lafan1")
        stats_path = compute_fake_stats(bvh_dir, data_cfg, skel, run_dir / "fake_stats.npz")
        cache_dir = None
        torch.manual_seed(0)
        ctx = freeze(build_context_transformer(cfg["model"], layout).to(device))      # 무작위 초기화 (동작 확인용)
        ctx_info = {"ckpt": None, "fake": True}
        print("[주의] --fake-data: 가짜 BVH + 학습 안 된 Context 로 동작만 확인합니다. 결과는 의미가 없습니다.")
    else:
        bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
        ctx, ctx_info = load_frozen_context(Path(cfg["context"]["ckpt"]), data_cfg, layout, device)
        stats_path = resolve_path(ctx_info["stats"])                                   # Context 와 같은 통계 (G8)
    norm = ZScoreNormalizer.load(stats_path).to(device=device, dtype=torch.float32)
    if norm.dim != layout.dim:
        raise ValueError(f"통계 차원 {norm.dim} 이 스켈레톤으로 계산한 D={layout.dim} 과 다릅니다.")

    def make_ds(split: str) -> LafanDataset:
        sp = data_cfg["splits"][split]
        return LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel,
                            start_frame=sp["start_frame"], contact_threshold=data_cfg["contact_vel_threshold"],
                            cache_dir=cache_dir, dtype=torch.float32, verbose=not args.fake_data)

    val_split = tcfg.get("val_split", "val")
    train_ds, val_ds = make_ds("train"), make_ds(val_split)
    window = data_cfg["splits"]["train"]["window"]
    ctx_len, min_trans, max_trans = data_cfg["context_len"], data_cfg["min_trans"], data_cfg["max_trans"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=int(tcfg.get("val_batch_size", 64)), shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    if len(train_loader) == 0:
        raise ValueError(f"train 클립 수({len(train_ds)})가 batch_size({tcfg['batch_size']})보다 적습니다.")
    val_trans = [t for t in tcfg.get("val_trans", [max_trans])
                 if ctx_len + t + 2 <= data_cfg["splits"][val_split]["window"]]

    # ----- 모델/최적화 -----
    det = build_detail_transformer(cfg["model"], layout, n_contact).to(device)
    base_lr = float(tcfg["base_lr"])
    optimizer = torch.optim.Adam(det.parameters(), lr=base_lr, betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
                                 weight_decay=float(tcfg.get("weight_decay", 0.0)))
    lr_factor = noam_lambda(int(tcfg["warmup_iters"]))
    amp_dtype = amp_dtype_from_str(tcfg.get("amp"), device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    accum = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = tcfg.get("grad_clip")
    max_iters = int(tcfg["max_iters"])

    iteration, epoch, best_val = 0, 0, math.inf
    if args.resume:
        ckpt_path = run_dir / f"{args.resume}.pth" if args.resume in ("last", "best") else resolve_path(args.resume)
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        prev_ctx = (ckpt.get("config") or {}).get("context_run", {}).get("ckpt")
        if not args.fake_data and prev_ctx and resolve_path(prev_ctx).resolve() != resolve_path(cfg["context"]["ckpt"]).resolve():
            raise ValueError(f"{ckpt_path} 는 Context {prev_ctx} 위에서 학습됐습니다 (지금 {cfg['context']['ckpt']}).")
        det.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        iteration, epoch, best_val = ckpt["iteration"], ckpt["epoch"], ckpt["best_val"]
        print(f"이어서 학습: {ckpt_path} (iteration {iteration})")
        if resume_external:
            # 새 폴더의 best.pth 는 "이 폴더에서 학습한 구간" 중 최고로 둔다 (첫 검증에서 반드시 저장).
            print(f"  다른 폴더에서 이어감 → best 기준 초기화 (원래 폴더 best val {best_val:.4f})")
            best_val = math.inf

    resolved = {"env": env["name"], "data": {k: str(v) if isinstance(v, Path) else v for k, v in data_cfg.items()},
                **copy.deepcopy(cfg), "context_run": {k: ctx_info.get(k) for k in ("ckpt", "iteration", "stats")},
                "fake_data": args.fake_data, "stats": str(stats_path)}
    if resume_external:
        resolved["resumed_from"] = {"ckpt": str(ckpt_path), "iteration": int(ckpt["iteration"])}
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)

    print(f"device={device}, amp={amp_dtype}, 관절 {skel.num_joints}, D={layout.dim}, 접촉 {n_contact}, "
          f"입력 {det.d_in} → 출력 {det.d_out} (포즈 {det.d_pose} + 접촉 {det.n_contact}), "
          f"파라미터 {count_parameters(det):,} (Context 고정 {sum(p.numel() for p in ctx.parameters()):,})")
    print(f"train 클립 {len(train_ds)} (window {window}), val 클립 {len(val_ds)}, "
          f"batch {tcfg['batch_size']} × accum {accum}, max_iters {max_iters}, base_lr {base_lr}")
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
    det.train()
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
            spec = sample_transition(ctx_len, min_trans, max_trans, window,
                                     float(tcfg.get("midway_targets_amount", 0.0)),
                                     float(tcfg.get("midway_targets_p", 0.0)), rng)
            pb = prepare_batch(batch, skel, norm, ctx_len, data_cfg["forward_world_axis"], device)
            contacts = batch["contacts"].to(device, non_blocking=True)
            losses = detail_forward(det, ctx, pb, contacts, spec, skel, layout, norm, loss_w, ctx_post, amp_dtype)
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
            torch.nn.utils.clip_grad_norm_(det.parameters(), float(grad_clip))
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
                  f"pos {avg['pos']:.3f} contact {avg['contact']:.3f} foot {avg['foot']:.3f} | "
                  f"{dt * 1000:.0f}ms/it{mem}")
            for k, v in avg.items():
                log.writerow([iteration, epoch, "train", k, f"{v:.6g}"])
            log.writerow([iteration, epoch, "train", "lr", f"{lr:.6g}"])
            log_file.flush()
            running, t0 = {}, time.time()

        if iteration % int(tcfg.get("val_interval", 2000)) == 0 or iteration == max_iters:
            check_frozen(ctx)
            res = validate(det, ctx, val_loader, val_trans, data_cfg, skel, layout, norm, loss_w, ctx_post, device)
            print("  [val] " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()
                                         if k.endswith(("_loss", "_gpos_l2", "_contact_acc"))))
            for k, v in res.items():
                log.writerow([iteration, epoch, "val", k, f"{v:.6g}"])
            log_file.flush()
            key = f"val{val_trans[-1]}_loss" if val_trans else None
            if key and res[key] < best_val:
                best_val = res[key]
                save_checkpoint(run_dir / "best.pth", det, optimizer, scaler, iteration, epoch, best_val, resolved)
                print(f"  best 갱신 ({key} = {best_val:.4f}) → best.pth")
            t0 = time.time()

        if iteration % int(tcfg.get("ckpt_interval", 5000)) == 0 or iteration == max_iters:
            save_checkpoint(run_dir / "last.pth", det, optimizer, scaler, iteration, epoch, best_val, resolved)

    log_file.close()
    print(f"끝: iteration {iteration}, best val {best_val:.4f}. 결과: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
