"""
[연구실] 테스트 샘플별 결과 영상: GT | PRED, 구간별 색 구분 (4단계, 교수님 미팅 피드백 [1][2])
==========================================================================================

이 스크립트가 하는 일
--------------------
1. 설정 읽기
     configs/viz/video.yaml   : 색, 구간 길이 (BEFORE 30, TARGET 유지 30), 배치, 출력 경로
     configs/eval/lengths.yaml: 체크포인트, 평가셋 (long: Subject 5, window = 10 + max(lengths) + 2, offset 40)
     <ckpt 폴더>/config.yaml  : 학습 때 쓴 설정 (context_len, 정규화 통계 경로 ...)
2. long 평가셋에서 클립을 고른다 (피드백 [2])
     - 기본: video.yaml 의 seed 로 n_samples 개를 무작위로 (앞에 BEFORE 프레임이 있는 클립 중에서)
     - long 평가셋은 길이와 무관하게 정해지므로 (window = 10 + max(lengths) + 2),
       --trans 를 바꿔도 같은 seed 면 "같은 시퀀스·같은 시작 프레임" 이 뽑힌다 → 길이별로 같은 샘플 비교
     - --index 로 직접 지정할 수도 있다
3. 생성: src.evaluation.inference.generate (= 학습 중 검증과 같은 순서, 후처리 식 23 포함)
     --stage detail 이면 두 단계 (Context + 후처리 → Detail, Detail 출력 후처리 없음) → outputs/videos/trans_<길이>_detail/
     같은 seed 면 Context 만 영상 (trans_<길이>/) 과 같은 클립이라 나란히 비교할 수 있다
4. 클립마다 mp4 하나: outputs/videos/trans_<길이>/<시퀀스>_<시작프레임>.mp4
     영상 = [BEFORE 원본 30] [CONTEXT 10] [GENERATED trans] [TARGET 1 + 유지 29], 모두 1배속 30fps
     GT 와 PRED 모두 구간 색 (회색/파랑/주황/빨강), GENERATED 동안 target 포즈 반투명 겹치기
5. 검증 출력: transition 평균 위치 오차 (cm), context/target 구간 PRED == GT (≈0),
   BEFORE 를 붙인 정답이 모델 창 정답과 같은 좌표인지 (≈0)
6. outputs/videos/trans_<길이>/samples.csv : 샘플 목록 (시퀀스, 시작 프레임, 클립 번호, 오차, 영상 경로)

<시작프레임> = 모델 창 첫 프레임 (첫 context 프레임) 의 원본 BVH 프레임 번호.

실행 예:
    python scripts/render_samples.py                             # seed 0 으로 5개, trans 30
    python scripts/render_samples.py --trans 150                 # 같은 5개 샘플, 생성 구간 5초
    python scripts/render_samples.py --n 10 --seed 1             # 설정 덮어쓰기
    python scripts/render_samples.py --index 1814                # 클립 직접 지정
    python scripts/render_samples.py --trans 30 --stage detail   # 두 단계 (Context → Detail) 같은 5개 샘플
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from matplotlib.animation import writers  # noqa: E402

from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    aligned_gt_with_before,
    build_eval_dataset,
    collate,
    generate,
    load_detail,
    load_run,
    max_trans_for_window,
    train_range_label,
)
from src.training.context import global_positions, gpos_l2  # noqa: E402
from src.visualization.skeleton_video import (  # noqa: E402
    VideoStyle,
    render_comparison_video,
    segment_labels,
    video_frame_index,
)

RANGE_EN = {"학습 범위 안": "in training range", "학습 범위 밖": "OUT of training range"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trans", type=int, default=30, help="transition 길이 (프레임, 30fps)")
    ap.add_argument("--index", type=int, nargs="+", default=None,
                    help="long 평가셋 클립 번호 직접 지정 (주면 --n/--seed 무시)")
    ap.add_argument("--n", type=int, default=None, help="샘플 수 (기본: video.yaml 의 n_samples)")
    ap.add_argument("--seed", type=int, default=None, help="샘플 고르기 seed (기본: video.yaml 의 seed)")
    ap.add_argument("--ckpt", type=Path, default=None, help="기본: configs/eval/lengths.yaml 의 ckpt")
    ap.add_argument("--stage", choices=["context", "detail"], default="context",
                    help="context = Context 만, detail = 두 단계 (Context → Detail, 3단계)")
    ap.add_argument("--detail-ckpt", type=Path, default=None, help="기본: configs/eval/lengths.yaml 의 detail_ckpt")
    ap.add_argument("--video-config", default="video", help="configs/viz/<이름>.yaml 또는 .yaml 경로")
    ap.add_argument("--eval-config", default="lengths", help="configs/eval/<이름>.yaml 또는 .yaml 경로")
    ap.add_argument("--postprocess", choices=["on", "off"], default=None,
                    help="후처리 (식 23) 켜기/끄기 (기본: video.yaml 의 postprocess)")
    ap.add_argument("--env", default=None, help="configs/env/<env>.yaml (기본: 학습 config.yaml 의 env)")
    ap.add_argument("--device", default=None, help="예) cpu, cuda:0 (기본: env 설정)")
    return ap.parse_args()


def select_samples(ds, n: int, seed: int, before: int) -> list[int]:
    """long 평가셋에서 영상용 클립 n 개를 고른다 (피드백 [2]).

    후보 = 앞에 BEFORE 프레임 (before 개) 이 같은 BVH 안에 있는 클립 (시작 프레임 ≥ before).
    long 평가셋의 클립 목록은 transition 길이와 무관하므로, 같은 seed 면 어떤 길이에서도 같은 클립이 뽑힌다.
    입력:  ds (long 평가셋), n, seed, before
    출력:  클립 번호 목록 (오름차순)
    """
    cand = [i for i in range(len(ds)) if ds.locate(i)[1] >= before]
    if not cand:
        raise ValueError(f"앞에 {before} 프레임이 있는 클립이 없습니다.")
    return sorted(random.Random(seed).sample(cand, min(n, len(cand))))


def write_manifest(results: list[dict], path: Path) -> None:
    """샘플 목록 CSV (같은 trans 폴더에 samples.csv). results: render_clips 출력."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


def boundary_jumps(gpos: torch.Tensor, gpos_gt: torch.Tensor, ctx: int, target: int) -> tuple[Tensor, Tensor]:
    """경계에서 "끊김" 크기 (cm): 생성 쪽 프레임 간 이동량 − 정답 쪽 프레임 간 이동량 (관절 평균).

    start : 마지막 context (ctx−1) → 첫 생성 프레임 (ctx)
    end   : 마지막 생성 프레임 (target−1) → target
    정답도 그 사이에 움직이므로 (달리기 ≈ 5 cm/프레임) 정답 이동량을 빼서 "추가로 튄 거리" 만 본다. ≈0 이면 자연스럽게 이어짐.
    입력:  gpos, gpos_gt : (B, L, J, 3)
    출력:  (start (B,), end (B,))
    """
    def step(g, a):                                            # (B,) 프레임 a → a+1 관절 평균 이동 거리
        return (g[:, a + 1] - g[:, a]).norm(dim=-1).mean(-1)
    return step(gpos, ctx - 1) - step(gpos_gt, ctx - 1), step(gpos, target - 1) - step(gpos_gt, target - 1)


def render_clips(run, ds, idx: list[int], trans: int, vcfg: dict, style: VideoStyle,
                 use_postprocess: bool = True, det=None) -> list[dict]:
    """클립 번호 목록 → 영상 저장 + 검증 값 출력. 출력: 클립별 요약 dict 목록.

    det : load_detail 결과. 주면 두 단계 결과 (Context 후처리는 Detail 학습 설정, Detail 출력 후처리 없음) 를
          그리고 use_postprocess 는 쓰지 않는다. 같은 클립의 Context 만 결과 오차도 함께 출력한다.

    좌표는 모두 학습용 정렬 좌표 (모델 창 마지막 context 프레임 기준 루트 XZ=0, +X 방향, cm).
    """
    ctx, fps = run.ctx, style.fps
    before, hold = int(vcfg["before_frames"]), int(vcfg["target_hold_frames"])
    L = ctx + trans + 2                                         # 모델 창 (target 다음 프레임 포함)

    # 1) 생성 (모델 창)  gen.gpos, gen.pb.gpos_gt : (B, L, J, 3)
    gen = generate(run, collate(ds, idx), trans, use_postprocess=use_postprocess, detail=det)
    ctx_gpos = None
    if det is not None:                                         # 비교용: Detail 입력이 된 Context 단계 결과
        ctx_gpos = global_positions(run.norm.denormalize(gen.y_ctx), gen.pb.offsets, run.skel, run.layout)
    jump_s, jump_e = boundary_jumps(gen.gpos, gen.pb.gpos_gt, ctx, gen.spec.target_idx)
    # 2) BEFORE 를 붙인 정답 (같은 정렬)  ext.gpos_gt : (B, before + L, J, 3)
    ext = aligned_gt_with_before(run, ds, idx, before, L)
    gt_full = ext.gpos_gt
    align_err = (gt_full[:, before:] - gen.pb.gpos_gt).norm(dim=-1).amax(dim=(1, 2))      # (B,)
    # 3) PRED = 정답에서 transition 구간만 생성으로 바꿈 (context, target 은 모델 입력이므로 정답 그대로)
    s = gen.spec.seq_slice
    pred_full = gt_full.clone()
    pred_full[:, before + s.start:before + s.stop] = gen.gpos[:, s]

    fidx = video_frame_index(before, ctx, trans, hold)        # 영상 프레임 → 늘린 시퀀스 프레임
    segs = segment_labels(before, ctx, trans, hold)
    head = run.skel.names.index("Head") if "Head" in run.skel.names else run.skel.num_joints - 1
    rng_label = train_range_label(trans, run.train_range)
    tgt = before + gen.spec.target_idx                         # 늘린 시퀀스에서 target 프레임 번호
    if det is not None:
        out_tmpl = vcfg["out_path_detail"]
        stage_txt = f"two-stage {det.name}/{det.ckpt_path.name} it {det.ckpt['iteration']}"
    else:
        out_tmpl = vcfg["out_path"] if use_postprocess else vcfg["out_path_postprocess_off"]
        stage_txt = f"context only, post-proc {'on' if use_postprocess else 'off'}"

    results = []
    for b, i in enumerate(idx):
        fi, start = ds.locate(i)
        name = ds.files[fi].stem
        frame0 = start + ds.start_frame                         # 모델 창 첫 프레임의 원본 BVH 프레임 번호
        one = slice(b, b + 1)
        l2 = float(gpos_l2(gen.gpos[one], gen.pb.gpos_gt[one], s))
        known = [*range(ctx), gen.spec.target_idx]
        known_err = float((gen.gpos[b, known] - gen.pb.gpos_gt[b, known]).norm(dim=-1).max())

        gt_v = gt_full[b, fidx].cpu().numpy()                    # (F, J, 3)
        pred_v = pred_full[b, fidx].cpu().numpy()
        err = np.linalg.norm(pred_v - gt_v, axis=-1).mean(-1)   # (F,) 프레임별 평균 관절 오차 (cm)
        info = []
        for f, (k, seg) in enumerate(zip(fidx, segs)):
            bvh_f = frame0 - before + int(k)
            extra = {"before": "", "context": f"ctx {k - before + 1}/{ctx}",
                     "generated": f"gen {k - before - ctx + 1}/{trans}",
                     "target": "target" if f == len(fidx) - hold else f"hold {f - (len(fidx) - hold)}/{hold - 1}"}[seg]
            info.append(f"frame {f:3d}/{len(fidx) - 1}  BVH #{bvh_f:<5d} {extra:>12s}")
        title = (f"{name}  start {frame0}  |  trans {trans} ({trans / fps:.1f}s, {RANGE_EN[rng_label]} "
                 f"{run.train_range[0]}-{run.train_range[1]})  |  transition mean err {l2:.2f} cm  |  "
                 f"{stage_txt}  |  {run.name}/{run.ckpt_path.name} it {run.ckpt['iteration']}")
        path = resolve_path(out_tmpl.format(trans=trans, seq=name, start=frame0))

        print(f"\n[{b + 1}/{len(idx)}] {name} 모델 창 BVH 프레임 {frame0}–{frame0 + gen.spec.target_idx} "
              f"(long 평가셋 클립 #{i}), trans {trans} ({rng_label})")
        print(f"  transition 평균 위치 오차 (gpos_l2): {l2:.4f} cm   "
              f"경계 점프 (정답 대비 추가 이동): context→생성 {float(jump_s[b]):+.1f} cm, "
              f"생성→target {float(jump_e[b]):+.1f} cm")
        ctx_l2 = None
        if ctx_gpos is not None:
            ctx_l2 = float(gpos_l2(ctx_gpos[one], gen.pb.gpos_gt[one], s))
            print(f"  (비교) 같은 클립 Context 단계 결과 오차: {ctx_l2:.4f} cm → Detail 후 {l2:.4f} cm")
        print(f"  context·target 프레임 PRED vs GT 최대 오차: {known_err:.2e} cm → "
              f"{'OK' if known_err < 1e-3 else '경고: 입력 프레임이 정답과 다름!'}")
        print(f"  BEFORE 붙인 정답 vs 모델 창 정답 (정렬 일치) 최대 오차: {float(align_err[b]):.2e} cm → "
              f"{'OK' if align_err[b] < 1e-3 else '경고: 정렬 변환이 다름!'}")
        tic = time.time()
        layout = render_comparison_video(path, gt_v, pred_v, segs, gt_full[b, tgt].cpu().numpy(),
                                         run.skel.parents, head, style, title, info, err,
                                         pred_label="PRED (Context + Detail)" if det is not None
                                         else "PRED (Context Transformer)")
        print(f"  저장: {path}  ({len(fidx)} 프레임 = {len(fidx) / fps:.1f}초, 배치 {layout}, "
              f"{time.time() - tic:.0f}초)")
        results.append({"seq": name, "start": frame0, "index": i, "trans": trans,
                        "train_range": rng_label, "stage": "detail" if det is not None else "context",
                        "postprocess": use_postprocess if det is None else "ctx" if det.ctx_postprocess else "off",
                        "gpos_l2": round(l2, 4),
                        "ctx_gpos_l2": round(ctx_l2, 4) if ctx_l2 is not None else "",
                        "jump_start_cm": round(float(jump_s[b]), 2), "jump_end_cm": round(float(jump_e[b]), 2),
                        "layout": layout, "path": str(path)})
    return results


def main() -> None:
    args = parse_args()
    if not writers.is_available("ffmpeg"):
        raise FileNotFoundError("ffmpeg 가 없습니다. sudo apt install ffmpeg")
    vcfg = load_named_config("viz", args.video_config)
    ecfg = load_named_config("eval", args.eval_config)
    style = VideoStyle.from_cfg(vcfg)
    run = load_run(args.ckpt or ecfg["ckpt"], env_name=args.env, device=args.device)
    ds, sp = build_eval_dataset(run, ecfg, "long")
    print(f"체크포인트 {run.ckpt_path} (iteration {run.ckpt['iteration']}), device={run.device}, "
          f"long 평가셋 {len(ds)} 클립 (window {sp['window']}, offset {sp['offset']})")
    if args.trans > max_trans_for_window(sp["window"], run.ctx):
        raise ValueError(f"trans {args.trans} > long 평가셋 최대 {max_trans_for_window(sp['window'], run.ctx)}. "
                         f"configs/eval/lengths.yaml 의 lengths 에 더 긴 길이를 추가하세요 (window 가 자동으로 커짐).")
    if args.index:
        idx = list(args.index)
        print(f"클립 직접 지정: {idx}")
    else:
        n = args.n if args.n is not None else int(vcfg["n_samples"])
        seed = args.seed if args.seed is not None else int(vcfg["seed"])
        idx = select_samples(ds, n, seed, int(vcfg["before_frames"]))
        print(f"seed {seed} 로 {len(idx)}개 선택 (길이와 무관하게 같은 클립): {idx}")
    pp = bool(vcfg["postprocess"]) if args.postprocess is None else args.postprocess == "on"
    det = None
    if args.stage == "detail":
        det = load_detail(args.detail_ckpt or ecfg["detail_ckpt"], run)
        print(f"두 단계: Detail {det.ckpt_path} (iteration {det.ckpt['iteration']}). "
              f"--postprocess 는 쓰지 않음 (Context 후처리 {'켬' if det.ctx_postprocess else '끔'}, Detail 출력 후처리 없음)")
    with torch.no_grad():
        results = render_clips(run, ds, idx, args.trans, vcfg, style, use_postprocess=pp, det=det)
    manifest = Path(results[0]["path"]).parent / "samples.csv"
    write_manifest(results, manifest)
    print(f"\n{len(results)}개 평균 transition 오차 (trans {args.trans}): "
          f"{np.mean([r['gpos_l2'] for r in results]):.4f} cm   샘플 목록: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
