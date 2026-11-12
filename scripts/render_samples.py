"""
[연구실] 테스트 샘플별 결과 영상: GT | PRED, 구간별 색 구분 (4단계, 교수님 미팅 피드백 [1][2])
==========================================================================================

이 스크립트가 하는 일
--------------------
1. 설정 읽기
     configs/viz/video.yaml   : 색, 구간 길이 (BEFORE 30, TARGET 유지 30), 배치, 출력 경로
     configs/eval/lengths.yaml: 체크포인트, 평가셋 (long: Subject 5, window = 10 + max(lengths) + 2, offset 40)
     <ckpt 폴더>/config.yaml  : 학습 때 쓴 설정 (context_len, 정규화 통계 경로 ...)
2. long 평가셋에서 클립을 고른다 (--index 로 직접 지정)
3. 생성: src.evaluation.inference.generate (= 학습 중 검증과 같은 순서, 후처리 식 23 포함)
4. 클립마다 mp4 하나: outputs/videos/trans_<길이>/<시퀀스>_<시작프레임>.mp4
     영상 = [BEFORE 원본 30] [CONTEXT 10] [GENERATED trans] [TARGET 1 + 유지 29], 모두 1배속 30fps
     GT 와 PRED 모두 구간 색 (회색/파랑/주황/빨강), GENERATED 동안 target 포즈 반투명 겹치기
5. 검증 출력: transition 평균 위치 오차 (cm), context/target 구간 PRED == GT (≈0),
   BEFORE 를 붙인 정답이 모델 창 정답과 같은 좌표인지 (≈0)

<시작프레임> = 모델 창 첫 프레임 (첫 context 프레임) 의 원본 BVH 프레임 번호.

실행 예:
    python scripts/render_samples.py --index 120                 # long 평가셋 클립 120, trans 30
    python scripts/render_samples.py --index 120 --trans 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import writers  # noqa: E402

from src.config import load_named_config, resolve_path  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    aligned_gt_with_before,
    build_eval_dataset,
    collate,
    generate,
    load_run,
    max_trans_for_window,
    train_range_label,
)
from src.training.context import gpos_l2  # noqa: E402
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
    ap.add_argument("--index", type=int, nargs="+", required=True, help="long 평가셋 클립 번호")
    ap.add_argument("--ckpt", type=Path, default=None, help="기본: configs/eval/lengths.yaml 의 ckpt")
    ap.add_argument("--video-config", default="video", help="configs/viz/<이름>.yaml 또는 .yaml 경로")
    ap.add_argument("--eval-config", default="lengths", help="configs/eval/<이름>.yaml 또는 .yaml 경로")
    ap.add_argument("--env", default=None, help="configs/env/<env>.yaml (기본: 학습 config.yaml 의 env)")
    ap.add_argument("--device", default=None, help="예) cpu, cuda:0 (기본: env 설정)")
    return ap.parse_args()


def render_clips(run, ds, idx: list[int], trans: int, vcfg: dict, style: VideoStyle) -> list[dict]:
    """클립 번호 목록 → 영상 저장 + 검증 값 출력. 출력: 클립별 요약 dict 목록.

    좌표는 모두 학습용 정렬 좌표 (모델 창 마지막 context 프레임 기준 루트 XZ=0, +X 방향, cm).
    """
    ctx, fps = run.ctx, style.fps
    before, hold = int(vcfg["before_frames"]), int(vcfg["target_hold_frames"])
    L = ctx + trans + 2                                         # 모델 창 (target 다음 프레임 포함)

    # 1) 생성 (모델 창)  gen.gpos, gen.pb.gpos_gt : (B, L, J, 3)
    gen = generate(run, collate(ds, idx), trans)
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
    out_tmpl = vcfg["out_path"]

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
                 f"{run.name}/{run.ckpt_path.name} it {run.ckpt['iteration']}")
        path = resolve_path(out_tmpl.format(trans=trans, seq=name, start=frame0))

        print(f"\n[{b + 1}/{len(idx)}] {name} 모델 창 BVH 프레임 {frame0}–{frame0 + gen.spec.target_idx} "
              f"(long 평가셋 클립 #{i}), trans {trans} ({rng_label})")
        print(f"  transition 평균 위치 오차 (gpos_l2): {l2:.4f} cm")
        print(f"  context·target 프레임 PRED vs GT 최대 오차: {known_err:.2e} cm → "
              f"{'OK' if known_err < 1e-3 else '경고: 입력 프레임이 정답과 다름!'}")
        print(f"  BEFORE 붙인 정답 vs 모델 창 정답 (정렬 일치) 최대 오차: {float(align_err[b]):.2e} cm → "
              f"{'OK' if align_err[b] < 1e-3 else '경고: 정렬 변환이 다름!'}")
        tic = time.time()
        layout = render_comparison_video(path, gt_v, pred_v, segs, gt_full[b, tgt].cpu().numpy(),
                                         run.skel.parents, head, style, title, info, err)
        print(f"  저장: {path}  ({len(fidx)} 프레임 = {len(fidx) / fps:.1f}초, 배치 {layout}, "
              f"{time.time() - tic:.0f}초)")
        results.append({"seq": name, "start": frame0, "index": i, "trans": trans, "gpos_l2": l2,
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
    with torch.no_grad():
        render_clips(run, ds, list(args.index), args.trans, vcfg, style)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
