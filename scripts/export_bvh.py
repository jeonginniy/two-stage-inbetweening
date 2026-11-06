"""
[연구실] Context Transformer 결과를 BVH 로 저장 (원본 / 생성 / SLERP)
==================================================================

이 스크립트가 하는 일
--------------------
1. scripts/render_compare.py 와 같은 코드로 클립을 고르고 생성한다
   (load_run → select_clips → infer_clips; 생성은 validate() 와 같은 eval_transition).
   --total_frames (기본 300) 길이로, transition 이 가운데 오도록 앞뒤를 같은 BVH 원본으로 채운다.
2. 클립마다 BVH 세 개 저장:  <out>/<파일>_f<모델창 첫 프레임>_trans<N>/{gt,pred,slerp}.bvh  + meta.json
     gt    : 데이터셋 원본 (월드 좌표 그대로)
     pred  : transition 구간만 모델 출력, 나머지는 원본
     slerp : transition 구간만 SLERP 기준선, 나머지는 원본
   - 계층 구조, OFFSET, End Site, 회전 순서, Frame Time 은 LaFAN1 원본 BVH 파일에서 그대로 가져온다.
   - 모델 출력 (정규화된 6D + 루트) → 역정규화 → 로컬 회전행렬 → 오일러 (원본과 같은 순서) 로 쓴다.
   - 학습용 정렬 (마지막 context 프레임 기준 루트 XZ=0, +X 방향) 을 uncanonicalize 로 되돌려
     원본 BVH 의 월드 좌표 (cm) 로 저장한다.
   - --side_offset (cm, 기본 150) : 생성은 +, SLERP 는 − 쪽으로, 정렬 기준 프레임의 진행 방향에 수직
     (수평) 으로 옮겨 저장한다 (0 이면 원래 자리). render_compare.py --layout side 와 같은 배치.
3. 검증 출력
     - 클립별 gpos_l2 (생성, SLERP), context 구간 원본 vs 생성 최대 오차 (≈0)
     - gt.bvh 를 다시 읽어 FK 한 관절 위치 vs LaFAN1 원본 BVH 같은 프레임 (원본 파일의 위치 채널로 FK) 최대 오차
     - pred.bvh / slerp.bvh 를 다시 읽어 FK 한 위치 vs 모델 출력 위치 (정렬 되돌림 + side offset) 최대 오차

[참고] LaFAN1 원본은 모든 관절이 6채널(위치+회전)이지만, 저장 파일은 루트 6채널 + 나머지 3채널(회전) 이다.
비루트 관절 위치 채널 = OFFSET 임은 1단계에서 확인했다 (docs/assumptions.md).

실행 예:
    python scripts/export_bvh.py                                   # best.pth, trans 30, 3개, 300 프레임
    python scripts/export_bvh.py --trans 15 --n 5 --seed 1 --side_offset 0
    # 그다음 Blender 로 렌더:
    blender --background --python scripts/blender_render.py -- experiments/ctx_base/bvh/<클립폴더>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # render_compare
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # src

import torch  # noqa: E402

from render_compare import FPS, infer_clips, load_run, select_clips, slerp_pose  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.data.bvh import BvhMotion, load_bvh, write_bvh  # noqa: E402
from src.data.pipeline import model_output_to_motion, motion_to_model_input  # noqa: E402
from src.data.preprocess import uncanonicalize_root  # noqa: E402
from src.data.skeleton import forward_kinematics, local_positions_from_root  # noqa: E402
from src.training.context import gpos_l2  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/ctx_base/best.pth"))
    ap.add_argument("--config", type=Path, default=None,
                    help="학습 때 저장된 config.yaml (기본: <ckpt 폴더>/config.yaml)")
    ap.add_argument("--trans", type=int, default=30, choices=[5, 15, 30], help="transition 길이")
    ap.add_argument("--n", type=int, default=3, help="저장할 클립 수")
    ap.add_argument("--seed", type=int, default=0, help="val 클립 고르기 시드 (render_compare.py 와 같으면 같은 클립)")
    ap.add_argument("--total_frames", "--total-frames", dest="total_frames", type=int, default=300)
    ap.add_argument("--side_offset", "--side-offset", dest="side_offset", type=float, default=150.0,
                    help="생성(+)/SLERP(−) 를 진행 방향에 수직으로 띄울 거리 (cm). 0 이면 원래 자리")
    ap.add_argument("--out", type=Path, default=Path("experiments/ctx_base/bvh"))
    ap.add_argument("--precision", type=int, default=6, help="BVH 소수점 자릿수")
    ap.add_argument("--env", default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def fk_world(motion_or_root, local_rot=None, offsets=None, parents=None) -> torch.Tensor:
    """FK → 전역 관절 위치 (F, J, 3), float64.

    BvhMotion 을 주면 파일의 로컬 위치 (위치 채널 그대로) 로, 아니면 루트 위치 + OFFSET 으로 계산.
    """
    if isinstance(motion_or_root, BvhMotion):
        m = motion_or_root
        return forward_kinematics(m.local_rot, m.local_pos, list(m.parents))[1]
    local_pos = local_positions_from_root(motion_or_root, offsets)
    return forward_kinematics(local_rot, local_pos, parents)[1]


def main() -> None:
    args = parse_args()
    run = load_run(args.ckpt, args.config, args.env, args.device)
    skel, layout, ds = run.skel, run.layout, run.val_ds
    ctx, trans, total = run.ctx, args.trans, args.total_frames
    pre, post, idx = select_clips(run, trans, total, args.n, args.seed)
    inf = infer_clips(run, idx, trans, pre, total, with_slerp=True)
    pb, spec = inf.pb, inf.spec
    s = spec.seq_slice
    t0, t1 = pre + s.start, pre + s.stop                         # 영상(=저장 BVH) 안의 transition [t0, t1)

    # ----- 정렬 변환: 늘린 시퀀스를 prepare_batch 와 같은 함수/기준 프레임으로 → CanonicalFrame -----
    ext = inf.ext_batch
    root_w = ext["root_pos"].to(run.device)                      # (B, total, 3) 월드
    rot_w = ext["local_rot"].to(run.device)                      # (B, total, J, 3, 3)
    frame = motion_to_model_input(root_w, rot_w, pre + ctx, skel.root_forward_local_axis,
                                  world_forward=run.data_cfg["forward_world_axis"]).frame

    # ----- 모델 창 → 월드 좌표 (역정규화 → 6D→회전행렬 → 정렬 되돌리기) -----
    pred_root_w, pred_rot_w = model_output_to_motion(inf.y, layout, normalizer=run.norm, frame=frame)
    sl_root_c, sl_rot_c = slerp_pose(pb, spec, layout)
    sl_root_w, sl_rot_w = uncanonicalize_root(sl_root_c, sl_rot_c, frame)

    def full(root_win, rot_win):
        r, q = root_w.clone(), rot_w.clone()
        r[:, t0:t1], q[:, t0:t1] = root_win[:, s], rot_win[:, s]
        return r, q

    # 진행 방향 (정렬 기준 프레임의 +X) 과 그 수평 수직 방향 (+Z), 월드 좌표
    inv = frame.rot_offset.transpose(-1, -2)[:, 0]               # (B, 3, 3)
    fwd_dir, side_dir = inv[..., 0], inv[..., 2]                 # (B, 3)
    tracks = {"gt": (root_w, rot_w, 0.0), "pred": (*full(pred_root_w, pred_rot_w), args.side_offset),
              "slerp": (*full(sl_root_w, sl_rot_w), -args.side_offset)}

    # 검증용 기대 위치 (월드): 모델 창 전역 위치를 정렬 되돌림 → 늘린 원본 위에 transition 만 덮기
    def to_world(p):                                             # (B, T, J, 3) 정렬 → 월드
        return (inv[:, None, None] @ p[..., None]).squeeze(-1) + frame.pos_offset[:, None]
    gt_world_fk = to_world(inf.pb_full.gpos_gt)
    expect = {"pred": to_world(inf.gpos), "slerp": to_world(inf.gpos_slerp)}

    out_root = resolve_path(args.out)
    originals: dict[int, BvhMotion] = {}
    worst = {"gt": 0.0, "pred": 0.0, "slerp": 0.0}
    print(f"저장 위치: {out_root}   (side_offset {args.side_offset:g} cm)")
    for b, (fi, start) in enumerate(inf.clips):
        path = ds.files[fi]
        if fi not in originals:
            originals[fi] = load_bvh(path)                       # 계층·OFFSET·End Site·회전 순서 원본
        org = originals[fi]
        if len(set(org.rotation_orders)) != 1:
            raise ValueError(f"{path.name}: 관절마다 회전 순서가 다릅니다 {set(org.rotation_orders)}")
        order = org.rotation_orders[0]
        frame0 = start + ds.start_frame                          # 모델 창 첫 프레임 (원본 BVH 번호)
        bvh_start = frame0 - pre                                 # 저장 BVH 첫 프레임 (원본 BVH 번호)
        clip_name = f"{path.stem}_f{frame0}_trans{trans}"
        clip_dir = out_root / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        one = slice(b, b + 1)
        l2 = float(gpos_l2(inf.gpos[one], pb.gpos_gt[one], s))
        l2s = float(gpos_l2(inf.gpos_slerp[one], pb.gpos_gt[one], s))
        ctx_err = float((inf.gpos[one, :ctx] - pb.gpos_gt[one, :ctx]).norm(dim=-1).max())
        print(f"\n[{b + 1}/{len(inf.clips)}] {clip_name}  (원본 프레임 {bvh_start}–{bvh_start + total - 1}, "
              f"transition = BVH 프레임 {t0}–{t1 - 1} (0부터))")
        print(f"  gpos_l2 생성 {l2:.4f}, SLERP {l2s:.4f} | context 원본 vs 생성 최대 오차 {ctx_err:.2e} → "
              f"{'OK' if ctx_err < 1e-4 else '경고: context 가 원본과 다름!'}")

        # ----- 저장 -----
        offsets = torch.from_numpy(org.offsets)
        for key, (root, rot, side) in tracks.items():
            root_b = root[b].double().cpu() + side * side_dir[b].double().cpu()
            write_bvh(clip_dir / f"{key}.bvh", org.names, list(org.parents), offsets, rot[b].double().cpu(),
                      root_b, org.frame_time, order=order, end_sites=org.end_sites, precision=args.precision)

        # ----- 검증 1: gt.bvh 재읽기 FK vs LaFAN1 원본 BVH 같은 프레임 (원본 위치 채널 그대로) -----
        saved = load_bvh(clip_dir / "gt.bvh")
        ref = fk_world(org)[bvh_start:bvh_start + total]
        e_gt = float((fk_world(saved) - ref).norm(dim=-1).max())
        # ----- 검증 2: pred/slerp 재읽기 FK vs 모델 출력 위치 (정렬 되돌림 + side offset) -----
        msg = [f"gt.bvh vs LaFAN1 원본 {e_gt:.2e} cm"]
        worst["gt"] = max(worst["gt"], e_gt)
        for key, sign in (("pred", 1.0), ("slerp", -1.0)):
            want = gt_world_fk[b].clone()
            want[t0:t1] = expect[key][b, s]
            want = want.double().cpu() + sign * args.side_offset * side_dir[b].double().cpu()
            got = fk_world(load_bvh(clip_dir / f"{key}.bvh"))
            e = float((got - want).norm(dim=-1).max())
            worst[key] = max(worst[key], e)
            msg.append(f"{key}.bvh vs 모델 출력 {e:.2e} cm")
        print("  재읽기 FK 최대 오차: " + ", ".join(msg))

        meta = {
            "clip": clip_name, "source_bvh": path.name, "source_start_frame": bvh_start,
            "frames": total, "fps": FPS, "units": "cm",
            "context": [pre, pre + ctx], "transition": [t0, t1], "target": t1,     # 0부터, [시작, 끝)
            "phases": [["original", 0, pre], ["context", pre, t0], ["transition", t0, t1],
                       ["target", t1, t1 + 1], ["original", t1 + 1, total]],
            "side_offset_cm": args.side_offset,
            "forward_dir": fwd_dir[b].tolist(), "side_dir": side_dir[b].tolist(),   # BVH 월드 좌표 (Y 위)
            "gpos_l2": l2, "gpos_l2_slerp": l2s,
            "ckpt": str(run.ckpt_path), "iteration": run.ckpt["iteration"],
        }
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  저장: {clip_dir}/{{gt,pred,slerp}}.bvh, meta.json")

    ok = worst["gt"] < 1e-2 and worst["pred"] < 1e-2 and worst["slerp"] < 1e-2
    print(f"\n재읽기 최대 오차 (전체): gt {worst['gt']:.2e}, pred {worst['pred']:.2e}, slerp {worst['slerp']:.2e} cm → "
          f"{'OK' if ok else '경고: 1e-2 cm 초과'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
