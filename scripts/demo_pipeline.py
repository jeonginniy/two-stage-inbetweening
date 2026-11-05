"""
[노트북에서 실행 가능] 데이터 파이프라인 데모
==========================================

이 스크립트가 하는 일
--------------------
1단계 파이프라인 전체를 한 번 돌려 보고, 각 단계의 텐서 shape 와 복원 오차를 출력한다.

    BVH 파일 ─load_bvh─▶ 로컬 회전/루트 위치 ─정렬+6D+z-score─▶ 모델 입력 x (1, T, 135)
             ─복원─▶ 로컬 회전/루트 위치 ─write_bvh─▶ 복원된 BVH 파일

입력 BVH:
    --bvh 를 주면 그 파일을 쓰고 (예: 연구실에서 실제 LaFAN1 파일),
    안 주면 LaFAN1 형식의 "가짜" BVH 를 만들어서 쓴다 (노트북용. 실제 데이터 아님).

결과 파일은 configs/env/<환경>.yaml 의 cache_dir 아래 demo/ 폴더에 저장된다.

실행 예:
    python scripts/demo_pipeline.py --env laptop
    python scripts/demo_pipeline.py --env lab --bvh /path/to/lafan1/walk1_subject1.bvh --start 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 프로젝트 루트를 import 경로에 추가

import torch  # noqa: E402

from src.config import load_data_config, load_env_config  # noqa: E402
from src.data.bvh import load_bvh, write_bvh  # noqa: E402
from src.data.pipeline import model_output_to_motion, motion_to_model_input  # noqa: E402
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 환경변수 TSI_ENV)")
    ap.add_argument("--bvh", type=Path, help="사용할 BVH 파일. 없으면 가짜 BVH 생성")
    ap.add_argument("--start", type=int, default=0, help="클립 시작 프레임")
    ap.add_argument("--window", type=int, default=None, help="클립 길이 T (기본: train window)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    data_cfg = load_data_config()
    out_dir = env["paths"]["cache_dir"] / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    T = args.window or data_cfg["splits"]["train"]["window"]
    ctx = data_cfg["context_len"]

    # 1) BVH 준비
    if args.bvh is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.helpers import write_fake_lafan_bvh
        bvh_path = out_dir / "fake_walk_subject0.bvh"
        write_fake_lafan_bvh(bvh_path, num_frames=max(T + args.start, 120))
        print(f"[1] 가짜 LaFAN1 형식 BVH 생성 (실제 데이터 아님): {bvh_path}")
    else:
        bvh_path = args.bvh
        print(f"[1] BVH 사용: {bvh_path}")

    # 2) 읽기
    m = load_bvh(bvh_path)
    skel.check_matches(m.names, m.parents, source=f"({bvh_path.name})")
    print(f"[2] load_bvh: local_rot {tuple(m.local_rot.shape)}, local_pos {tuple(m.local_pos.shape)}, "
          f"frame_time {m.frame_time:.6f}")

    # 3) 클립 하나 자르기 (배치 차원 B=1 추가)
    sl = slice(args.start, args.start + T)
    root, rot = m.root_pos[sl][None], m.local_rot[sl][None]
    print(f"[3] 클립: root_pos {tuple(root.shape)}, local_rot {tuple(rot.shape)}  (B, T, ...)")

    # 4) 모델 입력으로 변환 (데모에서는 이 클립 하나로 z-score 통계를 만든다. 실제는 compute_stats.py)
    raw = motion_to_model_input(root, rot, ctx, skel.root_forward_local_axis)
    stats = RunningStats(raw.x_raw.shape[-1])
    stats.update(raw.x_raw)
    norm = ZScoreNormalizer.from_stats(stats)
    inp = motion_to_model_input(root, rot, ctx, skel.root_forward_local_axis, normalizer=norm)
    f = ctx - 1
    print(f"[4] 정렬 후 프레임 {f} 의 루트 위치: {inp.root_pos[0, f].tolist()}  (x, z ≈ 0 이어야 함)")
    print(f"    x̃ (정규화 전) {tuple(inp.x_raw.shape)}  → x (z-score) {tuple(inp.x.shape)}   D = 22*6 + 3 = 135")

    # 5) 복원
    layout = FeatureLayout(skel.num_joints)
    p, r = model_output_to_motion(inp.x, layout, normalizer=norm, frame=inp.frame)
    offsets = torch.from_numpy(m.offsets)
    _, g0 = forward_kinematics(rot, local_positions_from_root(root, offsets), skel.parents)
    _, g1 = forward_kinematics(r, local_positions_from_root(p, offsets), skel.parents)
    print(f"[5] 복원 오차: 회전행렬 최대 {(r - rot).abs().max():.2e}, 루트 위치 최대 {(p - root).abs().max():.2e}, "
          f"FK 관절 위치 최대 {(g1 - g0).abs().max():.2e} (cm)")

    # 6) 복원 결과를 BVH 로 저장
    out_bvh = out_dir / f"{bvh_path.stem}_restored.bvh"
    write_bvh(out_bvh, m.names, m.parents, m.offsets, r[0], p[0], m.frame_time, end_sites=m.end_sites)
    print(f"[6] 복원된 BVH 저장: {out_bvh}")


if __name__ == "__main__":
    main()
