"""
[연구실 전용] 실제 LaFAN1 BVH 검증
================================

이 스크립트가 하는 일
--------------------
LaFAN1 폴더의 BVH 파일들(기본: 전부)에 대해 아래를 검사하고 표로 출력한다.
노트북에서는 데이터가 없어 확인하지 못한 "가정" 들을 실제 데이터로 확인하는 것이 목적이다.

  (a) 관절 이름/부모가 assets/skeleton_lafan1.yaml 과 같은가
  (b) 루트 외 관절의 위치 채널 = OFFSET 인가          ← 논문 3.2 "로컬 위치는 상수"
  (c) 파일마다 OFFSET(뼈 길이)이 같은가               ← 다르면 데이터셋이 파일별 offset 사용 (이미 지원)
  (d) BVH → 6D 표현 → 복원 오차 (회전, FK 관절 위치)  ← 논문 3.2 표현이 정보 손실 없는가
  (e) BVH 다시 쓰기 → 다시 읽기 오차                  ← write_bvh 가 맞는가
  (f) 루트 heading 가정 검사: 걷기/달리기 파일에서, 정렬 후 루트의 이동 방향이
      +X 와 얼마나 일치하는지 (cos 유사도 평균). 1 에 가까우면 "루트 로컬 +Y = 앞" 가정이 맞다.
      [가정 검증] skeleton_lafan1.yaml 의 root_forward_local_axis.

실행 예:
    python scripts/check_bvh_roundtrip.py --env lab
    python scripts/check_bvh_roundtrip.py --env lab --files walk1_subject1 run1_subject2
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_data_config, load_env_config, require_dir  # noqa: E402
from src.data.bvh import load_bvh, write_bvh  # noqa: E402
from src.data.pipeline import model_output_to_motion, motion_to_model_input  # noqa: E402
from src.data.representation import FeatureLayout  # noqa: E402
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root  # noqa: E402


def heading_alignment(m, skel: Skeleton, ctx: int, T: int = 50, stride: int = 25) -> float:
    """정렬 후 (마지막 context 프레임 → 그 다음 5프레임) 루트 이동 방향과 +X 의 cos 유사도 평균.

    움직임이 거의 없는 구간(이동 < 5cm)은 제외한다.
    """
    sims = []
    for s in range(0, m.num_frames - T, stride):
        root = m.root_pos[s:s + T][None]
        rot = m.local_rot[s:s + T][None]
        inp = motion_to_model_input(root, rot, ctx, skel.root_forward_local_axis)
        d = inp.root_pos[0, ctx - 1 + 5, [0, 2]] - inp.root_pos[0, ctx - 1, [0, 2]]
        if d.norm() > 5.0:
            sims.append((d[0] / d.norm()).item())
    return float(np.mean(sims)) if sims else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--files", nargs="*", help="검사할 파일 이름(확장자 제외). 없으면 전부")
    ap.add_argument("--clip", type=int, default=200, help="복원 검사에 쓸 프레임 수 (파일 앞부분)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    data_cfg = load_data_config()
    bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
    skel = Skeleton.from_yaml(data_cfg["skeleton"])
    ctx = data_cfg["context_len"]
    layout = FeatureLayout(skel.num_joints)

    files = sorted(bvh_dir.glob("*.bvh"))
    if args.files:
        files = [f for f in files if f.stem in set(args.files)]
    print(f"검사할 파일 {len(files)}개 (폴더: {bvh_dir})\n")

    header = f"{'file':28s} {'frames':>6s} {'skel':>5s} {'pos=off':>9s} {'rot_err':>9s} {'fk_err':>9s} {'bvh_err':>9s} {'heading':>8s}"
    print(header)
    print("-" * len(header))
    ref_offsets, offset_same, failures = None, True, 0
    for f in files:
        m = load_bvh(f)
        try:
            skel.check_matches(m.names, m.parents)
            skel_ok = "OK"
        except ValueError:
            skel_ok, failures = "FAIL", failures + 1
        offsets = torch.from_numpy(m.offsets)
        pos_off = (m.local_pos[:, 1:] - offsets[1:]).abs().max().item()
        if ref_offsets is None:
            ref_offsets = m.offsets
        elif not np.allclose(ref_offsets, m.offsets, atol=1e-4):
            offset_same = False

        # (d) 표현 왕복
        n = min(args.clip, m.num_frames)
        root, rot = m.root_pos[:n][None], m.local_rot[:n][None]
        inp = motion_to_model_input(root, rot, min(ctx, n), skel.root_forward_local_axis)
        p, r = model_output_to_motion(inp.x, layout, frame=inp.frame)
        rot_err = (r - rot).abs().max().item()
        _, g0 = forward_kinematics(rot, local_positions_from_root(root, offsets), skel.parents)
        _, g1 = forward_kinematics(r, local_positions_from_root(p, offsets), skel.parents)
        fk_err = (g1 - g0).abs().max().item()

        # (e) BVH 다시 쓰기 → 읽기
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "rt.bvh"
            write_bvh(tmp, m.names, m.parents, m.offsets, r[0], p[0], m.frame_time, end_sites=m.end_sites, precision=8)
            m2 = load_bvh(tmp)
        bvh_err = (m2.local_rot - rot[0]).abs().max().item()

        # (f) heading
        is_loco = any(k in f.stem for k in ("walk", "run", "sprint"))
        heading = f"{heading_alignment(m, skel, ctx):.3f}" if is_loco else "-"

        print(f"{f.stem:28s} {m.num_frames:6d} {skel_ok:>5s} {pos_off:9.2e} {rot_err:9.2e} {fk_err:9.2e} {bvh_err:9.2e} {heading:>8s}")

    print("\n요약")
    print(f"  스켈레톤 불일치 파일 수: {failures}")
    print(f"  모든 파일 OFFSET 동일: {offset_same}")
    print("  해석: pos=off 가 ~0 이면 '로컬 위치 = 상수' 가정 OK, rot_err/fk_err/bvh_err 가 ~1e-6 이하면 표현/복원 OK,")
    print("        heading 이 1 에 가까우면(예: > 0.9) '루트 로컬 +Y = 앞' 가정 OK. 음수/0 근처면 가정을 다시 봐야 함.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # 데이터 경로가 없을 때 (예: 노트북) 깔끔하게 종료
        sys.exit(f"[오류] {e}")
