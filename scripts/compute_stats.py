"""
[연구실 전용] z-score 정규화 통계 μ_x, σ_x 계산
==============================================

이 스크립트가 하는 일 (논문 3.2 마지막 문단 + 3.6.1)
--------------------------------------------------
"The normalization statistics μ_x and σ_x are calculated based on a sliding window
 of 50 frames with an offset of 20 frames."

1. train_stats 분할 (Subject 1–4, window 50, offset 20) 로 LafanDataset 을 만든다.
2. 각 클립을 pipeline.motion_to_model_input(normalizer=None) 으로 정렬 + 6D 변환 → x̃ (B, 50, 135)
3. 모든 클립의 모든 프레임에 대해 차원별 평균/표준편차 (RunningStats, float64)
4. <cache_dir>/stats/lafan1_context_stats.npz 로 저장 (2단계 학습에서 ZScoreNormalizer.load 로 사용)

실행 예:
    python scripts/compute_stats.py --env lab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_data_config, load_env_config, require_dir, resolve_device  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.pipeline import motion_to_model_input  # noqa: E402
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--split", default="train_stats", help="통계를 낼 분할 이름 (기본 train_stats)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None, help="저장 경로 (기본: <cache_dir>/stats/lafan1_context_stats.npz)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    cfg = load_data_config()
    device = resolve_device(env)
    bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
    cache_dir = env["paths"]["cache_dir"]
    sp = cfg["splits"][args.split]
    skel = Skeleton.from_yaml(cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)

    # 공식 코드처럼 통계는 float64 로 계산
    ds = LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel,
                      start_frame=sp["start_frame"], contact_threshold=cfg["contact_vel_threshold"],
                      cache_dir=cache_dir, dtype=torch.float64)
    print(f"[{args.split}] 클립 {len(ds)}개 (window {sp['window']}, offset {sp['offset']}), device={device}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=env["num_workers"])

    stats = RunningStats(layout.dim)
    for batch in loader:
        inp = motion_to_model_input(batch["root_pos"].to(device), batch["local_rot"].to(device),
                                    cfg["context_len"], skel.root_forward_local_axis,
                                    world_forward=cfg["forward_world_axis"])
        stats.update(inp.x_raw)

    norm = ZScoreNormalizer.from_stats(stats)
    out = args.out or (cache_dir / "stats" / "lafan1_context_stats.npz")
    norm.save(out, split=args.split, window=sp["window"], offset=sp["offset"],
              context_len=cfg["context_len"], num_samples=stats.n)
    print(f"저장: {out}  (샘플 프레임 수 {stats.n})")

    std = stats.std
    rs, ps = layout.rot_slice, layout.pos_slice
    print(f"  회전 6D 부분 σ: 최소 {std[rs].min():.4g}, 최대 {std[rs].max():.4g}")
    print(f"  루트 위치 μ: {stats.mean[ps].tolist()}")
    print(f"  루트 위치 σ: {std[ps].tolist()}")
    tiny = torch.nonzero(std < norm.min_std).flatten().tolist()
    print(f"  σ < {norm.min_std} 인 차원 (σ=1 로 대체됨): {tiny if tiny else '없음'}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # 데이터 경로가 없을 때 (예: 노트북) 깔끔하게 종료
        sys.exit(f"[오류] {e}")
