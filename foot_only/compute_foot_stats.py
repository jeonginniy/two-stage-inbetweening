"""
[연구실 전용] 발 위치 z-score 통계 계산 (발 전용 모델, STEP 2)
============================================================

하는 일
-------
1. train_stats 분할 (Subject 1–4, window 50, offset 20; scripts/compute_stats.py 와 같은 분할) 로 LafanDataset
2. 클립마다 기존 prepare_batch 로 정렬 (논문 3.6.1) + FK → 발 4관절 위치 (정렬 좌표, cm) (B, 50, 12)
3. 모든 프레임의 차원별 평균/표준편차 (RunningStats, float64, 모집단 표준편차)
4. <cache_dir>/stats/foot_only_stats.npz 로 저장 (ZScoreNormalizer.save 형식, meta 에 분할 정보)

val / benchmark (Subject 5) 는 쓰지 않는다 → 평가 데이터가 통계에 섞이지 않음.

실행 (프로젝트 루트에서):
    python foot_only/compute_foot_stats.py --env lab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from foot_only.data import compute_foot_stats, foot_names_xyz, load_foot_config  # noqa: E402
from src.config import load_data_config, load_env_config, require_dir, resolve_device  # noqa: E402
from src.data.dataset import LafanDataset  # noqa: E402
from src.data.representation import ZScoreNormalizer  # noqa: E402
from src.data.skeleton import Skeleton  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--config", default=None, help="발 전용 설정 yaml (기본 configs/foot_only.yaml)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None, help="저장 경로 (기본: <cache_dir>/stats/<foot_stats_file>)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    data_cfg = load_data_config()
    fcfg = load_foot_config(args.config)["data"]
    device = resolve_device(env)
    bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
    cache_dir = env["paths"]["cache_dir"]
    skel = Skeleton.from_yaml(data_cfg["skeleton"])

    ctx_stats = cache_dir / "stats" / fcfg["context_stats_file"]
    if not ctx_stats.exists():
        raise FileNotFoundError(f"{ctx_stats} 가 없습니다. 먼저 scripts/compute_stats.py 를 실행하세요.")
    # 발 위치는 정규화 전 값 (foot_raw) 만 쓰므로 입력 통계는 prepare_batch 에 넘기기만 한다
    ctx_norm = ZScoreNormalizer.load(ctx_stats).to(device=device, dtype=torch.float64)

    split = fcfg["stats_split"]
    sp = data_cfg["splits"][split]
    ds = LafanDataset(bvh_dir, sp["actors"], sp["window"], sp["offset"], skel, start_frame=sp["start_frame"],
                      contact_threshold=data_cfg["contact_vel_threshold"], cache_dir=cache_dir, dtype=torch.float64)
    print(f"[{split}] 클립 {len(ds)}개 (actors {sp['actors']}, window {sp['window']}, offset {sp['offset']}), "
          f"device={device}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=env["num_workers"])
    stats = compute_foot_stats(loader, skel, ctx_norm, data_cfg["context_len"], data_cfg["forward_world_axis"], device)

    norm = ZScoreNormalizer.from_stats(stats)
    out = args.out or (cache_dir / "stats" / fcfg["foot_stats_file"])
    norm.save(out, split=split, window=sp["window"], offset=sp["offset"], context_len=data_cfg["context_len"],
              num_samples=stats.n, names=list(foot_names_xyz()))
    print(f"저장: {out}  (샘플 프레임 수 {stats.n})")
    print(f"  {'차원':<12s} {'평균 (cm)':>10s} {'표준편차 (cm)':>12s}")
    for name, m, s in zip(foot_names_xyz(), stats.mean.tolist(), stats.std.tolist()):
        print(f"  {name:<12s} {m:10.3f} {s:12.3f}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(f"[오류] {e}")
