"""
[연구실 전용] 데이터 분할별 파일/프레임/클립 수 확인
================================================

이 스크립트가 하는 일
--------------------
configs/data/lafan1.yaml 의 splits (train, train_stats, val, benchmark) 마다
파일 수, 총 프레임 수, 슬라이딩 윈도우 클립 수를 계산하고,
논문 4.1 에 적힌 클립 수(train 404195, benchmark 2232)와 비교한다.

BVH 의 "Frames:" 줄만 읽으므로 빠르다 (전체 파싱 X).
train 클립 수가 논문과 다르면 window 41 (공식 context 모델 값)일 때의 수도 함께 출력한다.

실행 예:
    python scripts/inspect_dataset.py --env lab
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_data_config, load_env_config, require_dir  # noqa: E402
from src.data.dataset import list_bvh_files, num_windows  # noqa: E402


def count_frames(path: Path) -> int:
    """BVH 의 'Frames: N' 줄만 찾아서 N 을 돌려준다."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            mt = re.match(r"\s*Frames:\s*(\d+)", line)
            if mt:
                return int(mt.group(1))
    raise ValueError(f"{path}: Frames 줄이 없습니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    cfg = load_data_config()
    bvh_dir = require_dir(env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더")
    all_files = sorted(bvh_dir.glob("*.bvh"))
    frames = {p: count_frames(p) for p in all_files}
    print(f"BVH 파일 {len(all_files)}개 (LaFAN1 공개본은 77개), 총 프레임 {sum(frames.values())} "
          f"(논문 4.1: 496,672)\n")

    expected = cfg.get("expected_clip_counts", {})
    for name, sp in cfg["splits"].items():
        files = list_bvh_files(bvh_dir, sp["actors"])
        fr = [frames[p] - sp["start_frame"] for p in files]
        clips = sum(num_windows(n, sp["window"], sp["offset"]) for n in fr)
        line = (f"[{name:11s}] 배우 {sp['actors']}, 파일 {len(files)}개, 프레임 {sum(fr)}, "
                f"window {sp['window']}, offset {sp['offset']} → 클립 {clips}")
        if name in expected:
            ok = "일치" if clips == expected[name] else "불일치"
            line += f"   (논문 {expected[name]}: {ok})"
        print(line)
        if name == "train" and name in expected and clips != expected[name]:
            alt = sum(num_windows(n, 41, sp["offset"]) for n in fr)
            print(f"              참고: window 41 (공식 context 모델 값) 이면 클립 {alt}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # 데이터 경로가 없을 때 (예: 노트북) 깔끔하게 종료
        sys.exit(f"[오류] {e}")
