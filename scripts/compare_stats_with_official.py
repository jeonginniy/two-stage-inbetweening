"""
[연구실 전용, 검증용] 내 정규화 통계 vs 공식 코드의 통계 비교
=========================================================

이 스크립트가 하는 일
--------------------
공식 저장소에는 LaFAN1 으로 계산된 context 모델 통계 파일
    motion_inbetweening/tests/train_stats_context.pkl   ({"mean": (135,), "std": (135,)})
이 들어 있다. 내 scripts/compute_stats.py 결과와 차원별로 비교해서
"내 데이터 파이프라인(BVH 읽기 → 정렬 → 6D → 벡터 배치)이 공식 결과와 같은 값을 만드는지"
확인한다. 6D 펼침 순서를 공식과 같게 맞췄기 때문에 차원별 직접 비교가 가능하다.

규칙 준수: 공식 코드를 import 하거나 실행하지 않는다. 공식 "결과 파일" 만 읽는다.
안전: pickle 은 임의 코드 실행이 가능하므로, numpy 배열을 복원하는 데 필요한 클래스만
허용하는 제한된 Unpickler 를 쓴다.

실행 예:
    python scripts/compare_stats_with_official.py --env lab \
        --official /path/to/motion_inbetweening/tests/train_stats_context.pkl
    (--official 을 생략하면 configs/env 의 paths.official_repo/tests/train_stats_context.pkl)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_env_config  # noqa: E402

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    """numpy 배열 복원에 필요한 것만 허용하는 Unpickler."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"허용되지 않은 객체: {module}.{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="configs/env/<env>.yaml (없으면 TSI_ENV)")
    ap.add_argument("--official", type=Path, help="공식 train_stats_context.pkl 경로")
    ap.add_argument("--mine", type=Path, help="내 통계 .npz (기본: <cache_dir>/stats/lafan1_context_stats.npz)")
    args = ap.parse_args()

    env = load_env_config(args.env)
    official = args.official
    if official is None:
        repo = env["paths"].get("official_repo")
        if repo is None:
            sys.exit("--official 을 주거나 configs/env 의 paths.official_repo 를 설정하세요.")
        official = repo / "tests" / "train_stats_context.pkl"
    mine = args.mine or env["paths"]["cache_dir"] / "stats" / "lafan1_context_stats.npz"

    with open(official, "rb") as f:
        off = _SafeUnpickler(f).load()
    with np.load(mine) as f:
        my_mean, my_std = f["mean"], f["std"]
    o_mean, o_std = np.asarray(off["mean"], dtype=np.float64), np.asarray(off["std"], dtype=np.float64)

    print(f"공식: {official}\n내 것: {mine}\n")
    for name, a, b in (("mean", my_mean, o_mean), ("std", my_std, o_std)):
        d = np.abs(a - b)
        print(f"[{name}] shape 내 것 {a.shape} / 공식 {b.shape}")
        if a.shape != b.shape:
            continue
        print(f"   회전(0~131) 최대 차이 {d[:132].max():.3e},  루트 위치(132~134) 최대 차이 {d[132:].max():.3e}")
        worst = np.argsort(-d)[:5]
        for i in worst:
            print(f"   dim {i:3d}: 내 것 {a[i]: .6f}  공식 {b[i]: .6f}  차이 {d[i]:.3e}")
    print("\n해석: 차이가 ~1e-6 수준이면 파이프라인이 공식 결과와 같다. 크면 docs/assumptions.md 의 가정"
          "(6D 순서, 루트 forward 축, 정렬 기준 프레임)을 하나씩 의심해 볼 것.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # 파일 경로가 없을 때 깔끔하게 종료
        sys.exit(f"[오류] {e}")
