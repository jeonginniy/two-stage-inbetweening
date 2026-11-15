"""
[연구실] 발 전용 모델 v2 평가: 기존 foot_only/eval_foot.py 를 체크포인트·출력 폴더만 바꿔서 실행
==========================================================================================

기존 평가 스크립트는 수정하지 않는다. 설정 (configs/eval/foot_only.yaml) 을 읽는 함수만 바꿔 끼워서
foot_ckpt 와 out_dir 을 v2 것으로 바꾼다. 지표·평가셋·비교 방법은 foot_only 평가 (H10) 와 완전히 같다.
결과의 `foot_only` 행이 v2 모델이다 (기존 코드의 방법 이름).

실행: python foot_v2/eval_foot_v2.py      → outputs/eval/foot_v2/{metrics_long.csv, summary.csv, metrics.png, run_info.json}
      (--foot-ckpt, --out-dir 로 바꿀 수 있고, 나머지 옵션은 foot_only/eval_foot.py 와 같다)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import foot_only.eval_foot as base  # noqa: E402

DEFAULT_CKPT = "experiments/foot_v2/best.pth"     # [결정 I12] val30_loss 최소 (foot_only 와 같은 규칙)
DEFAULT_OUT = "outputs/eval/foot_v2"


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--foot-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    mine, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    orig = base.load_named_config

    def patched(group: str, name: str) -> dict:
        cfg = orig(group, name)
        if (group, name) == ("eval", "foot_only"):
            cfg = {**cfg, "foot_ckpt": mine.foot_ckpt, "out_dir": mine.out_dir}
        return cfg

    base.load_named_config = patched
    base.main()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
