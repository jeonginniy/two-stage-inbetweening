"""
[연구실] 발 전용 모델 v2 학습: foot_only 와 같은 학습 + 발 속도 손실 (I12)
======================================================================

기존 foot_only/train_foot.py 는 수정하지 않고 그대로 실행한다. 이 파일은 실행 전에 두 가지만 바꿔 끼운다:
    1. 학습 손실 함수  foot_forward → foot_v2.losses.foot_forward_v2 (L_vel 추가)
    2. 설정 읽기       configs/foot_v2.yaml (base: foot_only.yaml 위에 바뀐 값만 덮어씀)
그래서 모델·데이터·학습 설정·체크포인트 형식은 foot_only 와 완전히 같고, 기존 평가 코드 (load_foot_run) 로 읽을 수 있다.

검증 (val) 은 기존 validate 를 그대로 쓰므로 val{t}_loss = L_pos + L_contact (L_vel 없음) 이다.
→ log.csv 의 val 값과 best.pth 규칙 (val30_loss 최소) 이 experiments/foot_only 와 바로 비교된다.
   train 의 loss 에는 L_vel 이 들어 있고, log.csv 에 train `vel` (cm/frame) 이 추가로 기록된다.

실행 (프로젝트 루트, 24 ms/iteration 기준 약 3시간 20분):
    nohup python foot_v2/train_foot_v2.py --run-name foot_v2 > foot_v2.out 2>&1 &
    python foot_v2/train_foot_v2.py --run-name foot_v2 --resume last
    python foot_v2/train_foot_v2.py --run-name foot_v2_check_1k --max-iters 1000        # 짧게 동작 확인
나머지 옵션 (--env, --batch-size, --set, --fake-data 등) 은 foot_only/train_foot.py 와 같다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

import foot_only.train_foot as base  # noqa: E402
from feet_to_body.data import _merge  # noqa: E402
from foot_only.data import load_foot_config  # noqa: E402
from foot_v2.losses import foot_forward_v2  # noqa: E402
from src.config import resolve_path  # noqa: E402

DEFAULT_CONFIG = "configs/foot_v2.yaml"
# 끝난 실험 폴더에는 쓰지 않는다 (기존 목록 + 이 프로젝트에서 만든 폴더)
PROTECTED_RUNS = (*base.PROTECTED_RUNS, "foot_only", "foot_check_1k", "feet_to_body", "feet_to_body_pred",
                  "body_check_1k", "body_pred_check_1k")


def load_v2_config(path: str | Path | None = None) -> dict:
    """설정 yaml → dict. `base: <configs/ 안의 파일 이름>` 이 있으면 그 파일 위에 덮어쓴다 (기본 configs/foot_v2.yaml)."""
    p = resolve_path(path or DEFAULT_CONFIG)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b = cfg.pop("base", None)
    return _merge(load_foot_config(p.parent / b), cfg) if b else cfg


def main() -> None:
    if "--run-name" not in sys.argv:
        raise ValueError("--run-name 을 꼭 주세요 (예: --run-name foot_v2). 기본값 foot_only 는 기존 실험 폴더입니다.")
    base.PROTECTED_RUNS = PROTECTED_RUNS
    base.load_foot_config = load_v2_config       # 설정: foot_v2.yaml
    base.foot_forward = foot_forward_v2          # 학습 손실: + L_vel (validate 는 기존 foot_forward 그대로)
    base.main()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[오류] {e}")
