"""
학습 공통 도구: 학습률 스케줄, 체크포인트, 시드, 무한 DataLoader
=============================================================
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn


def noam_lambda(warmup_iters: int):
    """Noam 스케줄의 배율 함수. [공식코드]

        it < warmup : it × warmup^-1.5     (선형 증가)
        그 이후     : it^-0.5              (1/√it 감소)
    두 식은 it = warmup 에서 만난다. 실제 lr = base_lr × 배율.
    iteration 은 1 부터 센다 (0 이면 lr 이 0 이 되므로 1 로 올림).
    학습 스크립트는 매 iteration 마다 optimizer 의 lr 을 직접 이 값으로 설정한다.
    """
    w = float(warmup_iters)

    def f(it: int) -> float:
        it = max(int(it), 1)
        return it * w ** -1.5 if it < w else it ** -0.5

    return f


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infinite_batches(loader) -> Iterator[tuple[int, dict]]:
    """DataLoader 를 끝없이 돌며 (epoch, batch) 를 낸다. epoch 은 0 부터."""
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def amp_dtype_from_str(name: str | None, device: torch.device) -> torch.dtype | None:
    """설정 문자열 → autocast dtype. CPU 이거나 null 이면 None (사용 안 함)."""
    if name in (None, "", "none", "null", False):
        return None
    if device.type != "cuda":
        return None
    table = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"amp 는 null, fp16, bf16 중 하나: {name!r}")
    return table[name]


def save_checkpoint(path: str | Path, model: nn.Module, optimizer, scaler,
                    iteration: int, epoch: int, best_val: float, config: dict[str, Any]) -> None:
    """체크포인트 저장. 임시 파일에 쓴 뒤 이름을 바꿔서, 저장 중 중단돼도 옛 파일이 깨지지 않게 한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "iteration": iteration,
        "epoch": epoch,
        "best_val": best_val,
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
