"""
설정(config) 로더
=================

이 파일이 하는 일
-----------------
configs/ 폴더의 YAML 파일을 읽어서 파이썬 dict 로 돌려준다.
"데이터 경로와 GPU 설정은 코드에 직접 쓰지 않는다" 는 규칙을 지키기 위해,
경로/장치가 필요한 모든 코드는 반드시 이 모듈을 거쳐서 값을 얻는다.

사용 순서
---------
1. load_env_config(env)   : configs/env/<env>.yaml  (경로, device)  ← 환경마다 다름
2. load_data_config(name) : configs/data/<name>.yaml (window, context_len 등) ← 환경 무관
3. resolve_device(env_cfg): "cuda:0" 을 적었지만 GPU 가 없으면 cpu 로 대체

환경 이름은 (1) 함수 인자 / 스크립트의 --env, (2) 환경변수 TSI_ENV 순서로 정한다.
둘 다 없으면 일부러 에러를 낸다 (노트북 설정으로 연구실에서 잘못 돌리는 일을 막기 위해).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 = 이 파일(src/config.py)의 부모의 부모
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

# 환경 이름을 지정하는 환경변수 이름 (TSI = Two-Stage In-betweening)
ENV_VAR: str = "TSI_ENV"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 파일 하나를 dict 로 읽는다.

    입력:  path - YAML 파일 경로
    출력:  dict (파일이 비어 있으면 빈 dict)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(p: str | Path | None) -> Path | None:
    """설정 파일에 적힌 경로를 절대경로 Path 로 바꾼다.

    - None 이면 None 그대로 (예: 노트북에 LaFAN1 이 없을 때)
    - "~", "$HOME" 같은 표기를 펼친다
    - 상대경로면 프로젝트 루트 기준으로 해석한다
    """
    if p is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def available_envs() -> list[str]:
    """configs/env/ 에 있는 환경 이름 목록 (예: ["lab", "laptop"])."""
    return sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))


def load_env_config(env: str | None = None) -> dict[str, Any]:
    """실행 환경 설정을 읽는다 (configs/env/<env>.yaml).

    입력:  env - 환경 이름. None 이면 환경변수 TSI_ENV 를 본다.
    출력:  dict
        {
          "name": "lab",
          "paths": {"lafan1_bvh_dir": Path|None, "cache_dir": Path, ...},  # 절대경로로 변환됨
          "device": "cuda:0",
          "num_workers": 4,
        }
    """
    env = env or os.environ.get(ENV_VAR)
    if not env:
        raise ValueError(
            f"실행 환경이 지정되지 않았습니다. --env 인자 또는 환경변수 {ENV_VAR} 로 "
            f"지정하세요. 가능한 값: {available_envs()}"
        )
    path = CONFIG_DIR / "env" / f"{env}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 가능한 값: {available_envs()}")

    cfg = load_yaml(path)
    cfg["name"] = env
    cfg["paths"] = {k: resolve_path(v) for k, v in (cfg.get("paths") or {}).items()}
    cfg.setdefault("device", "cpu")
    cfg.setdefault("num_workers", 0)
    return cfg


def load_data_config(name: str = "lafan1") -> dict[str, Any]:
    """데이터 설정을 읽는다 (configs/data/<name>.yaml).

    출력 dict 의 "skeleton" 값은 절대경로 Path 로 바꿔서 돌려준다.
    """
    cfg = load_yaml(CONFIG_DIR / "data" / f"{name}.yaml")
    cfg["skeleton"] = resolve_path(cfg["skeleton"])
    return cfg


def require_dir(path: Path | None, what: str) -> Path:
    """경로가 설정돼 있고 실제로 존재하는 폴더인지 확인한다. 아니면 친절한 에러.

    입력:  path - 확인할 경로, what - 에러 메시지에 쓸 설명 (예: "LaFAN1 BVH 폴더")
    출력:  path (확인된 Path)
    """
    if path is None:
        raise FileNotFoundError(
            f"{what} 경로가 설정 파일에서 null 입니다. configs/env/<환경>.yaml 을 수정하세요."
        )
    if not path.is_dir():
        raise FileNotFoundError(
            f"{what} 폴더가 존재하지 않습니다: {path}\n"
            f"configs/env/<환경>.yaml 의 경로를 확인하세요."
        )
    return path


def resolve_device(env_cfg: dict[str, Any]):
    """설정의 device 문자열을 torch.device 로 바꾼다.

    "cuda..." 를 요청했는데 CUDA 를 쓸 수 없으면 경고를 내고 cpu 로 대체한다.
    출력:  torch.device
    """
    import torch  # torch 가 필요 없는 곳에서 이 모듈을 import 할 수 있도록 지연 import

    want = str(env_cfg.get("device", "cpu"))
    if want.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"device={want} 를 요청했지만 CUDA 를 쓸 수 없어 cpu 로 대체합니다.")
        return torch.device("cpu")
    return torch.device(want)
