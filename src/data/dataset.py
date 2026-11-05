"""
LaFAN1 데이터셋 (PyTorch Dataset)
================================

이 파일이 하는 일
-----------------
LaFAN1 BVH 폴더에서 지정한 배우(subject)의 파일만 골라 읽고,
슬라이딩 윈도우로 길이 T 의 클립을 잘라 하나씩 돌려준다 (논문 3.6.1, 4.1).

    파일 이름 규칙: <동작이름>_<배우>.bvh   예) walk1_subject1.bvh
    클립 수(파일당) = (F − window) // offset + 1     (F = 파일의 프레임 수, F ≥ window 일 때)

파일 하나를 처리하는 순서:
    load_bvh (bvh.py) → 스켈레톤 정의와 비교 (skeleton.py)
    → FK 로 월드 위치 계산 → 발 접촉 라벨 (preprocess.py)
    → (선택) data/cache/ 에 .npz 로 캐시 저장. 다음부터는 캐시를 읽는다.

__getitem__ 은 "원래 월드 좌표" 그대로의 클립을 준다. 정렬(canonicalize)과
6D 변환은 배치 단위로 pipeline.motion_to_model_input 에서 한다
(공식 코드와 같은 구조: 학습 루프에서 배치마다 정렬).

데이터 경로는 이 파일에 없다. 호출하는 쪽(scripts/*)이 configs/env/*.yaml 에서 읽어서 넘긴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.bvh import load_bvh
from src.data.preprocess import extract_foot_contacts
from src.data.skeleton import Skeleton, forward_kinematics, local_positions_from_root

CACHE_VERSION = 1  # 캐시 형식이 바뀌면 숫자를 올려서 옛 캐시를 무시하게 한다


def list_bvh_files(bvh_dir: str | Path, actors: Sequence[str]) -> list[Path]:
    """bvh_dir 에서 배우 이름이 actors 에 속하는 .bvh 파일 목록 (이름순 정렬).

    예) actors=["subject5"] → [aiming2_subject5.bvh, dance2_subject5.bvh, ...]
    """
    files = []
    for p in sorted(Path(bvh_dir).glob("*.bvh")):
        _, _, actor = p.stem.rpartition("_")
        if actor in actors:
            files.append(p)
    return files


def num_windows(num_frames: int, window: int, offset: int) -> int:
    """프레임 수 F 인 파일에서 나오는 클립 수. F < window 면 0."""
    if num_frames < window:
        return 0
    return (num_frames - window) // offset + 1


def process_bvh_file(path: Path, skeleton: Skeleton, contact_threshold: float) -> dict[str, np.ndarray]:
    """BVH 파일 하나 → 학습에 필요한 배열들 (float64 numpy).

    출력 dict:
        root_pos  : (F, 3)        루트 월드 위치
        local_rot : (F, J, 3, 3)  로컬 회전행렬
        offsets   : (J, 3)        뼈 offset (파일에서 읽은 값)
        contacts  : (F, 4)        발 접촉 라벨 [LFoot, LToe, RFoot, RToe]
    """
    motion = load_bvh(path)
    skeleton.check_matches(motion.names, motion.parents, source=f"({path.name})")
    offsets = torch.from_numpy(motion.offsets)

    # 발 접촉은 "원래 월드 좌표" 에서 계산한다 (정렬 전).
    # 논문 3.2 에 따라 루트 외 관절의 로컬 위치는 offset 으로 둔다.
    local_pos = local_positions_from_root(motion.root_pos, offsets)
    _, global_pos = forward_kinematics(motion.local_rot, local_pos, skeleton.parents)
    contacts = extract_foot_contacts(global_pos, skeleton.contact_joints, contact_threshold)

    return {
        "root_pos": motion.root_pos.numpy(),
        "local_rot": motion.local_rot.numpy(),
        "offsets": motion.offsets,
        "contacts": contacts.numpy(),
    }


class LafanDataset(Dataset):
    """LaFAN1 슬라이딩 윈도우 데이터셋.

    생성 인자:
        bvh_dir           : LaFAN1 BVH 폴더 (configs/env/*.yaml 의 paths.lafan1_bvh_dir)
        actors            : 사용할 배우 목록. 예) ["subject1", ..., "subject4"]
        window, offset    : 클립 길이 T, 슬라이딩 간격 (configs/data/lafan1.yaml 의 splits)
        skeleton          : Skeleton.from_yaml(...) (관절 순서/부모 검증, 발 관절 인덱스)
        start_frame       : 각 파일 앞에서 버릴 프레임 수
        contact_threshold : 발 접촉 임계값
        cache_dir         : None 이 아니면 파싱 결과를 <cache_dir>/lafan1_parsed/*.npz 로 캐시
        dtype             : 돌려줄 텐서 dtype

    __getitem__(i) 출력 dict:
        root_pos  : (T, 3)
        local_rot : (T, J, 3, 3)
        contacts  : (T, 4)
        offsets   : (J, 3)
        file_index: int, start: int   (어느 파일의 몇 번째 프레임부터인지.
                                       start 는 start_frame 을 잘라낸 뒤 기준 → 원본 프레임 = start + start_frame)
    DataLoader 로 묶으면 앞에 배치 차원 B 가 붙는다. 예) root_pos (B, T, 3)
    """

    def __init__(
        self,
        bvh_dir: str | Path,
        actors: Sequence[str],
        window: int,
        offset: int,
        skeleton: Skeleton,
        start_frame: int = 0,
        contact_threshold: float = 0.2,
        cache_dir: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ):
        super().__init__()
        self.window, self.offset, self.start_frame = window, offset, start_frame
        self.skeleton = skeleton
        self.dtype = dtype
        self.files = list_bvh_files(bvh_dir, actors)
        if not self.files:
            raise FileNotFoundError(f"{bvh_dir} 에 배우 {list(actors)} 의 .bvh 파일이 없습니다.")

        cache_root = Path(cache_dir) / "lafan1_parsed" if cache_dir is not None else None
        self.root_pos: list[Tensor] = []
        self.local_rot: list[Tensor] = []
        self.contacts: list[Tensor] = []
        self.offsets: list[Tensor] = []
        for path in self.files:
            arrays = self._load_one(path, cache_root, contact_threshold, verbose)
            s = start_frame
            self.root_pos.append(torch.as_tensor(arrays["root_pos"][s:], dtype=dtype))
            self.local_rot.append(torch.as_tensor(arrays["local_rot"][s:], dtype=dtype))
            self.contacts.append(torch.as_tensor(arrays["contacts"][s:], dtype=dtype))
            self.offsets.append(torch.as_tensor(arrays["offsets"], dtype=dtype))

        # 전역 클립 번호 → (파일 번호, 파일 안 클립 번호) 를 빠르게 찾기 위한 누적합
        self.frames = [r.shape[0] for r in self.root_pos]
        counts = [num_windows(f, window, offset) for f in self.frames]
        self._cum = np.cumsum([0] + counts)

    # ----- 파일 하나 읽기 (캐시 사용) -----
    def _load_one(self, path: Path, cache_root: Path | None, threshold: float, verbose: bool) -> dict:
        stat = path.stat()
        key = np.array([CACHE_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
        cache_file = cache_root / f"{path.stem}.npz" if cache_root is not None else None

        if cache_file is not None and cache_file.exists():
            with np.load(cache_file) as f:
                if np.array_equal(f["key"], key) and float(f["contact_threshold"]) == threshold:
                    return {k: f[k] for k in ("root_pos", "local_rot", "offsets", "contacts")}

        if verbose:
            print(f"[LafanDataset] BVH 처리 중: {path.name}")
        arrays = process_bvh_file(path, self.skeleton, threshold)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, key=key, contact_threshold=np.float64(threshold), **arrays)
        return arrays

    def __len__(self) -> int:
        return int(self._cum[-1])

    def locate(self, index: int) -> tuple[int, int]:
        """전역 클립 번호 → (파일 번호, 시작 프레임)."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self._cum, index, side="right") - 1)
        start = (index - int(self._cum[file_index])) * self.offset
        return file_index, start

    def __getitem__(self, index: int) -> dict:
        fi, start = self.locate(index)
        sl = slice(start, start + self.window)
        return {
            "root_pos": self.root_pos[fi][sl],
            "local_rot": self.local_rot[fi][sl],
            "contacts": self.contacts[fi][sl],
            "offsets": self.offsets[fi],
            "file_index": fi,
            "start": start,
        }
