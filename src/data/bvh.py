"""
BVH 파일 읽기 / 쓰기
====================

이 파일이 하는 일
-----------------
LaFAN1 모캡 데이터는 BVH 텍스트 파일이다. BVH 는 두 부분으로 되어 있다:

    HIERARCHY                       ← 관절 트리, 각 관절의 OFFSET(뼈), CHANNELS(저장되는 값 종류)
    ROOT Hips { OFFSET .. CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT LeftUpLeg { ... } ... }
    MOTION
    Frames: 7840
    Frame Time: 0.033333
    <프레임마다 한 줄: 모든 관절의 채널 값을 HIERARCHY 순서대로 나열>

1. parse_bvh()       : 텍스트 → BvhData (원시 숫자 그대로)
2. bvh_to_local()    : BvhData → 로컬 회전행렬 (F, J, 3, 3), 로컬 위치 (F, J, 3)
3. load_bvh()        : 1+2 를 한 번에 → BvhMotion
4. write_bvh()       : 로컬 회전행렬/위치 → BVH 파일 (복원 결과 확인, Blender 등에서 보기)

사용 순서
---------
src/data/dataset.py 가 load_bvh() 를 호출 → 로컬 회전/위치를 얻고
→ src/data/representation.py 에서 모델 입력 벡터로 바꾼다 (논문 3.2).

주의 / 가정
-----------
- 관절마다 채널 수와 회전 순서가 달라도 처리한다 (일반 BVH 지원).
- 위치 채널(Xposition...)이 있는 관절은 그 값을 로컬 위치로 쓰고, 없으면 OFFSET 을 쓴다.
  [가정] LaFAN1 은 모든 관절이 6채널(위치+회전)이고, 루트가 아닌 관절의 위치 채널 값은
  OFFSET 과 같다고 알려져 있다. 논문 3.2 도 "로컬 위치는 상수" 라고 하므로, 모델 입력에는
  루트 위치만 쓰고 나머지는 OFFSET 을 쓴다. 정말 같은지는 연구실에서
  scripts/check_bvh_roundtrip.py 가 검사한다.
- End Site 는 관절로 세지 않는다 (LaFAN1 22 관절 기준). 다시 쓸 때를 위해 offset 만 보관.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.rotations import euler_to_matrix, matrix_to_euler

_POS_CHANNELS = ("Xposition", "Yposition", "Zposition")
_ROT_CHANNELS = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


@dataclass
class BvhData:
    """BVH 파일 내용을 거의 그대로 담은 구조체 (아직 회전행렬로 바꾸기 전).

    속성:
        names      : 관절 이름, 길이 J
        parents    : (J,) int, 루트는 -1
        offsets    : (J, 3) float64, 부모 기준 뼈 offset
        channels   : 관절별 채널 이름 리스트. 예) [["Xposition",...,"Xrotation"], ["Zrotation",...]]
        end_sites  : {관절 인덱스: (3,) End Site offset}
        frame_time : 프레임 간격(초). LaFAN1 = 1/30
        motion     : (F, C) float64, C = 모든 관절 채널 수의 합
    """

    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    channels: list[list[str]]
    end_sites: dict[int, np.ndarray]
    frame_time: float
    motion: np.ndarray

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_frames(self) -> int:
        return self.motion.shape[0]

    def channel_columns(self) -> list[slice]:
        """관절 j 의 채널 값이 motion 의 몇 번째 열에 있는지 (slice 리스트)."""
        cols, start = [], 0
        for ch in self.channels:
            cols.append(slice(start, start + len(ch)))
            start += len(ch)
        return cols


@dataclass
class BvhMotion:
    """회전행렬로 변환된 BVH 동작 (데이터 파이프라인이 실제로 쓰는 형태).

    속성:
        names, parents, offsets, end_sites, frame_time : BvhData 와 같음
        local_rot : (F, J, 3, 3) float64 torch 텐서. 부모 기준 회전 (논문 r_t 의 원본)
        local_pos : (F, J, 3)    float64 torch 텐서. 부모 기준 위치 (루트는 월드 위치 p_t)
        rotation_orders : 관절별 오일러 순서 (예: "zyx"). 다시 저장할 때 사용
    """

    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    local_rot: Tensor
    local_pos: Tensor
    frame_time: float
    rotation_orders: list[str]
    end_sites: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.local_rot.shape[0]

    @property
    def root_pos(self) -> Tensor:
        """루트 월드 위치 (F, 3). 논문 3.2 의 p_t."""
        return self.local_pos[:, 0, :]


# -----------------------------------------------------------------------------
# 읽기
# -----------------------------------------------------------------------------
def parse_bvh(path: str | Path) -> BvhData:
    """BVH 텍스트를 파싱한다.

    입력:  path - .bvh 파일 경로
    출력:  BvhData  (motion shape: (F, C))

    파싱 방법: HIERARCHY 부분은 공백 기준 토큰으로 나눠 "{", "}" 로 깊이를 추적한다.
    스택에는 ("joint", 관절번호) 또는 ("end", 부모관절번호) 를 넣는다.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "MOTION" not in text:
        raise ValueError(f"{path}: MOTION 구역이 없습니다.")
    header, motion_text = text.split("MOTION", 1)

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}

    tokens = header.split()
    stack: list[tuple[str, int]] = []   # 현재 열려 있는 { } 블록들
    pending: tuple[str, int] | None = None  # 다음 "{" 가 열 블록
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "HIERARCHY":
            i += 1
        elif t in ("ROOT", "JOINT"):
            parent = stack[-1][1] if stack else -1
            if stack and stack[-1][0] != "joint":
                raise ValueError(f"{path}: End Site 안에 JOINT 가 있습니다.")
            names.append(tokens[i + 1])
            parents.append(parent)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            pending = ("joint", len(names) - 1)
            i += 2
        elif t == "End":                      # "End Site"
            pending = ("end", stack[-1][1])
            i += 2
        elif t == "{":
            if pending is None:
                raise ValueError(f"{path}: 예상치 못한 '{{'")
            stack.append(pending)
            pending = None
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            vals = [float(v) for v in tokens[i + 1:i + 4]]
            kind, idx = stack[-1]
            if kind == "joint":
                offsets[idx] = vals
            else:
                end_sites[idx] = np.array(vals, dtype=np.float64)
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            channels[stack[-1][1]] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            raise ValueError(f"{path}: 알 수 없는 토큰 {t!r}")

    # MOTION 구역: "Frames: N" "Frame Time: t" 다음에 숫자들
    mt = motion_text.split()
    if mt[0] != "Frames:" or mt[2] != "Frame" or mt[3] != "Time:":
        raise ValueError(f"{path}: MOTION 헤더 형식이 예상과 다릅니다: {mt[:5]}")
    num_frames = int(mt[1])
    frame_time = float(mt[4])
    num_channels = sum(len(c) for c in channels)
    values = np.array(mt[5:], dtype=np.float64)
    if values.size != num_frames * num_channels:
        raise ValueError(
            f"{path}: 값 개수 {values.size} ≠ 프레임 {num_frames} × 채널 {num_channels}"
        )

    return BvhData(
        names=names,
        parents=np.array(parents, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.float64),
        channels=channels,
        end_sites=end_sites,
        frame_time=frame_time,
        motion=values.reshape(num_frames, num_channels),
    )


def bvh_to_local(bvh: BvhData) -> tuple[Tensor, Tensor, list[str]]:
    """BvhData → 로컬 회전행렬, 로컬 위치.

    입력:  bvh - parse_bvh 결과 (motion: (F, C))
    출력:
        local_rot : (F, J, 3, 3) float64   오일러각(도) → 회전행렬 (euler_to_matrix)
        local_pos : (F, J, 3)    float64   위치 채널이 있으면 그 값, 없으면 OFFSET
        orders    : 길이 J, 관절별 오일러 순서 (회전 채널이 없으면 "zyx" 로 기록)
    """
    F, J = bvh.num_frames, bvh.num_joints
    motion = torch.from_numpy(bvh.motion)
    local_rot = torch.eye(3, dtype=torch.float64).repeat(F, J, 1, 1)
    local_pos = torch.from_numpy(bvh.offsets).clone().repeat(F, 1, 1)
    orders: list[str] = []

    for j, (ch, cols) in enumerate(zip(bvh.channels, bvh.channel_columns())):
        data = motion[:, cols]                              # (F, 이 관절의 채널 수)
        # 위치 채널
        for axis, name in enumerate(_POS_CHANNELS):
            if name in ch:
                local_pos[:, j, axis] = data[:, ch.index(name)]
        # 회전 채널 (파일에 적힌 순서 그대로가 오일러 순서)
        rot_names = [c for c in ch if c in _ROT_CHANNELS]
        if rot_names:
            if len(rot_names) != 3:
                raise ValueError(f"관절 {bvh.names[j]}: 회전 채널이 3개가 아닙니다: {rot_names}")
            order = "".join(_ROT_CHANNELS[c] for c in rot_names)
            euler = torch.stack([data[:, ch.index(c)] for c in rot_names], dim=-1)  # (F, 3)
            local_rot[:, j] = euler_to_matrix(euler, order, degrees=True)
            orders.append(order)
        else:
            orders.append("zyx")
    return local_rot, local_pos, orders


def load_bvh(path: str | Path) -> BvhMotion:
    """BVH 파일 → BvhMotion (회전행렬 형태). parse_bvh + bvh_to_local."""
    bvh = parse_bvh(path)
    local_rot, local_pos, orders = bvh_to_local(bvh)
    return BvhMotion(
        names=bvh.names,
        parents=bvh.parents,
        offsets=bvh.offsets,
        local_rot=local_rot,
        local_pos=local_pos,
        frame_time=bvh.frame_time,
        rotation_orders=orders,
        end_sites=bvh.end_sites,
    )


# -----------------------------------------------------------------------------
# 쓰기
# -----------------------------------------------------------------------------
def write_bvh(
    path: str | Path,
    names: Sequence[str],
    parents: Sequence[int],
    offsets: np.ndarray | Tensor,
    local_rot: Tensor,
    root_pos: Tensor,
    frame_time: float = 1.0 / 30.0,
    order: str = "zyx",
    end_sites: dict[int, np.ndarray] | None = None,
    precision: int = 6,
) -> None:
    """로컬 회전행렬 + 루트 위치 → BVH 파일.

    루트는 6채널(위치 3 + 회전 3), 나머지 관절은 회전 3채널로 저장한다.
    (논문 3.2 의 표현과 같은 정보량: 루트 위치 + 모든 관절 회전 + 고정 offset)

    입력:
        names, parents : 길이 J
        offsets        : (J, 3)
        local_rot      : (F, J, 3, 3)
        root_pos       : (F, 3)
        order          : 저장할 오일러 순서 (LaFAN1 은 "zyx")
        end_sites      : {관절 인덱스: (3,)} - 자식이 없는 관절에 붙일 End Site offset.
                         없으면 (0,0,0) 으로 쓴다 (많은 뷰어가 End Site 를 기대하기 때문).
    출력:  없음 (파일 저장)
    """
    offsets = np.asarray(offsets.cpu() if isinstance(offsets, Tensor) else offsets, dtype=np.float64)
    end_sites = end_sites or {}
    J = len(names)
    children: list[list[int]] = [[] for _ in range(J)]
    for j in range(1, J):
        children[parents[j]].append(j)
    rot_ch = " ".join(f"{a.upper()}rotation" for a in order)
    fmt = f"{{:.{precision}f}}"

    lines = ["HIERARCHY"]

    def write_joint(j: int, depth: int) -> None:
        ind = "\t" * depth
        lines.append(f"{ind}{'ROOT' if j == 0 else 'JOINT'} {names[j]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET " + " ".join(fmt.format(v) for v in offsets[j]))
        if j == 0:
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition {rot_ch}")
        else:
            lines.append(f"{ind}\tCHANNELS 3 {rot_ch}")
        for c in children[j]:
            write_joint(c, depth + 1)
        if not children[j]:
            es = end_sites.get(j, np.zeros(3))
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET " + " ".join(fmt.format(v) for v in es))
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    write_joint(0, 0)

    euler = matrix_to_euler(local_rot.to(torch.float64), order, degrees=True)   # (F, J, 3)
    frames = torch.cat([root_pos.to(torch.float64), euler.reshape(euler.shape[0], -1)], dim=-1)
    lines.append("MOTION")
    lines.append(f"Frames: {frames.shape[0]}")
    lines.append(f"Frame Time: {frame_time:.8f}")
    for row in frames.cpu().numpy():
        lines.append(" ".join(fmt.format(v) for v in row))

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
