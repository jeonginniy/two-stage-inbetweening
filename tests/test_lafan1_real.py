"""
[연구실 전용] 실제 LaFAN1 BVH 로 하는 테스트
=========================================

실행 방법 (연구실 Ubuntu):
    TSI_ENV=lab pytest -m lab -v

TSI_ENV 가 없거나 configs/env/<환경>.yaml 의 lafan1_bvh_dir 이 없으면 전부 skip 된다
(노트북에서는 skip 되는 것이 정상).

확인하는 것:
- walk1_subject1.bvh: 7840 프레임, 22 관절 (공식 코드 tests/test_data.py 에 적힌 값)
- 관절 이름/부모가 assets/skeleton_lafan1.yaml 과 같은가
- 루트 외 관절의 위치 채널 = OFFSET 인가 (논문 3.2 "로컬 위치는 상수" 가정)
- 실제 데이터에서 클립 → 모델 입력 → 복원 오차가 충분히 작은가
"""

import os

import pytest
import torch

from src.config import load_data_config, load_env_config
from src.data.bvh import load_bvh
from src.data.pipeline import model_output_to_motion, motion_to_model_input
from src.data.representation import FeatureLayout
from src.data.skeleton import Skeleton

pytestmark = pytest.mark.lab


@pytest.fixture(scope="module")
def walk1():
    env = os.environ.get("TSI_ENV")
    if not env:
        pytest.skip("TSI_ENV 미설정 (연구실에서 TSI_ENV=lab 으로 실행)")
    bvh_dir = load_env_config(env)["paths"].get("lafan1_bvh_dir")
    if bvh_dir is None or not (bvh_dir / "walk1_subject1.bvh").exists():
        pytest.skip(f"LaFAN1 BVH 없음: {bvh_dir}")
    return load_bvh(bvh_dir / "walk1_subject1.bvh")


def test_walk1_shape(walk1):
    assert walk1.local_rot.shape == (7840, 22, 3, 3)
    assert walk1.local_pos.shape == (7840, 22, 3)


def test_skeleton_matches_asset(walk1):
    skel = Skeleton.from_yaml(load_data_config()["skeleton"])
    skel.check_matches(walk1.names, walk1.parents, source="(walk1_subject1.bvh)")


def test_non_root_positions_equal_offsets(walk1):
    offsets = torch.from_numpy(walk1.offsets)
    diff = (walk1.local_pos[:, 1:] - offsets[1:]).abs().max().item()
    assert diff < 1e-4, f"루트 외 관절의 위치 채널이 OFFSET 과 다름 (최대 차이 {diff})"


def test_real_roundtrip(walk1):
    T = 50
    s = min(1000, walk1.num_frames - T)  # 파일 중간쯤의 50프레임
    root = walk1.root_pos[s:s + T][None]
    rot = walk1.local_rot[s:s + T][None]
    skel = Skeleton.from_yaml(load_data_config()["skeleton"])
    inp = motion_to_model_input(root, rot, 10, skel.root_forward_local_axis)
    p, r = model_output_to_motion(inp.x, FeatureLayout(22), frame=inp.frame)
    assert torch.allclose(p, root, atol=1e-8) and torch.allclose(r, rot, atol=1e-10)
