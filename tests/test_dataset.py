"""
데이터셋 테스트 (src/data/dataset.py) — 가짜 BVH 폴더로 실행 (데이터 없이 가능)

확인하는 것:
- 파일 이름의 배우(subjectN)로 걸러내는가
- 슬라이딩 윈도우 클립 수 = (F − window)//offset + 1 인가, start_frame 이 적용되는가
- 클립 내용이 원래 파일의 해당 구간과 같은가, shape 이 맞는가
- DataLoader 로 묶어서 pipeline 에 바로 넣을 수 있는가
- 캐시(.npz)를 만들고 다시 읽어도 결과가 같은가
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LafanDataset, list_bvh_files, num_windows
from src.data.pipeline import motion_to_model_input
from src.data.skeleton import Skeleton
from tests.helpers import SKELETON_YAML, write_fake_lafan_bvh

FRAMES = {"walk1_subject1": 80, "run1_subject2": 61, "walk1_subject5": 70, "dance1_subject1": 30}


@pytest.fixture
def fake_dir(tmp_path):
    truth = {}
    for i, (stem, n) in enumerate(FRAMES.items()):
        _, root, rot = write_fake_lafan_bvh(tmp_path / f"{stem}.bvh", n, seed=i)
        truth[stem] = (root, rot)
    (tmp_path / "readme.txt").write_text("not a bvh")
    return tmp_path, truth


def make_ds(path, **kw):
    args = dict(actors=["subject1", "subject2"], window=50, offset=20,
                skeleton=Skeleton.from_yaml(SKELETON_YAML), dtype=torch.float64, verbose=False)
    args.update(kw)
    return LafanDataset(path, **args)


def test_num_windows():
    assert num_windows(80, 50, 20) == 2      # 시작 0, 20  (40 이면 90 > 80)
    assert num_windows(50, 50, 1) == 1
    assert num_windows(30, 50, 1) == 0


def test_actor_filter(fake_dir):
    path, _ = fake_dir
    names = [p.stem for p in list_bvh_files(path, ["subject1"])]
    assert names == ["dance1_subject1", "walk1_subject1"]


def test_lengths_and_items(fake_dir):
    path, truth = fake_dir
    ds = make_ds(path)
    # dance1(30)=0, run1(61)=1, walk1(80)=2  → 3
    assert len(ds) == 3
    item = ds[2]
    assert item["root_pos"].shape == (50, 3)
    assert item["local_rot"].shape == (50, 22, 3, 3)
    assert item["contacts"].shape == (50, 4)
    assert item["offsets"].shape == (22, 3)
    # 마지막 클립 = walk1_subject1 의 20~69 프레임 (BVH 소수점 6자리 저장 오차 허용)
    assert ds.files[item["file_index"]].stem == "walk1_subject1" and item["start"] == 20
    root, rot = truth["walk1_subject1"]
    assert torch.allclose(item["root_pos"], root[20:70], atol=1e-5)
    assert torch.allclose(item["local_rot"], rot[20:70], atol=1e-5)
    with pytest.raises(IndexError):
        ds[3]


def test_start_frame(fake_dir):
    path, truth = fake_dir
    ds = make_ds(path, actors=["subject5"], window=65, offset=40, start_frame=5)
    assert len(ds) == 1                     # 70 - 5 = 65 프레임 → 1 클립
    root, _ = truth["walk1_subject5"]
    assert torch.allclose(ds[0]["root_pos"], root[5:70], atol=1e-5)


def test_dataloader_into_pipeline(fake_dir):
    path, _ = fake_dir
    ds = make_ds(path, window=42, offset=1, dtype=torch.float32)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert batch["root_pos"].shape == (4, 42, 3)
    assert batch["local_rot"].shape == (4, 42, 22, 3, 3)
    inp = motion_to_model_input(batch["root_pos"], batch["local_rot"], 10, ds.skeleton.root_forward_local_axis)
    assert inp.x.shape == (4, 42, 135) and inp.x.dtype == torch.float32


def test_cache_roundtrip(fake_dir, tmp_path_factory):
    path, _ = fake_dir
    cache = tmp_path_factory.mktemp("cache")
    ds1 = make_ds(path, cache_dir=cache)
    assert len(list((cache / "lafan1_parsed").glob("*.npz"))) == 3  # subject1,2 파일 3개
    ds2 = make_ds(path, cache_dir=cache)                             # 이번엔 캐시에서 읽음
    for i in range(len(ds1)):
        for k in ("root_pos", "local_rot", "contacts"):
            assert torch.equal(ds1[i][k], ds2[i][k])


def test_skeleton_mismatch_raises(fake_dir):
    path, _ = fake_dir
    skel = Skeleton.from_yaml(SKELETON_YAML)
    wrong = Skeleton(names=("X",) + skel.names[1:], parents=skel.parents)
    with pytest.raises(ValueError):
        make_ds(path, skeleton=wrong)
