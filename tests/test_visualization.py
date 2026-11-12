"""
4단계 시각화 도우미 테스트 (src/visualization/skeleton_video.py) — 데이터 없이 실행

확인 내용
    - 구간 라벨: [BEFORE | CONTEXT | GENERATED | TARGET 유지] 길이와 순서
    - 영상 프레임 → 늘린 정답 시퀀스 프레임 번호 (target 유지 구간은 target 프레임 반복)
    - auto 배치: 두 캐릭터가 가까우면 panels, 멀면 side
    - 설정 파일 configs/viz/video.yaml 이 VideoStyle 로 읽히는지
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_named_config
from src.visualization.skeleton_video import (
    VideoStyle,
    choose_layout,
    min_horizontal_distance,
    segment_labels,
    video_frame_index,
)


def test_segment_labels_order_and_length():
    segs = segment_labels(before=3, ctx=2, trans=4, hold=2)
    assert segs == ["before"] * 3 + ["context"] * 2 + ["generated"] * 4 + ["target"] * 2
    with pytest.raises(ValueError):
        segment_labels(1, 1, 1, hold=0)


def test_video_frame_index_holds_target():
    before, ctx, trans, hold = 3, 2, 4, 3
    idx = video_frame_index(before, ctx, trans, hold)
    target = before + ctx + trans                       # 늘린 시퀀스에서 target 프레임 번호 = 9
    assert len(idx) == len(segment_labels(before, ctx, trans, hold))
    assert list(idx[:target + 1]) == list(range(target + 1))
    assert all(i == target for i in idx[target:])


def test_choose_layout_auto():
    style = VideoStyle.from_cfg(load_named_config("viz", "video"))
    F, J = 5, 4
    gt = np.zeros((F, J, 3))
    gt[..., 1] = np.arange(J)                           # 세로로 선 관절 (수평 위치는 모두 원점)
    # 같은 자리 → side_offset 만큼 띄우면 떨어짐 → side
    assert choose_layout(gt, gt.copy(), style) == "side"
    # PRED 가 −Z 로 side_offset 만큼 움직여 있으면, 띄운 뒤 GT 와 겹침 → panels
    pred = gt.copy()
    pred[..., 2] -= style.side_offset_cm
    assert min_horizontal_distance(gt, pred + [0, 0, style.side_offset_cm]) == pytest.approx(0.0)
    assert choose_layout(gt, pred, style) == "panels"


def test_video_style_from_config_has_all_colors():
    cfg = load_named_config("viz", "video")
    style = VideoStyle.from_cfg(cfg)
    assert set(style.colors) >= {"before", "context", "generated", "target"}
    bad = dict(cfg, colors={"context": "blue"})
    with pytest.raises(ValueError):
        VideoStyle.from_cfg(bad)
