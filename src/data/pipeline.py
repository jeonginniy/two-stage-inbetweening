"""
데이터 파이프라인 전체 연결: 클립 → 모델 입력 → 원래 형태로 복원
================================================================

이 파일이 하는 일
-----------------
1단계의 목표인 "BVH 읽기 → 모델 입력 표현으로 변환 → 원래 형태로 복원" 중
뒤의 두 단계를 한 함수씩으로 묶었다. 2단계(Context Transformer 학습)에서는
이 두 함수만 부르면 된다.

    [dataset.py 가 준 배치]
      root_pos (B,T,3), local_rot (B,T,J,3,3)      ← 원래 월드 좌표
            │ motion_to_model_input()
            │   1) canonicalize_root   (논문 3.6.1: 마지막 context 프레임 기준 정렬)
            │   2) encode_pose         (논문 3.2: 6D 회전 + 루트 위치 → x̃, D=135)
            │   3) normalize           (논문 3.2: z-score → x)
            ▼
      ModelInput.x (B,T,135)   ── 2단계: mask/placeholder 처리 후 모델에 넣음 ──▶ y (B,T,135)
            │ model_output_to_motion()
            │   1) denormalize   2) decode_pose (Gram–Schmidt)   3) uncanonicalize_root
            ▼
      root_pos (B,T,3), local_rot (B,T,J,3,3)      ← 다시 원래 월드 좌표 (BVH 로 저장 가능)

주의: 결측 프레임 처리(루트 선형보간, 회전 0, mask 이어붙이기; 논문 3.3)는 모델 쪽 일이므로
2단계에서 구현한다. 여기서는 "완전한 정답 클립" 을 표현으로 바꾸는 것까지만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor

from src.data.preprocess import CanonicalFrame, canonicalize_root, uncanonicalize_root
from src.data.representation import FeatureLayout, ZScoreNormalizer, decode_pose, encode_pose


@dataclass
class ModelInput:
    """motion_to_model_input 의 결과.

    속성:
        x         : (B, T, D)        정규화된 입력 (논문의 x). normalizer 가 없으면 x_raw 와 같음
        x_raw     : (B, T, D)        정규화 전 (논문의 x̃)
        root_pos  : (B, T, 3)        정렬된 루트 위치 (손실 계산용 정답 p)
        local_rot : (B, T, J, 3, 3)  정렬된 로컬 회전 (손실 계산용 정답 r, FK 입력)
        frame     : CanonicalFrame   원래 월드 좌표로 되돌릴 때 필요
    """

    x: Tensor
    x_raw: Tensor
    root_pos: Tensor
    local_rot: Tensor
    frame: CanonicalFrame


def motion_to_model_input(
    root_pos: Tensor,
    local_rot: Tensor,
    context_len: int,
    forward_local: Sequence[float],
    normalizer: ZScoreNormalizer | None = None,
    world_forward: str = "x",
) -> ModelInput:
    """원래 월드 좌표의 클립 → 모델 입력.

    입력:
        root_pos      : (B, T, 3)
        local_rot     : (B, T, J, 3, 3)
        context_len   : context 프레임 수 t_ctx (논문 4.1: 10). 정렬 기준 = context_len - 1 번 프레임
        forward_local : 루트의 로컬 앞 축 (skeleton.root_forward_local_axis)
        normalizer    : z-score 정규화기. None 이면 정규화 생략 (통계 계산할 때 사용)
    출력:
        ModelInput (x: (B, T, J*6+3))
    """
    root_c, rot_c, frame = canonicalize_root(root_pos, local_rot, context_len - 1, forward_local, world_forward)
    x_raw = encode_pose(root_c, rot_c)
    x = normalizer.normalize(x_raw) if normalizer is not None else x_raw
    return ModelInput(x=x, x_raw=x_raw, root_pos=root_c, local_rot=rot_c, frame=frame)


def model_output_to_motion(
    x: Tensor,
    layout: FeatureLayout,
    normalizer: ZScoreNormalizer | None = None,
    frame: CanonicalFrame | None = None,
) -> tuple[Tensor, Tensor]:
    """모델 출력(또는 입력) 벡터 → 루트 위치, 로컬 회전행렬.

    입력:
        x          : (B, T, D)  정규화된 벡터 (normalizer 가 None 이면 x̃ 로 간주)
        layout     : FeatureLayout(num_joints=J)
        normalizer : 정규화를 되돌릴 때 사용
        frame      : 주면 원래 월드 좌표로, None 이면 정렬된 좌표 그대로 돌려준다
    출력:
        root_pos  : (B, T, 3)
        local_rot : (B, T, J, 3, 3)
    """
    x_raw = normalizer.denormalize(x) if normalizer is not None else x
    root_pos, local_rot = decode_pose(x_raw, layout)
    if frame is not None:
        root_pos, local_rot = uncanonicalize_root(root_pos, local_rot, frame)
    return root_pos, local_rot
