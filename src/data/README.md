# src/data/ — 1단계 데이터 파이프라인

BVH 를 읽어서 논문 3.2 의 모델 입력 `x ∈ R^{T×135}` 로 바꾸고, 다시 원래 동작으로 복원한다.

## 데이터 흐름

```
 LaFAN1 .bvh
     │ bvh.load_bvh                         오일러각(도) → 회전행렬
     ▼
 local_rot (F,22,3,3), root_pos (F,3)       ← 원래 월드 좌표
     │ dataset.LafanDataset                  슬라이딩 윈도우 (논문 3.6.1, 4.1) + 발 접촉 라벨
     ▼
 배치: root_pos (B,T,3), local_rot (B,T,22,3,3), contacts (B,T,4)
     │ pipeline.motion_to_model_input
     │   ├ preprocess.canonicalize_root      마지막 context 프레임에서 루트 XZ=0, +X 방향 (논문 3.6.1)
     │   ├ representation.encode_pose        6D 회전 132 + 루트 위치 3 = 135 (논문 3.2)
     │   └ ZScoreNormalizer.normalize        (x̃ − μ) / σ (논문 3.2)
     ▼
 x (B,T,135)  ───── 2단계: 모델 ─────▶  y (B,T,135)
     │ pipeline.model_output_to_motion
     │   ├ ZScoreNormalizer.denormalize
     │   ├ representation.decode_pose        6D → Gram–Schmidt → 회전행렬 (논문 3.2, 식 10)
     │   └ preprocess.uncanonicalize_root    원래 월드 좌표로
     ▼
 local_rot, root_pos ── bvh.write_bvh ──▶ .bvh  (Blender 등에서 확인)
```

## 파일별 역할

| 파일 | 역할 | 논문 |
|---|---|---|
| `rotations.py` | 오일러 ↔ 회전행렬 ↔ 6D ↔ 쿼터니언 변환 (torch, 미분 가능) | 3.2 |
| `skeleton.py` | `Skeleton` (관절 구조), `forward_kinematics`, `local_positions_from_root` | 3.2, 식 17 |
| `bvh.py` | BVH 파싱(`parse_bvh`, `load_bvh`)과 저장(`write_bvh`) | – |
| `preprocess.py` | 클립 정렬/역정렬, 발 접촉 라벨 | 3.6.1, 3.4 |
| `representation.py` | `FeatureLayout`, `encode_pose`/`decode_pose`, `RunningStats`, `ZScoreNormalizer` | 3.2, 식 10 |
| `dataset.py` | `LafanDataset` (배우 필터, 슬라이딩 윈도우, .npz 캐시) | 3.6.1, 4.1 |
| `pipeline.py` | 위를 묶은 `motion_to_model_input`, `model_output_to_motion` | 3.2, 3.6.1 |

## 1단계에서 하지 않은 것 (2단계에서 구현)

- 결측 프레임 처리: 루트 위치 선형보간, 회전 0, 마스크 `m_in` 이어붙이기 (논문 3.3)
- 어텐션 마스크, Keyframe 위치 인덱스 (논문 3.3, 3.3.1)
- 손실 함수 (논문 3.5) — 단, 손실에 필요한 FK 와 6D→행렬 변환은 이미 미분 가능하게 만들어 두었다.
