# tests/ — 테스트

```bash
pytest            # 노트북: 데이터 없이 도는 테스트 전부 (lab 테스트는 자동 skip)
pytest -m lab -v  # 연구실: TSI_ENV=lab 일 때 실제 LaFAN1 테스트
```

| 파일 | 데이터 필요 | 확인 내용 |
|---|---|---|
| `test_rotations.py` | ✗ | 오일러↔행렬 (6가지 순서, 짐벌락), 6D 배치/왕복/Gram–Schmidt/미분, 쿼터니언 |
| `test_skeleton_fk.py` | ✗ | 스켈레톤 YAML, 손계산 FK, 루트+offset 로컬 위치 |
| `test_bvh.py` | ✗ | 손으로 쓴 작은 BVH 파싱(+End Site), FK 손계산, 쓰기→읽기 왕복 |
| `test_representation.py` | ✗ | 135차원 배치, encode/decode, z-score, 정렬 성질(XZ=0, +X, 강체변환), 전체 파이프라인 왕복, 발 접촉 |
| `test_dataset.py` | ✗ (가짜 BVH 폴더) | 배우 필터, 클립 수, start_frame, DataLoader→pipeline, 캐시 |
| `test_context_model.py` | ✗ (가짜 BVH 폴더) | 2단계: 차원이 관절 수를 따름, 마스크/Keyframe PE/보간/합치기/후처리, 상대 위치 어텐션 직접 계산 비교, 마스크 동작, forward/backward, 과적합, 학습 스크립트 `--fake-data` |
| `test_visualization.py` | ✗ | 4단계: 구간 라벨 순서/길이, 영상 프레임 → 시퀀스 프레임 (target 유지), auto 배치 판단, video.yaml 읽기 |
| `test_lafan1_real.py` | ✔ 🔬 | walk1_subject1 = 7840 프레임, 스켈레톤 일치, 위치채널=OFFSET, 실제 데이터 왕복 |
| `helpers.py` | – | 가짜 LaFAN1 형식 데이터 생성 (22관절 계층 + 임의 offset + 사인파 동작). **실제 값 아님** |

## 실행 결과

```
1단계 노트북 (2026-11-05):   48 passed, 4 skipped
2단계 이 환경 (2026-11-05):  65 passed, 4 skipped   (Linux, Python 3.13, torch 2.15 CPU)
```
skip 된 4개는 `test_lafan1_real.py` (실제 데이터 필요).
