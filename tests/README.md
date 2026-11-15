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
| `test_detail_model.py` | ✗ (가짜 BVH 폴더) | 3단계: 차원이 스켈레톤을 따름 (136 → 135 + 4), 마스크 (그림 5b, target 이후 가림), Detail 입력 (제약 프레임 = 정답), 고정된 Context 에 gradient 없음, L_foot 가 ĉ 로 gradient 안 보냄, L_c / L_foot 손계산, 두 단계 추론, 과적합, `train_detail.py` 가짜 데이터 실행 |
| `test_evaluation_detail.py` | ✗ (가짜 BVH + tmp 의 가짜 학습 결과) | 4단계 두 단계 평가: `load_detail` 이 다른 Context·`--fake-data` 결과를 거부, `generate(detail=)` = `infer_two_stage`, 입력 프레임 = 정답, `eval_lengths` 의 `detail`/`detail_post` 지표 = 직접 계산한 값 |
| `test_foot_only.py` | ✗ (가짜 BVH 폴더) | 발 전용 모델: 발 관절 번호 = contact_joints, 발 위치 = FK 결과의 발 관절·월드 복원 일치, 입력 148 (전신 부분 = Context 입력, 발 위치는 아는 프레임만), 출력 16, 손실 손계산·transition 밖 정답, 배치 하나 과적합, validate 지표 키, `train_foot.py` 가짜 데이터 실행·덮어쓰기 거부·resume, 평가 (동작 이름, GT 위치 판정 접촉 = 라벨, 방법별 발 위치·GT 검산 행·발 미끄러짐이 전신 함수와 같음) |
| `test_feet_to_body.py` | ✗ (가짜 BVH 폴더) | 발 → 몸 모델: 발 조건 16 = 발 위치 12 ⊕ 접촉 4, 입력 152 (전신 부분·m_in = Context 입력, 발 조건은 모든 프레임), 어텐션 마스크 (attend_all), 출력 135, 손실 = 식 21 가중합·transition 밖 정답, 발 조건이 출력에 영향 (마스크가 있으면 그 프레임에만), 정답 전신의 발 오차 0, 배치 하나 과적합, validate 지표 키, `train_body.py` 가짜 데이터 실행·덮어쓰기 거부·resume·기존 실험 폴더 보호, 평가 (방법별 전신 shape, transition 밖 정답, 예측한 발 조건은 transition 에서만 다름), 설정 base 합치기, validate 의 cond_fn (cond_l2, foot_follow), best_mean.pth 저장 |
| `test_metrics.py` | ✗ | 4단계: 정답이면 지표 0, L2P 평행이동 불변, 쿼터니언 부호 정리, NPSS 손계산·부호 불변, 발 미끄러짐 (정지/바닥/높이), 평균 중심, 기준선 (context·target 유지, zero-velocity, SLERP) |
| `test_visualization.py` | ✗ | 4단계: 구간 라벨 순서/길이, 영상 프레임 → 시퀀스 프레임 (target 유지), auto 배치 판단, video.yaml 읽기 |
| `test_lafan1_real.py` | ✔ 🔬 | walk1_subject1 = 7840 프레임, 스켈레톤 일치, 위치채널=OFFSET, 실제 데이터 왕복 |
| `helpers.py` | – | 가짜 LaFAN1 형식 데이터 생성 (22관절 계층 + 임의 offset + 사인파 동작). **실제 값 아님** |

## 실행 결과

```
1단계 노트북 (2026-11-05):   48 passed, 4 skipped
2단계 이 환경 (2026-11-05):  65 passed, 4 skipped   (Linux, Python 3.13, torch 2.15 CPU)
3단계 연구실 (2026-11-12):   90 passed, 4 skipped
발 전용 연구실 (2026-11-13): 98 passed, 4 skipped
발 전용 평가 (2026-11-14):   101 passed, 4 skipped
발 → 몸 (2026-11-15):        113 passed, 4 skipped
```
skip 된 4개는 `test_lafan1_real.py` (실제 데이터 필요).
