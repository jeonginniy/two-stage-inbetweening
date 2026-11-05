# 가정과 미확인 사항 (Assumptions)

논문에 명시되지 않았거나, 노트북에 LaFAN1 이 없어 확인하지 못한 것을 모두 적는다.
"확인 방법" 의 스크립트는 연구실에서 실행한다.

표기: ✅ 확인됨 · 🟡 간접 근거 있음 (최종 확인 필요) · 🔬 연구실에서 확인 필요

## A. 데이터 / 스켈레톤

| # | 가정 | 근거 | 상태 | 확인 방법 |
|---|---|---|---|---|
| A1 | LaFAN1 22관절 이름·부모·순서는 `assets/skeleton_lafan1.yaml` 과 같다 | 공식 코드 테스트(22관절, 발 인덱스 3,4/7,8), LaFAN1 공개 정보 | 🔬 | `check_bvh_roundtrip.py` 의 `skel` 열, `pytest -m lab` |
| A2 | 루트가 아닌 관절의 BVH 위치 채널 값은 OFFSET 과 같다. 그래서 모델 입력에는 루트 위치만 넣고, 나머지는 OFFSET 을 쓴다 | 논문 3.2 "local position ... is constant" | 🔬 | `check_bvh_roundtrip.py` 의 `pos=off` 열 (≈0 이어야 함) |
| A3 | 루트(Hips)의 로컬 **+Y 축이 "앞"** 이다 (정렬할 때 이 축을 월드 +X 로 돌림) | 공식 코드 주석. 또 공식 통계 파일에서 정렬 후 루트 6D 의 2번째 열 평균이 (0.80, −0.05, −0.01) → 로컬 Y 가 거의 +X 를 향함, 루트 x 위치 평균이 +26cm (앞으로 걸어감) | 🟡 | `check_bvh_roundtrip.py` 의 `heading` 열 (걷기/달리기 파일에서 1 에 가까워야 함) |
| A4 | 파일마다 OFFSET 이 다를 수 있다고 보고, 데이터셋은 클립마다 그 파일의 offset 을 함께 준다 | 공식 코드도 파일별 위치값을 FK 에 씀 | 🔬 | `check_bvh_roundtrip.py` 요약의 "모든 파일 OFFSET 동일" |
| A5 | LaFAN1 의 발가락(LeftToe, RightToe) 로컬 회전은 사실상 상수다 | 공식 통계 파일에서 해당 6D 차원의 σ ≈ 2e-8 | 🟡 | `compute_stats.py` 가 출력하는 "σ < 1e-6 인 차원" |

## B. 표현 / 전처리 (논문 3.2, 3.6.1)

| # | 가정 | 근거 | 상태 |
|---|---|---|---|
| B1 | 6D = 회전행렬의 **앞 두 열**, 벡터 안 순서 `[R00, R01, R10, R11, R20, R21]` | 논문은 "rows or columns" 라고만 함. 공식 코드와 같은 선택 → 통계를 공식과 차원별로 비교 가능 | 결정 사항 |
| B2 | 벡터 배치 = `[관절0 6D, …, 관절21 6D, 루트 x,y,z]` (D = 135) | 논문 3.2 "flatten r_t and p_t ... concatenate", 공식 config 의 인덱스 (0~132 회전, 132~135 위치) | ✅ (공식 config 와 일치) |
| B3 | BVH 오일러: 채널 순서가 곧 곱 순서. `Zrotation Yrotation Xrotation` → R = Rz·Ry·Rx, 단위 도 | BVH 표준 관례 | ✅ (손계산 테스트) |
| B4 | "마지막 context 프레임" = 0-based 인덱스 `context_len − 1` (=9) | 논문 3.6.1 | 결정 사항 |
| B5 | 정렬은 루트 XZ 만 0 으로 (높이 Y 는 유지), 회전은 위쪽(Y)축 기준만 | 논문 3.6.1 "XZ coordinates are zeroed", "faces the X+ axis" | ✅ (테스트) |
| B6 | z-score 통계: train_stats 분할(window 50, offset 20)의 정렬된 모든 클립·모든 프레임에 대한 차원별 평균, **모집단** 표준편차(ddof=0) | 논문 3.6.1, 공식 코드 `np.std` | 결정 사항 |
| B7 | σ < 1e-6 인 차원은 σ = 1 로 대체 (0 나누기 방지). **공식 코드와 다름** (공식은 그대로 나눔) | A5 의 발가락 차원 때문. 두 방식 모두 그 차원 값이 ≈0 이 되므로 학습엔 영향이 거의 없을 것으로 예상 | 결정 사항 (2단계에서 영향 확인) |
| B8 | 발 접촉 라벨: 프레임 간 \|dx\|+\|dy\|+\|dz\| < 0.2 cm 이면 접촉. 원래 월드 좌표에서 계산, 마지막 프레임은 앞 프레임 복사 | 논문에 생성 방법 없음. 공식 코드 방식 | 가정 |

## C. 분할 / 하이퍼파라미터

| # | 내용 | 근거 |
|---|---|---|
| C1 | train window = **42** (= 10 + 30 + 2) | 논문 3.6.1. 공식 context config 는 41, detail config 는 42. 이유: Detail 학습 때 (고정된) Context 모델을 후처리(식 23)까지 돌리는데, 후처리가 target 다음 프레임 1개를 더 쓰기 때문 (`docs/context_detail_connection.md`). **2단계에서 Context 용으로 41/42 중 결정** |
| C2 | val 분할 (subject5, window 65, offset 160, start_frame 35), `min_trans = 5` | 논문에 없음. 공식 config 값 |
| C3 | 클립 수 = (F − window) // offset + 1 | 공식 코드와 같은 공식. 논문 수치(train 404195, benchmark 2232)와 맞는지 `inspect_dataset.py` 로 확인 🔬 |

## D. 노트북에서 확인한 것 / 못 한 것

- ✅ 회전 변환, 6D 왕복, FK, BVH 파싱(손으로 쓴 BVH), BVH 쓰기→읽기, 정렬/역정렬, 전체 파이프라인 왕복
  → 가짜 데이터로 `pytest` 48개 통과, 복원 오차 ~1e-14
- ✅ 연구실 스크립트 5개가 가짜 LaFAN1 형식 폴더에서 끝까지 실행됨 (값 자체는 가짜라 의미 없음)
- 🔬 실제 LaFAN1 에서의 모든 수치 (A1~A5, C3, 통계 비교) → `scripts/README.md` 의 "연구실 검증 순서"
- 공식 코드 폴더 `C:\Users\aa010\projects\motion_inbetweening` 가 노트북에 없어서, 참고용으로 GitHub 에서
  임시 폴더에 받아 **읽기만** 했다 (실행/수정/import 하지 않음).
