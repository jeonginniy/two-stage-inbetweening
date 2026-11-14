# 가정과 미확인 사항 (Assumptions)

논문에 명시되지 않았거나, 노트북에 LaFAN1 이 없어 확인하지 못한 것을 모두 적는다.
"확인 방법" 의 스크립트는 연구실에서 실행한다.

표기: ✅ 확인됨 · 🟡 간접 근거 있음 (최종 확인 필요) · 🔬 연구실에서 확인 필요

## 0. 1단계 연구실 검증 결과 (실제 LaFAN1)

연구실에서 `scripts/README.md` 의 검증 순서를 실행한 결과. **1단계 데이터 파이프라인 검증 통과.**

| 항목 | 스크립트 | 결과 | 판정 |
|---|---|---|---|
| 클립 수 | `inspect_dataset.py` | train **404195**, benchmark **2232** — 논문 4.1 과 일치 | ✅ C3 (train window 42 기준) |
| 정규화 통계 vs 공식 `train_stats_context.pkl` | `compute_stats.py` → `compare_stats_with_official.py` | 차원별 평균 차이 **~1e-12**, 표준편차 차이 **~1e-8** | ✅ B1, B2, B6, 정렬(B5, A3) 이 공식 코드와 같음 |
| 스켈레톤 불일치 | `check_bvh_roundtrip.py` `skel` | **0개** | ✅ A1 |
| 위치 채널 = OFFSET | `check_bvh_roundtrip.py` `pos=off` | **0** | ✅ A2 |
| 표현 → 복원 오차 | `check_bvh_roundtrip.py` `rot_err`, `fk_err`, `bvh_err` | **1e-13 이하** | ✅ 논문 3.2 표현에 정보 손실 없음 |
| heading (루트 로컬 +Y = 앞) | `check_bvh_roundtrip.py` `heading` | run/sprint/walk 파일에서 **0.47 ~ 0.88, 모두 양수** | ✅ A3 (아래 설명) |

heading 해석: 처음에 "0.9 보다 크면 OK" 라고 적었지만, 실제 값은 0.47~0.88 이다. 그래도 가정이 맞다고 본다.
- 이 값은 "마지막 context 프레임 다음 5프레임 동안의 루트 이동 방향" 과 +X 의 cos 평균이다.
  LaFAN1 의 걷기/달리기 파일에는 옆걸음, 뒷걸음, 제자리 회전이 많이 섞여 있어서 1 보다 작은 것이 자연스럽다.
- 앞 축을 잘못 골랐다면 평균이 0 근처(90° 틀림)나 음수(180° 틀림)여야 한다. 모든 파일이 양수다.
- 결정적으로, 정렬된 데이터로 낸 통계가 공식 통계와 1e-12 수준으로 같다. 공식 코드도 로컬 +Y 를 앞으로 쓰므로
  정렬 결과가 공식과 같다는 뜻이다.

연구실 결과로 보고되지 않은 것 (필요하면 나중에 확인): `check_bvh_roundtrip.py` 요약의 "모든 파일 OFFSET 동일" (A4),
`pytest -m lab`, 복원 BVH 를 Blender 로 본 결과.

## A. 데이터 / 스켈레톤

| # | 가정 | 근거 | 상태 | 확인 방법 |
|---|---|---|---|---|
| A1 | LaFAN1 22관절 이름·부모·순서는 `assets/skeleton_lafan1.yaml` 과 같다 | 공식 코드 테스트(22관절, 발 인덱스 3,4/7,8), LaFAN1 공개 정보 | ✅ 연구실: 불일치 0 | `check_bvh_roundtrip.py` 의 `skel` 열, `pytest -m lab` |
| A2 | 루트가 아닌 관절의 BVH 위치 채널 값은 OFFSET 과 같다. 그래서 모델 입력에는 루트 위치만 넣고, 나머지는 OFFSET 을 쓴다 | 논문 3.2 "local position ... is constant" | ✅ 연구실: `pos=off` 0 | `check_bvh_roundtrip.py` 의 `pos=off` 열 (≈0 이어야 함) |
| A3 | 루트(Hips)의 로컬 **+Y 축이 "앞"** 이다 (정렬할 때 이 축을 월드 +X 로 돌림) | 공식 코드 주석. 또 공식 통계 파일에서 정렬 후 루트 6D 의 2번째 열 평균이 (0.80, −0.05, −0.01) → 로컬 Y 가 거의 +X 를 향함, 루트 x 위치 평균이 +26cm (앞으로 걸어감) | ✅ 연구실: run/sprint/walk 에서 0.47~0.88 (전부 양수) + 통계가 공식과 일치. 0 절 설명 참고 | `check_bvh_roundtrip.py` 의 `heading` 열 |
| A4 | 파일마다 OFFSET 이 다를 수 있다고 보고, 데이터셋은 클립마다 그 파일의 offset 을 함께 준다 | 공식 코드도 파일별 위치값을 FK 에 씀 | 🟡 결과 미보고. 파일별 offset 을 쓰므로 같든 다르든 동작은 맞음 | `check_bvh_roundtrip.py` 요약의 "모든 파일 OFFSET 동일" |
| A5 | LaFAN1 의 발가락(LeftToe, RightToe) 로컬 회전은 사실상 상수다 | 공식 통계 파일에서 해당 6D 차원의 σ ≈ 2e-8 | ✅ (간접) 내 σ 가 공식과 ~1e-8 차이 → 내 발가락 σ 도 1e-6 보다 훨씬 작음 → B7 적용 대상 | `compute_stats.py` 가 출력하는 "σ < 1e-6 인 차원" |

## B. 표현 / 전처리 (논문 3.2, 3.6.1)

| # | 가정 | 근거 | 상태 |
|---|---|---|---|
| B1 | 6D = 회전행렬의 **앞 두 열**, 벡터 안 순서 `[R00, R01, R10, R11, R20, R21]` | 논문은 "rows or columns" 라고만 함. 공식 코드와 같은 선택 → 통계를 공식과 차원별로 비교 가능 | 결정 사항 |
| B2 | 벡터 배치 = `[관절0 6D, …, 관절21 6D, 루트 x,y,z]` (D = 135) | 논문 3.2 "flatten r_t and p_t ... concatenate", 공식 config 의 인덱스 (0~132 회전, 132~135 위치) | ✅ (공식 config 와 일치, 연구실 통계 차원별 일치) |
| B3 | BVH 오일러: 채널 순서가 곧 곱 순서. `Zrotation Yrotation Xrotation` → R = Rz·Ry·Rx, 단위 도 | BVH 표준 관례 | ✅ (손계산 테스트) |
| B4 | "마지막 context 프레임" = 0-based 인덱스 `context_len − 1` (=9) | 논문 3.6.1 | 결정 사항 |
| B5 | 정렬은 루트 XZ 만 0 으로 (높이 Y 는 유지), 회전은 위쪽(Y)축 기준만 | 논문 3.6.1 "XZ coordinates are zeroed", "faces the X+ axis" | ✅ (테스트) |
| B6 | z-score 통계: train_stats 분할(window 50, offset 20)의 정렬된 모든 클립·모든 프레임에 대한 차원별 평균, **모집단** 표준편차(ddof=0) | 논문 3.6.1, 공식 코드 `np.std` | ✅ 연구실: 공식 통계와 평균 ~1e-12, σ ~1e-8 차이 |
| B7 | σ < 1e-6 인 차원은 σ = 1 로 대체 (0 나누기 방지). **공식 코드와 다름** (공식은 그대로 나눔) | A5 의 발가락 차원 때문. 두 방식 모두 그 차원 값이 ≈0 이 되므로 학습엔 영향이 거의 없을 것으로 예상 | 결정 사항 (2단계에서 영향 확인) |
| B8 | 발 접촉 라벨: 프레임 간 \|dx\|+\|dy\|+\|dz\| < 0.2 cm 이면 접촉. 원래 월드 좌표에서 계산, 마지막 프레임은 앞 프레임 복사 | 논문에 생성 방법 없음. 공식 코드 방식 | 가정 |

## C. 분할 / 하이퍼파라미터

| # | 내용 | 근거 |
|---|---|---|
| C1 | train window = **42** (= 10 + 30 + 2) | 논문 3.6.1. 공식 context config 는 41, detail config 는 42 (`docs/context_detail_connection.md`). **2단계 결정: Context 도 42** (E1) |
| C2 | val 분할 (subject5, window 65, offset 160, start_frame 35), `min_trans = 5` | 논문에 없음. 공식 config 값 |
| C3 | 클립 수 = (F − window) // offset + 1 | 공식 코드와 같은 공식. ✅ 연구실: train 404195, benchmark 2232 로 논문과 일치 |

## D. 1단계: 노트북에서 확인한 것 / 연구실에서 확인한 것

- ✅ 회전 변환, 6D 왕복, FK, BVH 파싱(손으로 쓴 BVH), BVH 쓰기→읽기, 정렬/역정렬, 전체 파이프라인 왕복
  → 가짜 데이터로 `pytest` 48개 통과, 복원 오차 ~1e-14
- ✅ 연구실 스크립트 5개가 가짜 LaFAN1 형식 폴더에서 끝까지 실행됨 (값 자체는 가짜라 의미 없음)
- ✅ 실제 LaFAN1 에서의 수치 (A1~A3, A5, C3, 통계 비교) → 연구실에서 확인 완료 (0 절)
- 공식 코드 폴더 `C:\Users\aa010\projects\motion_inbetweening` 가 노트북에 없어서, 참고용으로 GitHub 에서
  임시 폴더에 받아 **읽기만** 했다 (실행/수정/import 하지 않음).

## E. 2단계 Context Transformer (논문 3.3, 3.5, 3.6)

코드: `src/models/`, `src/training/`, `scripts/train_context.py`, 설정 `configs/model/context.yaml`.
공식 코드는 이번에도 임시 폴더에 받아 **읽기만** 했다 (import/복사/실행 없음).

| # | 내용 | 근거 | 상태 |
|---|---|---|---|
| E1 | Context 학습 window = **42** (공식 context config 는 41) | 어텐션 key 를 context·제약·target 프레임으로 제한하므로 target 다음 프레임(41번)은 어떤 transition 출력에도 영향을 주지 않는다 → 41 과 결과가 같다. 42 로 하면 클립 수가 논문(404195)과 같고 Detail 학습(42)과 같은 데이터셋을 쓴다 | 결정 사항 |
| E2 | 입력/출력 차원은 설정에서 계산: D = J×6+3 (J = 스켈레톤 YAML 관절 수), 입력 = D + d_mask(1), 출력 = D | 논문 3.2, 3.3. 코드에 135/136/22 를 쓰지 않음 (`test_model_follows_joint_count`) | 결정 사항 |
| E3 | 어텐션 점수 스케일 1/√d_head (표준). 공식 코드는 1/√d_model. 설정 `attn_scale` 로 바꿀 수 있음 | 논문은 표준 Transformer 를 따른다고만 함 | 결정 사항 (성능 차이 나면 `d_model` 로 비교) |
| E4 | pre-LN 블록 = `x + Sublayer(LN(x))` (표준). 공식 코드는 `LN(x) + Sublayer(LN(x))` (residual 에 정규화된 값을 더함; 공식 코드 docstring 과도 다름) | 표준 pre-LN | 결정 사항 (공식과 다름) |
| E5 | 상대 위치 임베딩 r_{j−i}: 거리 값(스칼라)을 2층 MLP 에 넣어 d_head 벡터, 모든 헤드·층이 공유, 점수에 q_i·r_{j−i} 를 더함 | 논문 식 12–13, 공식 코드 (방향 j−i = key−query 를 공식 skew 트릭과 맞춤) | 공식과 같음 |
| E6 | Keyframe PE 입력 = [t − (t_ctx−1), t_tgt − t], 2층 MLP 출력을 인코더 출력에 더함 | 논문 식 11, 공식 코드 | 공식과 같음 |
| E7 | "회전 = 0" placeholder = **정규화된 공간의 0**. 루트 위치는 앵커(마지막 context, 제약, target) 사이 선형보간 (정규화 공간에서 해도 같음). target 이후 프레임은 0 | 논문 3.3, 공식 코드 | 가정 (공식과 같음) |
| E8 | 어텐션 마스크: 모든 query 가 context·제약·target 프레임만 key 로 봄 (target 이후 프레임도 안 봄) | 논문 그림 5a, 공식 코드 | 공식과 같음 |
| E9 | transition 길이는 **배치마다 하나** 를 [min_trans, max_trans] 에서 균등하게 뽑는다 (배치 안 클립은 같은 길이) | 공식 코드. 논문은 "randomly sample" 이라고만 함 | 가정 |
| E10 | 제약(midway) 프레임: int(0.1 × trans) 번 시도, 각 확률 0.5 로 transition 안 임의 프레임 추가, **전 관절 제약** (d_mask = 1). 부분 제약은 미구현 | 공식 config (`midway_targets_amount/p`, `constrained_slices [[null,null]]`) | 가정 |
| E11 | 손실 (식 21): L_rot(6D, 정규화 되돌린 값) + 0.01·L_pos(FK 전역 위치, cm) + 0.005·L_smooth(예측 전역 위치의 프레임 간 차이, [t_ctx−1, target] 구간), 모두 L1·원소 평균·transition 프레임만. 제약 프레임도 손실에 포함 | 논문 식 21 의 가중치, 공식 코드의 세부 정의 | 가정 (공식과 같음) |
| E12 | 손실 전에 transition 이외 프레임은 정답으로 덮어쓴다 (gradient 는 transition 출력으로만) | 논문 3.6.2, 공식 코드 | 공식과 같음 |
| E13 | 모델 크기: 6층, 8헤드, d_model 512, d_head 64, FFN 512, 인코더/디코더 은닉 512, dropout 0 → **1,038만 파라미터** | 공식 config (논문 수치는 이 환경에서 재확인 못 함) | 🔬 논문 4.1 구현 세부와 대조 필요 |
| E14 | 최적화: Adam, Noam 스케줄 lr = 0.0625 × min(it·8000^-1.5, it^-0.5) (최대 ≈ 7.0e-4), batch 32 | 공식 config/코드 | 가정 |
| E15 | 학습 길이 `max_iters` 500,000 | 논문에 없음, 공식은 2000 epoch(사실상 수동 중단) | 🔬 연구실에서 val 곡선 보고 조절 |
| E16 | 검증: val 분할(subject5, window 65, offset 160, start 35)을 앞에서 t_ctx+trans+2 프레임으로 잘라 trans ∈ {5,15,30} 에서 학습과 같은 손실 + 후처리 후 전역 위치 L2(cm). best 체크포인트는 trans 30 손실 기준 | 공식 코드는 L2P/L2Q/NPSS 로 고름. 공식 평가 지표는 4단계에서 구현 | 결정 사항 (임시) |
| E17 | 후처리 ψ (식 23): 공식 코드의 근사 (양 끝 선형 외삽과 예측의 차이 평균을 δ 로) | 공식 코드 | 가정. 학습에는 안 쓰고 검증/추론에만 |

### 2단계에서 이 환경(데이터 없음)으로 확인한 것

- ✅ 가짜 LaFAN1 형식 데이터로 `pytest` → **65 passed, 4 skipped** (skip = 실제 데이터 테스트).
  새 테스트 17개 (`tests/test_context_model.py`): 차원이 관절 수를 따름, 마스크/PE 입력/보간/합치기/후처리 값,
  상대 위치 어텐션 = 반복문 직접 계산, 모르는 프레임을 바꿔도 다른 프레임 출력 불변(마스크 동작),
  설정 크기 모델 forward/backward (손실 유한, 모든 파라미터 gradient), 작은 모델 과적합(손실 절반 이하),
  학습 스크립트 `--fake-data` 끝까지 실행 (체크포인트/로그 생성, 이어서 학습).
- 🔬 실제 학습, GPU 속도/VRAM, 학습된 모델의 품질 → 연구실 (`scripts/README.md` 의 "2단계: Context Transformer 학습")

## F. 4단계 평가·시각화 (교수님 미팅 피드백, 브랜치 `viz-update`)

코드: `src/evaluation/`, `src/visualization/`, `scripts/render_samples.py`, `scripts/eval_lengths.py`,
설정 `configs/eval/lengths.yaml`, `configs/viz/video.yaml`. 공식 코드는 지표 정의 확인을 위해 **읽기만** 했다.

| # | 내용 | 근거 | 상태 |
|---|---|---|---|
| F1 | **L2P**: 전역 관절 위치 (J×3) 를 z-score 한 뒤 프레임별 L2 → transition 프레임 평균 → 클립 평균. μ, σ 는 Subject 1–4, window 50, offset 20 (20214 클립) 을 **평균 중심 정렬** (루트 XZ 의 클립 평균 = 0, 마지막 context 프레임에서 +X 를 보게) 한 전역 위치의 차원별 평균·모집단 표준편차. 예측·정답은 학습용 정렬 좌표 그대로 쓴다 (같은 평행이동은 차이에서 상쇄) | 논문 4.1 은 Harvey et al. 2020 을 따른다고만 함. 공식 코드 (`benchmark.py`, `rmi.py`) 를 읽고 같은 정의로 새로 작성. 평균 중심 = 우리 정렬 결과에서 루트 XZ 평균만 뺀 것 (yaw 회전이 선형이라 순서 교환 가능) | ✅ interp/zerov 기준선 L2P 가 README 와 소수점 4자리까지 일치 (5/15/30/45) |
| F2 | **NPSS**: transition 구간 전역 쿼터니언 (J×4, 부호 정리) 의 시간축 FFT **실수부** 제곱을 파워로, 차원별 정규화 누적분포의 L1 거리 (EMD), GT 전체 파워로 가중 (가중치는 **데이터셋 전체** 합으로 정규화) 한 합 | 원 논문 (Gopalakrishnan 2019) 은 크기 (magnitude) 를 쓰지만 공식 README 수치는 실수부 기준 | ✅ 기준선 NPSS 가 README 와 소수점 4자리 일치 |
| F3 | **L2Q**: 전역 회전 쿼터니언 (J×4) 의 프레임별 L2 → transition 평균 → 클립 평균. 시간 방향 부호 뒤집힘 제거 (프레임 0 기준). 우리 `matrix_to_quat` 는 w ≥ 0 규약이라 공식 변환과 프레임 0 부호가 다를 수 있으나, GT·예측에 같은 함수를 쓰므로 L2Q·NPSS 는 같다 | 공식 코드 | ✅ 기준선 L2Q 가 README (소수점 2자리) 와 반올림 일치 |
| F3b | 후처리 있음/없음 둘 다 보고 (`model`, `model_nopost`). Context only 비교 대상은 README 의 "w/ post-processing", "WITHOUT post-processing" | README 에 두 수치가 모두 있음 | 결정 사항 |
| F4 | **발 미끄러짐** (논문에 없음): 발가락 (LeftToe, RightToe) 높이 h < H = 3 cm 인 프레임의 수평 속도 × clip(2 − 2^(h/H), 0, 1), transition 프레임·관절 평균 (cm/frame). 같은 값을 GT 로도 계산해 기준으로 보고 | Subject 5 실측: 발목 (Foot) 관절은 서 있을 때도 높이 7~8 cm (1% 분위 6.6 cm), 발가락은 1~3 cm (중앙값 2.5~2.8 cm) → 하나의 H 로 발목·발가락을 함께 다룰 수 없어 발가락만 사용. 가중치 형태는 Zhang et al. 2018 계열에서 흔히 쓰는 식 | 가정 (H 는 설정값, diffusion 모델과 비교할 때 같은 정의를 써야 함) |
| F5 | 긴 길이 평가셋: Subject 5, **window = 10 + max(lengths) + 2 = 312**, offset 40, start 0 → **2151 클립**. 모든 길이에서 같은 시작 프레임 (같은 context), target 만 뒤로 이동. 공식 README 와 비교하는 5/15/30/45 는 공식 벤치마크 (window 65, 2232 클립) 로도 따로 계산 | 공식 벤치마크 window 65 에는 trans ≤ 53 만 들어감. 길이별로 같은 샘플을 비교하려면 가장 긴 길이에 맞춘 window 가 필요 | 결정 사항 |
| F6 | 후처리 (식 23) 는 공식 코드의 근사 (E17): 시작 쪽과 끝 쪽 어긋남의 **평균** δ 를 transition 전체에 더한다. 학습 범위 밖 (예: trans 90) 에서 모델이 target 에 못 미치면 (run1_subject5 4800: 마지막 생성 프레임이 target 에서 ~2 m), δ ≈ 그 절반이 되어 **시작·끝 경계에서 각각 ~1 m 점프** 가 생긴다. 후처리를 끄면 끝 경계에서만 ~2 m 점프 | `render_samples.py` 실측 (trans 90, seed 0 샘플 5개): 후처리 켬 → 경계 점프 최대 +98.8 / +96.7 cm, 끔 → +3.2 / +197.1 cm. trans 30 에서는 ±1.5 cm 이내 | 확인됨. 영상·지표는 후처리 켬/끔 모두 제공 |
| F7 | 영상: 모두 1배속 30fps, `[BEFORE 원본 30 (회색)] [CONTEXT 10] [GENERATED trans] [TARGET 1 + 유지 29]`. BEFORE 는 모델 입력이 아니며, 모델 창과 같은 정렬 (마지막 context 프레임 기준) 로 그린다 | 피드백 [1] + 미팅 후 요청 (재생 속도를 바꾸면 경계가 자연스러운지 판단하기 어려움) | 결정 사항 |
| F8 | 모델 구조상 최대 transition 길이는 없다. 상대 위치 PE (식 12–13) 와 Keyframe PE (식 11) 가 거리 값 (스칼라) 을 MLP 에 넣는 방식이라 고정 길이 표가 없다. 학습 때 본 값은 상대 거리 ≤ 41, keyframe 거리 ≤ 31 이고, 그보다 큰 값은 MLP 외삽 (PReLU → 거의 선형) 이다. 재학습 없이 불가능한 길이는 없으며, 필요한 것은 클립 길이 ≥ ctx + trans + 2 (Subject 5 의 가장 짧은 파일 4742 프레임) 와 메모리 (어텐션 O(T²); T = 312, batch 64 에서 8 GB 안) 뿐 | 코드 (`src/models/layers.py`) | 확인됨 |
| F9 | 결과표의 "학습 범위 밖" = 학습 transition 범위 [min_trans, max_trans] = [5, 30] (학습 run 의 config.yaml) 밖. 공식 README 의 45 도 학습 범위 밖 | 피드백 [3] | 결정 사항 |
| F10 | **길이가 다른 결과끼리 비교할 때 주의**: (a) NPSS 는 transition 프레임 수만큼의 주파수 bin 에 대한 누적분포 L1 거리라서 길이가 길수록 값 자체가 커진다 → 같은 길이 안에서 방법끼리만 비교. (b) 발 미끄러짐은 덜 움직일수록 작아진다 (zero-velocity 는 항상 0). long 평가셋 1/4 표본 실측: trans 300 에서 모델의 루트 속도 1.91 cm/frame (GT 3.40), 발가락 속도 2.79 (GT 3.74) → trans 300 에서 미끄러짐이 줄어든 것은 개선이 아니라 덜 움직인 결과. 이동량과 함께 봐야 함 | 지표 정의 (F2, F4) + 실측 | 해석 주의 |

## G. 3단계 Detail Transformer (논문 3.4, 3.5, 3.6, 브랜치 `detail-transformer`)

코드: `src/models/detail_transformer.py`, `src/training/detail.py`, `src/training/losses.py` (L_c, L_foot 추가),
`scripts/train_detail.py`, 설정 `configs/model/detail.yaml`. 공식 코드는 **읽기만** 했다 (import/복사/실행 없음).

| # | 내용 | 근거 | 상태 |
|---|---|---|---|
| G1 | 구조 = Context 와 같은 층 (상대 위치 어텐션, pre-LN FFN, 6층·8헤드·512), **Keyframe PE 없음**, 출력 = 포즈 D + 발 접촉 C (= 스켈레톤 contact_joints 수, LaFAN1 4 → 139). 기존 `ContextTransformer` 와 체크포인트는 그대로 두고 새 클래스로 작성 | 논문 3.4, 3.6, 공식 config `d_out 139` | 논문과 같음 |
| G2 | 어텐션 마스크 (그림 5b): 모든 query 가 프레임 0..target 을 key 로 봄. **target 이후 프레임은 가림** | 논문 그림 5b 는 "자유롭게 본다" 고만 함. 공식 코드 `atten_mask[:, :target_idx + 1] = False` | 가정 (공식과 같음) |
| G3 | 입력 x_det = 정답 클립에서 transition 만 Context 출력, 제약 (midway) 프레임은 정답으로 되돌림 (전 관절 제약, d_mask 1). m_in 은 Context 와 같음. target 이후 프레임은 정답 (마스크로 가려짐) | 논문 3.4 | 논문과 같음 |
| G4 | Detail 입력을 만들 때 Context 출력은 **후처리 (식 23) 한 값**. Detail 출력에는 기본으로 후처리를 하지 않음 (`infer_two_stage(det_postprocess=False)`) | 논문 3.6.2 "post-processing is only applied during inference of the Context Transformer". 공식 코드: 학습·평가 모두 Context 후처리 켬, `eval_detail_model.py` 는 기본 후처리 없음 (README Detail 수치도 후처리 없음) | 가정 (공식과 같음) |
| G5 | 손실 (식 22): L_rot (6D, 역정규화) + 0.02·L_pos (FK, cm) + 0.05·L_c + 0.05·L_foot, 모두 L1·원소 평균·transition 프레임만. **스무딩 손실 없음** | 논문 3.5, 3.6.1. 공식 config 에 `sw 0.005` 가 있지만 Detail 손실 코드에는 쓰이지 않음 | 논문과 같음 |
| G6 | L_c = ĉ (sigmoid 확률) 과 정답 접촉 라벨 (B8, 0/1) 의 L1. L_foot = ĉ.detach() × 발 관절 4개 (LeftFoot, LeftToe, RightFoot, RightToe) 의 전역 속도 v_t = g_t − g_{t−1} (첫 transition 프레임은 마지막 context 프레임과의 차이) 의 L1 | 논문 식 19, 20 ("sever the gradient back-propagation on ĉ"), 공식 코드 `cal_c_loss`, `cal_f_loss` | 가정 (공식과 같음) |
| G7 | 접촉 출력: transition 프레임은 ĉ, 나머지 프레임은 정답 라벨 (손실은 transition 만이라 영향 없음) | 공식 코드 | 가정 |
| G8 | Detail 정규화 통계 = Context 통계 (`lafan1_context_stats.npz`) 재사용 | 공식 코드의 Detail 통계 (`train_stats_detail.pkl`) 는 Context 통계와 **같은 분할 (window 50, offset 20)·같은 정렬·같은 135차원 표현** 으로 계산 → 값이 같다. 우리 Context 통계는 공식 Context 통계와 1e-12 수준으로 일치 (0 절) | 가정 (근거 있음) |
| G9 | 학습률 base_lr **0.025** (Noam, warmup 8000 → 최대 ≈ 2.8e-4). Context (0.0625) 와 다름 | 논문에 없음. 공식 detail config `lr 0.025` | 가정 |
| G10 | 고정할 Context = `experiments/ctx_base/best.pth` (iteration 338000). 학습 중 eval 모드, `requires_grad=False`, `torch.no_grad` 로 실행 | 논문 3.6.1 "freeze its parameters" | 결정 사항 |
| G11 | transition 길이·제약 프레임은 Context 와 같은 방식으로 배치마다 무작위 (q 0.1, p 0.5), window 42, batch 32, Adam (0.9, 0.999) | 논문 3.6.1 | 논문과 같음 |
| G12 | 학습 길이 max_iters 500,000, best 체크포인트 = val trans 30 손실 최소 (Context 와 같은 기준, E16). val L2P 대신 손실을 쓰는 것은 일관성 때문 | 논문: "최대 200 epoch + early stopping" | 결정 사항 |
| G13 | Detail 출력은 **잔차가 아님**: transition 프레임을 Detail 출력 포즈로 그대로 덮어쓴다 (Context 출력 + Δ 가 아님). 그래서 학습 초기에는 Detail 결과가 Context 보다 훨씬 나쁘다 (300 iteration 에서 val30 위치 오차 45.6 cm vs Context 6.4 cm) | 논문 식 14 (y_det = D(·) 의 R, P). 공식 코드 `get_model_output`: `state_out[..., seq_slice, :] = model_out[..., seq_slice, rp_slice]` (읽기만 함) | 논문·공식과 같음 |
| G15 | **결과 차이 (미해결)**: 두 단계 L2P 개선이 README 보다 작다 (공식 벤치마크 trans 30: Context 만 1.0368 → 두 단계 1.0117, README 1.0354 → 0.8948). 학습 중 val30 위치 오차도 120k iteration 부터 Context 만과 거의 같은 값 (6.36 → 최종 6.20 cm). 가설: Context 가 train 분할에서 훨씬 정확해서 (`ctx_base/best.pth`, trans 5/15/30 위치 오차 train 0.61/1.55/3.75 cm, val 0.74/2.81/6.39 cm) Detail 은 평가 때보다 쉬운 입력으로 학습된다. Detail 자체도 train 손실 0.034, val30 손실 0.117 로 차이가 크다 | 실측 (train 2000 클립 무작위, val 560 클립) | **가설, 미확인**. 확인 방법 후보: 덜 학습된 Context 체크포인트로 Detail 학습, 공식 코드 Detail 학습 설정 재대조 |
| G16 | Detail 출력은 후처리하지 않으므로 (G4) 학습 범위 밖에서 Context 가 target 에 못 미친 거리가 **끝 경계 점프** 로 다시 나타난다 (Context 후처리가 양 끝으로 나눴던 것을 Detail 이 되돌림). trans 90 run1_subject5 4800: Context +후처리 시작/끝 +98.8/+96.7 cm → 두 단계 +75.8/+263.5 cm. 학습 범위 안 (trans 30) 에서는 ±0.8 cm | `render_samples.py --stage detail` 실측 (seed 0 샘플 5개) | 확인됨. `detail_post` 로 후처리한 지표도 제공 |
| G17 | 다음 실험 = **이어서 학습** (det_base 500k → det_cont 1M iteration, 같은 Noam 스케줄 계속). 근거: det_base best 가 iteration 498000 (사실상 마지막 검증) 이고 val30 오차가 6.36 → 6.20 cm 로 아직 줄던 중. 공식 Detail 설정 `total_epoch 2000` (≈ 2500만 iteration, best 만 저장) 에 비해 우리 500k 는 약 40 epoch. 공식 `dropout 0.0` × `dropout_iterations 150000` 스케줄은 0 이라 영향 없음 (확인). 새 폴더 best.pth 는 이어간 구간 (500k–1M) 의 최고 | 공식 configs/lafan1_detail_model.json (읽기만) | 결정 사항 (G15 확인용 실험 1) |
| G18 | **G17 결과**: det_cont (1M, best = iteration 978000) 공식 벤치마크 L2P trans 5/15/30/45 = 0.1136/0.4302/0.9909/1.8429 (det_base 0.1177/0.4425/1.0117/1.8686, README 0.1049/0.3943/0.8948/1.6777). val30 손실 0.1171 → 0.1147. 개선은 있으나 README 와의 차이 (trans 30 에서 13% → 11%) 의 대부분이 남는다. 학습 길이를 두 배로 늘린 효과가 이 정도이므로 G15 의 원인은 학습 부족보다 다른 쪽 (Context 의 train/val 차이, 설정 차이) 일 가능성이 크다. long 평가셋에서 trans 90 이상은 여전히 Context 만보다 나쁘다 (300: 21.80 → 23.34) | `eval_lengths.py --detail-ckpt experiments/det_cont/best.pth` 실측 | 확인됨 (G15 는 여전히 미해결) |
| G19 | **공식 Detail 학습 코드 재대조** (G15 원인 찾기, `train/detail_model.py`, `train/utils.py`, `train/context_model.py`, `model/__init__.py` 읽기만). **같음**: 입력 = 고정 Context (+후처리, 제약 프레임 포함 마스크) 출력에서 제약 프레임만 정답으로 되돌린 상태 + m_in (zscore 에 마스크 곱하지 않음), 어텐션 key 0..target, transition 출력만 교체 (residual 아님), 접촉 = sigmoid, transition 밖 접촉은 정답, 손실 L_rot (역정규화 6D) + 0.02 L_pos + 0.05 L_c (L1) + 0.05 L_foot (발·발끝 4관절, v_t = g_t − g_{t−1}, ĉ detach), 제약 프레임 샘플링 (0.1, 0.5), 배치마다 길이 하나 [5, 30], Noam (0.025, warmup 8000), batch 32, train window 42. Detail 통계 `train_stats_detail.pkl` 의 state 는 Context 통계와 같은 데이터셋·같은 특징 (rot 6D + 루트 위치) 이라 G14 와 같은 값. delta 통계는 계산만 하고 쓰지 않음. **다름**: (1) 어텐션 스케일 1/√d_model (우리 1/√d_head, E3), (2) pre-LN residual 에 정규화된 값을 더함 (E4), (3) best = val trans 30 의 gpos+gquat+npss (후처리 없음; 우리 val30 학습 손실), (4) 학습 길이 (2000 epoch 설정). (1)(2) 는 Context 에도 같이 있는데 Context 만은 README 와 일치했으므로 Detail 에서만 크게 작용할지는 불확실 | 공식 코드 읽기 (실행·복사 안 함) | 확인됨. 입력·손실·스케줄 쪽에서는 G15 원인이 될 차이를 찾지 못함. 남은 후보: (1)(2) 구조 차이, G15 의 Context train/val 차이 |
| G14 | 정규화 통계 경로는 Context 체크포인트 폴더의 `config.yaml` 의 `stats` 를 그대로 쓴다. 데이터 설정 (context_len, forward_world_axis, 스켈레톤 파일) 이 Context 학습 때와 다르면 학습을 시작하지 않는다 | G8 과 같은 이유 | 결정 사항 |

## H. 발 전용 In-betweening 모델 (브랜치 `feat/foot-only`, 코드 `foot_only/`)

교수님 피드백: transition 구간에서 전신 대신 발만 먼저 예측해 정확도를 확인 (이후 발 → 몸통 확장 예정).
비교 기준: ctx_base (같은 구조), ctx_base + det_cont (최종 Two-stage), SLERP (`interp`).

| 번호 | 내용 | 근거 | 상태 |
|---|---|---|---|
| H1 | 출력 = transition 프레임마다 발 4관절 위치 (LeftFoot 3, LeftToe 4, RightFoot 7, RightToe 8; `contact_joints` 순서, 12) + 접촉 logit 4. 좌표 = **정렬 월드** (마지막 context 프레임에서 루트 XZ=0, +X 를 봄; 기존 `gpos_gt` 와 같음, cm) | 루트 기준 좌표는 루트 궤적이 없으면 월드 위치를 복원할 수 없음. 정렬 월드는 `CanonicalFrame` 역변환만으로 복원 (`foot_to_world`), 땅에 붙은 발 = 고정 값 | 결정 사항 (STEP 1 승인) |
| H2 | 입력 = Context Transformer 입력 (전신 135, 모르는 프레임 0, 루트 선형보간) ⊕ **아는 프레임의 발 위치 12** (정규화, 모르는 프레임 0) ⊕ m_in → 148 | 출력이 발 위치이므로 경계 프레임의 발 위치를 직접 준다 (회전 → 위치 FK 를 모델이 배울 필요를 줄임). ctx_base 와 입력이 12차원 다름 | 결정 사항 (STEP 1 승인) |
| H3 | 발 위치 z-score 통계: train_stats 분할 (Subject 1–4, window 50, offset 20, 20214 클립, 1,010,700 프레임) 의 정렬 좌표 발 위치 차원별 평균·모집단 표준편차 → `<cache_dir>/stats/foot_only_stats.npz`. 표준편차 x ≈ 70, y ≈ 21, z ≈ 38–40 cm | 입력 통계 (scripts/compute_stats.py) 와 같은 분할. Subject 5 는 쓰지 않음 | 확인됨 (STEP 2) |
| H4 | 접촉 라벨은 **원래 월드 좌표** 에서 |dx|+|dy|+|dz| < 0.2 cm/frame (기존 데이터셋 라벨 그대로). L1 속도는 Y축 회전에 따라 값이 변해서, 정렬 좌표로 다시 계산하면 임계값 근처 프레임의 0.7% 가 달라진다 (월드 좌표로 계산하면 100% 일치, window 마지막 프레임 제외). 예측 발 위치로 접촉을 판정할 때도 월드 좌표로 되돌려 계산해야 같은 정의가 된다. STEP 0 보고의 "정렬해도 결과 같음" 은 틀린 설명이었음 | `foot_only/check_data.py` [4a] 실측 | 확인됨 |
| H5 | 접촉 라벨 특성: 접촉 프레임 35%. 접촉 프레임의 발 높이 중앙값 2.9 cm 이지만 15 cm 이상이 7.7% → 대부분 obstacles (장애물 위에 선 동작, 30.7%) 와 ground (5.7%) 라서 라벨 오류가 아님. 임계값이 엄격해서 달리기에서는 발목 (Foot) 접촉이 거의 0 이고 발가락만 짧게 잡히며, 걷기에서도 속도가 임계값 근처를 오가며 라벨이 끊긴다 → 접촉 F1 해석 시 주의 | `check_data.py` [4b][4c], `outputs/foot_only/check_data/contact_*.png` | 확인됨 (라벨 정의는 공식 코드 값 유지) |
| H6 | best 체크포인트 = **val trans 30 의 자기 학습 손실 (`val30_loss` = w_pos·L_pos + w_contact·BCE) 최소**. ctx_base 와 같은 규칙 ("val trans 30 에서의 자기 손실") 이지만 손실 식이 달라 값끼리는 비교 불가. 가중치 1:1 에서는 거의 발 위치 L1 이 결정. `val30_foot_l2` 등은 log 에만 기록 | ctx_base 와 절차를 맞춤. 대안 (val30 발 위치 L1 만) 은 접촉을 전혀 반영하지 않음 | 결정 사항 (STEP 3 승인) |
| H7 | 손실 = 1.0·L_pos (발 위치 cm L1, 12 좌표 평균) + 1.0·BCE (접촉 logit 4), transition 프레임만 (제약 프레임 포함). pos_weight 없음. 근거: ctx_base 발 4관절 L1 (val, trans 5/15/30) = 0.64 / 1.93 / 4.19 cm → 학습 범위 평균 수렴 ≈ 2 cm, 상수 예측 BCE 0.64 → 수렴 근처에서 접촉 항 ≈ 10–15%. train 접촉 비율 LeftFoot 29.4%, LeftToe 42.9%, RightFoot 28.3%, RightToe 42.7% (최대 2.5:1) | 실측 (train 406,819 프레임, ctx_base best.pth) | 결정 사항 (STEP 3 승인). 1,000 스텝 실측: 접촉 항 비중 7.5% (pos 7.5 cm, BCE 0.61), 발목 접촉 F1 0 (아직 항상 "비접촉" 예측) → 본 학습 후 관절별 F1 확인, 낮으면 pos_weight 검토 → 본 학습 결과 H9 |
| H8 | 학습 비용 (RTX 3070): 23 ms/iteration, VRAM 최대 341 MB, 파라미터 10,334,933 (ctx_base 10,38만과 거의 같음, 입력·출력 층만 다름) → 500k ≈ 3.2 시간 + 검증 | `train_foot.py --run-name foot_check_1k --max-iters 1000` | 확인됨 |
| H9 | 본 학습 결과 (`experiments/foot_only/`, 2026-11-13 20:38 → 23:57, 3시간 19분, 500k 정상 종료): best = iteration **454000** (val30_loss 3.5254; 마지막 500k 는 3.6175). val (560 클립, 제약 프레임·후처리 없음) trans 5/15/30: 발 L2 0.99 / 2.70 / 6.75 cm, 발 L1 0.49 / 1.31 / 3.21 cm, 접촉 acc (출력) 0.906 / 0.894 / 0.868, 접촉 acc (예측 위치를 월드 좌표로 판정) 0.854 / 0.848 / 0.806. trans 30 접촉 F1: LeftFoot 0.703, LeftToe 0.797, RightFoot 0.689, RightToe 0.804. **발목 F1 은 1k 의 0 에서 회복했지만 50k 이후 0.68–0.70 에서 오르지 않음**, 발끝보다 ~0.1 낮음. 위치로 판정한 발끝 접촉 acc (0.74) 가 출력 (0.84) 보다 낮음 (0.2 cm/frame 기준이 엄격) | `experiments/foot_only/log.csv`, `best.pth` | 확인됨. pos_weight 재학습은 평가 (관절·동작별 F1, `docs/foot_only_eval_plan.md`) 뒤 결정 (미정) |
| H10 | 평가 결과 (`foot_only/eval_foot.py` → `outputs/eval/foot_only/`, official 2232 / long 2151 클립). 검산: GT 발 위치로 판정한 접촉 acc 1.000, GT 발 미끄러짐 0.078; ctx_post / two_stage 발 미끄러짐 (trans 30: 0.476 / 0.281, 45: 0.909 / 0.555) 이 기존 평가와 같음. **발 위치 L2 (cm, official 5/15/30/45)**: foot-only 1.00 / 2.73 / 6.91 / 14.67, two-stage 0.96 / 3.17 / 7.81 / 15.82, ctx+후처리 1.28 / 4.00 / 8.89 / 16.98, SLERP 3.78 / 14.22 / 24.40 / 32.35 → trans 5 를 빼면 foot-only 가 가장 작다. long 60 / 90: foot-only 25.1 / 50.0, two-stage 26.8 / 52.5, ctx+후처리 27.9 / 52.1, SLERP 42.6 / 62.8 → 학습 범위 밖에서는 모두 크게 틀리고 차이가 줄어든다. **발 미끄러짐 (cm/frame)**: 모든 길이에서 foot-only 가 가장 작다 (trans 30: 0.191 vs two-stage 0.281, ctx 0.476; 90: 0.569 vs 1.164, 1.445). 그래도 GT 의 2.4배 (trans 30). **접촉 F1 (official 30)**: 출력 기준 foot-only 0.768 (발목 0.70, 발끝 0.80) vs two-stage 0.740 (0.65, 0.78). 위치 판정 기준 foot-only 0.583, two-stage 0.414, ctx 0.174, SLERP 0.246 → 위치 판정 F1 이 낮은 이유는 recall (발끝 0.41): 예측 발이 접촉 구간에서 0.2 cm/frame 이하로 멈춰 있지 못함. 발목 출력 접촉은 precision 0.75–0.79, recall 0.64–0.65 (발끝 0.86–0.88 / 0.73–0.76). 같은 차이가 Detail 접촉 출력에도 있음 (발목 F1 0.63–0.67). 동작별 (trans 30, 출력 F1 발목): walk 0.72–0.73, run 0.69–0.74 → 발목 차이는 동작보다 관절 쪽. run 의 위치 판정 발끝 F1 은 0.26–0.29 로 낮음 | `outputs/eval/foot_only/{summary.csv, metrics_long.csv, metrics.png}` | 확인됨. pos_weight 는 접촉 출력 recall 만 올릴 수 있고 위치 판정 recall (발 미끄러짐) 은 바꾸지 못함 → 재학습 여부 사용자 결정 대기 |

## I. 발 → 몸 모델 (브랜치 `feat/feet-to-body`, 코드 `feet_to_body/`)

교수님 피드백의 두 번째 단계 (발 → 몸). 발 전용 모델 (H 절) 과 별도 프로젝트로 진행한다.
1차 실험 = 정답 발 조건: "transition 구간의 발을 완벽히 알 때 전신이 얼마나 좋아지는가" 의 상한을 잰다.
비교 기준: ctx_base (구조·손실·학습 설정이 같고 발 조건만 없음).

| 번호 | 내용 | 근거 | 상태 |
|---|---|---|---|
| I1 | 입력 = Context Transformer 입력 (전신 135, 모르는 프레임 0, 루트 선형보간) ⊕ **모든 프레임의 발 조건 16** ⊕ m_in → 152. 발 조건 = 발 4관절 위치 12 (정렬 월드 좌표, 발 전용 모델과 같은 z-score 통계 `foot_only_stats.npz`) + 접촉 4 (0/1 라벨). 출력 = 전신 135 | 발 전용 모델의 출력 (위치 12 + 접촉 logit 4) 과 같은 형식·좌표라서, 다음 단계에서 예측한 발 (접촉은 확률) 을 그대로 넣을 수 있다. 접촉 포함은 사용자 결정 (2026-11-15) | 결정 사항 |
| I2 | 어텐션은 **모든 프레임을 key 로 본다** (`cond.attend_all: true`, key 마스크 없음). ctx_base 는 context·제약·target 프레임만 key (논문 그림 5a) | ctx_base 에서 transition 프레임을 가리는 이유는 그 프레임에 정보가 없기 때문이다. 여기서는 transition 프레임마다 발 정보가 있고, 가리면 각 프레임이 자기 발만 보고 이웃 프레임의 발 (발 궤적) 을 못 본다. ctx_base 의 transition 프레임에는 볼 정보가 없으므로 어텐션 범위 자체는 새 정보원이 아니다: ctx_base 대비 이득의 출처는 발 정보이고, 어텐션을 연 것은 그 정보를 쓰는 통로다 (상한을 재려면 발 정보를 가장 잘 쓰게 해야 함). 단, 발 없이 어텐션만 연 ctx 모델이 ctx_base 와 같은 성능인지는 확인하지 않았다 (추정). `attend_all=false` 학습은 "이웃 프레임의 발 궤적 없이 자기 발만 알아도 되는가" 라는 다른 질문이며, 결과가 기대와 다르거나 구조 차이를 따져야 할 때 돌린다 | 결정 사항 (2026-11-15 사용자 확인: 그대로 진행) |
| I3 | 손실 = ctx_base 와 같은 식 21 (w_rot 1, w_pos 0.01, w_smooth 0.005, transition 프레임만). 발을 따르게 하는 손실·IK 는 없다. best = `val30_loss` 최소 | 발 조건의 효과만 보기 위해 나머지를 같게 둠. 손실 식이 같아서 `val{t}_loss` 를 ctx_base 로그와 바로 비교할 수 있다 | 결정 사항 |
| I4 | 학습 설정 = ctx_base 와 같음 (batch 32, Noam 0.0625 / warmup 8000, 500k, transition 5–30, 제약 프레임 0.1 / 0.5, seed 0). 학습 중 발 조건은 항상 정답 (노이즈 없음) | 공정한 비교 (사용자 결정 2026-11-15). 정답 발로만 학습하므로, 예측한 발을 넣을 때는 학습·추론 입력 분포가 달라진다 (다음 단계에서 확인할 것) | 결정 사항 |
| I5 | 학습 비용 (RTX 3070): 36 ms/iteration, VRAM 최대 350 MB, 파라미터 10,398,028 → 500k ≈ 5 시간 + 검증. 1k 확인 (`body_check_1k`): val30_loss 0.2387, val30_gpos_l2 15.9 cm (학습 초기라 의미 없음, 동작 확인용) | `train_body.py --run-name body_check_1k --max-iters 1000` | 확인됨 |

