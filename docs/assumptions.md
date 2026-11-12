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
| F5 | 긴 길이 평가셋: Subject 5, **window = 10 + max(lengths) + 2 = 312**, offset 40, start 0 → **2151 클립**. 모든 길이에서 같은 시작 프레임 (같은 context), target 만 뒤로 이동. 공식 README 와 비교하는 5/15/30/45 는 공식 벤치마크 (window 65, 2232 클립) 로도 따로 계산 | 공식 벤치마크 window 65 에는 trans ≤ 53 만 들어감. 길이별로 같은 샘플을 비교하려면 가장 긴 길이에 맞춘 window 가 필요 | 결정 사항 |
| F6 | 후처리 (식 23) 는 공식 코드의 근사 (E17): 시작 쪽과 끝 쪽 어긋남의 **평균** δ 를 transition 전체에 더한다. 학습 범위 밖 (예: trans 90) 에서 모델이 target 에 못 미치면 (run1_subject5 4800: 마지막 생성 프레임이 target 에서 ~2 m), δ ≈ 그 절반이 되어 **시작·끝 경계에서 각각 ~1 m 점프** 가 생긴다. 후처리를 끄면 끝 경계에서만 ~2 m 점프 | `render_samples.py` 실측 (trans 90, seed 0 샘플 5개): 후처리 켬 → 경계 점프 최대 +98.8 / +96.7 cm, 끔 → +3.2 / +197.1 cm. trans 30 에서는 ±1.5 cm 이내 | 확인됨. 영상·지표는 후처리 켬/끔 모두 제공 |
| F7 | 영상: 모두 1배속 30fps, `[BEFORE 원본 30 (회색)] [CONTEXT 10] [GENERATED trans] [TARGET 1 + 유지 29]`. BEFORE 는 모델 입력이 아니며, 모델 창과 같은 정렬 (마지막 context 프레임 기준) 로 그린다 | 피드백 [1] + 미팅 후 요청 (재생 속도를 바꾸면 경계가 자연스러운지 판단하기 어려움) | 결정 사항 |
