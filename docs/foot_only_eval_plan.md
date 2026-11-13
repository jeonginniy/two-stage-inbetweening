# 발 전용 모델: 다음 평가 준비 메모

브랜치 `feat/foot-only`. 학습은 끝났고 (`experiments/foot_only/best.pth` = iteration 454000), 평가·시각화는
다음 작업이다. 이 문서는 그 작업을 시작할 때 필요한 결정·주의점을 모아 둔 것이다 (아직 코드 없음).
가정 번호는 `docs/assumptions.md` H 절.

## 1. 비교 대상

| 이름 | 체크포인트 | 발 위치를 얻는 방법 | 접촉을 얻는 방법 |
|---|---|---|---|
| foot-only | `experiments/foot_only/best.pth` | 모델 출력 (정렬 좌표 12) | (a) 접촉 출력 sigmoid > 0.5, (b) 예측 위치로 판정 |
| ctx_base | `experiments/ctx_base/best.pth` | 전신 출력 → FK → 발 4관절 | 예측 위치로 판정 |
| ctx_base + det_cont (최종 Two-stage) | `experiments/det_cont/best.pth` (`lengths.yaml` 기본 `detail_ckpt`) | Detail 출력 → FK → 발 4관절 | (a) Detail 접촉 출력, (b) 예측 위치로 판정 |
| SLERP (interp) | 없음 (`src/evaluation/baselines.py interpolation`) | 보간 전신 → FK → 발 4관절 | 예측 위치로 판정 |

- **공정한 비교 = (b) "예측 위치로 판정한 접촉"**. ctx_base 와 SLERP 에는 접촉 출력이 없다.
  (a) 는 접촉 출력이 있는 모델 (foot-only, det_cont) 끼리만 비교한다.
- **정해야 할 것: ctx_base 후처리 (식 23)**. foot-only 와 Detail 은 후처리를 하지 않는다.
  ctx_base 는 `model` (후처리 O, README 조건) 과 `model_nopost` 를 둘 다 내는 것을 제안한다.
  foot-only 에 후처리를 할지도 정해야 한다 (발 위치용 식 23 은 아직 없음).

## 2. 길이와 평가셋

- 길이: **5, 15, 30, 45** (학습 범위 5–30 + 공식 45) **+ 60, 90** (학습 범위 밖, 긴 구간 질문용).
- 평가셋: 기존 `configs/eval/lengths.yaml` 의 두 셋을 그대로 쓴다 (`inference.build_eval_dataset`).
  - `official`: 공식 벤치마크 2232 클립, trans ≤ 53 → 5, 15, 30, 45
  - `long`: 2151 클립, 모든 길이에서 같은 시작 프레임 → 30, 60, 90 (길이 사이 비교용)
- 학습 중 val 분할 (`val`, 560 클립) 수치는 log 확인용이다. 보고 수치는 위 두 셋에서 다시 계산한다.

## 3. 지표

| 지표 | 계산 | 주의 |
|---|---|---|
| 발 위치 L2 (cm) | transition 프레임, 발 4관절 (+ 관절별) 의 정렬 좌표 거리 평균 | 정규화된 L2P 가 아님. 4 방법 모두 같은 식으로 계산 |
| 접촉 F1 | 관절별 (LeftFoot, LeftToe, RightFoot, RightToe), **동작별** (최소 walk / run, 가능하면 전체 동작 접두어별) | 동작 이름 = `ds.files[batch["file_index"]].stem` 의 접두어 (예: `walk3_subject5` → walk) |
| 발 미끄러짐 (cm/frame) | 기존 `src/evaluation/metrics.py foot_skating_per_clip` 그대로 | 아래 4번 참고 |

- 접촉 라벨 정의는 바꾸지 않는다 (0.2 cm/frame, `extract_foot_contacts`).
- H5: 달리기에서는 발목 접촉 라벨이 거의 없다. 동작별 F1 에서 run 의 발목 F1 은 표본이 적어 흔들릴 수 있으니 접촉 프레임 수를 같이 적는다.

## 4. 구현 주의점

1. **접촉 판정은 월드 좌표로 (H4)**. 예측 위치 (정렬 좌표) 를 `foot_only.data.foot_to_world` 로 되돌린 뒤
   `extract_foot_contacts(world, range(4), 0.2)` 를 쓴다. 정렬 좌표에서 바로 계산하면 정의가 달라진다
   (L1 속도는 Y축 회전에 따라 값이 바뀜). 전신 방법 (ctx_base, det_cont, SLERP) 도 같은 경로를 쓴다.
   마지막 transition 프레임은 target 프레임 (정답) 과의 차이로 판정 (`foot_only/train_utils.py validate` 와 같음).
2. **발 미끄러짐 함수에 발 위치만 넘기기**. `foot_skating_per_clip(gpos, foot_idx, H, frames)` 는 (B, T, J, 3) 을 받는다.
   foot-only 출력은 (B, T, 4, 3) 이므로 `foot_idx` 를 발끝 2개의 **발 배열 안 번호 [1, 3]** 으로 준다
   (전신 방법은 [4, 8]; 함수는 `lengths.yaml foot_skating` 의 발끝 관절, H = 3.0 cm). 수평 속도 (XZ L2) 는
   Y축 회전에 불변이라 정렬 좌표 그대로 써도 된다. 첫 transition 프레임은 마지막 context 프레임 (정답) 과의 차이.
3. **체크포인트 불러오기**. `inference.load_run` 은 ContextTransformer (입력 136) 를 만들기 때문에 foot-only 를
   못 읽는다. `experiments/foot_only/config.yaml` (`model`, `data`, `foot_data` 절) 로 `build_foot_model` +
   두 정규화 통계를 만드는 로더를 `foot_only/` 에 새로 둔다. 클립 자르기 (ctx + trans + 2) 는 `prepare_window` 와 같게.
4. 평가 스크립트·결과는 새 파일 `foot_only/eval_foot.py` → `outputs/eval/foot_only/` 를 제안 (기존 eval 파일 수정 없이).

## 5. 학습 중 val 기록 (참고, best.pth = 454k)

| trans | 발 L1 (cm, `val{t}_pos`) | 발 L2 (cm) | 접촉 acc (출력) | 접촉 acc (위치 판정) | F1 발목 L / R | F1 발끝 L / R |
|---|---|---|---|---|---|---|
| 5 | 0.49 | 0.99 | 0.906 | 0.854 | 0.80 / 0.75 | 0.87 / 0.87 |
| 15 | 1.31 | 2.70 | 0.894 | 0.848 | 0.78 / 0.73 | 0.84 / 0.84 |
| 30 | 3.21 | 6.75 | 0.868 | 0.806 | 0.70 / 0.69 | 0.80 / 0.80 |

- STEP 3 설계 때 잰 ctx_base 의 발 4관절 L1 (val 분할) 은 0.64 / 1.93 / 4.19 cm 였다 (H7).
  foot-only 가 더 낮아 보이지만 측정 조건 (후처리 여부 등) 을 맞춘 비교가 아니므로 평가에서 다시 확인한다.
- 위치로 판정한 발끝 접촉 정확도 (0.74) 가 접촉 출력 (0.84–0.85) 보다 낮다: 0.2 cm/frame 기준이 엄격해서 예측 위치가
  조금만 흔들려도 비접촉이 된다. 발 미끄러짐 수치와 함께 볼 것.
- 발목 F1 은 50k 이후 0.68–0.70 에서 거의 그대로 (H9). 평가에서 관절·동작별 F1 을 본 뒤 pos_weight 재학습 여부를 정한다.
