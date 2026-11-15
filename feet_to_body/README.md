# feet_to_body/ — 발 → 몸 모델 (브랜치 `feat/feet-to-body`)

교수님 피드백의 두 번째 단계: **발을 먼저 알면 전신 예측이 얼마나 좋아지는가.**
발 전용 모델 (`foot_only/`) 과는 별도 프로젝트로 진행한다 (기존 결과·영상과 섞지 않음).

## 실험 내용

- 1차 실험 = **정답 (GT) 발 조건**: transition 구간의 정답 발 4관절 위치 + 접촉을 입력으로 주고 전신을 예측한다.
  결과는 "발을 완벽히 알 때" 의 상한이다. 상한이 좋으면 다음에 발 전용 모델이 예측한 발을 넣는다.
- 비교 기준 = `ctx_base` (Context Transformer). 구조·손실 (논문 식 21)·학습 설정·500k iteration 이 모두 같고,
  다른 점은 아래 두 가지뿐이다 (docs/assumptions.md I 절):

| | ctx_base | feet_to_body |
|---|---|---|
| 입력 | 전신 135 + m_in 1 = 136 | 전신 135 + **발 조건 16 (모든 프레임)** + m_in 1 = 152 |
| 어텐션 key | context·제약·target 프레임만 (논문 그림 5a) | **모든 프레임** (transition 프레임에도 발 정보가 있으므로) |
| 출력 / 손실 | 전신 135 / 식 21 | 같음 |

발 조건 16 = 발 위치 12 (LeftFoot, LeftToe, RightFoot, RightToe 의 xyz; 정렬 월드 좌표, 발 전용 모델과 같은 z-score 통계)
+ 접촉 4 (0/1). 발 전용 모델의 출력 형식과 같다.
모델이 준 발을 따르게 강제하는 장치 (IK 등) 는 없다. 따르는 정도는 `val{t}_foot_l2` 로 잰다.

## 파일

| 파일 | 내용 |
|---|---|
| `data.py` | `load_body_config`, `foot_condition` (정답 발 조건), `build_body_input` (입력 152), `attention_mask` |
| `model.py` | `build_body_model` (ContextTransformer, 입력 152 → 출력 135) |
| `train_utils.py` | `run_body_model`, `body_forward` (손실, 식 21), `body_foot_error`, `validate` |
| `train_body.py` | [실행] 학습 → `experiments/feet_to_body/{config.yaml, log.csv, last.pth, best.pth}` |
| `evaluate.py` | 평가 계산: `load_body_run`, `predicted_cond` (발 전용 모델 예측 → 발 조건), `predict_bodies` (방법별 전신) |
| `eval_body.py` | [실행] 평가 → `outputs/eval/feet_to_body/{metrics.csv, metrics.png, run_info.json}` |

설정: `configs/feet_to_body.yaml` (1차, 정답 발), `configs/feet_to_body_pred.yaml` (2차, 예측한 발; base 파일에서 바뀐 값만 적음). 배치 준비는 `foot_only/data.py` 의 `prepare_foot_batch` 를 import 해서 쓴다.

## 실행 순서 (프로젝트 루트)

```bash
source .venv/bin/activate && export TSI_ENV=lab
# 통계 2개는 이미 있으면 생략 (ctx_base / 발 전용 모델과 같은 파일)
python scripts/compute_stats.py
python foot_only/compute_foot_stats.py

# 본 학습 (RTX 3070: 36 ms/iteration, VRAM 350 MB → 500k 약 5시간 + 검증)
# --run-name 은 필수. 끝난 실험 폴더 (feet_to_body 포함) 이름은 거부한다
# 1차 (정답 발, 끝남): nohup python feet_to_body/train_body.py --run-name feet_to_body > feet_to_body.out 2>&1 &
# 2차 (예측한 발 조건으로 학습, I8; 42 ms/iteration → 약 6시간)
nohup python feet_to_body/train_body.py --config configs/feet_to_body_pred.yaml --run-name feet_to_body_pred \
    > feet_to_body_pred.out 2>&1 &
# 이어서: 같은 명령 + --resume last

# 짧은 확인 (다른 폴더에)
python feet_to_body/train_body.py --run-name body_check_1k --max-iters 1000 --set train.val_interval=1000
# 데이터 없이 동작 확인 (결과 의미 없음)
python feet_to_body/train_body.py --env laptop --fake-data --run-name body_smoke --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
```

`feet_to_body.out`, `feet_to_body_pred.out` (학습 로그) 은 커밋하지 않는다. 기존 실험 폴더 이름 (`ctx_base`, `det_*`, `foot_only` 등) 은 `--run-name` 으로 쓸 수 없다.

## log.csv (iteration, epoch, kind, name, value)

- train (100 iteration 마다): `loss`, `rot`, `pos` (cm), `smooth`, `lr`
- val (2000 iteration 마다, trans t = 5/15/30, 제약 프레임 없음, 정답 발 조건):
  - `val{t}_loss/rot/pos/smooth` (후처리 없음), `val{t}_gpos_l2` (전역 관절 위치 L2 cm, 후처리 포함)
    → `experiments/ctx_base/log.csv` 의 같은 이름과 **같은 식·같은 val 분할** 이라 바로 비교할 수 있다.
  - `val{t}_gpos_l2_nopost`, `val{t}_foot_l2` (+ `_<관절>`): 출력 전신의 발 vs 조건으로 준 발 (cm)
- `val{t}_foot_follow` (출력 발 vs 준 발), `val{t}_cond_l2` (준 발 vs 정답 발), `val_mean_loss` (val 손실 평균)
- best.pth = `val30_loss` 최소 (ctx_base 와 같은 규칙). best_mean.pth = `val_mean_loss` 최소 (I9).

## 평가 (`python feet_to_body/eval_body.py`, 약 1분)

설정 `configs/eval/feet_to_body.yaml` + `configs/eval/lengths.yaml` + `configs/eval/foot_only.yaml` (foot_ckpt).
지표는 기존 평가와 같은 함수 (L2P, L2Q, NPSS, 발 미끄러짐) + 전역 위치 L2, 출력 발 vs 정답 발, 출력 발 vs 준 발.
결과: `outputs/eval/feet_to_body/{metrics.csv, metrics.png, run_info.json}`.

방법: `body_gt` = 정답 발 조건 (상한), `body_pred` = 발 전용 모델이 예측한 발 조건 (학습 때 본 적 없는 입력, I4),
`_post` = 후처리 (식 23) 포함.

## 결과 (docs/assumptions.md I6, I7)

학습: 500k 정상 종료 (4시간 52분), best = iteration 164000 (`val30_loss` 0.0654; ctx_base best 0.0885).

L2P (공식 벤치마크 2232 클립 / long 평가셋 2151 클립). 발 → 몸은 후처리 없음 / 있음:

| trans | 발→몸, 정답 발 | 발→몸, 예측한 발 | Context + 후처리 | Two-stage | SLERP |
|---|---|---|---|---|---|
| 5 | 0.360 / 0.140 | 0.368 / 0.149 | 0.129 | **0.114** | 0.373 |
| 15 | 0.450 / **0.363** | 0.529 / 0.449 | 0.465 | 0.430 | 1.249 |
| 30 | **0.650** / 0.645 | 0.975 / 0.954 | 1.037 | 0.991 | 2.316 |
| 45 (밖) | 1.228 / **1.156** | 1.974 / 1.881 | 1.887 | 1.843 | 3.447 |
| 60 (밖, long) | 2.907 / **2.716** | 3.889 / 3.669 | 3.163 | 3.130 | 4.790 |
| 90 (밖, long) | 6.688 / **6.201** | 7.922 / 7.434 | 6.311 | 6.463 | 7.457 |

trans 30 의 다른 지표 (후처리 없음; GT 발 미끄러짐 0.078):

| 방법 | L2Q | NPSS | 발 미끄러짐 | 출력 발 vs 정답 발 (cm) | 출력 발 vs 준 발 (cm) |
|---|---|---|---|---|---|
| 발→몸, 정답 발 | **0.481** | **0.090** | **0.200** | **1.45** | 1.45 |
| 발→몸, 예측한 발 | 0.581 | 0.116 | 0.276 | 7.19 | 1.42 |
| Two-stage | 0.590 | 0.124 | 0.281 | 7.81 | - |
| Context + 후처리 | 0.606 | 0.127 | 0.476 | 8.89 | - |

- **상한은 크다**: 정답 발을 주면 trans 30 L2P 가 Two-stage 보다 34% 좋다 (0.650 vs 0.991).
- **예측한 발로는 이득이 거의 없다**: trans 30 에서만 Two-stage 보다 조금 좋고 (0.954 vs 0.991), 5 / 15 / 45 는 나쁘다.
  모델은 준 발을 잘 따르므로 (1.4 cm), 남은 오차는 발 전용 모델의 발 오차 (7 cm) 에서 온다.
- **짧은 구간이 약하다**: trans 5 는 정답 발을 줘도 Context 보다 나쁘다. 짧은 구간은 500k 까지 계속 배우는 중인데
  best (val30_loss 최소) 는 164k 에서 골라졌다 (I6). 참고로 last.pth 는 trans 5 가 0.122 (후처리) 로 좋아진다.
- **학습 범위 밖은 정답 발을 줘도 무너진다**: trans 90 L2P 6.2 (Context 6.3), 출력 발이 준 발과 30 cm 다르다.

## 2차 실험: 예측한 발로 학습 (I8, I9)

1차 모델은 준 발을 그대로 따라서, 예측한 발의 오차 (7 cm) 가 전신 오차로 넘어갔다. 2차는 학습 때부터
발 전용 모델 (고정) 의 예측을 발 조건으로 준다. 주의: 발 전용 모델은 학습 클립에서 발을 더 잘 맞혀서
(trans 30: 3.96 cm vs val 6.75 cm) 학습 때 보는 발 오차가 평가 때보다 작다.

학습 완료 (500k, `experiments/feet_to_body_pred/`): best.pth = iteration 152000 (val30_loss 0.0835),
best_mean.pth = iteration 492000 (val 평균 loss 0.0576). 평가 (I10):

```bash
python feet_to_body/eval_body.py --body-ckpt experiments/feet_to_body_pred/best.pth --out outputs/eval/feet_to_body_pred
python feet_to_body/eval_body.py --body-ckpt experiments/feet_to_body_pred/best_mean.pth --out outputs/eval/feet_to_body_pred_mean
```

L2P (예측한 발 조건 + 후처리, 공식 2232 클립 / 60, 90 은 long 평가셋):

| trans | 1차 best (164k) | 2차 best (152k) | 2차 best_mean (492k) | Context + 후처리 | Two-stage |
|---|---|---|---|---|---|
| 5 | 0.149 | 0.164 | 0.136 | 0.129 | **0.114** |
| 15 | 0.449 | 0.470 | 0.456 | 0.465 | **0.430** |
| 30 | 0.954 | **0.942** | 0.957 | 1.037 | 0.991 |
| 45 | 1.881 | 1.858 | 1.910 | 1.887 | **1.843** |
| 60 | 3.669 | 3.496 | 3.475 | 3.163 | **3.130** |
| 90 | 7.434 | 7.378 | 7.162 | **6.311** | 6.463 |

- **예측한 발로 학습해도 거의 달라지지 않았다** (trans 30: 0.954 → 0.942, 약 1%). 기대한 "발 오차를 덜 믿는" 효과는 없었다:
  trans 30 에서 출력 발이 준 발을 2.2 cm 이내로 따라가고 (1차 1.4 cm), 출력 발 오차 7.2 cm 는 준 발 오차 (6.9 cm) 와 같다.
- "발 먼저" 가 Two-stage 보다 좋은 길이는 **trans 30 뿐** (L2P −5%, L2Q 0.571 vs 0.590, NPSS 0.119 vs 0.124).
  trans 5, 15, 45 와 학습 범위 밖 (60, 90) 은 Two-stage 가 좋다.
- 발 미끄러짐 (trans 30, 후처리 포함): 2차 0.324, 1차 0.298, Two-stage 0.281, Context 0.476. Two-stage 보다 좋지 않다.
- best_mean.pth 는 짧은 구간을 고친다 (trans 5: 0.164 → 0.136) 대신 30 이상은 조금 나빠진다. 그래도 trans 5 는 Context (0.129) 보다 나쁘다.
- 같은 2차 모델에 정답 발을 주면 trans 30 L2P 0.681 (1차 0.645): 상한과의 차이 (0.94 vs 0.68) 는 발 예측 오차에서 온다.

## 발 오차 민감도 (I11, `python feet_to_body/eval_foot_mix.py`, 학습 없음 약 3분)

질문: 발 전용 모델의 오차를 얼마나 줄여야 Two-stage 를 확실히 이기는가. 발 조건을 정답 발과 예측한 발 사이에서
섞어 준다 (`(1 − α) × 정답 + α × 예측`, 설정 `configs/eval/feet_to_body.yaml` 의 `foot_mix`) → `outputs/eval/feet_to_body_mix/`.

1차 모델 (정답 발 학습), 공식 2232 클립, 후처리 포함. 칸 = L2P (괄호 = 준 발의 오차 cm):

| trans | α = 0 | 0.25 | 0.5 | 0.75 | 1 (예측 그대로) | Two-stage |
|---|---|---|---|---|---|---|
| 5 | 0.140 (0) | 0.140 (0.3) | 0.142 (0.5) | 0.145 (0.8) | 0.149 (1.0) | 0.114 |
| 15 | 0.363 (0) | 0.369 (0.7) | 0.388 (1.4) | 0.415 (2.0) | 0.449 (2.7) | 0.430 |
| 30 | 0.645 (0) | 0.678 (1.7) | 0.749 (3.5) | 0.844 (5.2) | 0.954 (6.9) | 0.991 |
| 45 | 1.156 (0) | 1.264 (3.7) | 1.440 (7.3) | 1.650 (11.0) | 1.881 (14.7) | 1.843 |

- 발 오차가 줄면 L2P 가 거의 비례해서 준다. trans 30 은 발 오차 6.9 → 5.2 cm 면 Two-stage 보다 15% 좋다.
- trans 15 는 2.0 cm 이하 (지금 2.7 cm), trans 45 는 약 14 cm 이하 (지금 14.7 cm) 여야 Two-stage 를 이긴다.
- trans 5 는 정답 발이어도 Two-stage 보다 나쁘다: 발이 아니라 몸 모델 쪽 한계.
- 주의: 섞어서 만든 오차는 지금 발 모델의 오차와 방향이 같고 크기만 작다 (I11).

## 영상 (I13, `python feet_to_body/render_body.py`)

설정 `configs/viz/feet_to_body.yaml` → `outputs/videos/feet_to_body/<클립>/trans<길이>.mp4` (walk3 / run1 / obstacles1, 30 / 90).
2×2 전신 스켈레톤: GT / Two-stage / 발 → 몸 (예측한 발, 2차 best) / 발 → 몸 (정답 발, 1차 best). 네 칸 같은 카메라,
스켈레톤 색 = 구간, 회색 = GT 겹침, 초록 점 = 모델에 준 발. 칸 제목에 그 클립의 발 미끄러짐과 전신 위치 오차.

| 클립 | 전신 위치 오차 (cm) Two-stage / 예측한 발 / 정답 발 | 발 미끄러짐 (cm/frame) |
|---|---|---|
| walk3 trans 30 | 2.8 / 3.0 / 2.2 | 0.64 / 0.93 / 0.26 |
| run1 trans 30 | 5.8 / 5.7 / 5.3 | 0.25 / 0.25 / 0.24 |
| obstacles1 trans 30 | 6.0 / 9.3 / 2.7 | 0.01 / 0.00 / 0.03 |
| walk3 trans 90 | 31.7 / 46.3 / 44.9 | 0.65 / 1.73 / 1.44 |
| run1 trans 90 | 139.8 / 107.3 / 94.8 | 2.52 / 2.05 / 0.90 |
| obstacles1 trans 90 | 27.4 / 27.1 / 27.4 | 0.04 / 0.00 / 0.00 |

## 발 전용 모델 v2 (I12, `foot_v2/`)

발 예측을 고치는 시도: foot_only 손실에 발 속도 항을 더해 다시 학습. 설명·결과는 [foot_v2/README.md](../foot_v2/README.md).
새 발 모델로 평가: `python feet_to_body/eval_body.py --foot-ckpt experiments/foot_v2/best.pth --out outputs/eval/feet_to_body_footv2`.

