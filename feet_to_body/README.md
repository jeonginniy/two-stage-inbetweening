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

학습 중 (2026-11-15 시작, `experiments/feet_to_body_pred/`). 끝나면 결과를 여기에 기록한다.
