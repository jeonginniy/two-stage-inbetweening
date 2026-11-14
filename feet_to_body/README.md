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

설정: `configs/feet_to_body.yaml`. 배치 준비는 `foot_only/data.py` 의 `prepare_foot_batch` 를 import 해서 쓴다.

## 실행 순서 (프로젝트 루트)

```bash
source .venv/bin/activate && export TSI_ENV=lab
# 통계 2개는 이미 있으면 생략 (ctx_base / 발 전용 모델과 같은 파일)
python scripts/compute_stats.py
python foot_only/compute_foot_stats.py

# 본 학습 (RTX 3070: 36 ms/iteration, VRAM 350 MB → 500k 약 5시간 + 검증)
nohup python feet_to_body/train_body.py > feet_to_body.out 2>&1 &
python feet_to_body/train_body.py --resume last          # 이어서

# 짧은 확인 (다른 폴더에)
python feet_to_body/train_body.py --run-name body_check_1k --max-iters 1000 --set train.val_interval=1000
# 데이터 없이 동작 확인 (결과 의미 없음)
python feet_to_body/train_body.py --env laptop --fake-data --run-name body_smoke --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
```

`feet_to_body.out` (학습 로그) 은 커밋하지 않는다. 기존 실험 폴더 이름 (`ctx_base`, `det_*`, `foot_only` 등) 은 `--run-name` 으로 쓸 수 없다.

## log.csv (iteration, epoch, kind, name, value)

- train (100 iteration 마다): `loss`, `rot`, `pos` (cm), `smooth`, `lr`
- val (2000 iteration 마다, trans t = 5/15/30, 제약 프레임 없음, 정답 발 조건):
  - `val{t}_loss/rot/pos/smooth` (후처리 없음), `val{t}_gpos_l2` (전역 관절 위치 L2 cm, 후처리 포함)
    → `experiments/ctx_base/log.csv` 의 같은 이름과 **같은 식·같은 val 분할** 이라 바로 비교할 수 있다.
  - `val{t}_gpos_l2_nopost`, `val{t}_foot_l2` (+ `_<관절>`): 출력 전신의 발 vs 조건으로 준 발 (cm)
- best = `val30_loss` 최소 (ctx_base 와 같은 규칙).

## 결과

학습 중 (2026-11-15 시작). 끝나면 여기와 docs/assumptions.md I 절에 기록한다.
