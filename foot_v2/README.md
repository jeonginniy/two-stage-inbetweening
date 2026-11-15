# foot_v2/: 발 전용 모델 v2 (발 속도 손실)

발 → 몸 실험 (`feet_to_body/`, assumptions I10, I11) 에서 전신 오차의 병목이 발 예측 정확도로 나와서,
발 전용 모델 (`foot_only/`) 을 고쳐 보는 시도. 기존 `foot_only/` 파일은 수정하지 않고 import 해서 쓴다.
결정·결과는 `docs/assumptions.md` I12, I14.

## 개념

- foot_only 손실: `L_pos (발 위치 L1, cm) + L_contact (접촉 BCE)` — 프레임마다 따로 맞춘다.
  그래서 접촉 구간에서 발이 멈춰 있지 못했다 (위치로 판정한 접촉 recall 발끝 0.41, H10).
- v2 손실: 위에 **발 속도 항**을 더한다 (논문 식 21 의 L_smooth 와 같은 생각).
  `L_vel = mean |(p̂_t − p̂_{t−1}) − (p_t − p_{t−1})|` (cm/frame), t = 첫 transition 프레임 … target 프레임 (경계 포함).
  접촉 구간은 정답 속도가 거의 0 이라 "딛은 발은 멈춰 있어라" 가 된다. 가중치 `w_vel = 1.0` (조정해 보지 않음).
- 모델·입력·데이터·학습 설정·체크포인트 형식은 foot_only 와 같다. 접촉 라벨 정의도 그대로.
- 검증은 기존 `validate` 그대로라 `val{t}_loss = L_pos + L_contact` 이고, `experiments/foot_only/log.csv` 와 바로 비교된다.
  best.pth 규칙도 같다 (val30_loss 최소). train 로그에만 `vel` (cm/frame) 이 추가된다.

## 파일

| 파일 | 내용 |
|---|---|
| `losses.py` | `foot_velocity_loss`, `foot_forward_v2` (기존 `foot_forward` + L_vel) |
| `train_foot_v2.py` | [실행] 학습. 기존 `foot_only/train_foot.py` 를 손실 함수·설정 읽기만 바꿔 끼워 실행 |
| `eval_foot_v2.py` | [실행] 평가. 기존 `foot_only/eval_foot.py` 를 체크포인트·출력 폴더만 바꿔 실행 |

설정: `configs/foot_v2.yaml` (`base: foot_only.yaml` 위에 `loss.w_vel` 만 추가).

## 실행 (프로젝트 루트, `export TSI_ENV=lab`)

```bash
nohup python foot_v2/train_foot_v2.py --run-name foot_v2 > foot_v2.out 2>&1 &    # 24 ms/it, 약 3시간 20분 (로그는 커밋하지 않음)
python foot_v2/train_foot_v2.py --run-name foot_v2 --resume last                 # 이어서
python foot_v2/eval_foot_v2.py                                                   # → outputs/eval/foot_v2/ (결과의 foot_only 행 = v2)
python feet_to_body/eval_body.py --foot-ckpt experiments/foot_v2/best.pth --out outputs/eval/feet_to_body_footv2
```

`--run-name` 은 꼭 줘야 하고, 기존 실험 폴더 이름은 거부된다.

## 결과

`experiments/foot_v2/` (500k, best.pth = iteration 424000, val30_loss 3.569; foot_only 는 3.525). 자세한 값은 assumptions I14.

발 자체 (공식 2232 클립, `outputs/eval/foot_v2/`):

| trans | 발 L2 (cm) foot_only → v2 | 발 미끄러짐 (GT 0.08) | 접촉 F1 (위치 판정) | 접촉 F1 (출력) |
|---|---|---|---|---|
| 5 | 1.00 → 0.95 | 0.102 → 0.095 | 0.722 → 0.756 | 0.841 → 0.841 |
| 15 | 2.73 → 2.73 | 0.110 → 0.106 | 0.710 → 0.740 | 0.818 → 0.820 |
| 30 | 6.91 → 6.84 | 0.191 → 0.178 | 0.583 → 0.612 | 0.768 → 0.768 |
| 45 | 14.67 → 14.69 | 0.334 → 0.294 | 0.414 → 0.459 | 0.684 → 0.674 |

v2 발을 조건으로 준 전신 (1차 발 → 몸 모델, 후처리 포함, `outputs/eval/feet_to_body_footv2/`):

| trans | L2P foot_only 발 → v2 발 | 전신 발 미끄러짐 | Two-stage L2P / 발 미끄러짐 |
|---|---|---|---|
| 5 | 0.149 → 0.150 | 0.183 → 0.180 | 0.114 / 0.140 |
| 15 | 0.449 → 0.453 | 0.209 → 0.198 | 0.430 / 0.179 |
| 30 | 0.954 → 0.950 | 0.298 → 0.275 | 0.991 / 0.281 |
| 45 | 1.881 → 1.893 | 0.703 → 0.641 | 1.843 / 0.555 |

- **발 위치 정확도는 오르지 않았다** (trans 30 발 L2 −1%). 그래서 전신 L2P 도 그대로다.
- 발 미끄러짐 (−7%) 과 위치로 판정한 접촉 (F1 +0.03) 만 조금 좋아졌고, 전신 발 미끄러짐도 그만큼 줄었다 (trans 30 −8%).
- 학습 범위 밖 (60, 90) 은 발 L2 가 조금 나빠졌다 (25.1 / 50.0 → 26.3 / 53.0 cm).
- 몸 모델을 v2 발로 다시 학습하지는 않았다 (2차 몸 모델에 v2 발을 줘도 같은 경향: trans 30 L2P 0.942 → 0.948).
