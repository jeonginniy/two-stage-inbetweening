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

학습 중 (2026-11-15 시작). 끝나면 여기에 기록한다.
