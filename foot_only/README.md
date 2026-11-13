# foot_only/: 발 전용 In-betweening 모델

transition 구간에서 전신 대신 **발 관절 4개 위치 + 발 접촉 4채널**만 예측하는 모델 (교수님 피드백: 발 정확도 먼저 확인,
이후 발 → 몸통 확장 예정). 기존 데이터 처리·FK·모델 코드는 import 해서 재사용하고, 이 폴더에는 새 파일만 둔다.
결정·가정은 `docs/assumptions.md` H 절, 설정은 `configs/foot_only.yaml`.

## 개념

- **출력**: 프레임마다 `[LeftFoot, LeftToe, RightFoot, RightToe]` (관절 3, 4, 7, 8) 의 xyz (12) + 접촉 logit (4).
- **좌표 (H1)**: 정렬 월드 좌표. 마지막 context 프레임에서 루트 XZ=0, 루트가 +X 를 보게 돌린 좌표 (cm, Y-up).
  기존 `PreparedBatch.gpos_gt` 와 같다. `foot_to_world` 로 원래 월드 좌표로 되돌린다 (루트 궤적 불필요).
- **입력 (H2)**: Context Transformer 입력 (전신 135, 모르는 프레임 0, 루트 선형보간) + 아는 프레임의 발 위치 12 + m_in = 148.
- **정규화 (H3)**: 입력은 ctx_base 통계, 발 위치는 train_stats 분할 (Subject 1–4) 로 따로 계산한 통계.
- **접촉 (H4, H5)**: 라벨 = 원래 월드 좌표에서 `|dx|+|dy|+|dz| < 0.2 cm/frame` (데이터셋 라벨 그대로).
  L1 속도는 회전에 따라 값이 변하므로, 예측 위치로 접촉을 판정할 때도 월드 좌표로 되돌려서 계산한다.
- **손실 (H7)**: `1.0 × L1(발 위치, cm) + 1.0 × BCE(접촉)`, transition 프레임만.
- **학습 설정**: ctx_base 와 같음 (batch 32, Noam 0.0625/8000, 500k, window 42, transition [5, 30], 제약 프레임 학습 때만).
  best = `val30_loss` 최소 (H6).

## 파일

| 파일 | 내용 |
|---|---|
| `data.py` | 발 관절 번호, `prepare_foot_batch`, `build_foot_input`, `foot_to_world`, `compute_foot_stats` |
| `model.py` | `build_foot_model` (ContextTransformer, 입력 148 → 출력 16), `split_output` |
| `train_utils.py` | `foot_forward` (손실), `validate` (log.csv 의 val 값) |
| `compute_foot_stats.py` | [실행] 발 위치 통계 → `<cache_dir>/stats/foot_only_stats.npz` |
| `check_data.py` | [실행] 데이터 검증 (shape, FK 일치, round-trip, 접촉 그래프 → `outputs/foot_only/check_data/`) |
| `train_foot.py` | [실행] 학습 → `experiments/foot_only/{config.yaml, log.csv, last.pth, best.pth}` |

## 실행 순서 (프로젝트 루트, `export TSI_ENV=lab`)

```bash
python scripts/compute_stats.py              # 입력 통계 (ctx_base 와 같은 파일, 이미 있으면 생략)
python foot_only/compute_foot_stats.py       # 발 위치 통계
python foot_only/check_data.py               # (선택) 데이터 검증
python foot_only/train_foot.py               # 학습 (약 3.5 시간)
python foot_only/train_foot.py --resume last # 중단 후 이어서
```

`experiments/foot_only/last.pth` 가 이미 있으면 `--resume` 없이는 시작하지 않는다 (덮어쓰기 방지).
기존 실험 이름 (ctx_base, det_base, det_cont 등) 은 `--run-name` 으로 쓸 수 없다.

## log.csv (iteration, epoch, kind, name, value)

- train (100 iteration 마다): `loss`, `pos` (cm), `contact` (BCE), `lr`
- val (2000 iteration 마다, trans t = 5/15/30, 제약 프레임·후처리 없음):
  `val{t}_loss/pos/contact`, `val{t}_foot_l2` (+ `_<관절>`), `val{t}_contact_acc` (+ `_<관절>`),
  `val{t}_contact_f1_<관절>`, `val{t}_poscontact_acc` (+ `_<관절>`: 예측 위치를 월드 좌표로 되돌려 판정한 접촉)
