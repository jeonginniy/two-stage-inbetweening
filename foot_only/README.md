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
| `evaluate.py` | 평가 계산: `load_foot_run`, `predict_feet` (방법별 발 위치), `FootMetricAccumulator` (발 L2, 접촉 F1, 발 미끄러짐) |
| `eval_foot.py` | [실행] 평가 → `outputs/eval/foot_only/{metrics_long.csv, summary.csv, metrics.png, run_info.json}` |
| `render_feet.py` | [실행] 영상 (발끝 궤적·높이·속도·접촉, GT / foot-only / Two-stage) → `outputs/videos/foot_only/<클립>/trans<길이>.mp4` |

## 실행 순서 (프로젝트 루트)

```bash
source .venv/bin/activate
export TSI_ENV=lab                           # 또는 매 명령에 --env lab
```

**1. 데이터 준비** (한 번만, 결과는 `configs/env/lab.yaml` 의 `cache_dir` 아래 `stats/`)

```bash
python scripts/compute_stats.py              # 입력 (전신) 통계 lafan1_context_stats.npz. ctx_base 와 같은 파일, 이미 있으면 생략
python foot_only/compute_foot_stats.py       # 발 위치 통계 foot_only_stats.npz (train_stats 분할, 약 1분)
python foot_only/check_data.py               # (선택) 데이터 검증: FK 일치, round-trip, 접촉 라벨 → outputs/foot_only/check_data/
```

BVH 파싱 결과는 `data/cache/lafan1_parsed/` 에 캐시되어 있어 ctx_base 를 학습한 PC 면 다시 읽지 않는다.

**2. 학습** (RTX 3070: 24 ms/it, VRAM 341 MB, 500k ≈ 3시간 20분)

```bash
nohup python foot_only/train_foot.py > foot_only.out 2>&1 &   # → experiments/foot_only/ (로그 foot_only.out 은 커밋하지 않음)
python foot_only/train_foot.py --max-iters 1000 --run-name foot_check_1k   # (선택) 짧게 동작 확인
python foot_only/train_foot.py --env laptop --fake-data --run-name foot_smoke --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64   # 데이터 없이 스모크 (끝나면 experiments/foot_smoke 삭제)
```

설정은 `configs/foot_only.yaml` 에서 읽고, `--batch-size`, `--max-iters`, `--set 절.키=값` 으로 덮어쓴다.
실제로 쓴 설정은 `experiments/<run>/config.yaml` 에 저장된다 (`data` = 데이터셋 설정, `foot_data` = 발 전용 설정).

**3. 이어서 학습** (중단됐을 때)

```bash
python foot_only/train_foot.py --resume last  # last.pth 의 모델·optimizer·iteration·best 값에서 계속
```

- `experiments/<run>/last.pth` 가 이미 있으면 `--resume` 없이는 시작하지 않는다 (덮어쓰기 방지).
- 기존 실험 이름 (ctx_base, ctx_smoke, det_base, det_cont) 은 `--run-name` 으로 쓸 수 없다.
- 학습 중에는 run 폴더에 쓰지 않는다. 중간 확인은 last.pth 를 복사해서, 최종 수치는 best.pth 로만.

## 학습 결과 (`experiments/foot_only/`, best.pth = iteration 454000)

val 분할 (560 클립, 제약 프레임·후처리 없음). 자세한 내용은 `docs/assumptions.md` H9.

| trans | 발 L2 (cm) | 접촉 acc (출력) | 접촉 acc (위치 판정, 월드 좌표) | F1 발목 L / R | F1 발끝 L / R |
|---|---|---|---|---|---|
| 5 | 0.99 | 0.906 | 0.854 | 0.80 / 0.75 | 0.87 / 0.87 |
| 15 | 2.70 | 0.894 | 0.848 | 0.78 / 0.73 | 0.84 / 0.84 |
| 30 | 6.75 | 0.868 | 0.806 | 0.70 / 0.69 | 0.80 / 0.80 |

- 발목 접촉 F1 은 50k 이후 0.68–0.70 에서 오르지 않는다 (발끝보다 ~0.1 낮음). pos_weight 재학습은 평가 뒤 결정.
- 평가 계획·결정: `docs/foot_only_eval_plan.md`.

## 평가 (`python foot_only/eval_foot.py`, 약 1분)

설정 `configs/eval/foot_only.yaml` + `configs/eval/lengths.yaml` (ctx_base, det_cont, 평가셋, 발 미끄러짐 설정).
모든 방법을 같은 정답 창에서 발 4관절 위치로 바꿔 같은 함수로 계산한다. 접촉 F1 은 두 가지:
`pos` = 예측 위치를 월드 좌표로 되돌려 라벨 규칙으로 판정 (모든 방법), `out` = 접촉 출력 > 0.5 (foot-only, Two-stage).
`gt` 행은 검산용 (위치 판정 acc 1.000, 발 미끄러짐 = GT 값).

공식 벤치마크 2232 클립 (trans 30) / long 평가셋 (trans 90). 결과 전체는 `docs/assumptions.md` H10:

| 방법 | 발 L2 (cm) 30 / 90 | 발 미끄러짐 30 / 90 (GT 0.08) | 접촉 F1 pos 30 | 접촉 F1 out 30 (발목 / 발끝) |
|---|---|---|---|---|
| foot-only | **6.91** / **50.0** | **0.191** / **0.569** | **0.583** | **0.768** (0.70 / 0.80) |
| Two-stage (ctx_base + det_cont) | 7.81 / 52.5 | 0.281 / 1.164 | 0.414 | 0.740 (0.65 / 0.78) |
| Context + 후처리 | 8.89 / 52.1 | 0.476 / 1.445 | 0.174 | - |
| Context, 후처리 없음 | 8.96 / 52.4 | 0.478 / 1.371 | 0.176 | - |
| SLERP | 24.40 / 62.8 | 1.632 / 1.169 | 0.246 | - |

- 학습 범위 안에서는 발 위치·발 미끄러짐·접촉 모두 foot-only 가 가장 좋다 (trans 5 발 L2 만 Two-stage 0.96 < 1.00).
- 학습 범위 밖 (60, 90) 에서는 모든 모델이 크게 틀리고 차이가 줄어든다.
- 위치로 판정한 접촉 F1 이 출력보다 훨씬 낮다: 예측 발이 접촉 구간에서 멈춰 있지 못함 (recall 발끝 0.41).

## 영상 (`python foot_only/render_feet.py`)

설정 `configs/viz/foot_only.yaml` (클립 walk3 / run1 / obstacles1, 길이 30 / 90, 색, 출력 경로).
전신 스켈레톤 대신 발이 땅에 제대로 서 있는지를 보는 화면: 왼쪽 = 위에서 본 발끝 궤적 (접촉 중이면 채운 점),
오른쪽 = LeftToe 높이 (회색 띠 = GT 접촉) / 수평 속도 / 접촉 띠 (GT 라벨 / out / pos). 계산은 평가와 같은 `predict_feet`.

| 클립 | 발 L2 (cm) foot-only / Two-stage | 발 미끄러짐 (cm/frame) | 보이는 것 |
|---|---|---|---|
| walk3 trans 30 | 2.6 / 3.2 | 0.37 / 0.64 | foot-only 는 딛은 발이 고정, Two-stage 는 2 cm 뜨고 미끄러짐 |
| run1 trans 30 | 4.2 / 6.7 | 0.21 / 0.25 | 둘 다 발 딛는 타이밍을 따라감 |
| obstacles1 trans 30 | 11.5 / 7.8 | 0.01 / 0.01 | 23 cm 장애물 위 발을 foot-only 는 7 cm 로 내림 (약점) |
| walk3 trans 90 | 38.9 / 31.4 | 0.98 / 0.65 | 70 프레임 서 있는 구간에서 foot-only 발이 돌아다님 |
| run1 trans 90 | 86.8 / 163.1 | 0.78 / 2.52 | 둘 다 target 에 못 감, Two-stage 발이 땅 아래 −18 cm |

## log.csv (iteration, epoch, kind, name, value)

- train (100 iteration 마다): `loss`, `pos` (cm), `contact` (BCE), `lr`
- val (2000 iteration 마다, trans t = 5/15/30, 제약 프레임·후처리 없음):
  `val{t}_loss/pos/contact`, `val{t}_foot_l2` (+ `_<관절>`), `val{t}_contact_acc` (+ `_<관절>`),
  `val{t}_contact_f1_<관절>`, `val{t}_poscontact_acc` (+ `_<관절>`: 예측 위치를 월드 좌표로 되돌려 판정한 접촉)
