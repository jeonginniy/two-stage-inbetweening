# Two-stage Transformer Motion In-betweening — 직접 구현

"Motion In-betweening via Two-stage Transformers" (Qin, Zheng, Zhou, ACM TOG 2022) 를
공부하며 직접 구현하는 프로젝트. 공식 코드(github.com/victorqin/motion_inbetweening)는
**읽고 참고만** 했고, 코드는 모두 새로 작성했다 (공식 코드 import/복사 없음).

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 데이터 파이프라인: BVH 읽기 → 모델 입력 표현 → 원래 형태로 복원 | ✅ 완료 (연구실 실제 LaFAN1 검증 통과) |
| **2** | **Context Transformer 구현 + 학습** | ✅ 구현·가짜 데이터 테스트 완료 / 🔬 실제 학습은 연구실 |
| 3 | Detail Transformer 구현 + 학습 | ✅ 구현·학습 완료. 최종 = `experiments/det_cont/best.pth` (1M 까지 이어서 학습, iteration 978000; 500k 판은 `det_base`). 결과는 아래 "3단계 결과" |
| 4 | 평가 (공식 README 수치와 비교) + 시각화 | 🔨 브랜치 `viz-update`: 구간별 색 비교 영상 ✅, 길이별 L2P/L2Q/NPSS/발 미끄러짐 ✅ (Context only) |
| 발→몸 | 발 조건 전신 예측 (교수님 피드백 2단계: 발 → 몸). 1차 = 정답 발 조건 (상한) | ✅ 학습·평가 완료 (브랜치 `feat/feet-to-body`, `experiments/feet_to_body/best.pth` = iteration 164000). 정답 발: trans 30 L2P 0.650 (Two-stage 0.991), 예측한 발: 0.954. 결과 표는 [feet_to_body/README.md](feet_to_body/README.md). 2차 (예측한 발로 학습, `experiments/feet_to_body_pred/`, best = 152000) 완료: trans 30 L2P 0.942 로 1차 (0.954) 와 거의 같음. Two-stage 보다 좋은 길이는 trans 30 뿐. 발 오차 민감도 (trans 30: 발 오차 6.9 → 5.2 cm 면 L2P 0.84)·영상 (`outputs/videos/feet_to_body/`)·발 전용 모델 v2 ([foot_v2/README.md](foot_v2/README.md)) |
| 발 | 발 전용 In-betweening (교수님 피드백: 발만 먼저 예측) | ✅ 학습·평가 완료 (브랜치 `feat/foot-only`, `experiments/foot_only/best.pth` = iteration 454000, `outputs/eval/foot_only/`), 발 영상 ✅ (`foot_only/render_feet.py`) |

### 1단계 결과 (연구실, 실제 LaFAN1)

클립 수 train 404195 / benchmark 2232 (논문과 일치), 공식 통계와 평균 차이 ~1e-12·표준편차 차이 ~1e-8,
스켈레톤 불일치 0, 위치채널=OFFSET 오차 0, 복원 오차 1e-13 이하, heading 0.47~0.88 (모두 양수).
자세한 내용은 [docs/assumptions.md](docs/assumptions.md) 0 절.

### 2단계 상태

- ✅ 구현: `src/models/` (Context Transformer, 상대 위치 어텐션, Keyframe PE), `src/training/` (입력·마스크, 손실, 후처리),
  `scripts/train_context.py`, 설정 `configs/model/context.yaml`.
  입력/출력 차원(136/135)과 관절 수(22)는 코드에 쓰지 않고 스켈레톤 설정에서 계산한다.
- ✅ 이 환경 (데이터 없음): `pytest` → **65 passed, 4 skipped**. 가짜 데이터로 shape, forward/backward,
  마스크 동작, 과적합, 학습 스크립트 끝까지 실행을 확인.
- 🔬 연구실: [scripts/README.md](scripts/README.md) 의 "2단계: Context Transformer 학습" 순서대로 실행.
  2단계에서 정한 것/가정은 [docs/assumptions.md](docs/assumptions.md) E 절.

### 4단계 결과 (브랜치 `viz-update`, 교수님 미팅 피드백)

지표 구현 확인: SLERP·zero-velocity 기준선이 공식 README 와 소수점 4자리 일치 (L2P, NPSS; L2Q 는 README 2자리 반올림 일치).
Context Transformer 만 평가 (Detail Transformer 미구현). `best.pth` (iteration 338000), 공식 벤치마크 2232 클립:

| trans | 학습 범위 | L2P (+후처리) / README | L2Q / README | NPSS / README |
|---|---|---|---|---|
| 5 | 안 | 0.1289 / 0.1288 | 0.1143 / 0.1143 | 0.0015 / 0.0015 |
| 15 | 안 | 0.4650 / 0.4623 | 0.3210 / 0.3154 | 0.0215 / 0.0211 |
| 30 | 안 | 1.0368 / 1.0354 | 0.6059 / 0.5898 | 0.1272 / 0.1210 |
| 45 | 밖 | 1.8874 / 1.9439 | 0.9145 / 0.9114 | 0.3458 / 0.3349 |

긴 길이 (long 평가셋 2151 클립, 모든 길이에서 같은 시작 프레임) L2P: 모델 (+후처리) vs SLERP

| trans (초) | 30 (1) | 60 (2) | 90 (3) | 150 (5) | 300 (10) |
|---|---|---|---|---|---|
| 모델 | 1.04 | 3.16 | 6.31 | 12.48 | 21.80 |
| SLERP | 2.34 | 4.79 | 7.46 | 12.26 | 20.23 |

- 학습 범위 (≤ 30) 를 넘으면 오차가 빠르게 커지고, **150 프레임 (5초) 부터는 단순 SLERP 보다 나빠진다** (L2Q 는 90 부터).
- 긴 구간에서 모델은 target 까지 이동하지 못하고 (trans 300 루트 속도 GT 의 56%), 공식 후처리 근사 (식 23) 가
  그 차이를 transition 양 끝의 점프로 나눈다 (trans 90 샘플: 시작·끝 각 ~1 m). 학습 범위 안에서는 경계 점프 ±1.5 cm.
- 결과 파일: `outputs/eval/ctx_base/metrics_by_length.{csv,png}`, 영상 `outputs/videos/trans_<길이>/`.
  해석 주의 (NPSS 길이 의존성, 발 미끄러짐 vs 이동량) 와 가정은 `docs/assumptions.md` F 절.

### 3단계 결과 (브랜치 `detail-transformer`, Detail Transformer)

고정된 Context (`ctx_base/best.pth`) 위에서 Detail 을 500k iteration 학습 (41 ms/it, VRAM 384 MB, 약 6시간).
`det_base/best.pth` (iteration 498000) 로 평가. 두 단계 = Context + 후처리 → Detail (Detail 출력 후처리 없음, README 와 같은 조건).

공식 벤치마크 2232 클립, L2P (L2Q / NPSS 는 `outputs/eval/det_base/metrics_by_length.csv`):

| trans | Context 만 | 두 단계 | README 두 단계 | 발 미끄러짐 Context 만 → 두 단계 (cm/frame, GT 0.08) |
|---|---|---|---|---|
| 5 | 0.1289 | 0.1177 | 0.1049 | 0.177 → 0.143 |
| 15 | 0.4650 | 0.4425 | 0.3943 | 0.315 → 0.186 |
| 30 | 1.0368 | 1.0117 | 0.8948 | 0.476 → 0.301 |
| 45 (밖) | 1.8874 | 1.8686 | 1.6777 | 0.909 → 0.620 |

- 두 단계는 모든 공식 길이에서 Context 만보다 좋고, **발 미끄러짐은 32–41% 줄어든다**.
- 하지만 L2P 개선 폭은 README 보다 작다 (trans 30: README −13.6%, 여기 −2.4% → 두 단계 결과가 README 보다 13% 나쁨).
  Context 만은 README 와 일치하므로 차이는 Detail 쪽. 가설: Context 가 train 분할에서 훨씬 정확해서
  (trans 30 위치 오차 train 3.7 cm vs val 6.4 cm) Detail 이 쉬운 입력으로만 학습됨 (docs/assumptions.md G15, 미확인).
- 긴 길이 (long 평가셋) L2P: 30 → 1.019 (Context 만 1.044), 60 → 3.166 (3.163), 90 → 6.49 (6.31), 150 → 13.03 (12.48),
  300 → 22.83 (21.80). **60 프레임부터는 Detail 이 위치를 고치지 못하고 90 부터는 약간 나빠진다.**
  Context 가 target 에 도달하지 못하는 한계를 Detail 이 넘지 못한다.
- 경계: trans 30 샘플 5개 경계 점프 ±0.8 cm. trans 90 에서는 Detail 출력 (후처리 없음) 의 끝 경계 점프가 커진다
  (run1_subject5 4800: Context +후처리 시작/끝 +99/+97 cm → 두 단계 +76/+264 cm). G16.
- 결과: `outputs/eval/det_base/`, `outputs/eval/det_base_long/`, 영상 `outputs/videos/two_stage/trans_{030,060,090}/`
  (클립마다 Context 만 / det_base / det_cont + 위아래 비교 영상, 영상별 설명은 그 폴더 `README.md`)

**이어서 학습 (det_cont, 500k → 1M iteration, `det_cont/best.pth` = iteration 978000)** — docs/assumptions.md G17, G18:

| trans | Context 만 | det_base | det_cont | README 두 단계 | 발 미끄러짐 det_base → det_cont |
|---|---|---|---|---|---|
| 5 | 0.1289 | 0.1177 | 0.1136 | 0.1049 | 0.143 → 0.140 |
| 15 | 0.4650 | 0.4425 | 0.4302 | 0.3943 | 0.186 → 0.179 |
| 30 | 1.0368 | 1.0117 | 0.9909 | 0.8948 | 0.301 → 0.281 |
| 45 (밖) | 1.8874 | 1.8686 | 1.8429 | 1.6777 | 0.620 → 0.555 |

- 모든 공식 길이에서 det_base 보다 좋아졌지만 폭은 작다 (trans 30 Context 대비 −2.4% → −4.4%, README −13.6%).
  val30 손실도 0.1171 → 0.1147 (2%) 만 줄었다. **학습 길이만으로는 README 와의 차이를 설명하기 어렵다.**
- long 평가셋 L2P (det_cont): 60 → 3.130 (Context 만 3.163), 90 → 6.46 (6.31), 150 → 13.11 (12.48), 300 → 23.34 (21.80).
  60 까지는 조금 나아졌고, 90 이상은 여전히 Context 만보다 나쁘다 (300 은 det_base 22.83 보다도 나쁨).
- 결과: `outputs/eval/det_cont_official/`, `outputs/eval/det_cont_long/`. det_base 결과·영상 사본은
  `outputs/archive/2026-11-13_ctx_base_det_base/`.

### 발 전용 모델 (브랜치 `feat/foot-only`)

transition 구간에서 전신 대신 **발 4관절 (LeftFoot, LeftToe, RightFoot, RightToe) 위치 + 접촉 4채널**만 예측한다.
Context Transformer 구조와 ctx_base 학습 설정을 그대로 쓰고 입출력만 바꿨다 (입력 148 = 전신 135 + 아는 프레임 발 위치 12 + 마스크,
출력 16). 손실 = 발 위치 L1 (cm) + 접촉 BCE. 코드·실행 방법은 [foot_only/README.md](foot_only/README.md), 결정·가정은
[docs/assumptions.md](docs/assumptions.md) H 절.

500k 학습 (3시간 19분), best = iteration 454000. val 분할 (제약 프레임·후처리 없음):

| trans | 발 L2 (cm) | 접촉 acc (출력) | 접촉 F1 발목 L / R | 접촉 F1 발끝 L / R |
|---|---|---|---|---|
| 5 | 0.99 | 0.906 | 0.80 / 0.75 | 0.87 / 0.87 |
| 15 | 2.70 | 0.894 | 0.78 / 0.73 | 0.84 / 0.84 |
| 30 | 6.75 | 0.868 | 0.70 / 0.69 | 0.80 / 0.80 |

- 발목 접촉 F1 은 발끝보다 ~0.1 낮고 50k 이후 오르지 않는다 (H9).

평가 (`python foot_only/eval_foot.py`, 공식 벤치마크 2232 클립 / long 평가셋, 발 4관절만 비교, H10):

| trans | 발 L2 (cm) foot-only / Two-stage / Context+후처리 / SLERP | 발 미끄러짐 (cm/frame, GT 0.08) 같은 순서 |
|---|---|---|
| 5 | 1.00 / **0.96** / 1.28 / 3.78 | **0.102** / 0.140 / 0.177 / 1.181 |
| 15 | **2.73** / 3.17 / 4.00 / 14.22 | **0.110** / 0.179 / 0.315 / 1.936 |
| 30 | **6.91** / 7.81 / 8.89 / 24.40 | **0.191** / 0.281 / 0.476 / 1.632 |
| 45 (밖) | **14.67** / 15.82 / 16.98 / 32.35 | **0.334** / 0.555 / 0.909 / 1.544 |
| 60 (밖, long) | **25.06** / 26.84 / 27.93 / 42.56 | **0.445** / 0.895 / 1.258 / 1.417 |
| 90 (밖, long) | **50.04** / 52.52 / 52.10 / 62.83 | **0.569** / 1.164 / 1.445 / 1.169 |

- 발만 예측하면 발 위치·발 미끄러짐 모두 Two-stage 보다 좋다 (trans 30 발 미끄러짐 −32%). 접촉 출력 F1 도 0.768 vs 0.740.
- 하지만 학습 범위 밖에서는 발 전용 모델도 같이 무너진다 (trans 90 발 L2 50 cm). 긴 구간 문제는 출력 차원을 줄여도 남는다.
- 영상 (`python foot_only/render_feet.py` → `outputs/videos/foot_only/`): 학습 범위 안에서는 foot-only 가 딛은 발을 더 잘 고정하지만,
  장애물 위처럼 높은 곳의 접촉은 땅 쪽으로 끌어내리고, trans 90 에서는 서 있는 발도 돌아다닌다 (클립별 설명 `foot_only/README.md`).
- 결과: `outputs/eval/foot_only/`, 계획·결정: [docs/foot_only_eval_plan.md](docs/foot_only_eval_plan.md).

## 폴더 구조

```
two_stage_inbetweening/
├── README.md                  ← 이 파일
├── requirements.txt           파이썬 패키지 목록
├── pytest.ini                 pytest 설정 (lab 마커, import 경로)
├── configs/                   설정 (경로/GPU 는 여기에서만 관리)
│   ├── env/laptop.yaml          노트북: LaFAN1 경로 null, device cpu
│   ├── env/lab.yaml             연구실: LaFAN1 경로(수정 필요), device cuda:0
│   ├── data/lafan1.yaml         context_len, window/offset, 분할, 접촉 임계값
│   ├── model/context.yaml       2단계: 모델 크기, batch size, 학습률, 손실 가중치
│   ├── model/detail.yaml        3단계: Detail 모델, 고정할 Context, 손실 가중치 (발 접촉·미끄러짐)
│   ├── eval/lengths.yaml        4단계: 평가 길이 목록, 평가셋, 체크포인트
│   └── viz/video.yaml           4단계: 구간 색, 영상 구성, 샘플 수·seed, 출력 경로
├── assets/
│   └── skeleton_lafan1.yaml     22관절 이름·부모, 발 관절, 루트 앞 축
├── data/                      전처리 캐시 (자동 생성, git 제외)
├── experiments/               학습 결과: 체크포인트, 로그 (자동 생성, git 제외)
├── outputs/                   4단계 결과: 영상, 평가 CSV/그래프 (자동 생성, git 제외)
├── src/                       라이브러리 코드
│   ├── config.py                configs/ 로더
│   └── data/                    1단계 파이프라인
│       ├── rotations.py           오일러 ↔ 행렬 ↔ 6D ↔ 쿼터니언        (논문 3.2)
│       ├── skeleton.py            스켈레톤, FK                         (논문 식 17)
│       ├── bvh.py                 BVH 읽기/쓰기
│       ├── preprocess.py          클립 정렬, 발 접촉 라벨              (논문 3.6.1, 3.4)
│       ├── representation.py      135차원 벡터, z-score               (논문 3.2, 식 10)
│       ├── dataset.py             LaFAN1 슬라이딩 윈도우 Dataset       (논문 3.6.1, 4.1)
│       └── pipeline.py            클립 → 모델 입력 → 복원 (한 함수씩)
│   ├── models/                  2단계
│   │   ├── layers.py              상대 위치 어텐션, FFN, Keyframe PE      (논문 식 11–13)
│   │   └── context_transformer.py Context Transformer                  (논문 3.3)
│   └── training/                2단계
│       ├── context_inputs.py      빈칸/마스크/보간/합치기/후처리         (논문 3.3, 3.6, 식 23)
│       ├── losses.py              L_rot, L_pos, L_smooth                (논문 식 21)
│       ├── context.py             배치 하나 학습/추론
│       └── utils.py               Noam 학습률, 체크포인트, AMP
│   ├── evaluation/              4단계
│   │   ├── inference.py           학습된 run 불러오기, 평가셋 (official / long), 임의 길이 생성
│   │   ├── metrics.py             L2P, L2Q, NPSS, 발 미끄러짐                 (논문 4.1)
│   │   └── baselines.py           SLERP 보간, zero-velocity 기준선
│   └── visualization/           4단계
│       └── skeleton_video.py      구간별 색 GT | PRED 비교 영상
├── foot_only/                 발 전용 모델 (데이터, 모델, 학습, 평가, 영상; configs/foot_only.yaml, configs/{eval,viz}/foot_only.yaml, README 참고)
├── feet_to_body/              발 → 몸 모델 (발 조건으로 전신 예측: 학습, 평가; configs/feet_to_body.yaml, configs/eval/feet_to_body.yaml, README 참고)
├── scripts/                   실행 스크립트 (데모, 연구실 검증, 통계 계산, train_context.py, train_detail.py)
├── tests/                     pytest 테스트 (+ 가짜 데이터 생성 도우미)
└── docs/                      정리 노트
    ├── assumptions.md           가정/미확인 사항 ← 꼭 읽기
    ├── data_representation.md   논문 3.2 정리
    ├── context_detail_connection.md  Context ↔ Detail 연결 정리
    └── foot_only_eval_plan.md   발 전용 모델 다음 평가 준비 메모
```

각 폴더의 `README.md` 에 더 자세한 설명이 있다.

### 제안했던 예시 구조에서 바꾼 점과 이유

- **`configs/env/` 와 `configs/data/` 분리**: 노트북/연구실마다 다른 값(경로, GPU)과
  어디서나 같은 값(window, context_len)을 나눴다. 환경을 옮길 때 `env/` 파일만 고르면 된다.
- **`scripts/` 추가**: `src/` 는 import 하는 라이브러리, `scripts/` 는 실행하는 진입점으로 나눴다.
  2단계의 `train_context.py` 등도 여기에 들어간다.
- **`tests/` 추가**: 데이터 없이 노트북에서 검증하기 위해. 실제 데이터 테스트는 `lab` 마커로 분리.
- **`docs/` 추가**: 가정 목록과 논문 정리 노트를 코드와 함께 관리.

## 설치

### 노트북 (Windows, CPU) — 이미 설치되어 있음

```powershell
cd C:\Users\aa010\projects\two_stage_inbetweening
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pyyaml pytest
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 연구실 (Ubuntu, RTX 3070)

```bash
cd two_stage_inbetweening
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyyaml pytest
pip install torch            # CUDA 빌드. 드라이버에 맞는 버전은 pytorch.org 에서 확인
```
그리고 `configs/env/lab.yaml` 의 `lafan1_bvh_dir` 을 실제 LaFAN1 폴더로 수정한다.

## 실행

```powershell
# 노트북
.\.venv\Scripts\python.exe -m pytest                                # 테스트
.\.venv\Scripts\python.exe scripts\demo_pipeline.py --env laptop     # 파이프라인 데모 (가짜 BVH)
```

```bash
# 연구실
export TSI_ENV=lab
pytest -m lab -v
python scripts/inspect_dataset.py
python scripts/check_bvh_roundtrip.py
python scripts/compute_stats.py
python scripts/compare_stats_with_official.py --official <공식repo>/tests/train_stats_context.pkl
python scripts/train_context.py --run-name ctx_base          # 2단계 학습 (자세히: scripts/README.md)
python scripts/train_detail.py --run-name det_base           # 3단계 학습 (고정된 Context 위에서)
python scripts/render_samples.py --trans 30                  # 4단계 구간별 색 비교 영상 (seed 고정 5개) → outputs/videos/trans_30/
python scripts/eval_lengths.py                               # 4단계 길이별 지표 → outputs/eval/ctx_base/ (자세히: scripts/README.md)
```

## 코드에서 쓰는 방법 (1단계 파이프라인)

```python
from src.config import load_data_config, load_env_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input, model_output_to_motion
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.data.skeleton import Skeleton

env, cfg = load_env_config("lab"), load_data_config()
skel = Skeleton.from_yaml(cfg["skeleton"])
sp = cfg["splits"]["train"]
ds = LafanDataset(env["paths"]["lafan1_bvh_dir"], sp["actors"], sp["window"], sp["offset"], skel,
                  cache_dir=env["paths"]["cache_dir"])
norm = ZScoreNormalizer.load(env["paths"]["cache_dir"] / "stats" / "lafan1_context_stats.npz")

inp = motion_to_model_input(batch["root_pos"], batch["local_rot"], cfg["context_len"],
                            skel.root_forward_local_axis, normalizer=norm)   # inp.x: (B, T, 135)
root_pos, local_rot = model_output_to_motion(y, FeatureLayout(22), norm, frame=inp.frame)
```

## 코드에서 쓰는 방법 (2단계 Context Transformer)

```python
from src.config import load_model_config
from src.models.context_transformer import build_context_transformer
from src.training.context import context_forward, infer_context, prepare_batch
from src.training.context_inputs import TransitionSpec

mcfg = load_model_config("context")
layout = FeatureLayout(skel.num_joints)                       # D = J*6+3 → 입력 D+1, 출력 D
model = build_context_transformer(mcfg["model"], layout)

pb = prepare_batch(batch, skel, norm, cfg["context_len"])     # 정렬 + 정규화 + 정답 FK
spec = TransitionSpec(context_len=10, target_idx=40)          # transition 10..39, target 40
losses = context_forward(model, pb, spec, skel, layout, norm, mcfg["loss"])   # {"loss","rot","pos","smooth"}
y = infer_context(model, pb.x_gt, spec, layout)               # 예측 + 후처리 (식 23), (B, T, D) 정규화 공간
```
