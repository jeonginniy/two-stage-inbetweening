# Two-stage Transformer Motion In-betweening — 직접 구현

"Motion In-betweening via Two-stage Transformers" (Qin, Zheng, Zhou, ACM TOG 2022) 를
공부하며 직접 구현하는 프로젝트. 공식 코드(github.com/victorqin/motion_inbetweening)는
**읽고 참고만** 했고, 코드는 모두 새로 작성했다 (공식 코드 import/복사 없음).

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 데이터 파이프라인: BVH 읽기 → 모델 입력 표현 → 원래 형태로 복원 | ✅ 완료 (연구실 실제 LaFAN1 검증 통과) |
| **2** | **Context Transformer 구현 + 학습** | ✅ 구현·가짜 데이터 테스트 완료 / 🔬 실제 학습은 연구실 |
| 3 | Detail Transformer 구현 + 학습 | 예정 |
| 4 | 평가 (공식 README 수치와 비교) + 시각화 | 🔨 진행 중 (브랜치 `viz-update`): 구간별 색 비교 영상 ✅ |

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
│   │   └── inference.py           학습된 run 불러오기, 평가셋 (official / long), 임의 길이 생성
│   └── visualization/           4단계
│       └── skeleton_video.py      구간별 색 GT | PRED 비교 영상
├── scripts/                   실행 스크립트 (데모, 연구실 검증, 통계 계산, train_context.py)
├── tests/                     pytest 테스트 (+ 가짜 데이터 생성 도우미)
└── docs/                      정리 노트
    ├── assumptions.md           가정/미확인 사항 ← 꼭 읽기
    ├── data_representation.md   논문 3.2 정리
    └── context_detail_connection.md  Context ↔ Detail 연결 정리
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
python scripts/render_samples.py --index 1814 --trans 30     # 4단계 구간별 색 비교 영상 → outputs/videos/
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
