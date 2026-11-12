# src/ — 라이브러리 코드

실행 스크립트(`scripts/`)와 테스트(`tests/`)가 import 해서 쓰는 코드만 둔다.
이 폴더의 파일은 직접 실행하지 않는다.

| 경로 | 역할 | 단계 |
|---|---|---|
| `config.py` | `configs/` 의 YAML 을 읽는다. 경로/device 는 반드시 여기를 거친다 | 1 |
| `data/` | 데이터 파이프라인: BVH → 모델 입력 → 복원 (`data/README.md`) | 1 |
| `models/` | Context Transformer (✅ 2단계), Detail Transformer (3단계 예정) | 2, 3 |
| `training/` | Context 입력/마스크, 손실 함수(논문 3.5), 학습 한 스텝, 학습 도구 | 2, 3 |
| `evaluation/` | `inference.py`: 학습된 run 불러오기, 평가셋 (official / long), 임의 길이 생성 | 4 |
| `visualization/` | `skeleton_video.py`: 구간별 색 GT \| PRED 비교 영상 (render_samples.py, render_compare.py 가 사용) | 4 |

import 는 프로젝트 루트 기준으로 한다: `from src.data.bvh import load_bvh`.
(`pytest.ini` 의 `pythonpath = .`, 스크립트 맨 위의 `sys.path.insert` 가 이를 가능하게 한다.)
