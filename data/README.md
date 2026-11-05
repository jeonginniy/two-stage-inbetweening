# data/ — 전처리 결과 캐시

이 폴더에는 **코드가 만든 파일만** 들어간다. 원본 LaFAN1 BVH 는 여기에 두지 않는다
(원본 위치는 `configs/env/<환경>.yaml` 의 `paths.lafan1_bvh_dir` 에서 관리).

캐시 위치 자체도 `configs/env/<환경>.yaml` 의 `paths.cache_dir` 로 바꿀 수 있다 (기본값 `data/cache`).

| 경로 | 만드는 코드 | 내용 |
|---|---|---|
| `cache/lafan1_parsed/<파일명>.npz` | `src/data/dataset.py` (LafanDataset) | BVH 하나를 파싱한 결과: `root_pos (F,3)`, `local_rot (F,22,3,3)`, `offsets (22,3)`, `contacts (F,4)`. 원본 파일 크기/수정시각이 바뀌면 자동으로 다시 만든다 |
| `cache/stats/lafan1_context_stats.npz` | `scripts/compute_stats.py` | z-score 통계 `mean (135,)`, `std (135,)` (논문 3.2, 3.6.1) |
| `cache/demo/` | `scripts/demo_pipeline.py` | 데모용 가짜 BVH 와 복원된 BVH |

- 캐시는 언제 지워도 된다. 다시 실행하면 새로 만든다.
- `cache/` 안의 파일은 git 에 올리지 않는다 (`.gitignore`).
