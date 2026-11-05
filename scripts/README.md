# scripts/ — 실행 스크립트

모든 스크립트는 **프로젝트 루트에서** 실행하고, `--env` 로 환경을 고른다
(또는 환경변수 `TSI_ENV`). 경로/device 는 `configs/env/<환경>.yaml` 에서 읽는다.

| 스크립트 | 어디서 | 하는 일 |
|---|---|---|
| `demo_pipeline.py` | 노트북 ✅ / 연구실 | 파이프라인 전체를 한 번 돌리고 각 단계 shape, 복원 오차 출력. `--bvh` 없으면 가짜 BVH 사용 |
| `inspect_dataset.py` | 연구실 🔬 | 분할별 파일/프레임/클립 수 → 논문 4.1 수치(train 404195, benchmark 2232)와 비교 |
| `check_bvh_roundtrip.py` | 연구실 🔬 | 실제 BVH 로 스켈레톤 일치, 위치채널=OFFSET, 표현/복원 오차, BVH 재저장 오차, heading 가정 검사 |
| `compute_stats.py` | 연구실 🔬 | z-score 통계 μ, σ 계산 → `data/cache/stats/lafan1_context_stats.npz` (논문 3.6.1) |
| `compare_stats_with_official.py` | 연구실 🔬 | 내 통계 vs 공식 저장소의 `tests/train_stats_context.pkl` 차원별 비교 |

🔬 = 실제 LaFAN1 이 필요. 노트북에서 실행하면 "경로가 null" 메시지를 내고 종료한다 (정상).

## 연구실 검증 순서 (권장)

```bash
export TSI_ENV=lab
python scripts/inspect_dataset.py          # 1. 파일/클립 수가 논문과 맞는지
python scripts/check_bvh_roundtrip.py      # 2. 가정(A1~A4) 확인, 복원 오차 확인
pytest -m lab -v                           # 3. 실제 데이터 테스트
python scripts/compute_stats.py            # 4. 정규화 통계 (처음엔 BVH 파싱+캐시 때문에 몇 분 걸림)
python scripts/compare_stats_with_official.py --official <공식repo>/tests/train_stats_context.pkl  # 5.
python scripts/demo_pipeline.py --bvh <LaFAN1>/walk1_subject1.bvh --start 1000   # 6. 복원 BVH 를 Blender 로 확인
```

`compare_stats_with_official.py` 는 공식 코드를 import/실행하지 않고 공식 "결과 파일" 만 읽는다.
pickle 은 임의 코드 실행 위험이 있어서 numpy 배열만 허용하는 제한된 Unpickler 를 쓴다.
