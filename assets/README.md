# assets/ — 스켈레톤 정의

| 파일 | 내용 |
|---|---|
| `skeleton_lafan1.yaml` | LaFAN1 22관절의 이름, 부모 관계, 발 접촉 관절, 루트의 "앞" 축 |

- 관절 순서가 곧 모델 입력 벡터에서 관절이 놓이는 순서다 (논문 3.2, `r_t ∈ R^{22×6}`).
- **뼈 길이(offset)는 적지 않았다.** 실제 값은 BVH 파일에서 읽는다
  (노트북에 LaFAN1 이 없어 실제 값을 볼 수 없었고, 추측으로 채우지 않았다).
- `src/data/dataset.py` 는 BVH 를 읽을 때마다 이 정의와 이름/부모가 같은지 확인하고,
  다르면 어떤 관절이 다른지 표로 보여 주며 멈춘다.

## 연구실에서 확인할 것

이 파일의 내용 중 아래 두 가지는 **가정**이다 (`docs/assumptions.md` A1, A3).

1. 관절 이름/부모 → `python scripts/check_bvh_roundtrip.py --env lab` 의 `skel` 열이 전부 OK 인지
2. `root_forward_local_axis: [0, 1, 0]` → 같은 스크립트의 `heading` 열이 1 에 가까운지
