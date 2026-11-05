# 논문 3.2 Data Representation 정리 — 모캡 관절 데이터가 입력 벡터가 되기까지

코드 위치는 괄호 안에 적었다.

## 1. 출발점: BVH 한 프레임

BVH 한 줄에는 22개 관절마다 (위치 3 + 오일러 회전 3) 값이 들어 있다 (LaFAN1 은 관절마다 6채널).

```
Hips: Xpos Ypos Zpos Zrot Yrot Xrot | LeftUpLeg: Xpos Ypos Zpos Zrot Yrot Xrot | ... (22개)
```

- **회전**: 부모 관절 기준의 로컬 회전, 오일러각(도).
  `Zrot Yrot Xrot` 은 R = Rz · Ry · Rx (벡터에 X 회전이 먼저 적용). (`rotations.euler_to_matrix`)
- **위치**: 루트(Hips)만 월드 위치로 의미가 있고, 나머지 관절은 부모 기준 위치 = 뼈 길이(OFFSET) 로 상수다.

## 2. 어떤 정보만 남기나

논문: 프레임 t 의 포즈 = **모든 관절의 로컬 회전 r_t ∈ R^{J×6}** + **루트의 월드 위치 p_t ∈ R^3**.

- 뼈 길이는 상수라서 넣지 않는다 → 출력에서도 뼈 길이가 바뀔 수 없음 (일관된 몸)
- 월드 회전/월드 위치가 아니라 **로컬** 회전을 쓰는 이유: 어떤 로컬 회전을 출력하든 FK 를 하면 항상
  "뼈가 연결된" 올바른 몸이 된다. (월드 위치를 직접 출력하면 뼈 길이가 늘었다 줄었다 할 수 있다.)
- 관절 위치가 필요할 때는 FK 로 계산한다: g_t = FK(p_t, r_t) (식 17, `skeleton.forward_kinematics`)

## 3. 왜 6D 회전인가

| 표현 | 차원 | 문제 |
|---|---|---|
| 오일러각 | 3 | 짐벌락, 359° 와 1° 가 수치상 멀다 (불연속) |
| 쿼터니언 | 4 | q 와 −q 가 같은 회전 (유일하지 않음), 불연속 |
| 회전행렬 | 9 | 연속이지만 네트워크 출력이 직교행렬이라는 보장이 없음 |
| **6D** | 6 | 연속 + 유일. 아무 6개 값이나 Gram–Schmidt 로 회전행렬로 만들 수 있음 |

- **행렬 → 6D**: 앞 두 열만 남긴다 (`rotations.matrix_to_6d`).
  세 번째 열은 앞 두 열의 외적이라 정보가 중복이기 때문.
- **6D → 행렬** (`rotations.sixd_to_matrix`):
  ```
  b1 = a1/|a1|,  b2 = normalize(a2 − (b1·a2) b1),  b3 = b1 × b2,  R = [b1 b2 b3]
  ```

## 4. 벡터로 펼치기 (D = 22 × 6 + 3 = 135)

```
인덱스:  0 ........ 5 | 6 ...... 11 | ... | 126 ..... 131 | 132 133 134
내용:    Hips 6D      | LeftUpLeg 6D | ... | RightHand 6D   | 루트 x  y  z
```
관절 j 의 6D 는 인덱스 [6j, 6j+6), 그 안의 순서는 [R00, R01, R10, R11, R20, R21].
(`representation.FeatureLayout`, `encode_pose`)

## 5. 클립 단위 정렬 (논문 3.6.1)

같은 "걷기" 라도 방 안의 위치/방향이 다르면 숫자는 완전히 다르다. 그래서 클립마다
**마지막 context 프레임 (인덱스 9)** 을 기준으로

1. 루트의 X, Z 를 빼서 0 으로 만들고 (높이 Y 는 그대로)
2. 루트의 "앞" 방향이 +X 를 보도록 Y축(위쪽)으로 회전한다.

몸 전체에 같은 강체 변환을 하는 것인데, 로컬 회전 표현에서는 **루트의 위치/회전만** 바꾸면 된다
(다른 관절은 부모 기준이라 그대로). (`preprocess.canonicalize_root`)
역변환 정보(위치 offset, 회전 R_y)를 저장해 두면 원래 월드 좌표로 정확히 되돌릴 수 있다.

## 6. z-score 정규화

회전 6D 값은 [−1, 1] 범위인데 루트 위치는 수십~수백 cm 라서 스케일이 너무 다르다.
차원마다 `x = (x̃ − μ) / σ`. μ, σ 는 학습 데이터(Subject 1–4, window 50, offset 20)의 정렬된 클립에서 구한다.
(`representation.RunningStats`, `ZScoreNormalizer`, `scripts/compute_stats.py`)

## 7. 다시 원래대로 (출력 해석, 식 10)

```
y (B,T,135) ─denormalize─▶ ỹ ─R(·): [0:132] → (22,6) → Gram–Schmidt → (22,3,3)
                               └P(·): [132:135] → 루트 위치
            ─uncanonicalize─▶ 원래 월드 좌표 ─write_bvh─▶ .bvh
```
(`pipeline.model_output_to_motion`)

## 8. shape 한눈에 보기 (배치 B, 클립 길이 T)

| 단계 | 텐서 | shape |
|---|---|---|
| BVH 로딩 | local_rot / root_pos | (F, 22, 3, 3) / (F, 3) |
| Dataset 클립 | root_pos, local_rot, contacts | (T, 3), (T, 22, 3, 3), (T, 4) |
| DataLoader 배치 | 〃 | (B, T, 3), (B, T, 22, 3, 3), (B, T, 4) |
| 6D | rot6d | (B, T, 22, 6) |
| 모델 입력 | x̃, x | (B, T, 135) |
| 복원 | root_pos, local_rot | (B, T, 3), (B, T, 22, 3, 3) |
| FK | global_pos | (B, T, 22, 3) |
