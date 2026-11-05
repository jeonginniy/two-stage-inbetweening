# Context Transformer 와 Detail Transformer 의 연결 (2·3단계 준비 노트)

논문 3.3, 3.4, 3.6.1, 3.6.2 와 공식 코드를 **읽고** 정리한 것. 구현은 2·3단계에서 한다.
"[공식코드]" 표시는 논문에 없고 공식 코드에서 확인한 내용.

## 1. 큰 그림

```
 정답 클립 (정렬됨)                       Context Transformer (1단계 생성)
 x_gt (B,T,135) ──[transition 구간 비우기]──▶ x_ctx ⊕ m_in ──▶ T_ctx ──▶ y_ctx
                                                                       │
                     후처리 ψ (식 23, 추론 때만): 양 끝 곡률 최소화 offset δ*
                     transition 이외 프레임은 정답으로 되돌림                │
                                                                       ▼
 x_det = x_gt 에서 transition 구간만 y_ctx 로 채움 ⊕ m_in ──▶ T_det ──▶ y_det (135) + 접촉 ĉ (4)
                                                      Detail Transformer (2단계 다듬기)
```

- **Context** 는 "빈칸 채우기": transition 프레임 값을 모른다 → 어텐션으로 아는 프레임만 본다.
- **Detail** 은 "다듬기": 모든 프레임에 값이 있다 → 모든 프레임이 서로 본다.
- 학습 순서: Context 를 먼저 학습 → **파라미터 고정(freeze)** → Detail 을 Context 의 출력으로 학습.

## 2. Context Transformer 입력 만들기 (논문 3.3)

| 프레임 | x_ctx 값 | m_in |
|---|---|---|
| context (0~9) | 정답 | 1 |
| transition (10 ~ t_tgt−1) | 루트 위치 = 마지막 context ↔ target 사이 **선형보간**, 회전 = 0 | 0 |
| 제약 프레임 (transition 중 일부, 선택) | 정답 (전체 또는 일부 관절) | 1 (부분 제약이면 차원별 m_in ∈ R^{T×D}) |
| target (t_tgt) | 정답 | 1 |
| target 이후 (윈도우 끝까지) | 사용 안 함 | 0 |

- [공식코드] 실제로는 정규화된 x 에 m_in 을 곱해서 모르는 값을 0 으로 만든 뒤, 루트 위치 차원만
  (정규화된 공간에서) 선형보간 값으로 다시 채운다. 입력 차원 = 135 + 1(마스크) = 136.
  → 즉 논문의 "회전 = 0" 은 **정규화된 값이 0** (= 데이터 평균 회전) 이라는 뜻으로 구현돼 있다.
  2단계에서 이 해석을 따를지 정해야 한다 (가정으로 기록할 것).
- 어텐션 마스크 (그림 5a): 모든 query 가 **context, 제약, target 프레임(key)만** 볼 수 있다.
- 위치 인코딩: Keyframe PE (식 11, 입력 = 각 프레임의 [마지막 context 로부터 거리, target 까지 거리])
  + Learned Relative PE (식 12–13, 어텐션 안에서).
- 출력 y_ctx (135) 에서 R(·), P(·) 로 회전/루트 위치 추출 (식 10) → 1단계의 `extract_rotation_6d`,
  `extract_root_position`, `decode_pose` 를 그대로 쓰면 된다.
- 손실 (식 21): L_rot + 0.01·L_pos + 0.005·L_smooth, transition 프레임에서만.
  [공식코드] 손실 전에 transition 이외 프레임은 정답으로 덮어쓴다.

## 3. 두 모델이 연결되는 지점 (논문 3.4, 3.6.2)

1. Context 출력에서 **transition 이외 프레임은 정답으로 되돌린다** (3.6.2).
2. 그러면 transition 양 끝에서 작은 "튐" 이 생길 수 있어 후처리 ψ 를 한다 (식 23):
   transition 전체에 같은 offset δ 를 더해, 마지막 context 프레임과 target 프레임에서의 곡률을 최소화.
   - [공식코드] 정확한 최적화 대신 근사: 양쪽 정답에서 선형 외삽한 값과 예측값의 차이를 평균해서 δ 로 씀.
   - target 쪽 외삽에 **target 다음 프레임**이 필요하다 → 이것이 window 가 42 (= 10+30+**2**) 인 이유.
   - [공식코드] Context 학습 config 는 window 41 (후처리 없이 학습), Detail 학습 config 는 42
     (Detail 학습 중 고정된 Context 를 후처리까지 포함해 추론하므로).
3. x_det 만들기 (3.4): 정답 클립에서 **transition 구간만** y_ctx 로 바꿔 넣는다.
   제약 프레임은 제약된 값은 정답, 나머지만 y_ctx.
4. m_in 은 Context 와 동일.

## 4. Detail Transformer (논문 3.4)

- 구조는 Context 와 비슷하지만 **Keyframe PE 없음** (이미 그럴듯한 동작이 있으므로 keyframe 과의 거리가 덜 중요).
- 어텐션 마스크 (그림 5b): 자리표시자 값이 없으므로 context·transition·target 이 모두 서로 본다.
  [공식코드] key 를 target 프레임까지만 허용 (target 이후 프레임은 가림).
- 출력: 135 + 발 접촉 4 = 139 차원 [공식코드 d_out=139]. 접촉 ĉ = sigmoid(C(y_det)) (식 15).
- 손실 (식 22): L_rot + 0.02·L_pos + 0.05·L_c + 0.05·L_foot.
  L_foot 는 ĉ 의 gradient 를 끊는다 (안 끊으면 ĉ 가 항상 0 으로 수렴).
- 발 접촉 정답 c 는 1단계 `LafanDataset` 이 이미 `contacts (B,T,4)` 로 준다.
- [공식코드] Detail 도 별도 통계 파일(train_stats_detail.pkl)을 쓰지만, 입력 상태 벡터는 Context 와 같은
  135차원 표현이다 (프레임 간 차이 delta 도 계산하지만 모델 입력에는 들어가지 않음).

## 5. 1단계 코드와의 연결점

| 2·3단계에서 필요한 것 | 1단계에서 준비된 것 |
|---|---|
| 정렬된 정답 클립, x_gt | `pipeline.motion_to_model_input(...).x`, `.root_pos`, `.local_rot` |
| R(·), P(·) | `representation.extract_rotation_6d`, `extract_root_position`, `decode_pose` |
| FK (L_pos, L_smooth, L_foot) | `skeleton.forward_kinematics` + `local_positions_from_root` (미분 가능) |
| 발 접촉 정답 | `LafanDataset` 의 `contacts` |
| L2Q 평가 (4단계) | `rotations.matrix_to_quat` |
| 결과 확인 | `pipeline.model_output_to_motion` → `bvh.write_bvh` |
