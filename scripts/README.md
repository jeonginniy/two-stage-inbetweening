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
| `train_context.py` | 연구실 🔬 / 노트북 (`--fake-data`) | **2단계** Context Transformer 학습 (논문 3.3, 3.5, 3.6). 체크포인트·로그를 `experiments/<run>/` 에 저장 |
| `train_detail.py` | 연구실 🔬 / 노트북 (`--fake-data`) | **3단계** Detail Transformer 학습 (논문 3.4, 3.5, 3.6). 고정된 Context (`experiments/ctx_base/best.pth`, 읽기만) 위에서 학습. 결과는 `experiments/<run>/` |
| `render_compare.py` | 연구실 🔬 | 학습된 모델로 val 클립 transition 생성 → 원본(파랑)/생성(빨강)/SLERP(회색, `--slerp`) 겹친 30fps mp4. 클립별 gpos_l2 와 context 일치 여부 출력 |
| `export_bvh.py` | 연구실 🔬 | 같은 클립을 원본/생성/SLERP BVH 로 저장 (LaFAN1 계층·offset, 월드 좌표 cm). 재읽기 FK 검증 출력 |
| `render_samples.py` | 연구실 🔬 | **4단계** 샘플별 mp4: GT \| PRED, 구간 색 (BEFORE 회색 / CONTEXT 파랑 / GENERATED 주황 / TARGET 빨강), 프레임 번호·구간 이름·범례, 생성 구간 동안 target 포즈 반투명. → `outputs/videos/trans_<길이>/<시퀀스>_<시작프레임>.mp4` |
| `eval_lengths.py` | 연구실 🔬 | **4단계** transition 길이별 L2P / L2Q / NPSS / 발 미끄러짐 (모델 ± 후처리, SLERP, zero-velocity), 공식 README 수치와 비교 → `outputs/eval/<run>/metrics_by_length.csv`, `.png`. `--render` 로 길이별 샘플 영상 |
| `blender_render.py` | 연구실 (Blender) | `blender --background --python ... -- <클립폴더>` 로 세 BVH 를 EEVEE mp4 렌더 |

🔬 = 실제 LaFAN1 이 필요. 노트북에서 실행하면 "경로가 null" 메시지를 내고 종료한다 (정상).

## 1단계: 연구실 검증 순서 — ✅ 완료 (결과는 `docs/assumptions.md` 0 절)

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

---

## 2단계: Context Transformer 학습 (연구실)

### 0. 준비 (한 번만)

```bash
cd two_stage_inbetweening
source .venv/bin/activate
git pull                                    # 2단계 코드 받기
export TSI_ENV=lab
pytest -q                                   # 65 passed, 4 skipped 이어야 함 (lab 테스트는 -m lab 로 따로)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, RTX 3070
ls data/cache/stats/lafan1_context_stats.npz   # 1단계 4번에서 만든 통계. 없으면 python scripts/compute_stats.py
```

### 1. 짧게 돌려 보기 (300 iteration)

설정·경로·GPU 가 맞는지, 손실이 줄어드는지, VRAM 을 얼마나 쓰는지 본다.

```bash
python scripts/train_context.py --run-name ctx_smoke --max-iters 300 \
    --set train.log_interval=50 train.val_interval=300
```

확인할 것:
- 첫 줄: `device=cuda:0, 관절 22, D=135, 입력 136 → 출력 135, 파라미터 10,389,836`
- `train 클립 404195 (window 42)` (1단계에서 확인한 논문 수치와 같아야 함)
- 로그 줄 끝의 `VRAM 최대 …MB` 와 `…ms/it`
- Subject 1–4 는 1단계 `compute_stats.py` 가 만든 BVH 파싱 캐시(`data/cache/lafan1_parsed/`)를 그대로 쓴다.
  val 용 Subject 5 는 처음 한 번 파싱하느라 시작이 조금 늦을 수 있다.

### 2. 본 학습

```bash
nohup python scripts/train_context.py --run-name ctx_base > ctx_base.out 2>&1 &
tail -f ctx_base.out
```

- 결과 폴더 `experiments/ctx_base/`
  - `config.yaml` : 이번 실행에 실제로 쓴 설정 전체 (덮어쓴 값 포함)
  - `log.csv`     : `iteration, epoch, kind(train/val), name, value` — 엑셀/pandas 로 그래프
  - `last.pth`    : `ckpt_interval`(5000) 마다 저장, `best.pth` : val trans 30 손실이 가장 낮을 때
- 중단 후 이어서: `python scripts/train_context.py --run-name ctx_base --resume last`
  (모델·Adam 상태·iteration 을 복원. 학습률도 iteration 에 맞게 이어짐)
- 길이: `train.max_iters` (기본 500,000). 논문에 학습 길이가 없어서 정한 값이다 (assumptions E15).
  val 손실이 더 이상 줄지 않으면 Ctrl+C 로 멈춰도 `best.pth` 가 남아 있다.

### 3. RTX 3070 (8GB) 에서 batch size / 메모리 조절

모든 값은 `configs/model/context.yaml` 의 `train:` 절에 있고, 실행할 때 덮어쓸 수 있다
(설정 파일을 고치지 않고 실험할 수 있게).

| 하고 싶은 것 | 방법 |
|---|---|
| batch 바꾸기 | `--batch-size 64` (= `--set train.batch_size=64`) |
| 메모리 부족 (CUDA out of memory) | batch 를 줄이고 같은 비율로 누적: `--batch-size 16 --grad-accum 2` → 유효 배치 32 유지 |
| 더 빠르게 / 메모리 절약 | `--amp bf16` (Ampere 지원, GradScaler 불필요) 또는 `--amp fp16`. 모델 부분만 혼합 정밀도, FK·손실은 float32 |
| DataLoader 가 병목 (GPU 사용률 낮음) | `--num-workers 8` (기본은 `configs/env/lab.yaml` 의 4) |
| 모델 크기 바꾸기 | `--set model.n_layer=4 model.d_model=256` |
| 학습률 | `--set train.base_lr=0.03 train.warmup_iters=4000` |
| 다른 설정 파일 | `configs/model/context.yaml` 을 복사해 고치고 `--config configs/model/my_ctx.yaml` |

참고:
- 기본 모델은 약 1,040만 파라미터, 클립 길이 42 라서 batch 32 는 8GB 에 크게 여유가 있을 것으로 예상하지만
  **아직 GPU 에서 재 보지 않았다.** 1번의 `VRAM 최대` 값을 보고 정한다. 여유가 크면 batch 를 키워도 되지만,
  Noam 학습률은 batch 와 무관하게 정해지므로 batch 를 크게 바꾸면 `base_lr` 도 다시 볼 것.
- `VRAM 최대` 는 PyTorch 가 텐서에 할당한 최대치다. `nvidia-smi` 에는 CUDA 컨텍스트 등 몇백 MB 가 더 보인다.

### 4. 연구실에서 봐 줄 결과 (3단계 전에 공유)

1. 1번 출력의 `VRAM 최대`, `ms/it`
2. 본 학습 `log.csv` 의 train `loss/rot/pos/smooth` 와 val `val5/15/30_loss`, `val*_gpos_l2` 추이
3. `best.pth` 의 iteration 과 그때의 `val30_gpos_l2` (cm)
   (공식 L2P/L2Q/NPSS 비교는 4단계에서 구현)

### 5. 결과 영상 (원본 vs 생성, 발표용)

```bash
pip install matplotlib                      # 처음 한 번 (ffmpeg 도 필요: sudo apt install ffmpeg)
python scripts/render_compare.py --slerp                  # best.pth, trans 30, 3개, 10초 → experiments/ctx_base/videos/
python scripts/render_compare.py --trans 15 --n 5 --seed 1 --slerp --layout overlay
python scripts/render_compare.py --slerp --total_frames 450 --offset 2.0
```

- 영상: `--total_frames` (기본 300 = 10초, 30fps). `원본 재생 | context (입력) | transition (생성) | target | 원본 재생`.
  transition 이 영상 가운데 오도록 앞뒤를 같은 BVH 의 원본으로 채운다 (여유가 없는 클립은 고르지 않음).
  앞뒤 프레임도 모델 창과 같은 정렬 (마지막 context 프레임 기준, `prepare_batch` 재사용) 이라 좌표가 이어진다.
- `--layout side` (기본): 원본(파랑) 가운데, 생성(빨강) +`--offset` m, SLERP(회색) −offset (진행 방향 +X 에 수직인 Z 로).
  `--layout overlay`: 겹쳐 그림. 아래 패널은 프레임별 위치 오차 (cm) 와 현재 프레임 커서.
- 설정은 `experiments/ctx_base/config.yaml` (`--ckpt` 폴더의 config.yaml) 에서 읽는다.
- 생성은 `validate()` 와 같은 함수 (`src.training.context.eval_transition`, `gpos_l2`) 를 쓴다.
  출력되는 클립별 `gpos_l2` 는 `log.csv` 의 `val{trans}_gpos_l2` 와 같은 식 (클립 몇 개 평균이라 값은 다를 수 있음).
- `context 구간 ... 최대 오차`, `앞뒤 원본 정렬 일치 ... 최대 오차` 가 ≈1e-5 cm 이하가 아니면 버그.
- 연구실 RTX 3070 에서 영상 1개 (300프레임) 약 18초.

### 6. BVH 저장 + Blender 렌더

```bash
python scripts/export_bvh.py                    # → experiments/ctx_base/bvh/<파일>_f<프레임>_trans30/{gt,pred,slerp}.bvh, meta.json
blender --background --python scripts/blender_render.py -- experiments/ctx_base/bvh/<클립폴더>          # → render.mp4
blender --background --python scripts/blender_render.py -- <클립폴더> --still 150   # 150번 프레임만 PNG (빠른 확인)
```

- 클립 선택·생성은 `render_compare.py` 의 `load_run / select_clips / infer_clips` 를 그대로 쓴다
  (`--seed`, `--trans`, `--total_frames` 가 같으면 같은 클립).
- 계층·OFFSET·End Site·회전 순서 (zyx) 는 LaFAN1 원본 BVH 에서 가져온다. 학습용 정렬을 되돌려 원본 월드 좌표 (cm) 로 저장.
  `--side_offset 150` (cm): 생성 +, SLERP − 를 진행 방향에 수직으로 옮김 (0 이면 원래 자리, 겹쳐 보기).
- 저장 파일은 루트 6채널 + 나머지 3채널 (원본은 모든 관절 6채널. 비루트 위치 채널 = OFFSET).
- 검증: `gt.bvh vs LaFAN1 원본` (재읽기 FK vs 원본 파일 같은 프레임), `pred/slerp.bvh vs 모델 출력` 이 ~1e-4 cm 이하.
- Blender 4.0 확인. 연구실 PC 에서 클립 하나 (300 프레임, 1280x720, EEVEE 16 샘플) 약 20초.
  BVH 는 scale 0.01 로 import, 뼈는 원기둥/구 메시 (Armature modifier). 라벨은 Noto Sans CJK (없으면 영어).

---

## 3단계: Detail Transformer 학습 (연구실, 브랜치 `detail-transformer`)

고정된 Context Transformer (설정 `configs/model/detail.yaml` 의 `context.ckpt`, 기본 `experiments/ctx_base/best.pth`) 를
읽기만 하고 그 위에서 Detail 만 학습한다. 정규화 통계는 Context 폴더의 `config.yaml` 에 적힌 파일을 그대로 쓴다 (G8, G14).
결과 폴더가 Context 폴더와 같거나 이미 `last.pth` 가 있으면 시작하지 않는다 (덮어쓰기 방지).

```bash
export TSI_ENV=lab
# 1) 짧게 돌려 보기
python scripts/train_detail.py --run-name det_smoke --max-iters 300 \
    --set train.log_interval=50 train.val_interval=300 train.ckpt_interval=300
# 2) 본 학습 (백그라운드)
nohup python scripts/train_detail.py --run-name det_base > det_base.out 2>&1 &
# 중단 후 이어서
python scripts/train_detail.py --run-name det_base --resume last
```

출력 읽는 법:
- train: `loss` (식 22 가중합), `rot`, `pos` (cm), `contact` (L_c), `foot` (L_foot)
- val (trans 5/15/30): `val{t}_loss`, `val{t}_gpos_l2` (두 단계 결과, cm), `val{t}_ctx_gpos_l2` (같은 클립의 Context 만 결과,
  비교 기준), `val{t}_contact_acc`. best = `val30_loss` 최소.
- Detail 은 잔차가 아니라 포즈 전체를 출력하므로 (G13) 학습 초기에는 `gpos_l2` 가 `ctx_gpos_l2` 보다 훨씬 크다. 정상이다.

300 iteration 실측 (RTX 3070, batch 32, fp32): **41 ms/it**, VRAM 최대 **384 MB**,
train loss 0.637 → 0.472 (50 → 300 iteration 평균), val30 위치 오차 45.6 cm (Context 6.4 cm).

## 4단계: 결과 영상 — 구간별 색 구분 (교수님 미팅 피드백 [1][2])

```bash
python scripts/render_samples.py                           # video.yaml 의 seed 0 으로 5개, trans 30 → outputs/videos/trans_30/
python scripts/render_samples.py --trans 90                # 같은 5개 샘플, 생성 구간 3초 → outputs/videos/trans_90/
python scripts/render_samples.py --trans 90 --postprocess off   # 후처리 끔 → outputs/videos/trans_90_nopost/
python scripts/render_samples.py --n 10 --seed 1           # 샘플 수/seed 덮어쓰기
python scripts/render_samples.py --index 1814              # 클립 직접 지정 (walk1_subject5, 프레임 1200)
python scripts/render_samples.py --trans 30 --stage detail # 두 단계 (Context → Detail), 같은 샘플 → outputs/videos/trans_30_detail/
```

`--stage detail` 은 `trans_<길이>/` (Context 만) 와 같은 클립을 그려서 나란히 비교할 수 있다. 콘솔과 `samples.csv` 에
같은 클립의 Context 단계 오차 (`ctx_gpos_l2`) 도 함께 남는다.

- 설정: `configs/viz/video.yaml` (색, BEFORE 30 / TARGET 유지 30 프레임, 배치, 출력 경로),
  `configs/eval/lengths.yaml` (체크포인트, 평가셋). 코드에 값을 쓰지 않는다.
- 영상 = `[BEFORE 원본 30] [CONTEXT 10] [GENERATED trans] [TARGET 1 + 유지 29]`, **모두 1배속 30fps**.
  GT 와 PRED 모두 구간 색을 따르고, GENERATED 동안 target 포즈를 반투명 빨강으로 겹쳐 그린다.
- 배치 `layout: auto`: 같은 축에 나란히 (PRED 를 진행 방향에 수직으로 150 cm) 그렸을 때 두 캐릭터가 20 cm 보다
  가까워지는 프레임이 있으면 좌우 두 패널 (축 범위 같음) 로 바꾼다. `side` / `panels` 로 고정할 수도 있다.
- 클립은 long 평가셋 (Subject 5, window 312, offset 40) 에서 seed 로 고른다. 이 평가셋의 클립 목록은 길이와 무관해서
  **`--trans` 를 바꿔도 같은 시퀀스·같은 시작 프레임** 이 뽑힌다 (seed 0: aiming2 6640, obstacles1 4040,
  pushAndStumble1 5480, run1 4800, walk3 5360). 후보는 앞에 BEFORE 30 프레임이 있는 클립.
- 폴더마다 `samples.csv`: 시퀀스, 시작 프레임, 클립 번호, 학습 범위 안/밖, 후처리, transition 평균 오차 (cm),
  경계 점프 `jump_start_cm` (마지막 context → 첫 생성), `jump_end_cm` (마지막 생성 → target), 배치, 영상 경로.
  경계 점프 = 생성 쪽 프레임 간 이동 − 정답 쪽 프레임 간 이동 (관절 평균). ≈0 이면 자연스럽게 이어짐.
- `postprocess` (식 23 후처리, 기본 켬): 학습 범위 밖에서는 모델이 target 에 못 미친 거리의 절반이 시작·끝 점프로
  나타난다 (assumptions F6). `--postprocess off` 로 모델 원래 출력을 볼 수 있다.
- 검증 출력: `context·target 프레임 PRED vs GT`, `BEFORE 붙인 정답 vs 모델 창 정답` 이 ≈1e-5 cm 이하가 아니면 버그.

## 4단계: 생성 구간 길이별 평가 (피드백 [3])

```bash
python scripts/eval_lengths.py --sets official --methods interp zerov   # 1. 기준선만: 지표 구현이 README 와 맞는지 먼저
python scripts/eval_lengths.py                                         # 2. 전체 (평가셋 2개 × 길이 8개 × 방법 4개)
python scripts/eval_lengths.py --render                                # + 길이별 같은 샘플 영상 (outputs/videos/trans_*/)
python scripts/eval_lengths.py --lengths 30 600 --sets long            # 설정보다 긴 길이 (long window 자동 확장)
```

- 길이 목록, 평가셋, 방법, 발 미끄러짐 H, README 비교 수치는 모두 `configs/eval/lengths.yaml`.
- 평가셋
  - `official`: 공식 벤치마크 (Subject 5, window 65, offset 40, 2232 클립). window 에 들어가는 trans ≤ 53 만 (5/15/30/45).
  - `long`: Subject 5, window 312 (= 10 + 300 + 2), offset 40, 2151 클립. **모든 길이에서 같은 시작 프레임** → 길이별 비교.
- 방법: `model` (+후처리 식 23), `model_nopost`, `interp` (SLERP), `zerov`,
  `detail` (두 단계: Context + 후처리 → Detail, Detail 출력 후처리 없음 = README Detail 수치와 같은 조건), `detail_post` (Detail 출력에도 후처리).
  `detail*` 는 `lengths.yaml` 의 `detail_ckpt` (기본 `experiments/det_base/best.pth`) 가 있을 때만 계산하고, 없으면 건너뛴다.
  Detail 학습 때 고정한 Context 가 `ckpt` 와 다르면 시작하지 않는다. Detail 을 넣으면 결과 폴더는 `outputs/eval/<det run>/`.
  README Detail 비교: `python scripts/eval_lengths.py --sets official --lengths 5 15 30 45 --methods model detail`
- 결과 (`outputs/eval/<run>/`)
  - `metrics_by_length.csv`: set, n_clips, trans, seconds, `train_range` (학습 범위 안/밖, 학습 범위 [5, 30]),
    method, L2P, L2Q, NPSS, foot_skate (cm/frame), foot_skate_gt, readme_L2P/L2Q/NPSS (official 만), batch_size
  - `metrics_by_length.png`: long 평가셋 길이 vs 지표 (x 로그, 학습 범위 음영, README 수치 ×)
  - `run_info.json`: 체크포인트, 설정, 시각
- L2P 통계 (Subject 1–4, 평균 중심 정렬) 는 처음 한 번 계산해서 `data/cache/stats/bench_stats_rmi_*.npz` 에 캐시.
- GPU 메모리가 부족하면 그 길이를 batch 절반으로 다시 계산하고 `← 줄임` 표시, CSV `batch_size` 에 실제 값.
- 지표 정의와 가정: `docs/assumptions.md` F1–F4, F8, F9.

### 데이터 없이 동작만 확인 (노트북/이 환경)

```bash
python scripts/train_context.py --env laptop --fake-data --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
```
가짜 LaFAN1 형식 BVH 를 `experiments/<run>/fake_lafan1/` 에 만들고 통계도 그 자리에서 계산한다. **학습 결과는 의미가 없다.**
`scripts/train_detail.py` 도 같은 옵션으로 돌린다 (고정할 Context 는 무작위 초기화한 같은 크기 모델).
