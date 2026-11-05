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

### 데이터 없이 동작만 확인 (노트북/이 환경)

```bash
python scripts/train_context.py --env laptop --fake-data --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
```
가짜 LaFAN1 형식 BVH 를 `experiments/<run>/fake_lafan1/` 에 만들고 통계도 그 자리에서 계산한다. **학습 결과는 의미가 없다.**
