# configs/ — 설정 파일

데이터 경로와 GPU 설정은 **코드에 직접 쓰지 않고 이 폴더에서만** 관리한다.
코드는 `src/config.py` 를 통해서만 이 값을 읽는다.

```
configs/
├── env/            실행 환경마다 다른 값 (경로, device)
│   ├── laptop.yaml   Windows 노트북, CPU. LaFAN1 경로는 null (데이터 없음)
│   └── lab.yaml      연구실 Ubuntu, cuda:0. ← lafan1_bvh_dir 을 실제 경로로 수정해야 함
├── data/           환경과 무관한 데이터 설정
│   └── lafan1.yaml   context_len, window/offset, 분할(배우), 발 접촉 임계값 등
└── model/          모델·학습 설정 (2단계~)
    └── context.yaml  Context Transformer 크기, batch size, AMP, 학습률, 손실 가중치
```

## 환경 고르기

1. 스크립트 인자: `python scripts/xxx.py --env lab`
2. 또는 환경변수: `export TSI_ENV=lab` (Windows PowerShell: `$env:TSI_ENV="laptop"`)

둘 다 없으면 일부러 에러를 낸다 (노트북 설정으로 연구실에서 잘못 도는 일을 막기 위해).

## 값의 출처 표시

`data/lafan1.yaml` 의 각 값 옆에 출처를 적었다.
- `[논문 x.x]` 논문에 적힌 값
- `[공식코드]` 논문에 없어서 공식 코드 config 에서 가져온 값 (= 가정, `docs/assumptions.md` 참고)

- `[결정]` 이 프로젝트에서 정한 값 (`docs/assumptions.md` E 절)

## 모델 입력/출력 차원

`model/context.yaml` 에는 입력/출력 차원과 관절 수를 **적지 않는다**. `data/lafan1.yaml` 이 가리키는
스켈레톤 파일의 관절 수 J 에서 D = J×6+3 (=135), 입력 = D+1 (=136), 출력 = D 를 계산한다.

## 학습 설정 덮어쓰기

`scripts/train_context.py --batch-size 16 --grad-accum 2 --amp bf16 --set model.n_layer=4` 처럼
실행할 때 바꿀 수 있다. 실제로 쓴 설정은 `experiments/<run>/config.yaml` 에 저장된다.

## 앞으로 추가될 것

- `model/detail.yaml` (3단계)
