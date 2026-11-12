"""
평가/시각화 공용: 학습된 Context Transformer 불러오기 → 평가셋 → 임의 길이 transition 생성 (4단계)
=========================================================================================

이 파일이 하는 일
-----------------
scripts/render_samples.py (영상) 와 scripts/eval_lengths.py (길이별 지표) 가 같이 쓰는 코드.
학습 코드(src/training/)는 바꾸지 않고 함수만 불러 쓴다. 체크포인트는 읽기만 한다.

쓰는 순서
---------
    run = load_run("experiments/ctx_base/best.pth")          # 1) 학습 때 저장된 config.yaml + 모델 + 정규화 통계
    ds, info = build_eval_dataset(run, eval_cfg, "long")     # 2) 평가셋 (official = 공식 벤치마크, long = 긴 길이용)
    batch = collate(ds, [0, 1, 2])                           # 3) 클립 몇 개를 배치로
    gen = generate(run, batch, trans=60)                     # 4) 자르기 → 정렬 → 정규화 → 모델 → (후처리) → FK

generate 는 학습 중 검증 (scripts/train_context.py 의 validate → src.training.context.eval_transition) 과
같은 순서를 따르고, 후처리 (논문 식 23) 를 켜고 끌 수 있게 한 것만 다르다.

transition 길이 제한
--------------------
모델 구조에는 최대 길이가 없다 (상대 위치 PE 식 12–13, Keyframe PE 식 11 모두 거리 "값" 을 MLP 에 넣음).
필요한 것은 클립이 context_len + trans + 2 프레임 이상인 것뿐이다 (+2 = target 과 그 다음 프레임; 후처리용).
학습 때 본 길이는 [min_trans, max_trans] = [5, 30] 이고, 그보다 긴 길이는 "학습 범위 밖" (외삽) 이다.

shape 표기: B = 배치, L = 모델 창 길이 (= ctx + trans + 2), J = 관절 수 (22), D = J*6+3 (135).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import default_collate

from src.config import load_env_config, load_yaml, require_dir, resolve_device, resolve_path
from src.data.dataset import LafanDataset
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.models.context_transformer import build_context_transformer
from src.training.context import PreparedBatch, global_positions, infer_context, prepare_batch
from src.training.context_inputs import TransitionSpec
from src.training.utils import load_checkpoint

FRAME_KEYS = ("root_pos", "local_rot", "contacts")   # 시간 축이 있는 배치 key (자를 때 사용)


# -----------------------------------------------------------------------------
# 1) 학습 결과 불러오기
# -----------------------------------------------------------------------------
@dataclass
class LoadedRun:
    """학습 결과 하나 (설정, 모델, 정규화 통계).

    cfg      : <run 폴더>/config.yaml 전체 (학습 때 실제로 쓴 값)
    data_cfg : cfg["data"] (context_len, min/max_trans, splits ...)
    env      : configs/env/<env>.yaml (경로, device)
    """

    cfg: dict
    data_cfg: dict
    env: dict
    device: torch.device
    skel: Skeleton
    layout: FeatureLayout
    norm: ZScoreNormalizer
    model: torch.nn.Module
    ckpt: dict
    ckpt_path: Path

    @property
    def ctx(self) -> int:
        """context 프레임 수 t_ctx (논문 4.1: 10)."""
        return int(self.data_cfg["context_len"])

    @property
    def train_range(self) -> tuple[int, int]:
        """학습 때 transition 길이 범위 [min_trans, max_trans] (논문 4.1: 최대 30)."""
        return int(self.data_cfg["min_trans"]), int(self.data_cfg["max_trans"])

    @property
    def name(self) -> str:
        """run 이름 = 체크포인트 폴더 이름 (예: ctx_base)."""
        return self.ckpt_path.parent.name


def load_run(ckpt: str | Path, config: str | Path | None = None, env_name: str | None = None,
             device: str | None = None) -> LoadedRun:
    """체크포인트 + 그 폴더의 config.yaml → LoadedRun.

    입력:
        ckpt     : 체크포인트 경로 (예: experiments/ctx_base/best.pth)
        config   : 설정 파일 경로. None 이면 <ckpt 폴더>/config.yaml
        env_name : configs/env/<env>.yaml 이름. None 이면 config.yaml 에 저장된 env
        device   : 예) "cpu", "cuda:0". None 이면 env 설정
    출력:  LoadedRun (model 은 eval 모드)
    """
    ckpt_path = resolve_path(ckpt)
    cfg_path = resolve_path(config) if config else ckpt_path.parent / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 {ckpt_path} 가 없습니다.")
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일 {cfg_path} 가 없습니다 (train_context.py 가 run 폴더에 저장).")
    cfg = load_yaml(cfg_path)
    if cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 결과입니다 (평가 의미 없음).")
    data_cfg = cfg["data"]
    env = load_env_config(env_name or cfg["env"])
    dev = torch.device(device) if device else resolve_device(env)

    skel = Skeleton.from_yaml(resolve_path(data_cfg["skeleton"]))
    layout = FeatureLayout(skel.num_joints)
    norm = ZScoreNormalizer.load(resolve_path(cfg["stats"])).to(device=dev, dtype=torch.float32)
    ck = load_checkpoint(ckpt_path, map_location=dev)
    model = build_context_transformer(cfg["model"], layout).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return LoadedRun(cfg, data_cfg, env, dev, skel, layout, norm, model, ck, ckpt_path)


# -----------------------------------------------------------------------------
# 2) 평가셋
# -----------------------------------------------------------------------------
def long_window(eval_cfg: dict, ctx: int) -> int:
    """긴 길이 평가셋의 window. "auto" 면 ctx + max(lengths) + 2 (예: 10 + 300 + 2 = 312)."""
    w = eval_cfg["eval_sets"]["long"]["window"]
    return ctx + max(eval_cfg["lengths"]) + 2 if w == "auto" else int(w)


def eval_split_cfg(run: LoadedRun, eval_cfg: dict, set_name: str) -> dict:
    """평가셋 이름 → {actors, window, offset, start_frame} (LafanDataset 인자).

    official : 학습 config 의 splits[<split>] 그대로 (공식 벤치마크: subject5, 65, 40, 0)
    long     : configs/eval/lengths.yaml 의 long (window auto = ctx + max(lengths) + 2)
    """
    sc = eval_cfg["eval_sets"][set_name]
    if "split" in sc:
        return dict(run.data_cfg["splits"][sc["split"]])
    return {"actors": list(sc["actors"]), "window": long_window(eval_cfg, run.ctx),
            "offset": int(sc["offset"]), "start_frame": int(sc["start_frame"])}


def build_eval_dataset(run: LoadedRun, eval_cfg: dict, set_name: str) -> tuple[LafanDataset, dict]:
    """평가셋 LafanDataset 과 그 설정 dict.

    BVH 파싱 결과는 data/cache/lafan1_parsed/ 캐시를 그대로 쓴다 (1단계에서 만든 것).
    출력:  (ds, split_cfg)   ds[i] = {"root_pos" (W,3), "local_rot" (W,J,3,3), "contacts" (W,4),
                                    "offsets" (J,3), "file_index", "start"}
    """
    sp = eval_split_cfg(run, eval_cfg, set_name)
    ds = LafanDataset(require_dir(run.env["paths"]["lafan1_bvh_dir"], "LaFAN1 BVH 폴더"), sp["actors"],
                      sp["window"], sp["offset"], run.skel, start_frame=sp["start_frame"],
                      contact_threshold=run.data_cfg["contact_vel_threshold"],
                      cache_dir=run.env["paths"]["cache_dir"], dtype=torch.float32, verbose=False)
    return ds, sp


def max_trans_for_window(window: int, ctx: int) -> int:
    """window 프레임 클립에서 계산할 수 있는 최대 transition 길이 (= window − ctx − 2)."""
    return window - ctx - 2


def collate(ds: LafanDataset, idx: Sequence[int]) -> dict:
    """클립 번호 목록 → 배치 dict (각 텐서 앞에 B 차원)."""
    return default_collate([ds[i] for i in idx])


# -----------------------------------------------------------------------------
# 3) 생성
# -----------------------------------------------------------------------------
@dataclass
class Generation:
    """배치 하나의 생성 결과. 좌표는 모두 학습용 정렬 좌표 (마지막 context 프레임 기준 루트 XZ=0, +X 방향).

    pb   : 정답 (PreparedBatch: x_gt (B,L,D) 정규화, x_gt_raw, gpos_gt (B,L,J,3), offsets (B,1,J,3))
    spec : TransitionSpec (context 0..ctx−1 | transition ctx..ctx+trans−1 | target ctx+trans)
    y    : (B, L, D) 예측 (정규화된 공간). transition 밖은 정답
    gpos : (B, L, J, 3) 예측 전역 관절 위치 (cm)
    """

    pb: PreparedBatch
    spec: TransitionSpec
    y: Tensor
    gpos: Tensor


def cut_batch(batch: dict, start: int, length: int) -> dict:
    """배치의 시간 축만 [start, start+length) 로 자른다 (offsets 등은 그대로)."""
    return {k: (v[:, start:start + length] if k in FRAME_KEYS else v) for k, v in batch.items()}


@torch.no_grad()
def generate(run: LoadedRun, batch: dict, trans: int, use_postprocess: bool = True) -> Generation:
    """LafanDataset 배치 → transition `trans` 프레임 생성 (논문 3.3, 후처리 식 23).

    순서 (학습 중 검증 eval_transition 과 같음):
      1) 클립 앞에서 L = ctx + trans + 2 프레임만 쓴다 (target 다음 프레임 = 후처리용; 모델은 key 마스크로 못 봄)
      2) prepare_batch: 정렬 (논문 3.6.1) → 6D+루트 135차원 → z-score
      3) infer_context: 입력 만들기 (빈칸 0, 루트 선형보간, m_in) → 모델 → transition 만 예측으로 → (후처리)
      4) 역정규화 → FK → 전역 관절 위치

    입력:
        batch : {"root_pos" (B,W,3), "local_rot" (B,W,J,3,3), "offsets" (B,J,3), ...}, W ≥ L
        trans : transition 길이 (1 이상, 학습 범위 밖이어도 됨)
    출력:  Generation
    """
    ctx = run.ctx
    L = ctx + trans + 2
    W = batch["root_pos"].shape[1]
    if W < L:
        raise ValueError(f"trans {trans}: 클립 길이 {W} < context {ctx} + trans + 2 = {L}")
    pb = prepare_batch(cut_batch(batch, 0, L), run.skel, run.norm, ctx,
                       run.data_cfg["forward_world_axis"], run.device)
    spec = TransitionSpec(ctx, ctx + trans)
    y = infer_context(run.model, pb.x_gt, spec, run.layout, use_postprocess=use_postprocess)
    gpos = global_positions(run.norm.denormalize(y), pb.offsets, run.skel, run.layout)
    return Generation(pb, spec, y, gpos)


@torch.no_grad()
def aligned_gt_with_before(run: LoadedRun, ds: LafanDataset, idx: Sequence[int], before: int,
                           length: int) -> PreparedBatch:
    """영상용: 클립 시작 `before` 프레임 전부터 `before + length` 프레임의 정답을, 모델 창과 "같은 정렬" 로.

    정렬 기준 프레임은 모델 창의 마지막 context 프레임 (= 늘린 시퀀스의 before + ctx − 1) 이라서
    늘린 시퀀스의 [before, before + length) 구간은 generate() 의 pb.gpos_gt 와 같은 좌표가 된다
    (오차 ≈ 1e-5 cm, scripts/render_samples.py 가 확인해서 출력).

    입력:  idx (B 개 클립 번호), before (앞에 붙일 프레임 수), length (모델 창 길이)
    출력:  PreparedBatch, gpos_gt (B, before + length, J, 3)
    """
    items = []
    for i in idx:
        fi, start = ds.locate(i)
        if start - before < 0:
            raise ValueError(f"클립 {i}: 앞에 {before} 프레임이 없습니다 (시작 프레임 {start}).")
        sl = slice(start - before, start + length)
        items.append({"root_pos": ds.root_pos[fi][sl], "local_rot": ds.local_rot[fi][sl],
                      "offsets": ds.offsets[fi]})
    ext = default_collate(items)
    return prepare_batch(ext, run.skel, run.norm, before + run.ctx, run.data_cfg["forward_world_axis"],
                         run.device)


def train_range_label(trans: int, train_range: tuple[int, int]) -> str:
    """결과표 표시: 학습 때 본 길이 범위 안이면 "학습 범위 안", 아니면 "학습 범위 밖"."""
    lo, hi = train_range
    return "학습 범위 안" if lo <= trans <= hi else "학습 범위 밖"


def describe(run: LoadedRun) -> dict[str, Any]:
    """로그용 요약 (체크포인트 iteration, best val, 학습 범위)."""
    return {"ckpt": str(run.ckpt_path), "iteration": int(run.ckpt["iteration"]),
            "best_val": float(run.ckpt["best_val"]), "context_len": run.ctx,
            "train_trans_range": list(run.train_range)}
