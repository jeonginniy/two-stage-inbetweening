"""
발 전용 모델 평가 라이브러리: 발 위치 L2 / 접촉 F1 (관절·동작별) / 발 미끄러짐
==============================================================================

docs/foot_only_eval_plan.md 의 평가를 계산하는 함수들. 실행은 foot_only/eval_foot.py.

사용 순서
    1) load_foot_run        experiments/foot_only/best.pth + config.yaml → 모델, 정규화 통계, 손실 가중치
    2) predict_feet         평가셋 배치 하나 → 정답 창 + 방법별 발 4관절 위치 (+ 접촉 확률)
    3) FootMetricAccumulator.add / .rows   클립별로 모았다가 전체·동작별 지표 표 (long 형식 행)

방법 (METHODS)
    foot_only   발 전용 모델 (후처리 없음)                     접촉 출력 있음
    ctx_post    Context Transformer + 후처리 (식 23)          README "w/ post-processing" 조건
    ctx_nopost  Context Transformer, 후처리 없음
    two_stage   Context (Detail 학습 설정대로 후처리) → Detail (Detail 출력 후처리 없음, G4)   접촉 출력 있음
    interp      SLERP 보간 (src/evaluation/baselines.py)
    gt          정답 발 위치 (검산용: 위치로 판정한 접촉이 라벨과 같아야 함)

모든 방법은 같은 정답 창 (inference.prepare_window, 클립 앞 ctx + trans + 2 프레임) 에서 발 4관절
[LeftFoot, LeftToe, RightFoot, RightToe] 의 정렬 좌표 위치 (B, L, 4, 3) 로 바꾼 뒤 같은 지표 함수를 쓴다.

지표 (transition 프레임 [ctx, ctx + trans) 만)
    foot_l2       발 위치 L2 오차 (cm), 정렬 좌표. 관절별 + 4관절 평균
    foot_skate    기존 metrics.foot_skating_per_clip (발끝 2개, 높이 H = 3 cm, cm/frame)
    contact (pos) 예측 위치 → 원래 월드 좌표 (foot_to_world) → extract_foot_contacts (0.2 cm/frame, H4) vs 라벨
    contact (out) 모델의 접촉 출력 sigmoid > 0.5 vs 라벨 (foot_only, two_stage 만)
    접촉은 precision / recall / F1 / acc 를 관절별 + 4관절 합산으로, 전체 클립과 동작 (파일 이름 접두어) 별로 낸다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from foot_only.data import FOOT_DIM, FOOT_JOINT_NAMES, N_FOOT, FootBatch, canonical_frame, foot_indices, foot_to_world
from foot_only.model import build_foot_model
from foot_only.train_utils import foot_forward
from src.config import load_yaml, resolve_path
from src.data.preprocess import extract_foot_contacts
from src.data.representation import FeatureLayout, ZScoreNormalizer
from src.evaluation.baselines import interpolation
from src.evaluation.inference import DetailRun, LoadedRun, cut_batch, prepare_window
from src.evaluation.metrics import foot_skating_per_clip, global_rot_pos
from src.training.context import run_model
from src.training.context_inputs import TransitionSpec, postprocess
from src.training.detail import run_detail
from src.training.utils import load_checkpoint

METHODS = ("foot_only", "ctx_post", "ctx_nopost", "two_stage", "interp")
METHOD_LABEL = {"foot_only": "Foot-only", "ctx_post": "Context (+post)", "ctx_nopost": "Context (no post)",
                "two_stage": "Two-stage (Context+Detail)", "interp": "Interpolation (SLERP)", "gt": "GT (check)"}


# -----------------------------------------------------------------------------
# 1) 체크포인트 불러오기
# -----------------------------------------------------------------------------
@dataclass
class FootRun:
    """학습된 발 전용 모델 하나.

    cfg       : <run 폴더>/config.yaml (train_foot.py 가 저장; data = 데이터셋, foot_data = 발 전용 data 절)
    model     : ContextTransformer (입력 148 → 출력 16), eval 모드
    ctx_norm  : 전신 입력 정규화 통계 (ctx_base 와 같은 파일)
    foot_norm : 발 위치 12 정규화 통계
    loss_w    : cfg["loss"] (foot_forward 가 손실을 계산할 때만 씀, 지표와 무관)
    """

    cfg: dict
    model: torch.nn.Module
    ctx_norm: ZScoreNormalizer
    foot_norm: ZScoreNormalizer
    loss_w: dict
    ckpt: dict
    ckpt_path: Path

    @property
    def name(self) -> str:
        return self.ckpt_path.parent.name


def load_foot_run(ckpt: str | Path, layout: FeatureLayout, device: torch.device) -> FootRun:
    """체크포인트 + 같은 폴더 config.yaml → FootRun.

    inference.load_run 은 ContextTransformer (입력 136) 를 만들기 때문에 발 전용 모델 (입력 148) 은 여기서 읽는다.
    """
    ckpt_path = resolve_path(ckpt)
    cfg_path = ckpt_path.parent / "config.yaml"
    for p in (ckpt_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 가 없습니다.")
    cfg = load_yaml(cfg_path)
    if cfg.get("fake_data"):
        raise ValueError(f"{cfg_path} 는 --fake-data 로 학습한 결과입니다 (평가 의미 없음).")
    ck = load_checkpoint(ckpt_path, map_location=device)
    model = build_foot_model(cfg["model"], layout).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ctx_norm = ZScoreNormalizer.load(resolve_path(cfg["stats"])).to(device=device, dtype=torch.float32)
    foot_norm = ZScoreNormalizer.load(resolve_path(cfg["foot_stats"])).to(device=device, dtype=torch.float32)
    return FootRun(cfg, model, ctx_norm, foot_norm, dict(cfg["loss"]), ck, ckpt_path)


def check_same_input_stats(foot: FootRun, ctx_run: LoadedRun) -> None:
    """발 전용 모델과 Context 의 입력 정규화 통계가 같은 값인지 (정답 창 pb 를 같이 쓰기 위한 조건)."""
    if not (torch.allclose(foot.ctx_norm.mean, ctx_run.norm.mean) and torch.allclose(foot.ctx_norm.std, ctx_run.norm.std)):
        raise ValueError(f"입력 통계가 다릅니다: {foot.cfg['stats']} vs {ctx_run.cfg['stats']}")


def motion_name(stem: str) -> str:
    """BVH 파일 이름 → 동작 이름 (앞쪽 영문자). 예) 'walk3_subject5' → 'walk', 'pushAndStumble1_subject5' → 'pushAndStumble'."""
    m = re.match(r"[A-Za-z]+", stem)
    return m.group(0) if m else stem


# -----------------------------------------------------------------------------
# 2) 배치 하나 → 방법별 발 위치
# -----------------------------------------------------------------------------
@dataclass
class FootPrediction:
    """배치 하나의 평가 재료. 좌표는 정렬 좌표 (cm), 프레임 수 L = ctx + trans + 2.

    spec    : TransitionSpec
    feet_gt : (B, L, 4, 3) 정답 발 위치
    labels  : (B, L, 4)    접촉 라벨 (데이터셋, 원래 월드 좌표 기준)
    frame   : 정렬 변환 (foot_to_world 용)
    preds   : {방법: (발 위치 (B, L, 4, 3), 접촉 확률 (B, L, 4) 또는 None)}. transition 밖은 정답
    """

    spec: TransitionSpec
    feet_gt: Tensor
    labels: Tensor
    frame: object
    preds: dict[str, tuple[Tensor, Tensor | None]]


@torch.no_grad()
def predict_feet(batch: dict, trans: int, ctx_run: LoadedRun, methods: Sequence[str],
                 foot: FootRun | None = None, det: DetailRun | None = None) -> FootPrediction:
    """평가셋 배치 → 방법별 발 4관절 위치.

    입력:  batch (LafanDataset 배치, 길이 W ≥ ctx + trans + 2), trans, ctx_run (ctx_base LoadedRun),
           methods (METHODS 의 부분집합), foot (foot_only 에 필요), det (two_stage 에 필요)
    전신 방법 계산은 scripts/eval_lengths.py eval_one_length 와 같다 (Context 는 한 번만 돌리고 후처리를 따로 적용).
    """
    pb, spec = prepare_window(ctx_run, batch, trans)
    L, dev = pb.x_gt.shape[1], pb.x_gt.device
    cut = cut_batch(batch, 0, L)
    idx = foot_indices(ctx_run.skel)
    feet_gt = pb.gpos_gt[:, :, idx]                                                    # (B, L, 4, 3)
    labels = cut["contacts"].to(device=dev, dtype=pb.x_gt.dtype)
    frame = canonical_frame(cut, ctx_run.skel, ctx_run.ctx, ctx_run.data_cfg["forward_world_axis"], dev)
    B = feet_gt.shape[0]

    def body_feet(x_raw: Tensor) -> Tensor:                                            # (B, L, D) 원값 → (B, L, 4, 3)
        return global_rot_pos(x_raw, pb.offsets, ctx_run.skel, ctx_run.layout)[1][:, :, idx]

    preds: dict[str, tuple[Tensor, Tensor | None]] = {}
    if "foot_only" in methods:
        foot_raw = feet_gt.reshape(B, L, FOOT_DIM)
        fb = FootBatch(pb=pb, foot_raw=foot_raw, foot=foot.foot_norm.normalize(foot_raw), contact=labels)
        _, fp, prob = foot_forward(foot.model, fb, spec, ctx_run.layout, foot.foot_norm, foot.loss_w)
        preds["foot_only"] = (fp.reshape(B, L, N_FOOT, 3), prob)
    if any(m in methods for m in ("ctx_post", "ctx_nopost", "two_stage")):
        y = run_model(ctx_run.model, pb.x_gt, spec, ctx_run.layout)                     # (B, L, D) 정규화
        y_post = postprocess(y, pb.x_gt, spec)                                           # 식 23
        if "ctx_nopost" in methods:
            preds["ctx_nopost"] = (body_feet(ctx_run.norm.denormalize(y)), None)
        if "ctx_post" in methods:
            preds["ctx_post"] = (body_feet(ctx_run.norm.denormalize(y_post)), None)
        if "two_stage" in methods:
            y_det, prob = run_detail(det.model, y_post if det.ctx_postprocess else y, pb.x_gt, spec)
            preds["two_stage"] = (body_feet(ctx_run.norm.denormalize(y_det)), prob)     # prob: contact_joints 순서
    if "interp" in methods:
        preds["interp"] = (body_feet(interpolation(pb.x_gt_raw, spec, ctx_run.layout)), None)
    preds["gt"] = (feet_gt, None)
    return FootPrediction(spec, feet_gt, labels, frame, preds)


def position_contacts(feet: Tensor, frame, threshold: float) -> Tensor:
    """정렬 좌표 발 위치 (B, L, 4, 3) → 위치로 판정한 접촉 (B, L, 4) 0/1.

    원래 월드 좌표로 되돌린 뒤 데이터셋 라벨과 같은 규칙 (|dx|+|dy|+|dz| (t→t+1) < threshold) [H4].
    마지막 transition 프레임 ctx+trans−1 은 target 프레임 (정답) 과의 차이로 판정된다.
    """
    B, L = feet.shape[:2]
    world = foot_to_world(feet.reshape(B, L, FOOT_DIM), frame)                         # (B, L, 4, 3)
    return extract_foot_contacts(world, range(N_FOOT), threshold)


# -----------------------------------------------------------------------------
# 3) 지표 모으기
# -----------------------------------------------------------------------------
class FootMetricAccumulator:
    """클립별 값을 모았다가 전체·동작별 지표를 낸다.

    생성 인자:
        toe_idx    : 발 미끄러짐에 쓸 관절의 "발 4관절 배열 안" 번호 (예: LeftToe, RightToe → [1, 3])
        height_cm  : 발 미끄러짐 높이 임계값 H (lengths.yaml foot_skating)
        threshold  : 접촉 속도 임계값 (0.2 cm/frame, 데이터셋 라벨과 같은 값)
    """

    def __init__(self, toe_idx: Sequence[int], height_cm: float, threshold: float):
        self.toe_idx, self.height_cm, self.threshold = list(toe_idx), float(height_cm), float(threshold)
        self.motions: list[str] = []
        self.l2: dict[str, list[np.ndarray]] = defaultdict(list)         # 방법 → (B, 4) 클립별 관절 평균 L2
        self.skate: dict[str, list[np.ndarray]] = defaultdict(list)      # 방법 → (B,)
        self.conf: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)   # (방법, pos|out) → (B, 4, 4) tp fp fn tn

    @staticmethod
    def _confusion(pred: Tensor, lab: Tensor) -> np.ndarray:
        """(B, n, 4) bool 둘 → (B, 4 관절, 4) [tp, fp, fn, tn] 개수."""
        parts = [pred & lab, pred & ~lab, ~pred & lab, ~pred & ~lab]
        return torch.stack([p.sum(dim=1) for p in parts], dim=-1).cpu().numpy().astype(np.int64)

    def add(self, motions: Sequence[str], fp: FootPrediction) -> None:
        """배치 하나 추가. motions: 클립마다 동작 이름 (B 개)."""
        s = fp.spec.seq_slice
        lab = fp.labels[:, s] > 0.5                                                     # (B, n, 4)
        self.motions.extend(motions)
        for m, (feet, prob) in fp.preds.items():
            self.l2[m].append((feet[:, s] - fp.feet_gt[:, s]).norm(dim=-1).mean(dim=1).cpu().numpy())
            self.skate[m].append(foot_skating_per_clip(feet, self.toe_idx, self.height_cm, s).cpu().numpy())
            pc = position_contacts(feet, fp.frame, self.threshold)[:, s] > 0.5
            self.conf[(m, "pos")].append(self._confusion(pc, lab))
            if prob is not None:
                self.conf[(m, "out")].append(self._confusion(prob[:, s] > 0.5, lab))

    def rows(self) -> list[dict]:
        """long 형식 행: {method, source, motion, joint, metric, value}.

        motion = "all" 또는 동작 이름, joint = "all" (4관절 평균·합산) 또는 관절 이름.
        metric: foot_l2, foot_skate, n_clips (source = "-") / precision, recall, f1, acc, n_pos, n_frames (source = pos|out)
        F1 등은 개수를 먼저 합친 뒤 계산 (micro). 4관절 "all" 도 개수 합산.
        """
        motions = np.array(self.motions)
        groups = [("all", np.ones(len(motions), bool))] + [(g, motions == g) for g in sorted(set(self.motions))]
        out = []
        for method in self.l2:
            l2 = np.concatenate(self.l2[method])                                        # (N, 4)
            sk = np.concatenate(self.skate[method])                                     # (N,)
            for g, sel in groups:
                base = dict(method=method, source="-", motion=g)
                out.append(dict(base, joint="all", metric="n_clips", value=int(sel.sum())))
                out.append(dict(base, joint="all", metric="foot_l2", value=float(l2[sel].mean())))
                for j, name in enumerate(FOOT_JOINT_NAMES):
                    out.append(dict(base, joint=name, metric="foot_l2", value=float(l2[sel, j].mean())))
                out.append(dict(base, joint="all", metric="foot_skate", value=float(sk[sel].mean())))
            for src in ("pos", "out"):
                if (method, src) not in self.conf:
                    continue
                cf = np.concatenate(self.conf[(method, src)])                           # (N, 4, 4)
                for g, sel in groups:
                    c = cf[sel].sum(axis=0)                                             # (4 관절, 4)
                    for name, cnt in [(n, c[j]) for j, n in enumerate(FOOT_JOINT_NAMES)] + [("all", c.sum(axis=0))]:
                        tp, fpos, fn, tn = (float(v) for v in cnt)
                        n = tp + fpos + fn + tn
                        vals = {"precision": tp / (tp + fpos) if tp + fpos else float("nan"),
                                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                                "f1": 2 * tp / (2 * tp + fpos + fn) if tp + fn else float("nan"),
                                "acc": (tp + tn) / n if n else float("nan"),
                                "n_pos": int(tp + fn), "n_frames": int(n)}
                        for k, v in vals.items():
                            out.append(dict(method=method, source=src, motion=g, joint=name, metric=k, value=v))
        return out
