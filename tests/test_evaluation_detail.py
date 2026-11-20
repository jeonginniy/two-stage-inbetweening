"""
4단계 평가 코드의 두 단계 (Context → Detail) 부분 테스트 — 가짜 데이터 (LaFAN1·학습 결과 불필요)
=========================================================================================

tmp 폴더에 "가짜 학습 결과" 두 개를 만든다 (실제 experiments/ 는 건드리지 않음):
    ctx_run/  config.yaml + best.pth   (무작위 초기화 작은 Context Transformer)
    det_run/  config.yaml + best.pth   (무작위 초기화 작은 Detail Transformer, context_run.ckpt = ctx_run/best.pth)
config.yaml 형식은 scripts/train_context.py, scripts/train_detail.py 가 저장하는 것과 같은 key 만 쓴다.

확인하는 것:
- load_detail: 고정했던 Context 가 같으면 불러오고, 다르면 / --fake-data 결과면 거부
- generate(detail=...): 두 단계 결과 = infer_two_stage 직접 호출 결과, context·target 프레임은 정답,
  y_ctx / contact 가 채워짐 (Context 만이면 None)
- eval_lengths.eval_one_length 의 detail 방법 지표 = generate(detail) 출력으로 직접 계산한 지표 (같은 경로인지)
- detail_post 는 Detail 출력에 식 (23) 후처리를 한 것
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import load_data_config
from src.data.dataset import LafanDataset
from src.data.pipeline import motion_to_model_input
from src.data.representation import FeatureLayout, RunningStats, ZScoreNormalizer
from src.data.skeleton import Skeleton
from src.evaluation.inference import collate, generate, load_detail, load_run, prepare_window
from src.evaluation.metrics import MetricAccumulator, global_quats, global_rot_pos
from src.models.context_transformer import build_context_transformer
from src.models.detail_transformer import build_detail_transformer
from src.training.context_inputs import postprocess
from src.training.detail import infer_two_stage
from src.training.utils import save_checkpoint
from tests.helpers import write_fake_lafan_bvh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TINY = dict(d_mask=1, n_layer=1, n_head=2, d_model=32, d_head=8, d_ff=32,
            d_encoder_h=32, d_decoder_h=32, dropout=0.0, attn_bias=False, attn_scale="d_head")
WINDOW = 50                                                      # 10 + trans (≤ 38) + 2


def _save_run(folder: Path, model: torch.nn.Module, cfg: dict) -> Path:
    """config.yaml + best.pth 저장 → best.pth 경로."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    opt = torch.optim.Adam(model.parameters())
    save_checkpoint(folder / "best.pth", model, opt, None, 10, 0, 1.0, cfg)
    return folder / "best.pth"


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    """가짜 BVH (subject5) + 통계 + 가짜 Context/Detail 학습 결과 → (run, det, ds, tmp)."""
    tmp = tmp_path_factory.mktemp("eval_detail")
    bvh = tmp / "bvh"
    bvh.mkdir()
    for i, stem in enumerate(["walk1_subject5", "run1_subject5"]):
        write_fake_lafan_bvh(bvh / f"{stem}.bvh", 120, seed=i)
    data_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in load_data_config().items()}
    skel = Skeleton.from_yaml(PROJECT_ROOT / data_cfg["skeleton"])
    layout = FeatureLayout(skel.num_joints)
    ds = LafanDataset(bvh, ["subject5"], WINDOW, 20, skel, contact_threshold=data_cfg["contact_vel_threshold"],
                      verbose=False)
    stats = RunningStats(layout.dim)
    for b in DataLoader(ds, batch_size=16):
        stats.update(motion_to_model_input(b["root_pos"], b["local_rot"], data_cfg["context_len"],
                                           skel.root_forward_local_axis).x_raw)
    stats_path = tmp / "stats.npz"
    ZScoreNormalizer.from_stats(stats).save(stats_path)

    torch.manual_seed(0)
    ctx_cfg = {"env": "laptop", "data": data_cfg, "model": dict(TINY), "fake_data": False, "stats": str(stats_path)}
    ctx_ckpt = _save_run(tmp / "ctx_run", build_context_transformer(TINY, layout), ctx_cfg)
    det_cfg = {"env": "laptop", "data": data_cfg, "context": {"ckpt": str(ctx_ckpt), "postprocess": True},
               "model": dict(TINY), "context_run": {"ckpt": str(ctx_ckpt), "iteration": 10, "stats": str(stats_path)},
               "fake_data": False, "stats": str(stats_path)}
    _save_run(tmp / "det_run", build_detail_transformer(TINY, layout, len(skel.contact_joints)), det_cfg)

    run = load_run(ctx_ckpt, env_name="laptop", device="cpu")
    det = load_detail(tmp / "det_run" / "best.pth", run)
    return run, det, ds, tmp, det_cfg


def test_load_detail_checks_context(fake_runs):
    run, det, ds, tmp, det_cfg = fake_runs
    assert det.ctx_postprocess is True and det.name == "det_run"
    assert not det.model.training
    # 다른 Context 위에서 학습된 Detail → 거부
    other = dict(det_cfg, context_run=dict(det_cfg["context_run"], ckpt=str(tmp / "other_ctx" / "best.pth")))
    _save_run(tmp / "det_other", det.model, other)
    with pytest.raises(ValueError, match="Context"):
        load_detail(tmp / "det_other" / "best.pth", run)
    # --fake-data 결과 → 거부
    _save_run(tmp / "det_fake", det.model, dict(det_cfg, fake_data=True))
    with pytest.raises(ValueError, match="fake"):
        load_detail(tmp / "det_fake" / "best.pth", run)
    with pytest.raises(FileNotFoundError):
        load_detail(tmp / "none" / "best.pth", run)


def test_generate_two_stage(fake_runs):
    run, det, ds, _, _ = fake_runs
    batch = collate(ds, [0, 1, 2])
    trans = 20
    gen = generate(run, batch, trans, detail=det)
    pb, spec = prepare_window(run, batch, trans)
    y_ctx, y_det, prob = infer_two_stage(run.model, det.model, pb.x_gt, spec, run.layout, ctx_postprocess=True)
    assert torch.allclose(gen.y, y_det) and torch.allclose(gen.y_ctx, y_ctx) and torch.allclose(gen.contact, prob)
    L = run.ctx + trans + 2
    assert gen.gpos.shape == (3, L, run.skel.num_joints, 3)
    assert gen.contact.shape == (3, L, len(run.skel.contact_joints))
    known = [*range(run.ctx), spec.target_idx]
    assert torch.allclose(gen.gpos[:, known], gen.pb.gpos_gt[:, known], atol=1e-3)   # 입력 프레임 = 정답
    only_ctx = generate(run, batch, trans)
    assert only_ctx.y_ctx is None and only_ctx.contact is None
    assert torch.allclose(only_ctx.y, gen.y_ctx)                  # Context 만 (+후처리) = Detail 입력


def test_eval_one_length_detail_matches_generate(fake_runs):
    from eval_lengths import eval_one_length
    run, det, ds, _, _ = fake_runs
    trans = 15
    J = run.skel.num_joints
    mean, std = torch.zeros(J * 3), torch.ones(J * 3)
    foot_idx = [run.skel.names.index("LeftToe"), run.skel.names.index("RightToe")]
    res = eval_one_length(run, ds, trans, ["model", "detail", "detail_post"], 8, mean, std, foot_idx, 3.0, det=det)
    assert set(res) == {"model", "detail", "detail_post", "_gt"}
    assert set(res["_gt"]) == {"foot_skate", "jitter", "root_speed"}          # GT 기준값 (J8)

    def metrics_from(fn):
        acc = MetricAccumulator(mean, std, foot_idx, 3.0)
        for b in DataLoader(ds, batch_size=8, shuffle=False):
            pb, spec = prepare_window(run, b, trans)
            grot_gt, gpos_gt = global_rot_pos(pb.x_gt_raw, pb.offsets, run.skel, run.layout)
            grot, gpos = global_rot_pos(run.norm.denormalize(fn(b, pb, spec)), pb.offsets, run.skel, run.layout)
            acc.add(gpos, gpos_gt, global_quats(grot), global_quats(grot_gt), spec.seq_slice)
        return acc.result()

    ref = metrics_from(lambda b, pb, spec: generate(run, b, trans, detail=det).y)
    ref_post = metrics_from(lambda b, pb, spec: postprocess(generate(run, b, trans, detail=det).y, pb.x_gt, spec))
    for k in ("L2P", "L2Q", "NPSS", "foot_skate", "boundary_jump", "jitter", "root_speed"):
        assert res["detail"][k] == pytest.approx(ref[k], rel=1e-4, abs=1e-6), k
        assert res["detail_post"][k] == pytest.approx(ref_post[k], rel=1e-4, abs=1e-6), k
    assert res["detail"]["n_clips"] == len(ds)
