"""발 전용 모델 v2 (발 속도 손실, I12) 테스트. 데이터 없이 실행된다."""

import shutil
import subprocess
import sys
from pathlib import Path

import torch

from foot_v2.losses import foot_velocity_loss
from src.training.context_inputs import TransitionSpec

ROOT = Path(__file__).resolve().parents[1]


def test_velocity_loss_zero_for_gt_and_counts_boundaries():
    """정답이면 0. transition 전체를 같은 값만큼 옮기면 (속도는 같고 경계 2곳만 튐) 경계 항만 남는다."""
    torch.manual_seed(0)
    gt = torch.randn(2, 20, 12)
    spec = TransitionSpec(10, 15)                       # transition = 프레임 10–14, target = 15
    assert float(foot_velocity_loss(gt, gt, spec)) == 0.0
    pred = gt.clone()
    pred[:, spec.seq_slice] += 3.0                      # 경계 2곳에서 3 cm 씩 튐, 차이 6개 중 2개
    assert torch.isclose(foot_velocity_loss(pred, gt, spec), torch.tensor(3.0 * 2 / 6), atol=1e-5)


def test_v2_config_merges_base():
    from foot_v2.train_foot_v2 import load_v2_config
    cfg = load_v2_config()
    assert cfg["loss"]["w_vel"] > 0 and cfg["loss"]["w_pos"] == 1.0 and cfg["model"]["n_layer"] == 6


def test_train_script_fake_data(tmp_path):
    """가짜 BVH 로 학습 스크립트가 끝까지 돌고 train `vel` 이 기록되는지, 기존 실험 이름·이름 없음은 거부되는지."""
    run_name = f"pytest_footv2_{tmp_path.name}"
    script = [sys.executable, str(ROOT / "foot_v2" / "train_foot_v2.py"), "--env", "laptop", "--fake-data"]
    opts = ["--batch-size", "4", "--num-workers", "0", "--max-iters", "6", "--set", "model.n_layer=1",
            "model.d_model=32", "model.d_ff=32", "model.d_encoder_h=32", "model.d_decoder_h=32", "model.d_head=8",
            "model.n_head=2", "train.log_interval=3", "train.val_interval=6", "train.ckpt_interval=6"]
    run_dir = ROOT / "experiments" / run_name
    try:
        r = subprocess.run(script + ["--run-name", run_name] + opts, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        log = (run_dir / "log.csv").read_text()
        assert ",train,vel," in log and "val30_loss" in log and (run_dir / "best.pth").exists()
        import yaml
        assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["loss"]["w_vel"] > 0
        r = subprocess.run(script + ["--run-name", "foot_only"] + opts, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "기존 실험 폴더" in (r.stdout + r.stderr)
        r = subprocess.run(script + opts, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "--run-name" in (r.stdout + r.stderr)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
