# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research context

The user is an undergraduate researcher in an intelligent control / navigation lab, studying **motion in-betweening**: given context frames (start poses) and a target frame (end pose), fill the motion in between. They are learning by reimplementing Qin et al. 2022 (paper PDF at `docs/motion_inbetweening_qin2022.pdf` — gitignored, must never be committed). The user is still learning, so new code needs thorough Korean comments: file header (what it is, order of use), every function's inputs/outputs with tensor shapes, and the paper section/equation it implements.

Current goal (professor meeting feedback, branch `viz-update`): find how far the model stays natural as the transition gets longer. The professor wants this as evidence for why Transformer in-betweening stays smooth while diffusion-based models break down on long gaps — so long-transition behavior, boundary continuity, and foot skating matter.

Status (2026-11):
- Context Transformer trained: 500k iterations, `experiments/ctx_base/` (`best.pth` = iteration 338000, `last.pth`, `log.csv`, `config.yaml`). Never overwrite anything in `experiments/`; checkpoint compatibility and training code must stay untouched.
- Detail Transformer (stage 3) on branch `detail-transformer`: model/inputs/losses (`src/models/detail_transformer.py`, `src/training/detail.py`), `scripts/train_detail.py` (frozen context = `ctx_base/best.pth`). Full training `experiments/det_base/` started 2026-11-12 (~6 h, 41 ms/it, log in `det_base.out` — also leave uncommitted). Eval: `eval_lengths.py` methods `detail`/`detail_post`, `render_samples.py --stage detail`; `inference.load_detail` refuses a Detail trained on a different context checkpoint.
- Stage 4 on `viz-update`: segment-colored GT|PRED videos (`scripts/render_samples.py`), L2P/L2Q/NPSS/foot-skating by transition length (`scripts/eval_lengths.py`). Metric implementation verified: interp/zero-velocity baselines match the official README to 4 decimals; Context-only matches README within ~1–5%.
- Finding: beyond the training range (trans > 30) the model fails to reach the target; the official post-processing approximation (eq. 23) then splits the gap into jumps at both transition boundaries (assumptions F6).
- `configs/env/lab.yaml` (local paths) `ctx_base.out` and `det_base.out` are intentionally uncommitted — leave them out of commits. Push only after the user confirms.

## Project

From-scratch reimplementation of "Motion In-betweening via Two-stage Transformers" (Qin, Zheng, Zhou, TOG 2022) on LaFAN1. The official code (github.com/victorqin/motion_inbetweening) is **read for reference only — never import, copy, or execute it**. The only allowed contact is reading its result files (e.g. `train_stats_context.pkl` via the restricted numpy-only Unpickler in `scripts/compare_stats_with_official.py`).

Stages: 1 data pipeline (done, verified on real LaFAN1) → 2 Context Transformer (trained) → 3 Detail Transformer (planned, `configs/model/detail.yaml`) → 4 evaluation + visualization (`src/evaluation/`, `src/visualization/`, in progress on `viz-update`). Progress tables live in `README.md`.

All docs, docstrings, comments, and commit messages are in Korean; follow that convention. Code comments cite paper sections/equations (e.g. `(논문 3.3, 식 23)`).

## Commands

Run everything from the project root (`pytest.ini` sets `pythonpath = .`; scripts do `sys.path.insert` so `from src... import` works).

```bash
source .venv/bin/activate
pytest                                         # all data-free tests (lab tests auto-skip); expected 90 passed, 4 skipped
pytest tests/test_context_model.py -k mask -v  # single test file / test
TSI_ENV=lab pytest -m lab -v                   # tests needing real LaFAN1 (tests/test_lafan1_real.py)
```

Every script requires an environment: `--env lab|laptop` or `export TSI_ENV=lab`. With neither, config loading errors on purpose. This machine is the lab box (Ubuntu, RTX 3070 8GB); `configs/env/lab.yaml` has real local paths.

```bash
python scripts/compute_stats.py                        # z-score stats → data/cache/stats/lafan1_context_stats.npz (required before training)
python scripts/train_context.py --run-name ctx_base    # → experiments/ctx_base/{config.yaml,log.csv,last.pth,best.pth}
python scripts/train_context.py --run-name ctx_base --resume last
python scripts/train_context.py --batch-size 16 --grad-accum 2 --amp bf16 --set model.n_layer=4 train.base_lr=0.03
# Smoke test without data (generates fake LaFAN1-format BVHs; results meaningless):
python scripts/train_context.py --env laptop --fake-data --max-iters 20 --batch-size 8 \
    --set model.n_layer=2 model.d_model=64 model.d_ff=64 model.d_encoder_h=64 model.d_decoder_h=64
python scripts/render_samples.py --trans 90            # stage 4: per-sample segment-colored GT|PRED mp4 → outputs/videos/trans_90/
python scripts/eval_lengths.py                         # stage 4: metrics by length → outputs/eval/ctx_base/metrics_by_length.{csv,png}
python scripts/eval_lengths.py --sets official --methods interp zerov   # metric sanity check vs README baselines
python scripts/render_compare.py --slerp               # older presentation video GT/pred/SLERP (needs matplotlib + ffmpeg)
python scripts/export_bvh.py                           # GT/pred/SLERP BVHs
blender --background --python scripts/blender_render.py -- <clip_dir>
```

`scripts/README.md` is the authoritative runbook (verification order, memory tuning, what to report).

## Configuration rules

- Paths and device live **only** in `configs/env/*.yaml`, read **only** through `src/config.py` (`load_env_config`, `load_data_config`, `load_model_config`, `resolve_device`). Relative paths resolve against the project root. Never hardcode paths/devices.
- Stage-4 settings (colors, lengths, N/seed, output paths, README reference numbers) live in `configs/viz/video.yaml` and `configs/eval/lengths.yaml`, read via `load_named_config(group, name)`; outputs go to `outputs/` (gitignored).
- `configs/data/lafan1.yaml`: env-independent data settings (context_len 10, max/min trans, splits with window/offset/start_frame/actors).
- Never hardcode dimensions: J comes from `assets/skeleton_lafan1.yaml`; D = J·6+3 (135), model input = D + d_mask (136), output = D. `FeatureLayout(J)` gives the slices. A test (`test_model_follows_joint_count`) enforces this.
- Config values are annotated with provenance: `[논문 x.x]` paper, `[공식코드]` taken from official config (= an assumption), `[결정]` project decision. Any new assumption or deviation from paper/official code must be recorded in `docs/assumptions.md` (section E for stage 2; add a new section for stage 3) and tagged likewise.
- The config actually used for a run is saved to `experiments/<run>/config.yaml`; `render_compare.py`/`export_bvh.py` read model settings from there, not from `configs/`.

## Architecture (big picture)

**Data flow** (`src/data/`, detailed diagram in `src/data/README.md`):
BVH → `bvh.load_bvh` (Euler deg → rotation matrices, `local_rot (F,22,3,3)`, `root_pos (F,3)`) → `dataset.LafanDataset` (actor filter + sliding windows + foot contacts, parsed BVHs cached as `.npz` under `data/cache/lafan1_parsed/`) → `pipeline.motion_to_model_input` (canonicalize so the **last context frame** has root XZ=0 facing +X; encode 6D rot ×22 + root pos = 135; z-score) → model → `pipeline.model_output_to_motion` (inverse, 6D → Gram–Schmidt → matrices, un-canonicalize). Root local +Y is the forward axis.

**Stage 2** (`src/models/`, `src/training/`):
- `training/context_inputs.py`: `TransitionSpec(context_len, target_idx, constrained frames)` drives everything — `sample_transition` (one transition length per batch), `data_mask`, `key_mask` (queries attend only to context/constraint/target keys), `keyframe_positions`, `build_context_input` (unknown rotations = 0 in normalized space, root pos linearly interpolated between anchors, append mask channel), `merge_prediction` (overwrite non-transition frames with GT), `postprocess` (ψ, eq. 23; inference/val only).
- `models/layers.py`: relative-position multi-head attention (scalar distance → shared 2-layer MLP → d_head, added as q·r), Keyframe PE MLP, FFN. `models/context_transformer.py`: pre-LN blocks `x + Sublayer(LN(x))` (intentionally differs from official), `build_context_transformer(model_cfg, layout)`.
- `training/context.py`: `prepare_batch` (canonicalize + normalize + GT FK), `context_forward` (losses), `infer_context`, `eval_transition`/`gpos_l2` — the same functions are reused by `train_context.py` validation and by `render_compare.py`/`export_bvh.py`, keep them shared rather than duplicating logic.
- `training/losses.py`: eq. 21, L1 on transition frames only — L_rot on denormalized 6D, L_pos on FK global positions (cm), L_smooth on frame differences.
- `training/utils.py`: Noam LR, checkpoints, AMP (model only; FK and losses stay float32).

**Stage 4** (`src/evaluation/`, `src/visualization/`): `inference.load_run` rebuilds model+normalizer from a run's saved `config.yaml`; `prepare_window`/`generate` cut clips to `ctx + trans + 2` frames (target+1 needed by post-processing) and reuse the training functions. Two eval sets: `official` (benchmark window 65, 2232 clips, trans ≤ 53) and `long` (window `ctx + max(lengths) + 2` = 312, 2151 clips, same start frames for every length — video samples are drawn from it so they match across lengths). `metrics.py` follows the official RMI definitions (L2P z-scored with mean-centered bench stats from Subject 1–4 window 50/offset 20, cached in `data/cache/stats/`). The model has no hard max length (both PEs feed scalar distances to MLPs); lengths > 30 are extrapolation, labeled "학습 범위 밖".

`docs/context_detail_connection.md` describes how stage 3 (Detail Transformer, window 42 — why context uses 42 not the official 41 is assumption E1) plugs into this.

## Testing notes

`tests/helpers.py` generates fake LaFAN1-format data (correct 22-joint hierarchy, random offsets, sinusoidal motion) so all non-`lab` tests run without data. Tests marked `@pytest.mark.lab` need `TSI_ENV=lab` and skip otherwise. Verification scripts print error magnitudes; round-trip/alignment errors should be ~1e-5 cm or below, otherwise it's a bug.
