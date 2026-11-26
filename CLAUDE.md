# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research context

The user is an undergraduate researcher in an intelligent control / navigation lab, studying **motion in-betweening**: given context frames (start poses) and a target frame (end pose), fill the motion in between. They are learning by reimplementing Qin et al. 2022 (paper PDF at `docs/motion_inbetweening_qin2022.pdf` — gitignored, must never be committed). The user is still learning, so new code needs thorough Korean comments: file header (what it is, order of use), every function's inputs/outputs with tensor shapes, and the paper section/equation it implements.

Current goal (professor meeting feedback, branch `viz-update`): find how far the model stays natural as the transition gets longer. The professor wants this as evidence for why Transformer in-betweening stays smooth while diffusion-based models break down on long gaps — so long-transition behavior, boundary continuity, and foot skating matter.

Status (2026-11):
- Context Transformer trained: 500k iterations, `experiments/ctx_base/` (`best.pth` = iteration 338000, `last.pth`, `log.csv`, `config.yaml`). Never overwrite anything in `experiments/`; checkpoint compatibility and training code must stay untouched.
- Detail Transformer (stage 3) on branch `detail-transformer`: model/inputs/losses (`src/models/detail_transformer.py`, `src/training/detail.py`), `scripts/train_detail.py` (frozen context = `ctx_base/best.pth`). Trained 500k: `experiments/det_base/` (`best.pth` = iteration 498000; log `det_base.out` — leave uncommitted). Result: two-stage beats context-only on every official length and cuts foot skating 32–41%, but the L2P gain is much smaller than README (trans 30: 1.0117 vs README 0.8948); beyond trans 60 Detail no longer helps (assumptions G15, G16). Continued training `experiments/det_cont/` (resumed from det_base/last.pth to 1M, `best.pth` = iteration 978000; log `det_cont.out` — leave uncommitted): trans 30 L2P 0.9909, still ~11% worse than README, so training length is not the main cause (G18). Eval: `eval_lengths.py` methods `detail`/`detail_post`, `render_samples.py --stage detail`; `inference.load_detail` refuses a Detail trained on a different context checkpoint.
- Foot-only model on branch `feat/foot-only` (code `foot_only/`, config `configs/foot_only.yaml`, assumptions section H): predicts only the 4 foot joints' positions (aligned-world coords, cm) + 4 contact logits over the transition, reusing `ContextTransformer` (input 148 = body 135 + known-frame foot positions 12 + mask, output 16) with ctx_base training settings; loss = foot L1 (cm) + contact BCE. Trained 500k → `experiments/foot_only/` (`best.pth` = iteration 454000; val trans 30 foot L2 6.75 cm, contact F1 ankle ~0.70 / toe ~0.80, H9; log `foot_only.out`, leave uncommitted). Evaluated with `python foot_only/eval_foot.py` (`foot_only/evaluate.py`, config `configs/eval/foot_only.yaml` + `lengths.yaml`) → `outputs/eval/foot_only/`: foot-only beats two-stage on foot L2 and foot skating at every length except trans-5 L2, but also collapses beyond trans 30 (H10). Contacts from predicted positions are judged in world coords via `foot_to_world`. Videos: `python foot_only/render_feet.py` (config `configs/viz/foot_only.yaml`) → `outputs/videos/foot_only/<clip>/trans<len>.mp4` (top-down toe paths, toe height/speed, contact strips; GT vs foot-only vs two-stage). Expected `pytest`: 101 passed, 4 skipped. Comparison targets: ctx_base, ctx_base + det_cont (final two-stage; `lengths.yaml` default `detail_ckpt`), SLERP. Contact labels are defined in original world coords (L1 speed is not rotation-invariant, H4).
- Feet-to-body model on branch `feat/feet-to-body` (separate project from foot-only; code `feet_to_body/`, config `configs/feet_to_body.yaml`, assumptions section I): same ContextTransformer / loss (eq. 21) / training settings as ctx_base, but the input also carries GT foot condition on every frame (foot positions 12 + contacts 4 → input 152, output 135) and attention sees all frames (`cond.attend_all`, I2). First experiment = GT feet (upper bound of "feet first"). `python feet_to_body/train_body.py` → `experiments/feet_to_body/` (500k done, `best.pth` = iteration 164000; log `feet_to_body.out`, leave uncommitted); `val{t}_loss/gpos_l2` are directly comparable with `experiments/ctx_base/log.csv`. Evaluated with `python feet_to_body/eval_body.py` (config `configs/eval/feet_to_body.yaml`) → `outputs/eval/feet_to_body/` (I6, I7): with GT feet, trans-30 L2P 0.650 vs two-stage 0.991 (large upper bound); with foot-only predicted feet 0.954 (barely better at 30, worse at 5/15/45); short transitions learn slowly (trans 5 worse than Context at best.pth, which the val30 rule picked early); beyond the training range it collapses even with GT feet. Second run (I8, I9): `python feet_to_body/train_body.py --config configs/feet_to_body_pred.yaml --run-name feet_to_body_pred` trains with the frozen foot-only model's predicted feet as the condition (`cond.source: foot_model`; config overlay via `base:`), also saves `best_mean.pth` (mean val loss over 5/15/30); 500k done → `experiments/feet_to_body_pred/` (`best.pth` = iteration 152000, `best_mean.pth` = 492000; log `feet_to_body_pred.out`, leave uncommitted). Result (I10, `outputs/eval/feet_to_body_pred{,_mean}/`): training on predicted feet changed almost nothing (trans-30 L2P with predicted feet 0.942 vs 0.954 first run; two-stage 0.991, context 1.037); the body model still follows the given feet, so foot prediction error is the bottleneck; feet-first beats two-stage only at trans 30, not at 5/15/45 or beyond the range, and foot skating is not better than two-stage. `--run-name` is required and finished run folders are refused. Follow-ups (2026-11-15): foot-error sensitivity `python feet_to_body/eval_foot_mix.py` (GT/predicted feet mixed, no training, I11: L2P falls almost linearly with foot error; trans 30 needs foot error 6.9 → ~5.2 cm for a 15% gain over two-stage); videos `python feet_to_body/render_body.py` (config `configs/viz/feet_to_body.yaml`, 2×2 GT / two-stage / feet→body predicted feet / GT feet → `outputs/videos/feet_to_body/`, I13); foot model v2 in `foot_v2/` (foot-only + foot velocity loss `w_vel`, config `configs/foot_v2.yaml`, I12) — runs the unmodified `foot_only/train_foot.py` / `eval_foot.py` with the loss function / config reader swapped in (`python foot_v2/train_foot_v2.py --run-name foot_v2` → `experiments/foot_v2/`, log `foot_v2.out` leave uncommitted; `python foot_v2/eval_foot_v2.py`; body eval with other feet via `eval_body.py --foot-ckpt`). foot_v2 result (500k, `best.pth` = iteration 424000, I14): foot L2 unchanged (trans 30: 6.91 → 6.84 cm), foot skating −7% and position-judged contact F1 +0.03; with v2 feet the body L2P is unchanged (0.954 → 0.950) and body foot skating −8% (0.275 ≈ two-stage 0.281) — the velocity loss did not fix foot position accuracy. Expected `pytest`: 116 passed, 4 skipped.
- Long-transition retraining (2026-11-20/21, branch `feat/feet-to-body`, assumptions section J, MotionBricks used as reference only): new options in existing files only, all defaulting to ctx_base behavior — `data.max_trans`/window 312 + `train.trans_dist: log_uniform` (J1–J2), `train.midway_none_p`/`midway_max` (J3), `loss.w_vel`/`w_foot`/`w_skate` (J4, J12; `losses.velocity_loss`/`foot_skate_loss`), `train.postprocess_mode: ramp` (J5, `postprocess(mode="ramp")`), `model.pe_scale` (J6), `train.best_metric: mean` (J7), `train.loss_on_postprocess` (J13, `context_forward(post_mode=...)`; validation uses the same), `train_context.py --reset-best`; new metrics boundary jump / jitter / root speed (J8); foot-lock two-bone IK post-process `inference.foot_lock` (methods `model_lock`/`detail_lock`, `render_samples.py --lock --tag`, settings `lengths.yaml` `foot_lock`, J9/J9b/J14). Final run `experiments/ctx_long_pp/` (resumed from pp_skate 150k to 500k, `best.pth` = iteration 295000; `train.out` there). Result (`outputs/eval/ctx_long_pp/`, J section end): long-set L2P trans 90/150/300 6.31/12.48/21.80 → 4.22/8.41/17.75 (beats SLERP at every length), boundary jump 75 → 0.75 cm/frame, foot skating with lock 0.10–0.37 at trans 60–300; but trans ≤ 30 is 7–16% worse than ctx_base, so use Two-stage for ≤ 30 and ctx_long_pp + lock for ≥ 45 (J16). Models trained with `loss_on_postprocess` must always be post-processed. 10-s motion is slower/smoother than GT (root speed 64%, jitter 1/4 of GT). Other run folders `ctx_long` (160k, ablation), `pp_skate`, `pp_skate10`, `ft_skate`, `ft_foot05`, `pilot_{a,b,c}` are experiments — leave untouched. Expected `pytest`: 127 passed, 4 skipped.
- Stage 4 on `viz-update`: segment-colored GT|PRED videos (`scripts/render_samples.py`), L2P/L2Q/NPSS/foot-skating by transition length (`scripts/eval_lengths.py`). Metric implementation verified: interp/zero-velocity baselines match the official README to 4 decimals; Context-only matches README within ~1–5%.
- Finding: beyond the training range (trans > 30) the model fails to reach the target; the official post-processing approximation (eq. 23) then splits the gap into jumps at both transition boundaries (assumptions F6).
- `configs/env/lab.yaml` (local paths) `ctx_base.out` and `det_base.out` are intentionally uncommitted — leave them out of commits. Push only after the user confirms.

## Project

From-scratch reimplementation of "Motion In-betweening via Two-stage Transformers" (Qin, Zheng, Zhou, TOG 2022) on LaFAN1. The official code (github.com/victorqin/motion_inbetweening) is **read for reference only inside this repo**. Never import it, never copy it into this repo, and never run it from this repo or its `.venv`.

**Exception for comparison** (allowed since 2026-11-26): the official code may be **run** in its own clone `~/projects/motion_inbetweening` (commit `fa9b6dc`) with its own conda env `inbetween` (Python 3.8, torch 1.8.2+cu111). Rules:
- Never modify the clone's existing files. New scripts go only in its `ext/`, and they run the original scripts or functions as they are (e.g. `ext/run_no_tf32.py` via `runpy`).
- Never mix the two environments.
- Never overwrite this repo's `outputs/` or `experiments/`.
- Bring back numbers and result files only (CSV/JSON/logs; pickles only through the restricted Unpickler). Never bring back code.
- Turn TF32 off when exact numbers matter. On this Ampere GPU, torch 1.8.2 enables TF32 by default, which shifts metrics by up to 2e-4. With TF32 off, the README numbers reproduce to 4 decimals.

The runbook and results are in `~/projects/motion_inbetweening/ext/RUNBOOK_ko.md` and assumption G20.

Outside this exception, the only allowed contact is reading the official code's result files (e.g. `train_stats_context.pkl` via the restricted numpy-only Unpickler in `scripts/compare_stats_with_official.py`).

Stages: 1 data pipeline (done, verified on real LaFAN1) → 2 Context Transformer (trained) → 3 Detail Transformer (planned, `configs/model/detail.yaml`) → 4 evaluation + visualization (`src/evaluation/`, `src/visualization/`, in progress on `viz-update`). Progress tables live in `README.md`.

All docs, docstrings, comments, and commit messages are in Korean; follow that convention. Code comments cite paper sections/equations (e.g. `(논문 3.3, 식 23)`).

## Commands

Run everything from the project root (`pytest.ini` sets `pythonpath = .`; scripts do `sys.path.insert` so `from src... import` works).

```bash
source .venv/bin/activate
pytest                                         # all data-free tests (lab tests auto-skip); expected 127 passed, 4 skipped
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
