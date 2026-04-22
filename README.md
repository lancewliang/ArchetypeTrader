# ArchetypeTrader

A PyTorch implementation of the three-phase hierarchical reinforcement learning framework for trading, based on the paper submitted to **AAAI 2026**. The original paper studies cryptocurrency trading with reusable strategic archetypes. The current codebase keeps that three-phase research structure, but has also evolved into an AL-focused experimental implementation with profit-aware Phase I training, a modular PPO-style Phase II trainer, and a fuller evaluation / audit pipeline.

> **Disclaimer:** This codebase is for research and engineering experiments only. It is not investment advice and should not be used directly for live trading. Several implementation choices intentionally go beyond the paper and are documented below.

> **Paper:** *ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading* [[PDF]](AAAI26_ArchetypeTrader.pdf)
> Chuqiao Zong, Molei Qin, Haochong Xia, Bo An — Nanyang Technological University, Singapore
>
> Code comments reference specific sections, equations, and algorithms from the paper for traceability. Where the paper is ambiguous or the current code has moved beyond the paper, this README lists the difference explicitly.

## Overview

ArchetypeTrader follows the original three-phase pipeline:

1. **Phase I — Archetype Discovery**: A dynamic programming planner (Algorithm 1) generates demonstration trajectories under a single-trade / single-change constraint. A VQ encoder-codebook-decoder model compresses these trajectories into discrete trading archetypes.

2. **Phase II — Archetype Selection**: A horizon-level RL agent selects one archetype at the start of each 72-step trading horizon. A frozen decoder then turns the selected archetype code into step-by-step base actions.

3. **Phase III — Archetype Refinement**: A step-level RL agent can refine the selected archetype's base actions with a regret-aware reward signal, while allowing at most one effective adjustment per horizon.

Current repository status:
- The available dataset in this workspace is `AL`; ETH is still present in config but the ETH feather files are not included in this checkout.
- The feature pipeline now uses `fixed features + optional cycle feature sets` instead of a hard-coded 45-dimensional state. CLI parsing defaults to `--cycle-feature-sets middle`, which gives `state_dim=57`.
- The default pipeline script currently trains Phase I and Phase II, then evaluates Phase II on val/test with a DP baseline. Phase III is implemented but commented out in `run_pipeline.sh`.

```
Feather data -> Feature Pipeline (fixed + cycle features)
    -> DP Planner -> cached DP trajectories
    -> VQ Encoder-Codebook-Decoder (Phase I)
    -> Phase I validation + profit-gated checkpoint selection
    -> Selection Agent (Phase II, PPO-style) -> Frozen Decoder -> Base Actions
    -> Optional Refinement Agent (Phase III, AdaLN) -> Final Trading Actions
    -> Evaluation Engine (TR, Sharpe, Calmar, Sortino, MDD, Volatility)
    -> PortfolioTracker + TradeAuditor + Backtrader Cross-Validation
```

## Project Structure

```
ArchetypeTrader/
├── data/
│   ├── AL/                             # Available feather data in this checkout
│   │   ├── df_train.feather
│   │   ├── df_val.feather
│   │   └── df_test.feather
│   └── AL_10s/
├── src/
│   ├── config.py                       # Global hyperparameters (dataclass + CLI)
│   ├── data/
│   │   ├── dataset.py                  # TrajectoryDataset + normalization stats
│   │   └── feature_pipeline.py         # Feather loading and fixed/cycle features
│   ├── env/
│   │   └── trading_env.py              # MDP trading environment with 5-level LOB slippage
│   ├── phase1/
│   │   ├── dp_planner.py               # Algorithm 1: single-trade DP planner + sampling
│   │   ├── vq_encoder.py               # LSTM encoder with temporal attention pooling
│   │   ├── vq_decoder.py               # BiLSTM decoder with constrained decoding
│   │   ├── codebook.py                 # VQ codebook with k-means init and dead-code reset
│   │   ├── checkpoint.py               # Profit-gated checkpoint selection
│   │   ├── validation.py               # Phase I artifact validation
│   │   └── env_validation.py           # Environment-level archetype validation
│   ├── phase2/
│   │   ├── selection_agent.py          # Horizon-level Actor-Critic selector
│   │   ├── rollout.py                  # Batched decoding and vectorized horizon execution
│   │   ├── evaluation.py               # Validation and selector diagnostics
│   │   └── diagnostics.py              # Execution / archetype histograms
│   ├── phase3/
│   │   ├── refinement_agent.py         # Step-level Actor-Critic with AdaLN
│   │   ├── policy_adapter.py           # Eq. 6: final action computation
│   │   ├── adaln.py                    # Adaptive Layer Normalization
│   │   └── regret_reward.py            # Eq. 8 + top-5 hindsight optimal adaptations
│   ├── evaluation/
│   │   ├── metrics.py                  # TR / AVOL / MDD / ASR / ACR / ASoR
│   │   ├── inference_runner.py         # Phase II / optional Phase III inference
│   │   ├── model_loader.py             # Centralized model loading
│   │   ├── portfolio_tracker.py        # Cross-horizon portfolio and cash management
│   │   ├── trade_auditor.py            # Post-evaluation trade statistics and checks
│   │   └── bt_verifier.py              # Backtrader replay verification
│   └── utils/
│       ├── gpu_guard.py
│       ├── logger.py
│       ├── normalizer.py
│       └── progress.py
├── scripts/
│   ├── train_phase1.py                 # Phase I: DP trajectories + VQ training
│   ├── train_phase2.py                 # Phase II: PPO-style selector training
│   ├── train_phase3.py                 # Phase III: regret-aware refinement training
│   ├── evaluate.py                     # Evaluation on val/test, optionally with DP baseline
│   ├── analyze_dataset.py              # Dataset shift / DP oracle analysis
│   ├── diagnose_archetype.py           # Phase II archetype diagnostics
│   └── diagnose_train_trajectories.py  # AL/batch_001 quick diagnostic script
├── tests/                              # Unit tests + property-based tests
├── docs/                               # Development and optimization logs
├── run_pipeline.sh                     # Current end-to-end Phase I -> Phase II -> eval script
├── requirements.txt
└── result/
    └── {PAIR}/{BATCH_ID}/
        ├── dp_trajectories/
        ├── phase1_archetype_discovery/
        ├── phase2_archetype_selection/
        ├── phase3_archetype_refinement/
        ├── phase2_eval_val/
        ├── phase2_eval_test/
        ├── dp_val/
        └── dp_test/
```

## Setup

```bash
conda create -n ArchetypeTrade python=3.12
conda activate ArchetypeTrade
pip install -r requirements.txt
pip install torch  # install PyTorch separately per your CUDA version
```

Dependencies in `requirements.txt`:
- `pyarrow>=14.0.0` — feather file I/O
- `numpy>=1.24.0`
- `polars>=0.20.0` — high-performance DataFrame operations
- `pandas>=2.0.0`
- `tqdm>=4.64.0` — progress bars

Additional tools used by optional scripts/tests:
- `pytest`, `hypothesis` — testing
- `backtrader` — independent replay verification
- `scipy` — dataset analysis script

## Usage

### Full Pipeline (Recommended)

Run the current Phase I -> Phase II -> Phase II evaluation pipeline:

```bash
bash run_pipeline.sh AL batch_001 --cycle-feature-sets middle
# Logs saved to logs/AL/batch_001/AL_pipeline_YYYYMMDD_HHMMSS.log
```

Notes:
- The first positional argument is `PAIR`; default is `AL`.
- The second positional argument is `BATCH_ID`; default is `batch_001`.
- Extra arguments after those two are forwarded to every Python stage.
- Phase III is implemented, but currently commented out in `run_pipeline.sh`.

### Individual Phases

Training runs sequentially — each phase depends on the previous one:

```bash
# Phase I: Generate DP trajectories + train VQ encoder-codebook-decoder
python scripts/train_phase1.py --pair AL --train-batch-id batch_001 --cycle-feature-sets middle

# Phase II: Train archetype selection agent
python scripts/train_phase2.py --pair AL --train-batch-id batch_001 --cycle-feature-sets middle

# Evaluate Phase II on val/test with DP baseline
python scripts/evaluate.py --pair AL --train-batch-id batch_001 \
  --split val test --stage-label phase2_eval --with-dp \
  --cycle-feature-sets middle

# Optional Phase III: Train refinement agent
python scripts/train_phase3.py --pair AL --train-batch-id batch_001 \
  --cycle-feature-sets middle --beta1 0.5

# Optional Phase III evaluation
python scripts/evaluate.py --pair AL --train-batch-id batch_001 \
  --split val test --stage-label phase3_eval \
  --cycle-feature-sets middle
```

Diagnostic scripts:

```bash
python scripts/analyze_dataset.py --pair AL
python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val --cycle-feature-sets middle
python scripts/diagnose_train_trajectories.py
```

`diagnose_train_trajectories.py` is currently a quick AL/batch_001 diagnostic script with hard-coded paths.

### Key CLI Arguments

All optional; defaults are defined in `src/config.py` and `parse_args()`:

| Argument | Default | Description |
|---|---:|---|
| `--pair` | `AL` in current training scripts | Single trading pair; overrides `Config.pairs` |
| `--train-batch-id` | `default` (`batch_001` in `run_pipeline.sh`) | Isolates result directories |
| `--cycle-feature-sets` | `middle` via CLI | Comma-separated feature groups: `short,middle,long` |
| `--horizon` | 72 | Steps per trading horizon |
| `--commission-rate` | 0.0002 | Evaluation commission rate |
| `--num-trajectories` | 50000 | DP demonstration trajectories |
| `--phase1-epochs` | 400 | Phase I training epochs |
| `--pretrain-epochs` | 10 | Phase A continuous latent pretraining epochs |
| `--latent-dim` | 32 | VQ latent/code dimension |
| `--lstm-hidden-dim` | 256 | Phase I encoder/decoder hidden dimension |
| `--phase1-start-sampling-mode` | `hybrid_stratified_importance` | DP start-index sampling mode |
| `--phase2-total-steps` | 1000000 | Selection agent training steps |
| `--selection-alpha` | 0.5 | Initial imitation/KL coefficient |
| `--phase2-alpha-schedule` | `linear` | `selection_alpha` schedule |
| `--phase2-alpha-final-ratio` | 0.0 | Final alpha as initial-alpha ratio |
| `--phase2-imitation-min-raw-return` | 0.0 | Apply imitation only to horizons whose raw return is above this threshold |
| `--phase3-total-steps` | 1000000 | Refinement agent training steps |
| `--phase3-num-envs` | 16 | Horizons collected per Phase III batch |
| `--beta1` | 0.5 | Regret coefficient beta1 |
| `--beta2` | 1.0 | Hindsight CE coefficient in Phase III |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Phase I batch size |

## Evaluation Pipeline

The evaluation system goes beyond simple metric computation:

1. **Phase II / optional Phase III inference** (`inference_runner.py`): Runs frozen Phase I and Phase II models, and optionally the Phase III refinement model.
2. **Portfolio tracking** (`portfolio_tracker.py`): Manages cash, long/short positions, average hold prices, short debt, horizon settlement, and final liquidation.
3. **Trade audit** (`trade_auditor.py`): Computes detailed trade statistics and consistency checks from exported operation records.
4. **Backtrader cross-validation** (`bt_verifier.py`): Replays the same position sequence through Backtrader as an independent verification engine.
5. **DP baseline** (`evaluate_pair_dp`): Runs the DP planner on the same split and reports the model-vs-DP TR gap.
6. **CSV export**: Per-step operation logs are exported in chunks for external analysis.

Example output paths:

```
result/AL/batch_001/phase2_eval_val/AL_results.json
result/AL/batch_001/phase2_eval_test/AL_results.json
result/AL/batch_001/dp_val/AL_dp_results.json
result/AL/batch_001/dp_test/AL_dp_results.json
```

## Testing

```bash
python -m pytest tests/ -v
```

The test suite covers core components with unit tests and property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/). It includes checks for:
- Feature dimension invariants and feature-set resolution
- Trading environment position/reward/cost invariants
- DP single-trade constraint and small-case optimality
- VQ nearest-neighbor quantization and constrained decoder behavior
- Codebook collapse detection and dead-code reset behavior
- At most one refinement adjustment per horizon
- Regret reward and hindsight-optimal adaptation logic
- Evaluation metric formulas (TR, AVOL, MDD, ASR, ACR, ASoR)
- Portfolio tracking, trade audit, and Backtrader verification logic

## Key Hyperparameters

| Parameter | Current Code Default | Paper Value | Notes |
|---|---:|---:|---|
| State dim | 57 with CLI default `middle` | 45 | **Differs** — current state dim is `24 fixed + selected cycle features` |
| Fixed feature dim | 24 | 45 total state dim | **Differs** — feature definitions have changed |
| Action space | {short, flat, long} | {0, 1, 2} | Matches |
| Horizon h | 72 | 72 | Matches |
| Evaluation commission | 0.0002 | 0.0002 | Matches paper fee when explicit config is used |
| DP / training commission | 0.0008 | 0.0002 | **Differs** — higher fee used as training/trajectory safety margin |
| `TradingEnv.COMMISSION_RATE` fallback | 0.0003 | 0.0002 | **Implementation caution** — main entrypoints pass config explicitly |
| Codebook size K | 10 | 10 | Matches |
| Latent dim | 32 | 16 | **Differs** — expanded bottleneck |
| LSTM hidden dim | 256 | 128 | **Differs** — larger models |
| VQ commitment beta0 | 0.25 | 0.25 | Matches |
| Phase I trajectories | 50000 | 30000 | **Differs** — increased data coverage |
| Phase I epochs | 400 | 100 | **Differs** — longer training/checkpoint search |
| Pretrain epochs | 10 | N/A | **Addition** — continuous latent pretraining |
| Selection alpha | 0.5 -> 0.0 linear schedule | 1.0 | **Differs** — annealed imitation/KL |
| Phase II steps | 1000000 | 3000000 | **Differs** — current docs report shorter successful runs |
| Phase III steps | 1000000 | 1000000 | Matches |
| Annualization factor | 52560 | 52560 | Matches 10-minute bars |

Common state dimensions:

| CLI `--cycle-feature-sets` | State dim |
|---|---:|
| none via direct `Config()` | 24 |
| `short` | 54 |
| `middle` | 57 |
| `long` | 54 |
| `short,middle` | 73 |
| `middle,long` | 70 |
| `short,middle,long` | 84 |

## Supported Trading Pairs

| Pair | Current Code Max Position (m) | Data in this checkout | Paper Max Position (m) | Notes |
|---|---:|---|---:|---|
| AL | 10 | Yes | N/A | Current experimental mainline |
| ETH/USDT | 100 | No | 100 | Config still supports ETH, but data must be provided |
| BTC/USDT | Not configured | No | 8 | Paper asset; not in current `max_positions` default |
| DOT/USDT | Not configured | No | 2500 | Paper asset; not in current `max_positions` default |
| BNB/USDT | Not configured | No | 200 | Paper asset; not in current `max_positions` default |

## Deviations from Paper

This section documents known differences between the paper and the current codebase. These include engineering enhancements, intentional design changes, and implementation cautions discovered during recent experiments.

### Phase I — Archetype Discovery

| Aspect | Paper | Current Code | Type |
|---|---|---|---|
| Input state | Fixed 45-dimensional market state | 24 fixed features + optional `short/middle/long` cycle feature sets | Design change |
| DP chunks | Sample fixed-length chunks | Sliding-window legal starts with `uniform`, `stratified`, or `hybrid_stratified_importance` sampling | Enhancement |
| DP cache | Not specified | `.npz` cache stores pair, horizon, gamma, state_dim, commission, sampling metadata; incompatible caches are backed up | Enhancement |
| DP commission | Same paper commission | Separate `dp_commission_rate=0.0008` to filter higher-quality trajectories | Design change |
| Encoder architecture | LSTM-based encoder | LSTM + temporal attention pooling over all hidden states | Enhancement |
| Decoder architecture | Decoder not fully specified | BiLSTM decoder conditioned on state sequence and code vector | Enhancement |
| Decoder inference | Not specified | `decode_with_single_trade_constraint()` searches best single-change sequence over decoder logits | Enhancement |
| Decoder constraint semantics | Algorithm 1 starts from flat and constrains action changes through DP state `c` | DP planner follows Algorithm 1; decoder post-processing only enforces at most one intra-sequence action change, so execution diagnostics still track position changes/direct flips relative to the flat env start | Implementation caution |
| Dataset normalization | Not discussed | Phase I stores z-score `norm_stats` for downstream Phase II/III/evaluation | Enhancement |
| Training strategy | End-to-end VQ training with Eq. (4) loss | Two-stage training: continuous latent pretraining, then full VQ training | Enhancement |
| Codebook initialization | Not discussed | Direction-aware k-means with profit-aware init/reset | Enhancement |
| Codebook collapse | Not discussed | Dead-code reset from recent/high-return latent samples | Enhancement |
| Profit semantics | Not explicitly constrained | Return bucket auxiliary objective + light usage-profit alignment + codebook separation | Enhancement |
| Checkpoint selection | Not specified | Periodic checkpoint evaluation plus strict profit gate before materializing `{PAIR}_vq_model.pt` | Enhancement |

The key experimental finding in `docs/2026-04-19_phase1_dp_structure_optimization_log.md` is that reconstruction quality alone is not enough: Phase II improves only when Phase I archetype semantics align with return structure. The current effective recipe is return bucket objective + light usage-profit alignment + strict profit gate.

### Phase II — Archetype Selection

| Aspect | Paper | Current Code | Type |
|---|---|---|---|
| Objective function | Eq. (5): horizon reward + alpha KL to demonstration label | PPO clipped surrogate + value loss + entropy bonus + imitation/KL term | Enhancement |
| Selection alpha | Constant alpha=1 in experiment setup | `selection_alpha=0.5` with linear schedule to 0 by default | Design change |
| Imitation target | VQ-assigned demonstration archetype | One-hot NLL equivalent of KL, masked by raw horizon return (`phase2_imitation_min_raw_return=0.0` by default) | Enhancement |
| Return scale | Raw horizon return | Batch-standardized returns and normalized advantages | Engineering change |
| Actor / critic update | Not specified | Separate actor and critic optimizers; critic uses higher LR | Engineering change |
| Rollout | Not specified | Batched decoder inference and vectorized horizon execution | Engineering change |
| Inference action | Paper describes policy distribution | Validation/evaluation use greedy argmax over archetype probabilities | Design change |
| Diagnostics | Not discussed | learned/random/oracle/fixed baselines, archetype histograms, execution cost/turnover/direct-flip stats | Enhancement |
| Code organization | Single selector concept | Refactored into `config.py`, `data_loader.py`, `rollout.py`, `evaluation.py`, `diagnostics.py`, `checkpoint.py` | Engineering change |

Implementation caution: Phase II currently indexes DP trajectory cache directly by `horizon_indices` for imitation labels. Current Phase I trajectory generation uses randomly sampled sliding-window starts, so those cached trajectories are not guaranteed to be the exact same non-overlapping windows used by Phase II rollouts. If strict Eq. (5) same-window labeling is required, this alignment protocol should be revisited.

### Phase III — Archetype Refinement

| Aspect | Paper | Current Code | Type |
|---|---|---|---|
| Pipeline status | Third phase is part of the full method | Implemented, but commented out in current `run_pipeline.sh` | Current workflow choice |
| Responsibility | Local action refinement after fixed Phase II selection | Same; Phase III does not reselect archetypes | Matches |
| `tau_remain` | Absolute remaining steps | Normalized `(h - step_idx) / h` in training and inference | Design change |
| `R_arche` | Raw cumulative reward | Normalized by `notional = m * p_0` for stable context scale | Enhancement |
| Architecture | Uses AdaLN conceptually | 3-layer MLP with residual + LayerNorm + AdaLN conditioning | Enhancement |
| Objective | Eq. (9) with regret-aware reward and CE | PPO clipped surrogate + value loss + hindsight CE + entropy | Enhancement |
| Batch collection | Not specified | Collects multiple horizons per update via `phase3_num_envs` to improve GPU utilization | Engineering change |
| Hindsight optimal | Top-5 DP adaptations | Vectorized candidate evaluation with LOB-aware costs | Enhancement |
| Beta tuning / checkpointing | `beta1` tuned over `{0.3, 0.5, 0.7}` by validation; `beta2=1` | CLI exposes `--beta1/--beta2`, default `beta1=0.5`; training saves the final model for that beta and does not run a built-in beta sweep | Design change |

`docs/2026-04-12_phase3_optimization_log.md` records that an AL run completed Phase III training but produced the same final operation sequence as Phase II evaluation. The current priority in this codebase is therefore to improve Phase I/II before expanding Phase III into a high-level selector-correction module.

### Global / Config

| Aspect | Paper | Current Code | Type |
|---|---|---|---|
| Main data asset | BTC/ETH/DOT/BNB crypto pairs | AL data is currently present; ETH config remains but data is absent | Design change |
| Dataset split dates | Paper reports fixed train/val/test calendar ranges | Current `FeaturePipeline` loads already-split feather files; `Config.train_start/val_start/test_start` are metadata labels and do not filter rows | Implementation caution |
| Result layout | Not specified | `result/{PAIR}/{BATCH_ID}/{stage}` for parallel experiment isolation | Enhancement |
| Feature configuration | Fixed state definition | CLI `--cycle-feature-sets` controls state dim; checkpoint loaders enforce state_dim consistency | Enhancement |
| Commission handling | Single commission rate | Separate evaluation, training, and DP commission rates | Design change |
| Evaluation infrastructure | Paper metrics | PortfolioTracker, TradeAuditor, BacktraderVerifier, DP baseline, chunked CSV export | Enhancement |
| Dataset analysis | Not discussed | `scripts/analyze_dataset.py` writes shift/oracle reports to `docs/` | Enhancement |
| Paper baselines / ablations | Reports DQN/PPO/CDQNRP/CLSTM-PPO/EarnHFT/MacroHFT/IV/MACD and VQ/refinement/regret ablations | Current repo focuses on the ArchetypeTrader pipeline; those external baselines and full paper ablation runners are not implemented as first-class scripts | Scope difference |
| Hardware setup | Paper experiments use 4 RTX-4090 GPUs | Scripts run single-process on the available CUDA/CPU device; GPU guard utilities reduce memory risk on smaller hardware | Engineering change |
| Paper reproduction guards / comments | Paper values are defined by the manuscript | Some inline comments and legacy `PAPER_PHASE1_SPEC`/"strict paper" log text are stale or disabled; use `Config`, current code paths, and this README as the source of truth | Documentation caution |
| Development logs | Not part of paper | `docs/` records Phase I/II/III optimization decisions and experiment outcomes | Documentation |

### Summary of Current Cautions

1. **Phase III is optional in the current script**: `run_pipeline.sh` currently stops after Phase II evaluation unless the Phase III block is uncommented or run manually.
2. **Pair defaults can surprise evaluation**: `Config.pairs` contains `["AL", "ETH"]`, so use `--pair AL` when ETH data is not available.
3. **Feature-set consistency matters**: use the same `--cycle-feature-sets` for Phase I, Phase II, Phase III, evaluation, and diagnostics.
4. **Phase II label alignment should be reviewed for strict paper reproduction**: current DP cache sampling and Phase II non-overlapping horizon indexing are not guaranteed to refer to identical windows.
5. **Phase III hindsight code uses a class-level commission fallback in one path**: `src/phase3/regret_reward.py` reads `env.COMMISSION_RATE` in `compute_top5_hindsight_optimal`; review this if comparing Phase III training costs precisely.
6. **Evaluation defaults can load Phase III**: `scripts/evaluate.py` uses Phase III unless `--stage-label phase2_eval` is passed; this matters because the default pipeline does not train Phase III.
7. **Config date fields are metadata for pre-split files**: the code loads `df_train/df_val/df_test.feather` directly and does not apply date filtering from `Config`.
8. **Historical docs include intermediate experiments**: for example, `docs/2026-04-03_phase1_decoder_optimization_log.md` contains older MLP/LSTM/teacher-forcing variants; current code uses BiLSTM + constrained decoding.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zong2026archetypetrader,
  title     = {ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading},
  author    = {Zong, Chuqiao and Qin, Molei and Xia, Haochong and An, Bo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

## License

This project is an academic implementation for research purposes.
