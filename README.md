# ArchetypeTrader

A PyTorch implementation of the three-phase hierarchical reinforcement learning framework for cryptocurrency trading, based on the paper submitted to **AAAI 2026**. The system discovers reusable trading archetypes from historical data via dynamic programming and vector quantization, then deploys them through hierarchical RL agents for real-time trading.

> **Disclaimer:** This codebase has NOT been validated with real trading data yet. Correctness of the implementation cannot be guaranteed at this stage. Please stay tuned for updates.

> **Paper:** *ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading* [[PDF]](AAAI26_ArchetypeTrader.pdf)
> Chuqiao Zong, Molei Qin, Haochong Xia, Bo An — Nanyang Technological University, Singapore
>
> This codebase is generated from the above research paper (AAAI 2026). Code comments reference specific sections, equations, and algorithms from the paper for traceability. Where the paper is ambiguous or lacks implementation details, the code includes `[NOTE]` annotations.

## Overview

ArchetypeTrader operates on 10-minute cryptocurrency bars (BTC/ETH/DOT/BNB vs USDT) with 5-level limit order book (LOB) data. It follows a three-phase pipeline:

1. **Phase I — Archetype Discovery**: A dynamic programming planner (Algorithm 1) generates optimal demonstration trajectories under a single-trade constraint. A VQ encoder-decoder compresses these trajectories into K=10 discrete trading archetypes stored in a learnable codebook.

2. **Phase II — Archetype Selection**: A horizon-level RL agent (PPO-style Actor-Critic) selects the best archetype at the start of each 72-step trading horizon. A frozen decoder then generates step-by-step micro-actions from the selected archetype code.

3. **Phase III — Archetype Refinement**: A step-level RL agent fine-tunes the selected archetype's actions using a regret-aware reward signal, with at most one adjustment per horizon. Adaptive Layer Normalization (AdaLN) conditions the agent on archetype context.

```
Historical Data -> Feature Pipeline (45-dim) -> DP Planner -> 30k Trajectories
    -> VQ Encoder-Decoder (Phase I) -> Codebook (K=10 archetypes)
    -> Selection Agent (Phase II, PPO) -> Frozen Decoder -> Micro Actions
    -> Refinement Agent (Phase III, AdaLN) -> Final Trading Actions
    -> Evaluation Engine (TR, Sharpe, Calmar, Sortino, MDD, Volatility)
    -> Backtrader Cross-Validation + Trade Audit
```

## Project Structure

```
ArchetypeTrader/
├── data/
│   ├── ETH/                            # Per-pair feather data (train/val/test)
│   │   ├── df_train.feather
│   │   ├── df_val.feather
│   │   └── df_test.feather
│   └── feature_list/                   # Feature name references (.npy)
│       ├── single_features.npy
│       └── trend_features.npy
├── src/
│   ├── config.py                       # Global hyperparameters (dataclass + CLI)
│   ├── data/
│   │   ├── dataset.py                  # TrajectoryDataset (PyTorch Dataset)
│   │   └── feature_pipeline.py         # Feather data loading via polars
│   ├── env/
│   │   └── trading_env.py              # MDP trading environment with LOB slippage
│   ├── phase1/
│   │   ├── dp_planner.py              # Algorithm 1: Single-trade DP planner
│   │   ├── vq_encoder.py             # LSTM encoder with temporal attention pooling
│   │   ├── vq_decoder.py             # BiLSTM decoder with single-trade constraint
│   │   ├── codebook.py               # VQ codebook with dead-code reset & k-means init
│   │   ├── validation.py             # Phase I artifact validation
│   │   └── env_validation.py         # Phase I environment validation
│   ├── phase2/
│   │   └── selection_agent.py         # Actor-Critic archetype selector
│   ├── phase3/
│   │   ├── refinement_agent.py        # Step-level Actor-Critic with AdaLN
│   │   ├── policy_adapter.py          # Eq. 6: final action computation
│   │   ├── adaln.py                   # Adaptive Layer Normalization
│   │   └── regret_reward.py           # Eq. 8: regret-aware reward + top-5 DP
│   ├── evaluation/
│   │   ├── metrics.py                 # TR / AVOL / MDD / ASR / ACR / ASoR
│   │   ├── inference_runner.py        # Full three-phase inference loop
│   │   ├── model_loader.py            # Centralized model loading for all phases
│   │   ├── portfolio_tracker.py       # Cross-horizon portfolio & cash management
│   │   ├── trade_auditor.py           # Post-evaluation trade statistics & checks
│   │   └── bt_verifier.py            # Backtrader cross-validation engine
│   └── utils/
│       └── logger.py                  # Logging utilities
├── scripts/
│   ├── train_phase1.py                # Phase I: DP trajectories + VQ training
│   ├── train_phase2.py                # Phase II: PPO-style selection agent
│   ├── train_phase3.py                # Phase III: Regret-aware refinement agent
│   └── evaluate.py                    # Full three-phase evaluation on test set
├── tests/                             # Unit tests + property-based tests (Hypothesis)
├── docs/                              # Development logs
├── run_pipeline.sh                    # End-to-end train + evaluate pipeline script
├── requirements.txt                   # Python dependencies
└── result/                            # Artifacts: trajectories, checkpoints, evaluations
    └── {PAIR}/
        ├── dp_trajectories/
        ├── phase1_archetype_discovery/
        ├── phase2_archetype_selection/
        ├── phase3_archetype_refinement/
        └── evaluation/                # Metrics JSON + per-step CSV + audit reports
```

## Setup

```bash
conda create -n ArchetypeTrade python=3.12
conda activate ArchetypeTrade
pip install -r requirements.txt
pip install torch  # install PyTorch separately per your CUDA version
```

Dependencies (from `requirements.txt`):
- `pyarrow>=14.0.0` — feather file I/O
- `numpy>=1.24.0`
- `polars>=0.20.0` — high-performance DataFrame operations
- `tqdm>=4.64.0` — progress bars
- `torch>=2.0.0` — PyTorch (install separately)
- `pytest`, `hypothesis` — testing (optional)
- `backtrader` — cross-validation (optional, for `bt_verifier`)

## Usage

### Full Pipeline (Recommended)

Run all phases sequentially with logging:

```bash
bash run_pipeline.sh ETH
# Logs saved to logs/ETH_pipeline_YYYYMMDD_HHMMSS.log
```

### Individual Phases

Training runs sequentially — each phase depends on the previous one:

```bash
# Phase I: Generate DP trajectories + train VQ encoder-decoder
python scripts/train_phase1.py --pair ETH

# Phase II: Train archetype selection agent (PPO-style)
python scripts/train_phase2.py --pair ETH

# Phase III: Train refinement agent with regret-aware reward
python scripts/train_phase3.py --pair ETH --beta1 0.5

# Evaluate on test set (2024-01-01 to 2024-09-01)
python scripts/evaluate.py --pair ETH
```

### Key CLI Arguments

All optional; defaults defined in `src/config.py`:

| Argument | Default | Description |
|---|---|---|
| `--pair` | `ETH` | Trading pair (BTC/ETH/DOT/BNB) |
| `--horizon` | 72 | Steps per trading horizon |
| `--num-trajectories` | 30000 | DP demonstration trajectories |
| `--phase1-epochs` | 300 | VQ encoder-decoder training epochs |
| `--pretrain-epochs` | 10 | Phase A continuous latent pretraining epochs |
| `--phase2-total-steps` | 800000 | Selection agent training steps |
| `--phase3-total-steps` | 1000000 | Refinement agent training steps |
| `--beta1` | 0.5 | Regret coefficient beta1 in {0.3, 0.5, 0.7} |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Batch size |

## Evaluation Pipeline

The evaluation system goes beyond simple metric computation:

1. **Three-phase inference** (`inference_runner.py`): Runs frozen Phase I/II/III models sequentially on the test set, with cross-horizon portfolio tracking.
2. **Portfolio tracking** (`portfolio_tracker.py`): Manages cash, positions, average hold prices, and settlement across horizon boundaries.
3. **Trade audit** (`trade_auditor.py`): Computes detailed trade statistics (win rate, avg PnL, turnover) and runs consistency checks.
4. **Backtrader cross-validation** (`bt_verifier.py`): Replays the exact same trade signals through Backtrader as an independent verification engine, comparing position sequences and final PnL.
5. **CSV export**: Per-step operation logs exported in chunks for external analysis.

## Testing

```bash
python -m pytest tests/ -v
```

The test suite covers all components with property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/). Property-based tests verify formal correctness properties such as:
- Feature dimension invariants and concatenation preservation
- Position state invariant (P_t in {-m, 0, m})
- Reward formula correctness (Eq. 1)
- DP single-trade constraint and optimality (brute-force verified for small inputs)
- VQ nearest-neighbor quantization correctness
- At most one refinement adjustment per horizon
- Evaluation metric formulas (TR, AVOL, MDD, ASR, ACR, ASoR)
- Codebook collapse detection and dead-code reset behavior

## Key Hyperparameters

| Parameter | Code Default | Paper Value | Notes |
|---|---|---|---|
| State dim | 45 (36 + 9) | 45 | Matches paper |
| Action space | {short, flat, long} | {0, 1, 2} | Matches paper |
| Horizon h | 72 | 72 | Matches paper |
| Commission rate delta | 0.0004 | 0.0002 | **Differs** — see Deviations |
| Codebook size K | 10 | 10 | Matches paper |
| Latent dim | 16 | 16 | Matches paper |
| LSTM hidden dim | 128 | 128 | Matches paper |
| VQ commitment beta0 | 0.25 | 0.25 | Matches paper |
| KL penalty alpha | 1.0 | 1.0 | Matches paper |
| Regret beta1 | 0.5 | {0.3, 0.5, 0.7} | Matches paper |
| Phase I epochs | 300 | 100 | **Differs** — extended for convergence |
| Pretrain epochs | 10 | N/A | **Addition** — not in paper |
| Annualization factor | 52560 | 52560 | Matches paper |

## Supported Trading Pairs

| Pair | Code Max Position (m) | Paper Max Position (m) | Notes |
|---|---|---|---|
| BTC/USDT | 8 | 8 | Matches |
| ETH/USDT | 5 | 100 | **Differs** — see Deviations |
| DOT/USDT | 2500 | 2500 | Matches |
| BNB/USDT | 200 | 200 | Matches |


## Deviations from Paper

This section documents all known differences between the current codebase and the paper. These include engineering enhancements not described in the paper, intentional design changes, and known bugs.

### Phase I — Archetype Discovery

| Aspect | Paper | Code | Type |
|---|---|---|---|
| Encoder architecture | Standard LSTM, last hidden state projected to z_e | LSTM + **temporal attention pooling** over all hidden states, then projected to z_e | Enhancement |
| Decoder architecture | Unspecified LSTM directionality | **BiLSTM** (bidirectional), output dim = 2 * hidden_dim | Enhancement |
| Decoder inference | Not specified | **Single-trade constraint post-processing**: searches optimal single-change-point split over BiLSTM logits via prefix/suffix log-prob sums | Enhancement |
| Training strategy | End-to-end VQ training with Eq. (4) loss | **Two-stage training**: Phase A (pretrain_epochs=10, L_rec only, no VQ) followed by Phase B (full VQ loss). Phase A collects z_e samples for codebook initialization | Enhancement |
| Codebook initialization | Not discussed | **Direction-aware k-means**: groups trajectories by dominant direction (long/short/flat), runs k-means within each group, then assigns centroids to codebook entries | Enhancement |
| Codebook collapse | Not discussed | **Dead-code reset**: monitors per-code usage counts each epoch; resets unused codes by reinitializing from recent z_e samples with noise | Enhancement |
| Training epochs | 100 | 300 (default) | Config change |

### Phase II — Archetype Selection

| Aspect | Paper | Code | Type |
|---|---|---|---|
| Objective function | Eq. (5): environment reward + alpha * KL(ground-truth label \|\| policy) | **PPO-style**: clipped surrogate objective + value loss + entropy bonus + KL penalty as additional term | Enhancement |
| Action selection (inference) | Sample from policy distribution | **Greedy** (argmax over action_probs) in Phase III training and evaluation scripts | Design change |
| Network architecture | Not specified | Two-layer MLP (128 -> 64) + ReLU, Actor-Critic with shared backbone, separate policy head (Linear -> Softmax) and value head (Linear) | Implementation detail |

### Phase III — Archetype Refinement

| Aspect | Paper | Code | Type |
|---|---|---|---|
| tau_remain definition | tau_remain = t + h - tau (absolute remaining steps) | Training: `float(h - step_idx)` (absolute). Inference: `(h - step_idx) / h` (normalized to [0,1]). **Train/inference mismatch.** | Bug |
| Objective function | Eq. (9): J' = E[sum gamma^tau * r_ref - beta2 * L(a_hat_ref, pi_ref)] | `loss = policy_loss + value_loss + beta2 * ce_loss`, where policy_loss = -log_prob * advantage (Actor-Critic), value_loss = MSE(V, G). Paper does not mention value loss term explicitly | Enhancement |
| RL episode termination | Terminates when adapter chooses non-zero action | Consistent with paper. Remaining steps executed with base actions to compute full horizon return R for Eq. (8) | Matches |

### Global / Config

| Aspect | Paper | Code | Type |
|---|---|---|---|
| Commission rate delta | 0.0002 (0.02%) | `config.py` default = 0.0004 (0.04%). `TradingEnv` class default = 0.0002. **Inconsistency between config and env class.** | Bug |
| ETH max position | 100 | 5 (in `config.py`) | Config mismatch |
| LOB slippage | Mentioned as execution loss O_t, no implementation detail | Full 5-level LOB walk implementation: walks ask/bid book levels, fills at each price, computes slippage = fill_cash - abs_delta * mark_price. Handles partial fills and direct flips (long->short split into close+open) | Enhancement |
| Evaluation infrastructure | Not discussed | Full evaluation pipeline: PortfolioTracker (cross-horizon cash/position management), TradeAuditor (statistics + consistency checks), BacktraderVerifier (independent cross-validation via Backtrader replay) | Enhancement |
| Trajectory caching | Not discussed | DP trajectories cached as `.npz` with full metadata (pair, horizon, gamma, seed, data shape). Incompatible caches auto-detected and backed up | Enhancement |

### Summary of Known Bugs

1. **tau_remain normalization mismatch**: Training uses absolute step count `float(h - step_idx)`, but inference normalizes to `(h - step_idx) / h`. This means the refinement agent sees different input distributions at train vs. test time.
2. **Commission rate inconsistency**: `Config.commission_rate` defaults to 0.0004, but `TradingEnv.COMMISSION_RATE` class constant is 0.0002. Which value is used depends on whether `commission_rate` is passed to the env constructor.
3. **ETH max position**: `Config.max_positions["ETH"]` is set to 5, while the paper specifies 100.

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
