# ArchetypeTrader

A PyTorch implementation of the three-phase reinforcement learning framework for cryptocurrency trading, based on the paper submitted to **AAAI 2026**. The system discovers reusable trading archetypes from historical data via dynamic programming and vector quantization, then deploys them through hierarchical RL agents for real-time trading.

> **Paper:** *ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading* [[PDF]](AAAI26_ArchetypeTrader.pdf)
> Chuqiao Zong, Molei Qin, Haochong Xia, Bo An — Nanyang Technological University, Singapore
>
> This codebase is generated from the above research paper (AAAI 2026). Code comments reference specific sections, equations, and algorithms from the paper for traceability. Where the paper is ambiguous or lacks implementation details, the code includes `[NOTE]` annotations.

## Overview

ArchetypeTrader operates on 10-minute cryptocurrency bars (BTC/ETH/DOT/BNB vs USDT) with 25-level limit order book (LOB) data. It follows a three-phase pipeline:

1. **Phase I — Archetype Discovery**: A dynamic programming planner (Algorithm 1) generates optimal demonstration trajectories under a single-trade constraint. A VQ encoder-decoder compresses these trajectories into K=10 discrete trading archetypes stored in a learnable codebook.

2. **Phase II — Archetype Selection**: A horizon-level RL agent selects the best archetype at the start of each 72-step trading horizon. A frozen decoder then generates step-by-step micro-actions from the selected archetype code.

3. **Phase III — Archetype Refinement**: A step-level RL agent fine-tunes the selected archetype's actions using a regret-aware reward signal, with at most one adjustment per horizon. Adaptive Layer Normalization (AdaLN) conditions the agent on market context.

```
Historical Data → Feature Pipeline → DP Planner → 30k Trajectories
    → VQ Encoder-Decoder (Phase I) → Codebook (K=10 archetypes)
    → Selection Agent (Phase II) → Frozen Decoder → Micro Actions
    → Refinement Agent (Phase III) → Final Trading Actions
    → Evaluation Engine (TR, Sharpe, Calmar, Sortino, MDD, Volatility)
```

## Project Structure

```
ArchetypeTrader/
├── data/feature_list/          # Market feature data (.npy)
│   ├── single_features.npy     # 36-dim per-step features (LOB, OHLCV, technicals)
│   └── trend_features.npy      # 9-dim 60-period trend indicators
├── src/
│   ├── config.py               # Global hyperparameters (dataclass + CLI override)
│   ├── data/                   # Feature pipeline & PyTorch dataset
│   ├── env/                    # MDP trading environment
│   ├── phase1/                 # DP planner, VQ encoder, decoder, codebook
│   ├── phase2/                 # Selection agent (horizon-level RL)
│   ├── phase3/                 # Refinement agent, policy adapter, AdaLN
│   ├── evaluation/             # Metrics engine (TR/AVOL/MDD/ASR/ACR/ASoR)
│   └── utils/                  # Logger
├── scripts/
│   ├── train_phase1.py         # Phase I: DP trajectories + VQ training
│   ├── train_phase2.py         # Phase II: Selection agent training
│   ├── train_phase3.py         # Phase III: Refinement agent training
│   └── evaluate.py             # Full three-phase evaluation on test set
├── tests/                      # Unit tests + property-based tests (278 tests)
└── result/                     # Artifacts: trajectories, checkpoints, evaluations
```

## Setup

```bash
conda create -n ArchetypeTrade python=3.12
conda activate ArchetypeTrade
pip install torch numpy pytest hypothesis
```

## Usage

Training runs sequentially — each phase depends on the previous one:

```bash
# Phase I: Generate DP trajectories + train VQ encoder-decoder
python scripts/train_phase1.py --pair BTC

# Phase II: Train archetype selection agent
python scripts/train_phase2.py --pair BTC

# Phase III: Train refinement agent with regret-aware reward
python scripts/train_phase3.py --pair BTC --beta1 0.5

# Evaluate on test set (2024-01-01 to 2024-09-01)
python scripts/evaluate.py --pair BTC
```

Key CLI arguments (all optional, defaults in `src/config.py`):

| Argument | Default | Description |
|---|---|---|
| `--pair` | all 4 pairs | Trading pair (BTC/ETH/DOT/BNB) |
| `--horizon` | 72 | Steps per trading horizon |
| `--num-trajectories` | 30000 | DP demonstration trajectories |
| `--phase1-epochs` | 100 | VQ encoder-decoder training epochs |
| `--phase2-total-steps` | 3000000 | Selection agent training steps |
| `--phase3-total-steps` | 1000000 | Refinement agent training steps |
| `--beta1` | 0.5 | Regret coefficient β₁ ∈ {0.3, 0.5, 0.7} |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Batch size |

## Testing

The test suite includes 278 tests: unit tests for all components plus 23 property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/) (100 examples each).

```bash
python -m pytest tests/ -v
```

Property-based tests verify formal correctness properties such as:
- Feature dimension invariants and concatenation preservation
- Position state invariant (P_t ∈ {-m, 0, m})
- Reward formula correctness (Eq. 1)
- DP single-trade constraint and optimality (brute-force verified for small inputs)
- VQ nearest-neighbor quantization correctness
- At most one refinement adjustment per horizon
- Evaluation metric formulas (TR, AVOL, MDD, ASR, ACR, ASoR)

## Key Hyperparameters

| Parameter | Value | Paper Reference |
|---|---|---|
| State dim | 45 (36 single + 9 trend) | Section 3.1 |
| Action space | {short, flat, long} | Section 3.1 |
| Horizon h | 72 steps | Section 3.1 |
| Commission rate δ | 0.02% | Section 3.1 |
| Codebook size K | 10 archetypes | Section 4.1 |
| Latent dim | 16 | Section 4.1 |
| LSTM hidden dim | 128 | Section 4.1 |
| VQ commitment β₀ | 0.25 | Section 4.1 |
| KL penalty α | 1.0 | Section 4.2 |
| Regret β₁ | {0.3, 0.5, 0.7} | Section 4.3 |
| Annualization factor m | 52560 | Section 5 |

## Supported Trading Pairs

| Pair | Max Position (m) |
|---|---|
| BTC/USDT | 8 |
| ETH/USDT | 100 |
| DOT/USDT | 2500 |
| BNB/USDT | 200 |

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

<!-- TODO: Add arXiv / conference URL once available -->

## License

This project is an academic implementation for research purposes.
