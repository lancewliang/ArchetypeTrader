"""Shared lightweight helpers for Phase II tests."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import polars as pl
import torch
import yaml

from src.config.phase2_config import (
    Phase1ArtifactsConfig,
    Phase2Config,
    Phase2SelectionPolicyConfig,
    PPOConfig,
    RollingValidationConfig,
    SelectorNetworkConfig,
)
from src.data.phase2_dataset import Phase2Dataset
from src.data.phase2_horizon_index import Phase2HorizonEntry
from src.models.archetype_selector import ArchetypeSelector
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.models.vq_archetype import ArchetypeDecoder
from src.rl.actor_critic import ActorCritic
from src.trainers.phase2_trainer import Phase2Trainer, Phase2TrainerArtifacts
from src.trading.cost_model import LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.reward_alignment import RewardAlignment


FEATURE_COLUMNS = ["feature_return_1", "feature_vol_4", "feature_momentum_8"]


def make_market_frame(num_rows: int = 32, *, start: float = 100.0) -> pl.DataFrame:
    """Create a deterministic in-memory market frame with five LOB levels."""
    prices = np.linspace(start, start + num_rows - 1, num_rows, dtype=np.float32)
    data = {
        "timestamp": list(range(num_rows)),
        "close": prices.tolist(),
        "feature_return_1": np.linspace(0.0, 0.01, num_rows).tolist(),
        "feature_vol_4": np.linspace(0.1, 0.2, num_rows).tolist(),
        "feature_momentum_8": np.linspace(-0.02, 0.02, num_rows).tolist(),
    }
    for level in range(1, 6):
        data[f"ask{level}_price"] = (prices + 0.01 * level).tolist()
        data[f"ask{level}_size"] = [100.0] * num_rows
        data[f"bid{level}_price"] = (prices - 0.01 * level).tolist()
        data[f"bid{level}_size"] = [100.0] * num_rows
    return pl.DataFrame(data)


def input_schema() -> dict:
    return {"feature_columns": list(FEATURE_COLUMNS), "price_column": "close"}


def make_entries(
    count: int = 4,
    *,
    horizon: int = 4,
    split: str = "train",
    labeled: bool = False,
    gap_bars: int = 0,
) -> List[Phase2HorizonEntry]:
    entries: List[Phase2HorizonEntry] = []
    for i in range(count):
        start = i * horizon
        entries.append(
            Phase2HorizonEntry(
                sample_id=f"{split}_{i}",
                horizon_start=start,
                horizon_end=start + horizon - 1,
                split=split,
                is_gap=gap_bars > 0,
                gap_bars=gap_bars,
                code_label=(i % 3 if labeled else None),
                is_labeled=labeled,
                timestamp_start=str(start),
            )
        )
    return entries


def write_phase1_artifacts(
    artifact_root: Path,
    *,
    pair: str = "TEST",
    phase1_batch_id: str = "smoke_phase1",
    horizon: int = 4,
    num_codes: int = 3,
    feature_dim: int = 3,
    code_dim: int = 4,
    hidden_dim: int = 8,
) -> Path:
    """Write a minimal, valid Phase I artifact directory for Phase II tests."""
    p1_dir = Path(artifact_root) / pair / phase1_batch_id / "phase1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    decoder = ArchetypeDecoder(feature_dim, code_dim, hidden_dim)
    torch.save(decoder.state_dict(), p1_dir / "decoder.pt")
    torch.save(torch.randn(num_codes, code_dim) * 0.05, p1_dir / "codebook.pt")
    (p1_dir / "input_schema.json").write_text(json.dumps(input_schema()), encoding="utf-8")
    (p1_dir / "feature_provenance.json").write_text("{}", encoding="utf-8")
    (p1_dir / "checkpoint_manifest.json").write_text("[]", encoding="utf-8")
    phase1_config = {
        "horizon": horizon,
        "dp": {
            "max_position": 1,
            "cost_config": {
                "reward_alignment": "paper_formula",
                "commission_rate": 0.0,
                "book_levels": 5,
                "insufficient_depth_policy": "reject_transition",
            },
        },
    }
    (p1_dir / "phase1_config.yaml").write_text(
        yaml.safe_dump(phase1_config), encoding="utf-8"
    )
    report = {
        "fatal_collapse": False,
        "code_assignment_drift_warning": False,
        "hindsight_bias_warning": "ok",
        "config_hash": "test_hash",
        "code_usage": {"counts": [10] * num_codes},
    }
    (p1_dir / "phase1_report.json").write_text(json.dumps(report), encoding="utf-8")
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        pl.DataFrame(
            {
                "sample_id": [f"p1_{split}_{i}" for i in range(count)],
                "start_index": [i * horizon for i in range(count)],
                "code_label": [i % num_codes for i in range(count)],
            }
        ).write_ipc(p1_dir / f"horizon_labels_{split}.feather")
    return p1_dir


def make_config(
    tmp_path: Path,
    *,
    horizon: int = 4,
    num_envs: int = 1,
    rollout_length: int = 2,
    total_timesteps: int = 4,
    phase2_batch_id: str = "smoke_phase2",
) -> Phase2Config:
    artifact_root = Path(tmp_path) / "artifacts"
    write_phase1_artifacts(artifact_root, horizon=horizon)
    return Phase2Config(
        pair="TEST",
        phase1_batch_id="smoke_phase1",
        phase2_batch_id=phase2_batch_id,
        artifact_root=str(artifact_root),
        horizon=horizon,
        num_envs=num_envs,
        rollout_length=rollout_length,
        total_timesteps=total_timesteps,
        device="cpu",
        fast_eval_max_horizons=2,
        selector_network=SelectorNetworkConfig(hidden_dims=[8], use_layer_norm=False),
        ppo=PPOConfig(update_epochs=1, minibatch_size=2, target_kl=None),
        selection_policy=Phase2SelectionPolicyConfig(
            max_drawdown=999.0,
            min_sharpe=-1_000_000.0,
            max_turnover_ratio=999.0,
            max_action_dominance_ratio=1.0,
            min_active_archetype_ratio=0.0,
            max_fold_volatility=999.0,
        ),
        rolling_validation=RollingValidationConfig(
            enabled=True,
            num_folds=2,
            max_fold_volatility=1_000_000_000_000_000.0,
        ),
        phase1_artifacts=Phase1ArtifactsConfig(
            artifact_root=str(artifact_root),
            pair="TEST",
            phase1_batch_id="smoke_phase1",
        ),
    )


def make_dataset(
    config: Phase2Config,
    *,
    split: str = "train",
    count: int = 4,
    labeled: bool = False,
    gap_bars: int = 0,
) -> Phase2Dataset:
    frame = make_market_frame(max(count * config.horizon + 2, config.horizon + 2))
    entries = make_entries(
        count,
        horizon=config.horizon,
        split=split,
        labeled=labeled,
        gap_bars=gap_bars,
    )
    return Phase2Dataset(frame, entries, input_schema(), config, reward_alignment="paper_formula")


def make_frozen_policy(num_codes: int = 3) -> Phase1FrozenPolicy:
    decoder = ArchetypeDecoder(feature_dim=3, code_dim=4, hidden_dim=8)
    codebook = torch.randn(num_codes, 4) * 0.05
    return Phase1FrozenPolicy(decoder, codebook, device="cpu")


def make_trading_env(*, commission_rate: float = 0.0, slippage_multiplier: float = 1.0) -> TradingEnv:
    return TradingEnv(
        cost_model=LobDepthCostModel(
            commission_rate=commission_rate,
            book_levels=5,
            slippage_multiplier=slippage_multiplier,
        ),
        reward_alignment=RewardAlignment("paper_formula"),
        max_position=1,
    )


def make_actor_critic(state_dim: int = 4, num_codes: int = 3) -> ActorCritic:
    selector = ArchetypeSelector(
        state_dim=state_dim,
        num_codes=num_codes,
        config=SelectorNetworkConfig(hidden_dims=[8], use_layer_norm=False),
    )
    return ActorCritic(selector)


def bias_actor_to_code(actor_critic: ActorCritic, code_id: int) -> None:
    """Make deterministic actor output choose one code."""
    with torch.no_grad():
        for param in actor_critic.selector.parameters():
            param.zero_()
        actor_critic.selector.actor_head.bias[code_id] = 10.0


def write_market_splits(base_dir: Path) -> tuple[Path, Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        base_dir / "market_train.feather",
        base_dir / "market_val.feather",
        base_dir / "market_test.feather",
    )
    for path, rows, start in zip(paths, [36, 24, 24], [100.0, 200.0, 300.0]):
        make_market_frame(rows, start=start).write_ipc(path)
    return paths


def run_smoke_phase2_training(tmp_path: Path, *, phase2_batch_id: str = "smoke_phase2") -> Phase2TrainerArtifacts:
    """Run a tiny real Phase II training flow and return artifacts."""
    train, val, _test = write_market_splits(Path(tmp_path) / "market")
    config = make_config(
        tmp_path,
        horizon=4,
        num_envs=1,
        rollout_length=2,
        total_timesteps=4,
        phase2_batch_id=phase2_batch_id,
    )
    config = replace(
        config,
        train_file=str(train),
        val_file=str(val),
    )
    return Phase2Trainer(config).run()
