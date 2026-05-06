"""Replay Phase I KL/demo labels through the frozen Phase I policy.

This is a local diagnostic test script for questions like:

    Do non_overlap_horizon_labels_train.feather labels actually make money
    when Phase II executes them via the frozen Phase I decoder?

It intentionally depends on local artifacts and is not meant for regular CI.
Run it with ``--assert-positive`` when you want a test-style non-zero exit if
the decoded KL/demo replay is not profitable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from src.config.phase2_config import (  # noqa: E402
    Phase1ArtifactsConfig,
    Phase2Config,
)
from src.preprocess_data.processed_store import Phase1ProcessedStore  # noqa: E402
from src.data.phase2_dataset import Phase2Dataset  # noqa: E402
from src.data.phase2_label_loader import Phase2LabelLoader  # noqa: E402
from src.models.phase1_frozen_policy import Phase1FrozenPolicy  # noqa: E402
from src.trading.cost_model import LobDepthCostModel  # noqa: E402
from src.trading.env import TradingEnv  # noqa: E402
from src.trading.reward_alignment import RewardAlignment  # noqa: E402
from src.utils.feather_io import atomic_write_json, read_json, write_ipc  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Phase I non-overlap KL/demo labels with frozen decoder and "
            "write per-step profitability diagnostics."
        )
    )
    parser.add_argument("--pair", default="FU")
    parser.add_argument("--phase1-batch-id", default="batch_03")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: artifacts/{pair}/{phase1_batch_id}/phase1/"
            "kl_demo_profit_diagnostic"
        ),
    )
    parser.add_argument(
        "--max-horizons",
        type=int,
        default=None,
        help="Limit replay horizon count for smoke checks.",
    )
    parser.add_argument(
        "--assert-positive",
        action="store_true",
        help="Exit with status 1 if decoded KL/demo total reward is <= 0.",
    )
    return parser


def _load_phase1_runtime_config(phase1_dir: Path) -> tuple[int, int, dict[str, Any], str]:
    config_path = phase1_dir / "phase1_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Phase I config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        phase1_config = yaml.safe_load(f) or {}

    dp_config = phase1_config.get("dp", {}) or {}
    cost_config = dp_config.get("cost_config", {}) or {}
    horizon = int(dp_config.get("horizon", 72))
    max_position = int(dp_config.get("max_position", 1))
    reward_alignment = str(cost_config.get("reward_alignment", "paper_formula"))
    return horizon, max_position, dict(cost_config), reward_alignment


def _make_phase2_config(
    *,
    pair: str,
    phase1_batch_id: str,
    artifact_root: str,
    horizon: int,
    max_position: int,
    device: str,
) -> Phase2Config:
    return Phase2Config(
        pair=pair,
        phase1_batch_id=phase1_batch_id,
        phase2_batch_id=f"{phase1_batch_id}_kl_demo_profit_diagnostic",
        train_file="",
        val_file="",
        test_file="",
        artifact_root=artifact_root,
        horizon=horizon,
        max_position=max_position,
        device=device,
        phase1_artifacts=Phase1ArtifactsConfig(
            artifact_root=artifact_root,
            pair=pair,
            phase1_batch_id=phase1_batch_id,
        ),
    )


def replay_kl_demo_labels(
    *,
    config: Phase2Config,
    split: str,
    cost_config: dict[str, Any],
    reward_alignment: str,
    max_horizons: Optional[int],
) -> dict[str, Any]:
    phase1_dir = config.phase1_dir()
    manifest_path = phase1_dir / "data_process_manifest.json"
    label_path = phase1_dir / f"non_overlap_horizon_labels_{split}.feather"

    for required in (
        manifest_path,
        label_path,
        phase1_dir / f"non_overlap_horizons_{split}.feather",
        phase1_dir / f"non_overlap_dp_teacher_{split}.feather",
        phase1_dir / "decoder.pt",
        phase1_dir / "codebook.pt",
        phase1_dir / "input_schema.json",
    ):
        if not required.exists():
            raise FileNotFoundError(f"missing required artifact: {required}")

    store = Phase1ProcessedStore(phase1_dir)
    records = store.load_non_overlap_records(manifest_path, split)
    if max_horizons is not None:
        records = records[: max(0, int(max_horizons))]

    input_schema = read_json(phase1_dir / "input_schema.json")
    dataset = Phase2Dataset.from_phase1_records(
        records,
        input_schema,
        config,
        reward_alignment=reward_alignment,
    )
    dataset.horizon_entries = Phase2LabelLoader(config).load_and_join(
        dataset.horizon_entries,
        split,
        label_path,
    )

    frozen_policy = Phase1FrozenPolicy.load(
        phase1_dir / "decoder.pt",
        phase1_dir / "codebook.pt",
        device=config.device,
    )
    cost_model = LobDepthCostModel(
        commission_rate=float(cost_config.get("commission_rate", 0.0002)),
        book_levels=int(cost_config.get("book_levels", 5)),
        insufficient_depth_policy=str(
            cost_config.get("insufficient_depth_policy", "reject_transition")
        ),
    )
    trading_env = TradingEnv(
        cost_model=cost_model,
        reward_alignment=RewardAlignment(reward_alignment),
        max_position=config.max_position,
    )

    step_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    prev_terminal_position = 0
    total_reward = 0.0
    total_cost = 0.0
    total_boundary_cost = 0.0
    total_teacher_return = 0.0
    action_matches = 0
    action_compared = 0
    label_counts: Counter[int] = Counter()
    decoded_action_counts: Counter[int] = Counter()

    for horizon_idx, entry in enumerate(dataset.horizon_entries):
        if not entry.is_labeled or entry.code_label is None:
            continue
        code_label = int(entry.code_label)
        label_counts[code_label] += 1

        horizon_states = dataset.get_horizon_states(horizon_idx)
        horizon_inputs = dataset.get_horizon_inputs(horizon_idx)
        frozen_policy.reset(code_id=code_label)
        decoded_actions: list[int] = []
        for state_t in horizon_states[: config.horizon]:
            out = frozen_policy.decode_step(state_t)
            decoded_actions.append(int(out.action))

        init_pos = (
            prev_terminal_position
            if config.horizon_schedule.position_continuity
            else 0
        )
        trading_env.reset(horizon_inputs, initial_position=init_pos)
        rewards, infos = trading_env.replay(decoded_actions)

        teacher_actions = list(records[horizon_idx].actions or [])
        teacher_rewards = list(records[horizon_idx].rewards or [])
        teacher_return = float(sum(teacher_rewards))
        horizon_reward = float(sum(rewards))
        cost_paid = float(sum(info.fee + info.slippage for info in infos))
        boundary_cost = float(infos[0].fee + infos[0].slippage) if infos else 0.0
        final_position = int(infos[-1].filled_position) if infos else init_pos

        prev_terminal_position = final_position
        total_reward += horizon_reward
        total_cost += cost_paid
        total_boundary_cost += boundary_cost
        total_teacher_return += teacher_return

        compared_this_horizon = 0
        matches_this_horizon = 0
        for t, action in enumerate(decoded_actions[: len(rewards)]):
            decoded_action_counts[int(action)] += 1
            teacher_action = (
                int(teacher_actions[t]) if t < len(teacher_actions) else None
            )
            teacher_reward = (
                float(teacher_rewards[t]) if t < len(teacher_rewards) else None
            )
            if teacher_action is not None:
                compared_this_horizon += 1
                action_compared += 1
                if int(action) == teacher_action:
                    matches_this_horizon += 1
                    action_matches += 1
            info = infos[t]
            step_rows.append(
                {
                    "split": split,
                    "horizon_index": horizon_idx,
                    "sample_id": entry.sample_id,
                    "phase1_sample_id": entry.phase1_sample_id,
                    "start_index": int(entry.horizon_start),
                    "t": t,
                    "code_label": code_label,
                    "decoded_action": int(action),
                    "teacher_action": teacher_action,
                    "action_match_teacher": (
                        bool(int(action) == teacher_action)
                        if teacher_action is not None
                        else None
                    ),
                    "step_reward": float(rewards[t]),
                    "teacher_step_reward": teacher_reward,
                    "fee": float(info.fee),
                    "slippage": float(info.slippage),
                    "filled_position": int(info.filled_position),
                    "nav": float(info.nav),
                    "execution_row": int(info.execution_row),
                    "markout_row": int(info.markout_row),
                    "rejected": bool(info.rejected),
                }
            )

        horizon_rows.append(
            {
                "split": split,
                "horizon_index": horizon_idx,
                "sample_id": entry.sample_id,
                "phase1_sample_id": entry.phase1_sample_id,
                "start_index": int(entry.horizon_start),
                "end_index": int(entry.horizon_end),
                "code_label": code_label,
                "initial_position": int(init_pos),
                "final_position": final_position,
                "decoded_horizon_reward": horizon_reward,
                "teacher_return": teacher_return,
                "reward_minus_teacher": horizon_reward - teacher_return,
                "cost_paid": cost_paid,
                "boundary_cost": boundary_cost,
                "num_steps": len(rewards),
                "decoded_action_match_teacher_ratio": (
                    matches_this_horizon / compared_this_horizon
                    if compared_this_horizon
                    else None
                ),
            }
        )

    num_horizons = len(horizon_rows)
    num_steps = len(step_rows)
    summary = {
        "pair": config.pair,
        "phase1_batch_id": config.phase1_batch_id,
        "split": split,
        "phase1_dir": str(phase1_dir),
        "horizon": config.horizon,
        "max_position": config.max_position,
        "reward_alignment": reward_alignment,
        "cost_config": cost_config,
        "num_horizons_total": len(dataset.horizon_entries),
        "num_labeled_horizons_replayed": num_horizons,
        "num_steps_replayed": num_steps,
        "decoded_kl_demo_total_reward": total_reward,
        "decoded_kl_demo_avg_reward_per_horizon": (
            total_reward / num_horizons if num_horizons else 0.0
        ),
        "decoded_kl_demo_avg_reward_per_step": (
            total_reward / num_steps if num_steps else 0.0
        ),
        "decoded_kl_demo_total_cost": total_cost,
        "decoded_kl_demo_total_boundary_cost": total_boundary_cost,
        "dp_teacher_total_return": total_teacher_return,
        "decoded_minus_dp_teacher_total": total_reward - total_teacher_return,
        "profitable": total_reward > 0,
        "positive_horizon_ratio": (
            sum(1 for row in horizon_rows if row["decoded_horizon_reward"] > 0)
            / num_horizons
            if num_horizons
            else 0.0
        ),
        "decoded_action_match_teacher_ratio": (
            action_matches / action_compared if action_compared else 0.0
        ),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "decoded_action_counts": {
            str(k): int(v) for k, v in sorted(decoded_action_counts.items())
        },
    }
    return {
        "summary": summary,
        "horizon_rows": horizon_rows,
        "step_rows": step_rows,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path, split: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"kl_demo_{split}_summary.json"
    horizon_path = output_dir / f"kl_demo_{split}_horizon_rewards.feather"
    step_path = output_dir / f"kl_demo_{split}_step_rewards.feather"

    summary = dict(payload["summary"])
    summary["output_files"] = {
        "summary": str(summary_path),
        "horizon_rewards": str(horizon_path),
        "step_rewards": str(step_path),
    }
    atomic_write_json(summary, summary_path)
    write_ipc(pl.DataFrame(payload["horizon_rows"]), horizon_path)
    write_ipc(pl.DataFrame(payload["step_rows"]), step_path)
    return summary["output_files"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    phase1_dir = (
        Path(args.artifact_root)
        / args.pair
        / args.phase1_batch_id
        / "phase1"
    )
    horizon, max_position, cost_config, reward_alignment = _load_phase1_runtime_config(
        phase1_dir
    )
    config = _make_phase2_config(
        pair=args.pair,
        phase1_batch_id=args.phase1_batch_id,
        artifact_root=args.artifact_root,
        horizon=horizon,
        max_position=max_position,
        device=args.device,
    )
    payload = replay_kl_demo_labels(
        config=config,
        split=args.split,
        cost_config=cost_config,
        reward_alignment=reward_alignment,
        max_horizons=args.max_horizons,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else phase1_dir / "kl_demo_profit_diagnostic"
    )
    output_files = write_outputs(payload, output_dir, args.split)
    summary = payload["summary"]

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("output_files=" + json.dumps(output_files, ensure_ascii=False, sort_keys=True))

    if args.assert_positive and summary["decoded_kl_demo_total_reward"] <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
