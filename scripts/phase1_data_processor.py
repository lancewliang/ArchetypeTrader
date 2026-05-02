"""Phase I offline data processor.

This command generates sampled horizon artifacts and DP teacher labels for
manifest-mode Phase I training. Training should load the manifest produced here
instead of re-reading raw market data or re-running DP.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.phase1_config import (  # noqa: E402
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    Phase1Config,
    Phase1DataProcessConfig,
    SamplingHealthConfig,
    StratificationConfig,
    TrainingConfig,
)
from src.data.data_augmentation import TemporalContrastiveBuilder  # noqa: E402
from src.data.demo_store import Phase1DemoStore  # noqa: E402
from src.data.horizon_builder import HorizonBuilder  # noqa: E402
from src.data.market_reader import MarketFileReader  # noqa: E402
from src.data.phase1_processed_store import (  # noqa: E402
    Phase1ProcessedStore,
    stable_hash,
)
from src.trainers.phase1_trainer import Phase1FatalError, Phase1Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase I 数据预处理入口")
    p.add_argument("--pair", required=True)
    p.add_argument("--data-batch-id", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--factor-profile", default="short")
    p.add_argument("--factor-list-file", default=None)
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--num-demos", type=int, default=30000)
    p.add_argument(
        "--sampling-strategy",
        choices=["stratified_uniform", "stratified_proportional"],
        default="stratified_uniform",
    )
    p.add_argument(
        "--stratification-mode",
        choices=["hindsight_horizon", "prospective_past"],
        default="hindsight_horizon",
    )
    p.add_argument("--diagnostic-pair-batch-id", default=None)
    p.add_argument("--allow-missing-prospective-diagnostic", action="store_true")
    p.add_argument("--risk-acknowledged-by", default=None)
    p.add_argument("--expected-sign-off-followup-batch-id", default=None)
    p.add_argument("--prospective-lookback-minutes", type=int, default=1440)
    p.add_argument("--sampling-min-gap-between-samples", type=int, default=None)
    p.add_argument("--sampling-max-overlap-ratio", type=float, default=None)
    p.add_argument("--sampling-flat-low-vol-max-ratio", type=float, default=None)
    p.add_argument("--split-boundary-embargo", type=int, default=None)
    p.add_argument("--next-row-split-boundary-embargo", type=int, default=None)
    p.add_argument("--sampling-health-warn-only", action="store_true")
    p.add_argument("--sampling-allow-overlap-relaxation", action="store_true")
    p.add_argument(
        "--reward-alignment",
        choices=["paper_formula", "next_row_execution"],
        default="paper_formula",
    )
    p.add_argument("--max-position", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--local-smoke-relaxed-guardrails",
        action="store_true",
        help="仅用于本地小样本 smoke: 放宽采样 guardrail，生产实验不要使用。",
    )
    return p


def build_data_process_config(args: argparse.Namespace) -> Phase1DataProcessConfig:
    cost = CostConfig(reward_alignment=args.reward_alignment)
    dp = DPConfig(horizon=args.horizon, cost_config=cost, max_position=args.max_position)
    strat = StratificationConfig(
        mode=args.stratification_mode,
        prospective_lookback_minutes=args.prospective_lookback_minutes,
        diagnostic_pair_batch_id=args.diagnostic_pair_batch_id,
    )
    sampling_health = SamplingHealthConfig()
    overrides = {}
    if args.sampling_min_gap_between_samples is not None:
        overrides["min_gap_between_samples"] = args.sampling_min_gap_between_samples
    if args.sampling_max_overlap_ratio is not None:
        overrides["max_overlap_ratio"] = args.sampling_max_overlap_ratio
    if args.sampling_flat_low_vol_max_ratio is not None:
        overrides["flat_low_vol_max_ratio"] = args.sampling_flat_low_vol_max_ratio
    if args.split_boundary_embargo is not None:
        overrides["split_boundary_embargo"] = args.split_boundary_embargo
    if args.next_row_split_boundary_embargo is not None:
        overrides["next_row_split_boundary_embargo"] = args.next_row_split_boundary_embargo
    if args.sampling_health_warn_only:
        overrides["warn_only"] = True
    if args.sampling_allow_overlap_relaxation:
        overrides["allow_overlap_relaxation"] = True
    if overrides:
        sampling_health = replace(sampling_health, **overrides)
    if args.local_smoke_relaxed_guardrails:
        sampling_health = SamplingHealthConfig(
            max_no_trade_ratio=1.0,
            flat_low_vol_max_ratio=1.0,
            min_gap_between_samples=1,
            max_overlap_ratio=1.0,
            split_boundary_embargo=0,
            next_row_split_boundary_embargo=0,
            warn_only=True,
        )
    return Phase1DataProcessConfig(
        pair=args.pair,
        data_batch_id=args.data_batch_id,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        artifact_root=args.artifact_root,
        factor_profile=args.factor_profile,
        factor_list_file=args.factor_list_file,
        horizon=args.horizon,
        num_demos=args.num_demos,
        sampling_strategy=args.sampling_strategy,
        stratification=strat,
        sampling_health=sampling_health,
        data_augmentation=DataAugmentationConfig(),
        dp=dp,
        seed=args.seed,
        allow_missing_prospective_diagnostic=args.allow_missing_prospective_diagnostic,
        risk_acknowledged_by=args.risk_acknowledged_by,
        expected_sign_off_followup_batch_id=args.expected_sign_off_followup_batch_id,
    )


def assert_prospective_diagnostic(args: argparse.Namespace) -> None:
    if args.stratification_mode == "prospective_past":
        return
    if args.diagnostic_pair_batch_id:
        return
    if args.allow_missing_prospective_diagnostic:
        if not args.risk_acknowledged_by or not args.expected_sign_off_followup_batch_id:
            print(
                "[error] allow_missing_prospective_diagnostic 需配套 "
                "--risk-acknowledged-by + --expected-sign-off-followup-batch-id",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return
    print(
        "[error] hindsight 主实验缺少 --diagnostic-pair-batch-id; "
        "请同时配套一个 prospective_past BATCH_ID 或显式风险声明。",
        file=sys.stderr,
    )
    raise SystemExit(2)


class Phase1DataProcessor:
    """Offline processor that writes Phase I manifest-mode training inputs."""

    def __init__(self, config: Phase1DataProcessConfig) -> None:
        self.config = config
        self.trainer = Phase1Trainer(self._trainer_config())

    def run(self) -> Path:
        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.trainer._seed_everything()
        self.trainer._check_prospective_diagnostic()

        frames = MarketFileReader().read_split(
            self.config.train_file, self.config.val_file, self.config.test_file
        )
        schema_validator = self.trainer._build_schema_validator()
        schema = schema_validator.validate(frames["train"])
        for split in ("val", "test"):
            schema_validator.validate_against_schema(frames[split], schema)
        schema_path = schema_validator.write_schema_json(
            schema, artifacts_dir / "input_schema.json"
        )
        schema_hash = stable_hash(schema.to_dict())

        train_horizons, train_window_path = self.trainer._build_horizons_for_split(
            "train", frames["train"], schema, artifacts_dir
        )
        val_horizons, val_window_path = self.trainer._build_horizons_for_split(
            "val", frames["val"], schema, artifacts_dir
        )
        test_horizons, test_window_path = self.trainer._build_horizons_for_split(
            "test", frames["test"], schema, artifacts_dir
        )

        if self.config.data_augmentation.temporal_contrastive.enabled:
            tc = self.config.data_augmentation.temporal_contrastive
            builder = HorizonBuilder(
                self.config.horizon,
                schema,
                self.config.dp.cost_config.reward_alignment,
            )
            shifted, _pairs = TemporalContrastiveBuilder(
                shift_bars=tc.shift_bars,
                pair_ratio=tc.pair_ratio,
                max_pairs=tc.max_pairs,
                require_same_strata=tc.require_same_strata,
                seed=self.config.seed,
            ).build_pairs(train_horizons, frames["train"], builder, pair=self.config.pair)
            train_horizons = list(train_horizons) + list(shifted)

        train_horizons, train_reject = self.trainer._generate_demos(train_horizons)
        val_horizons, val_reject = self.trainer._generate_demos(val_horizons)
        test_horizons, test_reject = self.trainer._generate_demos(test_horizons)

        input_file_audit = {
            split: _file_audit(path)
            for split, path in {
                "train": self.config.train_file,
                "val": self.config.val_file,
                "test": self.config.test_file,
            }.items()
        }
        data_process_hash = stable_hash(
            {
                "pair": self.config.pair,
                "input_file_audit": input_file_audit,
                "factor_profile": self.config.factor_profile,
                "factor_list_file": self.config.factor_list_file,
                "feature_source": schema.feature_source or {},
                "horizon": self.config.horizon,
                "num_demos": self.config.num_demos,
                "sampling_strategy": self.config.sampling_strategy,
                "stratification": asdict(self.config.stratification),
                "sampling_health": asdict(self.config.sampling_health),
                "data_augmentation": asdict(self.config.data_augmentation),
                "seed": self.config.seed,
            }
        )
        dp_teacher_hash = stable_hash(
            {
                "data_process_hash": data_process_hash,
                "dp": asdict(self.config.dp),
            }
        )

        store = Phase1ProcessedStore(artifacts_dir)
        split_records = {
            "train": (train_horizons, train_window_path, train_reject),
            "val": (val_horizons, val_window_path, val_reject),
            "test": (test_horizons, test_window_path, test_reject),
        }
        split_payload = {}
        for split, (records, window_path, reject_stats) in split_records.items():
            sampled_path = store.save_sampled_horizons(
                split,
                records,
                schema_hash=schema_hash,
                data_process_hash=data_process_hash,
            )
            teacher_path = store.save_dp_teacher(
                split,
                records,
                reject_stats,
                schema_hash=schema_hash,
                data_process_hash=data_process_hash,
                dp_teacher_hash=dp_teacher_hash,
            )
            reject_path = store.save_reject_stats(split, reject_stats)
            split_payload[split] = {
                "window_index_path": _relative_to_artifact(window_path, artifacts_dir),
                "sampled_horizons_path": _relative_to_artifact(sampled_path, artifacts_dir),
                "dp_teacher_path": _relative_to_artifact(teacher_path, artifacts_dir),
                "reject_stats_path": _relative_to_artifact(reject_path, artifacts_dir),
                "num_horizons": len(records),
            }

        Phase1DemoStore(artifacts_dir, data_process_hash, schema_hash).save_demos(
            train_horizons
        )
        manifest = {
            "version": 1,
            "phase": "phase1_data_process",
            "pair": self.config.pair,
            "data_batch_id": self.config.data_batch_id,
            "artifact_dir": str(artifacts_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_files": {
                "train": self.config.train_file,
                "val": self.config.val_file,
                "test": self.config.test_file,
            },
            "input_file_audit": input_file_audit,
            "input_schema_path": _relative_to_artifact(schema_path, artifacts_dir),
            "schema_hash": schema_hash,
            "data_process_hash": data_process_hash,
            "dp_teacher_hash": dp_teacher_hash,
            "feature_source": schema.feature_source or {},
            "splits": split_payload,
        }
        return store.write_manifest(manifest)

    def _trainer_config(self) -> Phase1Config:
        return Phase1Config(
            pair=self.config.pair,
            train_batch_id=self.config.data_batch_id,
            train_file=self.config.train_file,
            val_file=self.config.val_file,
            test_file=self.config.test_file,
            artifact_root=self.config.artifact_root,
            factor_profile=self.config.factor_profile,
            factor_list_file=self.config.factor_list_file,
            horizon=self.config.horizon,
            num_demos=self.config.num_demos,
            sampling_strategy=self.config.sampling_strategy,
            stratification=self.config.stratification,
            sampling_health=self.config.sampling_health,
            data_augmentation=self.config.data_augmentation,
            dp=self.config.dp,
            training=TrainingConfig(seed=self.config.seed),
            allow_missing_prospective_diagnostic=(
                self.config.allow_missing_prospective_diagnostic
            ),
            risk_acknowledged_by=self.config.risk_acknowledged_by,
            expected_sign_off_followup_batch_id=(
                self.config.expected_sign_off_followup_batch_id
            ),
        )


def _file_audit(path: str) -> dict:
    target = Path(path)
    stat = target.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _relative_to_artifact(path: Path, artifact_dir: Path) -> str:
    try:
        return str(Path(path).relative_to(artifact_dir))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert_prospective_diagnostic(args)
    config = build_data_process_config(args)
    try:
        manifest = Phase1DataProcessor(config).run()
    except Phase1FatalError as exc:
        print(f"[fatal] Phase I 数据预处理终止: {exc}", file=sys.stderr)
        return 1
    print(f"data_process_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
