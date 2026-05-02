"""Phase I offline data processor.

This command generates sampled horizon artifacts and DP teacher labels for
manifest-mode Phase I training. Training should load the manifest produced here
instead of re-reading raw market data or re-running DP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config.phase1_config import (  # noqa: E402
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    Phase1DataProcessConfig,
    SamplingHealthConfig,
    StratificationConfig,
    TrainingConfig,
)
from src.data.data_augmentation import TemporalContrastiveBuilder  # noqa: E402
from src.data.demo_store import Phase1DemoStore  # noqa: E402
from src.data.feature_registry import (  # noqa: E402
    default_factor_list_path,
    load_feature_selection,
)
from src.data.horizon_builder import HorizonBuilder  # noqa: E402
from src.data.market_reader import MarketFileReader  # noqa: E402
from src.data.phase1_processed_store import (  # noqa: E402
    Phase1ProcessedStore,
    stable_hash,
)
from src.data.sampling_health import SamplingHealthChecker, SamplingHealthError  # noqa: E402
from src.data.schema import InputSchemaValidator  # noqa: E402
from src.data.stratified_sampler import StratifiedWindowSampler  # noqa: E402
from src.data.window_indexer import SlidingWindowIndexer, WindowIndexEntry  # noqa: E402
from src.planners.demo_generator import Phase1DemoGenerator  # noqa: E402
from src.planners.single_trade_dp import SingleTradeDPPlanner  # noqa: E402
from src.trading.cost_model import LobDepthCostModel  # noqa: E402
from src.trading.reward_alignment import RewardAlignment  # noqa: E402
from src.utils.feather_io import write_ipc  # noqa: E402
from src.utils.run_logging import configure_run_logger  # noqa: E402


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


class Phase1FatalError(RuntimeError):
    pass


class Phase1DataProcessor:
    """Offline processor that writes Phase I manifest-mode training inputs."""

    def __init__(self, config: Phase1DataProcessConfig) -> None:
        self.config = config
        self._logger = logging.getLogger(
            f"archetype.data_process.{config.pair}.{config.data_batch_id}"
        )

    def run(self) -> Path:
        self._logger.info("Phase1运行开始 pair=%s batch_id=%s horizon=%d num_demos=%d",
                          self.config.pair, self.config.data_batch_id,
                          self.config.horizon, self.config.num_demos)
        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("产物目录已创建 dir=%s", artifacts_dir)

        _seed_everything(self.config.seed)
        self._logger.info("随机种子已设置 seed=%d", self.config.seed)

        self._logger.info("正在检查前瞻诊断配置 mode=%s diagnostic_batch_id=%s",
                          self.config.stratification.mode,
                          self.config.stratification.diagnostic_pair_batch_id)
        _check_prospective_diagnostic(self.config)

        self._logger.info("正在读取市场数据 train=%s val=%s test=%s",
                          self.config.train_file, self.config.val_file, self.config.test_file)
        frames = MarketFileReader().read_split(
            self.config.train_file, self.config.val_file, self.config.test_file
        )
        self._logger.info("市场数据读取完成 train_rows=%d val_rows=%d test_rows=%d",
                          frames["train"].height, frames["val"].height, frames["test"].height)

        self._logger.info("正在构建Schema验证器 profile=%s factor_list=%s",
                          self.config.factor_profile, self.config.factor_list_file)
        schema_validator = _build_schema_validator(self.config)
        schema = schema_validator.validate(frames["train"])
        for split in ("val", "test"):
            schema_validator.validate_against_schema(frames[split], schema)
        schema_path = schema_validator.write_schema_json(
            schema, artifacts_dir / "input_schema.json"
        )
        schema_hash = stable_hash(schema.to_dict())
        self._logger.info("Schema验证完成 schema_path=%s schema_hash=%s",
                          schema_path, schema_hash)

        self._logger.info("正在构建时间窗口 split=train")
        train_horizons, train_window_path = self._build_horizons_for_split(
            "train", frames["train"], schema, artifacts_dir
        )
        self._logger.info("正在构建时间窗口 split=val")
        val_horizons, val_window_path = self._build_horizons_for_split(
            "val", frames["val"], schema, artifacts_dir
        )
        self._logger.info("正在构建时间窗口 split=test")
        test_horizons, test_window_path = self._build_horizons_for_split(
            "test", frames["test"], schema, artifacts_dir
        )

        if self.config.data_augmentation.temporal_contrastive.enabled:
            self._logger.info("数据增强功能已启用 时间对比学习")
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
            self._logger.info("数据增强已应用 原始=%d 增强=%d 总计=%d",
                              len(train_horizons) - len(shifted), len(shifted), len(train_horizons))
        else:
            self._logger.info("数据增强功能未启用")

        self._logger.info("正在生成演示样本 split=train num_horizons=%d", len(train_horizons))
        train_horizons, train_reject = _generate_demos(self.config, train_horizons)
        self._logger.info("演示样本生成完成 split=train 接受=%d 拒绝率=%.4f",
                          len(train_horizons), train_reject.dataset_reject_rate)

        self._logger.info("正在生成演示样本 split=val num_horizons=%d", len(val_horizons))
        val_horizons, val_reject = _generate_demos(self.config, val_horizons)
        self._logger.info("演示样本生成完成 split=val 接受=%d 拒绝率=%.4f",
                          len(val_horizons), val_reject.dataset_reject_rate)

        self._logger.info("正在生成演示样本 split=test num_horizons=%d", len(test_horizons))
        test_horizons, test_reject = _generate_demos(self.config, test_horizons)
        self._logger.info("演示样本生成完成 split=test 接受=%d 拒绝率=%.4f",
                          len(test_horizons), test_reject.dataset_reject_rate)

        self._logger.info("正在计算数据哈希")
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
        self._logger.info("哈希计算完成 data_process_hash=%s dp_teacher_hash=%s",
                          data_process_hash, dp_teacher_hash)

        self._logger.info("正在保存产物文件")
        store = Phase1ProcessedStore(artifacts_dir)
        split_records = {
            "train": (train_horizons, train_window_path, train_reject),
            "val": (val_horizons, val_window_path, val_reject),
            "test": (test_horizons, test_window_path, test_reject),
        }
        split_payload = {}
        for split, (records, window_path, reject_stats) in split_records.items():
            self._logger.info("正在保存分集产物 split=%s num_records=%d", split, len(records))
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

        self._logger.info("正在保存演示存储 train=%d val=%d test=%d",
                          len(train_horizons), len(val_horizons), len(test_horizons))
        demo_store = Phase1DemoStore(artifacts_dir, data_process_hash, schema_hash)
        demo_store.save_demos(train_horizons, split="train")
        demo_store.save_demos(val_horizons, split="val")
        demo_store.save_demos(test_horizons, split="test")

        self._logger.info("正在写入清单文件")
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
        manifest_path = store.write_manifest(manifest)
        self._logger.info("Phase1运行完成 manifest_path=%s", manifest_path)
        return manifest_path

    def _build_horizons_for_split(
        self,
        split: str,
        frame,
        schema,
        artifacts_dir: Path,
    ):
        indexer = SlidingWindowIndexer(
            horizon=self.config.horizon,
            reward_alignment=self.config.dp.cost_config.reward_alignment,
            prospective_lookback_minutes=self.config.stratification.prospective_lookback_minutes,
        )
        all_entries = indexer.enumerate(
            frame, stratification_mode=self.config.stratification.mode
        )
        embargo = _active_split_boundary_embargo(self.config)
        entries, boundary_excluded = _filter_split_boundary_entries(
            all_entries, frame_height=frame.height, embargo=embargo
        )
        if boundary_excluded:
            self._logger.info(
                "窗口索引边界禁运已应用 split=%s 排除=%d 合格=%d embargo=%d",
                split,
                boundary_excluded,
                len(entries),
                embargo,
            )
        num_samples = _num_samples_for_split(self.config, split, len(entries))
        if split == "train":
            _check_overlap_health_feasibility(self.config, num_samples, entries)
        self._logger.info(
            "窗口索引枚举完成 split=%s 候选=%d 合格=%d 目标样本=%d",
            split,
            len(all_entries),
            len(entries),
            num_samples,
        )

        sampler = StratifiedWindowSampler(
            strategy=self.config.sampling_strategy,
            min_gap_between_samples=self.config.sampling_health.min_gap_between_samples,
            flat_low_vol_max_ratio=self.config.sampling_health.flat_low_vol_max_ratio,
            allow_overlap_relaxation=self.config.sampling_health.allow_overlap_relaxation,
            seed=self.config.seed
            + (1 if split == "val" else 2 if split == "test" else 0),
        )
        prospective = self.config.stratification.mode == "prospective_past"
        labels = [
            StratifiedWindowSampler.assign_strata(e, prospective=prospective)
            for e in entries
        ]
        sampled = sampler.sample(entries, num_samples=num_samples, strata_labels=labels)
        effective_min_gap = sampler.last_effective_min_gap_between_samples
        overlap_relaxation_applied = sampler.last_overlap_relaxation_applied
        self._logger.info(
            "窗口采样完成 split=%s 采样=%d 独立层=%d 实际最小间隔=%d 重叠放宽=%s",
            split,
            len(sampled),
            len({s.strata_label for s in sampled}),
            effective_min_gap,
            overlap_relaxation_applied,
        )

        if split == "train":
            checker = SamplingHealthChecker(
                horizon=self.config.horizon,
                max_overlap_ratio=self.config.sampling_health.max_overlap_ratio,
                min_gap_between_samples=self.config.sampling_health.min_gap_between_samples,
                split_boundary_embargo=embargo,
                flat_low_vol_max_ratio=self.config.sampling_health.flat_low_vol_max_ratio,
                warn_only=self.config.sampling_health.warn_only,
                effective_min_gap_between_samples=effective_min_gap,
                overlap_relaxation_applied=overlap_relaxation_applied,
            )
            report = checker.check(
                sampled=sampled,
                split_boundaries={"train_end_row": frame.height - 1},
                strata_labels=[s.strata_label for s in sampled],
            )
            self._logger.info(
                "采样健康检查通过 split=%s 重叠率=%.6f 最小间隔=%s 低波动占比=%.6f 警告=%d",
                split,
                report.window_overlap_ratio,
                report.min_sample_gap,
                report.flat_low_vol_sample_ratio,
                len(report.sampling_health_warnings),
            )

        index_frame = indexer.to_frame(all_entries)
        sampled_starts = {s.window_start for s in sampled}
        eligible_starts = {e.window_start for e in entries}
        index_frame = index_frame.with_columns(
            [
                (index_frame["window_start"].is_in(list(eligible_starts))).alias(
                    "is_boundary_eligible"
                ),
                (index_frame["window_start"].is_in(list(sampled_starts))).alias(
                    "is_sampled"
                ),
            ]
        )
        path = write_ipc(index_frame, artifacts_dir / f"window_index_{split}.feather")

        builder = HorizonBuilder(
            self.config.horizon, schema, self.config.dp.cost_config.reward_alignment
        )
        horizons = builder.build(frame, sampled, pair=self.config.pair, split=split)
        self._logger.info(
            "phase1_horizons_built split=%s horizons=%d window_index_path=%s",
            split,
            len(horizons),
            path,
        )
        return horizons, path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_prospective_diagnostic(config: Phase1DataProcessConfig) -> None:
    if config.stratification.mode == "prospective_past":
        return
    if not config.stratification.require_prospective_diagnostic:
        return
    if config.stratification.diagnostic_pair_batch_id is None:
        if not config.allow_missing_prospective_diagnostic:
            raise Phase1FatalError(
                "缺少 diagnostic_pair_batch_id; 主实验不可启动。"
                "传入 --allow-missing-prospective-diagnostic + "
                "--risk-acknowledged-by + --expected-sign-off-followup-batch-id 才能放行。"
            )
        if (
            not config.risk_acknowledged_by
            or not config.expected_sign_off_followup_batch_id
        ):
            raise Phase1FatalError(
                "allow_missing_prospective_diagnostic=True 时必须显式声明"
                "risk_acknowledged_by 与 expected_sign_off_followup_batch_id。"
            )


def _build_schema_validator(config: Phase1DataProcessConfig) -> InputSchemaValidator:
    factor_path = default_factor_list_path(config.pair, config.factor_profile)
    has_factor_file = bool(config.factor_list_file) or factor_path.exists()
    if has_factor_file:
        spec = load_feature_selection(
            pair=config.pair,
            profile=config.factor_profile,
            factor_list_file=config.factor_list_file,
        )
        return InputSchemaValidator(
            price_column=spec.price_column,
            feature_columns=spec.feature_columns,
            feature_source=spec.to_dict(),
        )
    return InputSchemaValidator(
        feature_source={
            "mode": "legacy_auto_numeric",
            "pair": config.pair,
            "profile": config.factor_profile,
            "factor_list_path": str(factor_path),
        }
    )


def _active_split_boundary_embargo(config: Phase1DataProcessConfig) -> int:
    if config.dp.cost_config.reward_alignment == "paper_formula":
        return config.sampling_health.split_boundary_embargo
    return config.sampling_health.next_row_split_boundary_embargo


def _filter_split_boundary_entries(
    entries: List[WindowIndexEntry],
    frame_height: int,
    embargo: int,
) -> Tuple[List[WindowIndexEntry], int]:
    if embargo <= 0:
        return list(entries), 0
    max_markout_row = frame_height - 1 - embargo
    eligible = [e for e in entries if e.last_markout_row <= max_markout_row]
    return eligible, len(entries) - len(eligible)


def _num_samples_for_split(
    config: Phase1DataProcessConfig, split: str, num_entries: int
) -> int:
    if split == "train":
        return min(config.num_demos, num_entries)
    return min(num_entries, min(64, max(1, config.num_demos // 16)))


def _check_overlap_health_feasibility(
    config: Phase1DataProcessConfig,
    num_samples: int,
    entries: List[WindowIndexEntry],
) -> None:
    max_overlap = config.sampling_health.max_overlap_ratio
    if (
        config.sampling_health.warn_only
        or num_samples <= 1
        or len(entries) <= 1
        or max_overlap >= 1.0
    ):
        return
    span = entries[-1].window_start - entries[0].window_start
    if span <= 0:
        return
    required_mean_gap = config.horizon * (1.0 - max_overlap)
    if required_mean_gap <= 0:
        return
    min_possible_overlap = max(
        0.0,
        (config.horizon - (span / (num_samples - 1))) / config.horizon,
    )
    if min_possible_overlap <= max_overlap:
        return
    max_samples_for_overlap = int(span // required_mean_gap + 1)
    raise SamplingHealthError(
        "采样健康检查不可行: "
        f"num_samples={num_samples} 在 eligible_span={span} rows, "
        f"horizon={config.horizon} 下 window_overlap_ratio 理论下限约 "
        f"{min_possible_overlap:.3f} > max={max_overlap}; "
        f"请将 --num-demos 降到 <= {max_samples_for_overlap}，"
        "或显式传 --sampling-health-warn-only / 提高 --sampling-max-overlap-ratio。"
    )


def _generate_demos(config: Phase1DataProcessConfig, horizons):
    cost_model = LobDepthCostModel(
        commission_rate=config.dp.cost_config.commission_rate,
        book_levels=config.dp.cost_config.book_levels,
        insufficient_depth_policy=config.dp.cost_config.insufficient_depth_policy,
    )
    alignment = RewardAlignment(config.dp.cost_config.reward_alignment)
    planner = SingleTradeDPPlanner(
        cost_model=cost_model,
        reward_alignment=alignment,
        max_position=config.dp.max_position,
        gamma=config.dp.gamma,
    )
    gen = Phase1DemoGenerator(
        planner=planner,
        health=config.dp.cost_config.reject_transition_health,
    )
    return gen.generate(horizons)


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

    logger, log_path = configure_run_logger(
        phase="data_process",
        pair=config.pair,
        batch_id=config.data_batch_id,
    )

    artifacts_dir = config.artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = Phase1DataProcessor(config).run()
    except Phase1FatalError as exc:
        logger.exception("phase1_fatal_error error=%s", exc)
        print(f"[fatal] Phase I 数据预处理终止: {exc}", file=sys.stderr)
        return 1
    print(f"data_process_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
