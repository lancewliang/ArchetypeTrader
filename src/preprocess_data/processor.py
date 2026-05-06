"""Phase I offline data processor."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.preprocess_data.config import (  # noqa: E402
    Phase1DataProcessConfig,
)
from src.preprocess_data.data_augmentation import TemporalContrastiveBuilder  # noqa: E402
from src.preprocess_data.demo_store import Phase1DemoStore  # noqa: E402
from src.preprocess_data.feature_registry import (  # noqa: E402
    default_factor_list_path,
    load_feature_selection,
)
from src.preprocess_data.feature_provenance import write_feature_provenance_json  # noqa: E402
from src.preprocess_data.horizon_builder import HorizonBuilder  # noqa: E402
from src.preprocess_data.market_reader import MarketFileReader  # noqa: E402
from src.preprocess_data.processed_store import (  # noqa: E402
    Phase1ProcessedStore,
    stable_hash,
)
from src.preprocess_data.sampling_health import SamplingHealthChecker, SamplingHealthError  # noqa: E402
from src.preprocess_data.schema import InputSchemaValidator  # noqa: E402
from src.preprocess_data.stratified_sampler import StratifiedWindowSampler  # noqa: E402
from src.preprocess_data.stratified_sampler import SampledHorizon  # noqa: E402
from src.preprocess_data.window_indexer import SlidingWindowIndexer, WindowIndexEntry  # noqa: E402
from src.planners.demo_generator import Phase1DemoGenerator, RejectStats  # noqa: E402
from src.planners.single_trade_dp import SingleTradeDPPlanner  # noqa: E402
from src.trading.cost_model import LobDepthCostModel  # noqa: E402
from src.trading.reward_alignment import RewardAlignment  # noqa: E402
from src.utils.feather_io import write_ipc  # noqa: E402
from src.utils.seed_init import seed_everything  # noqa: E402


class Phase1FatalError(RuntimeError):
    """Phase I 数据处理的可预期致命错误。"""

    pass


@dataclass
class SplitHorizonBuildResult:
    """单个 split 构建完成后的产物和审计信息。"""

    horizons: List[Any]
    window_index_path: Path
    all_entries: List[WindowIndexEntry]
    eligible_entries: List[WindowIndexEntry]
    sampled_horizons: List[SampledHorizon]
    full_time_pool_entries: List[WindowIndexEntry]
    strata_by_start: Dict[int, str]
    num_eligible_windows: int
    sampling_applied: bool
    labeling_mode: str
    sample_source_counts: Dict[str, int]
    sampling_health_warnings: List[str]


@dataclass
class SplitWindowIndexContext:
    """窗口枚举后的公共上下文，供 train 采样和 eval 标注分支复用。"""

    indexer: SlidingWindowIndexer
    all_entries: List[WindowIndexEntry]
    eligible_entries: List[WindowIndexEntry]
    labels: List[str]
    strata_by_start: Dict[int, str]
    target_horizons: int
    embargo: int


class Phase1DataProcessor:
    """Offline processor that writes Phase I manifest-mode training inputs."""

    def __init__(self, config: Phase1DataProcessConfig) -> None:
        """保存配置并创建带 pair/batch 维度的 logger。

        Processor 实例只持有运行配置，具体产物路径、随机种子和数据读取都在
        ``run`` 中展开，方便测试单个构建阶段。
        """
        self.config = config
        self._logger = logging.getLogger(
            f"archetype.data_process.{config.pair}.{config.data_batch_id}"
        )

    def run(self) -> Path:
        """执行完整 Phase I 离线数据预处理流水线。

        流程按可审计顺序展开：读数据、校验 schema、构建 horizon、运行 DP
        teacher、保存 split 产物和 manifest。这样训练阶段可以直接消费 manifest，
        避免重复枚举窗口或重新跑 DP。
        """
        self._logger.info("Phase1运行开始 pair=%s batch_id=%s horizon=%d num_demos=%d",
                          self.config.pair, self.config.data_batch_id,
                          self.config.horizon, self.config.num_demos)
        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("产物目录已创建 dir=%s", artifacts_dir)

        seed_everything(self.config.seed)
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

        # 分阶段构建和生成，避免同时持有所有 split 的 horizons
        # Phase 1: 构建并处理 train
        self._logger.info("正在构建时间窗口 split=train")
        train_result = self._build_sampled_horizons_for_split(
            "train", frames["train"], schema, artifacts_dir
        )
        train_horizons = train_result.horizons
        train_window_path = train_result.window_index_path

        # if self.config.data_augmentation.temporal_contrastive.enabled:
        #     self._logger.info("数据增强功能已启用 时间对比学习")
        #     tc = self.config.data_augmentation.temporal_contrastive
        #     builder = HorizonBuilder(
        #         self.config.horizon,
        #         schema,
        #         self.config.dp.cost_config.reward_alignment,
        #     )
        #     shifted, _pairs = TemporalContrastiveBuilder(
        #         shift_bars=tc.shift_bars,
        #         pair_ratio=tc.pair_ratio,
        #         max_pairs=tc.max_pairs,
        #         require_same_strata=tc.require_same_strata,
        #         seed=self.config.seed,
        #     ).build_pairs(train_horizons, frames["train"], builder, pair=self.config.pair)
        #     train_horizons = list(train_horizons) + list(shifted)
        #     self._logger.info("数据增强已应用 原始=%d 增强=%d 总计=%d",
        #                       len(train_horizons) - len(shifted), len(shifted), len(train_horizons))
        # else:
        
        self._logger.info("数据增强功能未启用")

        self._logger.info(
            "正在生成演示样本 split=train num_horizons=%d dp_workers=%s chunksize=%d",
            len(train_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        train_horizons, train_reject = _generate_demos(self.config, train_horizons)
        self._logger.info("演示样本生成完成 split=train 接受=%d 拒绝率=%.4f",
                          len(train_horizons), train_reject.dataset_reject_rate)
        train_horizons, train_reject, train_coverage = self._ensure_train_min_coverage(
            train_horizons,
            train_reject,
            train_result,
            frames["train"],
            schema,
        )
        _log_reward_distribution(self._logger, train_horizons, "train")

        full_time_train_horizons: List[Any] = []
        full_time_train_reject: RejectStats | None = None
        if self.config.time_distribution_sampling.enabled:
            full_time_labels = [
                train_result.strata_by_start[entry.window_start]
                for entry in train_result.full_time_pool_entries
            ]
            full_time_sampled = _sampled_from_entries(
                train_result.full_time_pool_entries,
                full_time_labels,
                source="full_time",
            )
            full_time_builder = HorizonBuilder(
                self.config.horizon,
                schema,
                self.config.dp.cost_config.reward_alignment,
            )
            full_time_train_horizons = full_time_builder.build(
                frames["train"],
                full_time_sampled,
                pair=self.config.pair,
                split="train",
            )
            self._logger.info(
                "正在生成full-time train演示样本 num_horizons=%d dp_workers=%s chunksize=%d",
                len(full_time_train_horizons),
                self.config.dp_workers,
                self.config.dp_worker_chunksize,
            )
            full_time_train_horizons, full_time_train_reject = _generate_demos(
                self.config, full_time_train_horizons
            )
            self._logger.info(
                "full-time train演示样本生成完成 接受=%d 拒绝率=%.4f",
                len(full_time_train_horizons),
                full_time_train_reject.dataset_reject_rate,
            )

        self._logger.info("正在构建non-overlap时间窗口 split=train")
        non_overlap_train_result = self._build_non_overlap_horizons_for_split(
            "train", frames["train"], schema, artifacts_dir
        )
        non_overlap_train_horizons = non_overlap_train_result.horizons
        self._logger.info(
            "正在生成non-overlap演示样本 split=train num_horizons=%d dp_workers=%s chunksize=%d",
            len(non_overlap_train_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        non_overlap_train_horizons, non_overlap_train_reject = _generate_demos(
            self.config, non_overlap_train_horizons
        )
        self._logger.info(
            "non-overlap演示样本生成完成 split=train 接受=%d 拒绝率=%.4f",
            len(non_overlap_train_horizons),
            non_overlap_train_reject.dataset_reject_rate,
        )

        # 释放 train_frame 内存（后续不再需要）
        del frames["train"]
        gc.collect()

        # Phase 2: 构建并处理 val
        self._logger.info("正在构建时间窗口 split=val")
        val_result = self._build_eval_horizons_for_split(
            "val", frames["val"], schema, artifacts_dir
        )
        val_horizons = val_result.horizons
        val_window_path = val_result.window_index_path

        self._logger.info(
            "正在生成演示样本 split=val num_horizons=%d dp_workers=%s chunksize=%d",
            len(val_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        val_horizons, val_reject = _generate_demos(self.config, val_horizons)
        self._logger.info("演示样本生成完成 split=val 接受=%d 拒绝率=%.4f",
                          len(val_horizons), val_reject.dataset_reject_rate)
        _log_reward_distribution(self._logger, val_horizons, "val")

        self._logger.info("正在构建non-overlap时间窗口 split=val")
        non_overlap_val_result = self._build_non_overlap_horizons_for_split(
            "val", frames["val"], schema, artifacts_dir
        )
        non_overlap_val_horizons = non_overlap_val_result.horizons
        self._logger.info(
            "正在生成non-overlap演示样本 split=val num_horizons=%d dp_workers=%s chunksize=%d",
            len(non_overlap_val_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        non_overlap_val_horizons, non_overlap_val_reject = _generate_demos(
            self.config, non_overlap_val_horizons
        )
        self._logger.info(
            "non-overlap演示样本生成完成 split=val 接受=%d 拒绝率=%.4f",
            len(non_overlap_val_horizons),
            non_overlap_val_reject.dataset_reject_rate,
        )

        # 释放 val_frame 内存
        del frames["val"]
        gc.collect()

        # Phase 3: 构建并处理 test
        self._logger.info("正在构建时间窗口 split=test")
        test_result = self._build_eval_horizons_for_split(
            "test", frames["test"], schema, artifacts_dir
        )
        test_horizons = test_result.horizons
        test_window_path = test_result.window_index_path

        self._logger.info(
            "正在生成演示样本 split=test num_horizons=%d dp_workers=%s chunksize=%d",
            len(test_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        test_horizons, test_reject = _generate_demos(self.config, test_horizons)
        self._logger.info("演示样本生成完成 split=test 接受=%d 拒绝率=%.4f",
                          len(test_horizons), test_reject.dataset_reject_rate)
        _log_reward_distribution(self._logger, test_horizons, "test")

        self._logger.info("正在构建non-overlap时间窗口 split=test")
        non_overlap_test_result = self._build_non_overlap_horizons_for_split(
            "test", frames["test"], schema, artifacts_dir
        )
        non_overlap_test_horizons = non_overlap_test_result.horizons
        self._logger.info(
            "正在生成non-overlap演示样本 split=test num_horizons=%d dp_workers=%s chunksize=%d",
            len(non_overlap_test_horizons),
            self.config.dp_workers,
            self.config.dp_worker_chunksize,
        )
        non_overlap_test_horizons, non_overlap_test_reject = _generate_demos(
            self.config, non_overlap_test_horizons
        )
        self._logger.info(
            "non-overlap演示样本生成完成 split=test 接受=%d 拒绝率=%.4f",
            len(non_overlap_test_horizons),
            non_overlap_test_reject.dataset_reject_rate,
        )

        # 释放 test_frame 内存
        del frames["test"]
        gc.collect()

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
                "no_trade_control": asdict(self.config.no_trade_control),
                "time_distribution_sampling": asdict(
                    self.config.time_distribution_sampling
                ),
                "eval_labeling": asdict(self.config.eval_labeling),
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

        feature_provenance_path = write_feature_provenance_json(
            schema,
            artifacts_dir / "feature_provenance.json",
            metadata={
                "created_by": "scripts/pre_process_data.py",
                "pair": self.config.pair,
                "data_batch_id": self.config.data_batch_id,
                "schema_hash": schema_hash,
                "data_process_hash": data_process_hash,
            },
        )
        self._logger.info(
            "feature_provenance_written path=%s features=%d",
            feature_provenance_path,
            len(schema.feature_columns),
        )

        self._logger.info("正在保存产物文件")
        store = Phase1ProcessedStore(artifacts_dir)
        split_records = {
            "train": (train_horizons, train_window_path, train_reject, train_result),
            "val": (val_horizons, val_window_path, val_reject, val_result),
            "test": (test_horizons, test_window_path, test_reject, test_result),
        }
        non_overlap_records = {
            "train": (non_overlap_train_horizons, non_overlap_train_reject),
            "val": (non_overlap_val_horizons, non_overlap_val_reject),
            "test": (non_overlap_test_horizons, non_overlap_test_reject),
        }
        split_payload = {}
        for split, (records, window_path, reject_stats, build_result) in split_records.items():
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

            no_records, no_reject = non_overlap_records[split]
            no_horizons_path = store.save_non_overlap_horizons(
                split,
                no_records,
                schema_hash=schema_hash,
                data_process_hash=data_process_hash,
            )
            no_teacher_path = store.save_non_overlap_dp_teacher(
                split,
                no_records,
                no_reject,
                schema_hash=schema_hash,
                data_process_hash=data_process_hash,
                dp_teacher_hash=dp_teacher_hash,
            )

            split_payload[split] = {
                "window_index_path": _relative_to_artifact(window_path, artifacts_dir),
                "sampled_horizons_path": _relative_to_artifact(sampled_path, artifacts_dir),
                "dp_teacher_path": _relative_to_artifact(teacher_path, artifacts_dir),
                "reject_stats_path": _relative_to_artifact(reject_path, artifacts_dir),
                "num_horizons": len(records),
                "num_eligible_windows": build_result.num_eligible_windows,
                "num_labeled_windows": len(records),
                "labeling_mode": build_result.labeling_mode,
                "sampling_applied": build_result.sampling_applied,
                "augmentation_applied": bool(
                    split == "train"
                    and self.config.data_augmentation.temporal_contrastive.enabled
                ),
                "sample_source_counts": _sample_source_counts(records),
                "sampling_health_warnings": build_result.sampling_health_warnings,
                "non_overlap_horizons_path": _relative_to_artifact(no_horizons_path, artifacts_dir),
                "non_overlap_dp_teacher_path": _relative_to_artifact(no_teacher_path, artifacts_dir),
                "non_overlap_num_horizons": len(no_records),
            }
            if split == "train":
                split_payload[split]["coverage_after_dp"] = train_coverage
                if full_time_train_reject is not None:
                    full_time_sampled_path = store.save_sampled_horizons(
                        "full_time_train",
                        full_time_train_horizons,
                        schema_hash=schema_hash,
                        data_process_hash=data_process_hash,
                    )
                    full_time_teacher_path = store.save_dp_teacher(
                        "full_time_train",
                        full_time_train_horizons,
                        full_time_train_reject,
                        schema_hash=schema_hash,
                        data_process_hash=data_process_hash,
                        dp_teacher_hash=dp_teacher_hash,
                    )
                    split_payload[split]["full_time_sampled_horizons_path"] = (
                        _relative_to_artifact(full_time_sampled_path, artifacts_dir)
                    )
                    split_payload[split]["full_time_dp_teacher_path"] = (
                        _relative_to_artifact(full_time_teacher_path, artifacts_dir)
                    )
                    split_payload[split]["full_time_num_horizons"] = len(
                        full_time_train_horizons
                    )

        self._logger.info("正在保存演示存储 train=%d val=%d test=%d",
                          len(train_horizons), len(val_horizons), len(test_horizons))
        demo_store = Phase1DemoStore(artifacts_dir, data_process_hash, schema_hash)
        demo_store.save_demos(train_horizons, split="train")
        demo_store.save_demos(val_horizons, split="val")
        demo_store.save_demos(test_horizons, split="test")

        self._logger.info("正在保存non-overlap演示存储 train=%d val=%d test=%d",
                          len(non_overlap_train_horizons), len(non_overlap_val_horizons),
                          len(non_overlap_test_horizons))
        demo_store.save_non_overlap_demos(non_overlap_train_horizons, split="train")
        demo_store.save_non_overlap_demos(non_overlap_val_horizons, split="val")
        demo_store.save_non_overlap_demos(non_overlap_test_horizons, split="test")

        # 所有数据保存完成后，释放 val/test 内存（train 在函数返回后自然释放）
        self._logger.info("演示存储保存完成，释放 val/test 内存")
        del val_horizons, test_horizons
        del non_overlap_train_horizons, non_overlap_val_horizons, non_overlap_test_horizons
        gc.collect()

        self._logger.info("正在写入清单文件")
        manifest = {
            "version": 2,
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
            "feature_provenance_path": _relative_to_artifact(
                feature_provenance_path, artifacts_dir
            ),
            "schema_hash": schema_hash,
            "data_process_hash": data_process_hash,
            "dp_teacher_hash": dp_teacher_hash,
            "feature_source": schema.feature_source or {},
            "splits": split_payload,
        }
        manifest_path = store.write_manifest(manifest)
        self._logger.info("Phase1运行完成 manifest_path=%s", manifest_path)
        return manifest_path

    def _prepare_window_index_context(
        self,
        split: str,
        frame,
        *,
        requires_sampling: bool,
    ) -> SplitWindowIndexContext:
        """完成采样和全量标注共享的窗口前处理。

        这里统一做 stride=1 枚举、split boundary embargo、目标数量计算和
        strata label 分配，确保 train 与 val/test 只在后续选择策略上不同。
        """
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
        if requires_sampling:
            _check_overlap_health_feasibility(self.config, num_samples, entries)
        self._logger.info(
            "窗口索引枚举完成 split=%s 候选=%d 合格=%d 目标样本=%d",
            split,
            len(all_entries),
            len(entries),
            num_samples,
        )

        prospective = self.config.stratification.mode == "prospective_past"
        labels = StratifiedWindowSampler.assign_strata_batch(
            entries, prospective=prospective
        )
        strata_by_start = {
            entry.window_start: label for entry, label in zip(entries, labels)
        }
        return SplitWindowIndexContext(
            indexer=indexer,
            all_entries=all_entries,
            eligible_entries=entries,
            labels=labels,
            strata_by_start=strata_by_start,
            target_horizons=num_samples,
            embargo=embargo,
        )

    def _build_sampled_horizons_for_split(
        self,
        split: str,
        frame,
        schema,
        artifacts_dir: Path,
    ) -> SplitHorizonBuildResult:
        """构建需要采样约束的 train horizon 集合。

        train 不是全量标注，而是从 eligible 窗口中按时间覆盖、分层和 min-gap
        选出 ``num_demos`` 预算内的样本，并在落盘前运行采样健康检查。
        """
        context = self._prepare_window_index_context(
            split, frame, requires_sampling=True
        )
        sampled, full_time_pool_entries, effective_min_gap, overlap_relaxation_applied = (
            self._sample_train_windows(
                context.eligible_entries,
                context.labels,
                context.target_horizons,
            )
        )
        labeling_mode = "sampled_train"
        self._logger.info(
            "窗口采样完成 split=%s horizon窗口数=%d 分层标签数=%d 模式=%s "
            "采样实际最小起点间隔=%d 重叠放宽=%s",
            split,
            len(sampled),
            len({s.strata_label for s in sampled}),
            labeling_mode,
            effective_min_gap,
            overlap_relaxation_applied,
        )

        checker = SamplingHealthChecker(
            horizon=self.config.horizon,
            max_overlap_ratio=self.config.sampling_health.max_overlap_ratio,
            min_gap_between_samples=self.config.sampling_health.min_gap_between_samples,
            split_boundary_embargo=context.embargo,
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
        sampling_health_warnings = list(report.sampling_health_warnings)
        self._logger.info(
            "采样健康检查通过 split=%s 重叠率=%.6f 最小间隔=%s 低波动占比=%.6f 警告=%d",
            split,
            report.window_overlap_ratio,
            report.min_sample_gap,
            report.flat_low_vol_sample_ratio,
            len(report.sampling_health_warnings),
        )

        return self._finalize_split_horizon_build(
            split=split,
            frame=frame,
            schema=schema,
            artifacts_dir=artifacts_dir,
            context=context,
            sampled=sampled,
            full_time_pool_entries=full_time_pool_entries,
            sampling_applied=True,
            labeling_mode=labeling_mode,
            sampling_health_warnings=sampling_health_warnings,
        )

    def _build_eval_horizons_for_split(
        self,
        split: str,
        frame,
        schema,
        artifacts_dir: Path,
    ) -> SplitHorizonBuildResult:
        """构建不参与训练采样的 val/test horizon 集合。

        默认 ``horizon_stride`` 每 ``horizon`` 个 eligible 起点保留一个窗口，
        让目标样本数等于 ``eligible // horizon``；兼容模式 ``all_eligible``
        仍可完整标注所有边界合格窗口。
        """
        context = self._prepare_window_index_context(
            split, frame, requires_sampling=False
        )
        if self.config.eval_labeling.apply_sampling:
            raise Phase1FatalError(
                "eval_labeling.apply_sampling=True 与 eval label 契约冲突。"
            )
        if self.config.eval_labeling.apply_augmentation:
            raise Phase1FatalError(
                "eval_labeling.apply_augmentation=True 与 eval label 契约冲突。"
            )
        eval_entries, eval_labels, labeling_mode, sample_source = _select_eval_label_entries(
            self.config, split, context.eligible_entries, context.labels
        )
        sampled = _sampled_from_entries(
            eval_entries,
            eval_labels,
            source=sample_source,
        )
        self._logger.info(
            "窗口标注集合完成 split=%s horizon窗口数=%d 分层标签数=%d 模式=%s "
            "eligible窗口数=%d 评估步长=%d 最小间隔采样=未应用",
            split,
            len(sampled),
            len({s.strata_label for s in sampled}),
            labeling_mode,
            len(context.eligible_entries),
            self.config.horizon if labeling_mode == "horizon_stride" else 1,
        )

        return self._finalize_split_horizon_build(
            split=split,
            frame=frame,
            schema=schema,
            artifacts_dir=artifacts_dir,
            context=context,
            sampled=sampled,
            full_time_pool_entries=[],
            sampling_applied=False,
            labeling_mode=labeling_mode,
            sampling_health_warnings=[],
        )

    def _build_non_overlap_horizons_for_split(
        self,
        split: str,
        frame,
        schema,
        artifacts_dir: Path,
    ) -> SplitHorizonBuildResult:
        """构建 non-overlap horizon 集合，供 Phase 2 使用。

        以 horizon 为步长从 eligible 窗口中选取互不重叠的窗口，
        保证相邻窗口的 window_start 间隔 >= horizon，无数据泄漏。
        """
        context = self._prepare_window_index_context(
            split, frame, requires_sampling=False
        )
        non_overlap_entries = _full_time_pool_entries(
            context.eligible_entries,
            horizon=self.config.horizon,
            mode="non_overlap",
            stride=self.config.horizon,
        )
        non_overlap_labels = [
            context.strata_by_start[entry.window_start]
            for entry in non_overlap_entries
        ]
        sampled = _sampled_from_entries(
            non_overlap_entries,
            non_overlap_labels,
            source="non_overlap",
        )
        labeling_mode = "non_overlap"
        self._logger.info(
            "non_overlap窗口选择完成 split=%s horizon窗口数=%d 分层标签数=%d 模式=%s",
            split,
            len(sampled),
            len({s.strata_label for s in sampled}),
            labeling_mode,
        )

        return self._finalize_split_horizon_build(
            split=split,
            frame=frame,
            schema=schema,
            artifacts_dir=artifacts_dir,
            context=context,
            sampled=sampled,
            full_time_pool_entries=[],
            sampling_applied=False,
            labeling_mode=labeling_mode,
            sampling_health_warnings=[],
        )

    def _finalize_split_horizon_build(
        self,
        *,
        split: str,
        frame,
        schema,
        artifacts_dir: Path,
        context: SplitWindowIndexContext,
        sampled: List[SampledHorizon],
        full_time_pool_entries: List[WindowIndexEntry],
        sampling_applied: bool,
        labeling_mode: str,
        sampling_health_warnings: List[str],
    ) -> SplitHorizonBuildResult:
        """把已选择的窗口写成 split 产物并构建 HorizonRecord。

        采样分支和 all_eligible 分支共享 window_index 审计字段、HorizonBuilder
        对齐逻辑和返回结构，所以收口到这里避免两套落盘逻辑发生漂移。
        """
        index_frame = context.indexer.to_frame(context.all_entries)
        sampled_starts = {s.window_start for s in sampled}
        eligible_starts = {e.window_start for e in context.eligible_entries}
        source_by_start = {s.window_start: s.sample_source for s in sampled}
        import polars as pl

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
        if sampled:
            source_frame = pl.DataFrame(
                {
                    "window_start": list(source_by_start.keys()),
                    "sample_source": list(source_by_start.values()),
                }
            )
            index_frame = index_frame.join(source_frame, on="window_start", how="left")
        else:
            index_frame = index_frame.with_columns(pl.lit(None).alias("sample_source"))
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
        return SplitHorizonBuildResult(
            horizons=horizons,
            window_index_path=path,
            all_entries=context.all_entries,
            eligible_entries=context.eligible_entries,
            sampled_horizons=sampled,
            full_time_pool_entries=full_time_pool_entries,
            strata_by_start=context.strata_by_start,
            num_eligible_windows=len(context.eligible_entries),
            sampling_applied=sampling_applied,
            labeling_mode=labeling_mode,
            sample_source_counts=_sample_source_counts(horizons),
            sampling_health_warnings=sampling_health_warnings,
        )

    def _sample_train_windows(
        self,
        entries: List[WindowIndexEntry],
        labels: List[str],
        num_samples: int,
    ) -> Tuple[List[SampledHorizon], List[WindowIndexEntry], int, bool]:
        """从 train eligible 窗口中选出最终训练样本。

        这里先保证 full-time 时间覆盖，再用分层 opportunity 采样补足预算；
        这样训练集既包含完整时间分布，也保留收益/波动 strata 的代表性。
        """
        if not self.config.time_distribution_sampling.enabled:
            sampler = StratifiedWindowSampler(
                strategy=self.config.sampling_strategy,
                min_gap_between_samples=self.config.sampling_health.min_gap_between_samples,
                flat_low_vol_max_ratio=self.config.sampling_health.flat_low_vol_max_ratio,
                allow_overlap_relaxation=self.config.sampling_health.allow_overlap_relaxation,
                seed=self.config.seed,
            )
            sampled = sampler.sample(
                entries, num_samples=num_samples, strata_labels=labels
            )
            for item in sampled:
                object.__setattr__(item, "sample_source", "opportunity")
            return (
                sampled,
                [],
                sampler.last_effective_min_gap_between_samples,
                sampler.last_overlap_relaxation_applied,
            )

        full_time_pool = _full_time_pool_entries(
            entries,
            horizon=self.config.horizon,
            mode=self.config.time_distribution_sampling.full_time_mode,
            stride=self.config.time_distribution_sampling.full_time_stride,
        )
        full_time_quota = int(
            np.ceil(num_samples * self.config.time_distribution_sampling.min_train_ratio)
        )
        full_time_quota = min(full_time_quota, len(full_time_pool), num_samples)
        full_time_selected = _take_evenly_spaced(full_time_pool, full_time_quota)
        labels_by_start = {
            entry.window_start: label for entry, label in zip(entries, labels)
        }
        full_time_sampled = _sampled_from_entries(
            full_time_selected,
            [labels_by_start[e.window_start] for e in full_time_selected],
            source="full_time",
        )

        selected_starts = [s.window_start for s in full_time_sampled]
        opportunity_entries, opportunity_labels = _filter_entries_away_from_starts(
            entries,
            labels,
            selected_starts,
            min_gap=self.config.sampling_health.min_gap_between_samples,
        )
        opportunity_quota = max(0, num_samples - len(full_time_sampled))
        opportunity_sampled: List[SampledHorizon] = []
        effective_min_gap = self.config.sampling_health.min_gap_between_samples
        overlap_relaxation_applied = False
        if opportunity_quota > 0 and opportunity_entries:
            sampler = StratifiedWindowSampler(
                strategy=self.config.sampling_strategy,
                min_gap_between_samples=self.config.sampling_health.min_gap_between_samples,
                flat_low_vol_max_ratio=self.config.sampling_health.flat_low_vol_max_ratio,
                allow_overlap_relaxation=self.config.sampling_health.allow_overlap_relaxation,
                seed=self.config.seed,
            )
            opportunity_quota = min(opportunity_quota, len(opportunity_entries))
            opportunity_sampled = sampler.sample(
                opportunity_entries,
                num_samples=opportunity_quota,
                strata_labels=opportunity_labels,
            )
            for item in opportunity_sampled:
                object.__setattr__(item, "sample_source", "opportunity")
            effective_min_gap = sampler.last_effective_min_gap_between_samples
            overlap_relaxation_applied = sampler.last_overlap_relaxation_applied

        sampled = _merge_sampled_sources(full_time_sampled, opportunity_sampled)
        if len(sampled) < num_samples:
            shortfall = num_samples - len(sampled)
            spare_entries, spare_labels = _filter_entries_away_from_starts(
                entries,
                labels,
                [s.window_start for s in sampled],
                min_gap=max(1, effective_min_gap),
            )
            spare_pairs = _take_evenly_spaced(
                list(zip(spare_entries, spare_labels)), shortfall
            )
            extra = _sampled_from_entries(
                [pair[0] for pair in spare_pairs],
                [pair[1] for pair in spare_pairs],
                source="opportunity",
            )
            sampled = _merge_sampled_sources(sampled, extra)

        full_time_ratio = (
            sum(1 for s in sampled if s.sample_source in {"full_time", "both"})
            / max(len(sampled), 1)
        )
        if full_time_ratio + 1e-12 < self.config.time_distribution_sampling.min_train_ratio:
            self._logger.warning(
                "full_time_sample_ratio_below_target split=train ratio=%.6f target=%.6f pool=%d num_samples=%d",
                full_time_ratio,
                self.config.time_distribution_sampling.min_train_ratio,
                len(full_time_pool),
                num_samples,
            )
        return sampled[:num_samples], full_time_pool, effective_min_gap, overlap_relaxation_applied

    def _ensure_train_min_coverage(
        self,
        train_horizons: List[Any],
        train_reject: RejectStats,
        train_result: SplitHorizonBuildResult,
        train_frame,
        schema,
    ) -> Tuple[List[Any], RejectStats, Dict[str, Any]]:
        """在 DP 后回填 no-trade 和低机会样本的最低覆盖。

        DP 过滤可能让训练集过度偏向有交易/高收益窗口；这里从未使用的 full-time
        pool 中补候选，并替换掉部分 opportunity 样本，让 codebook 看到足够的
        no-trade 与低机会场景。
        """
        control = self.config.no_trade_control
        initial = _coverage_stats(
            train_horizons,
            low_opportunity_return_quantile=control.low_opportunity_return_quantile,
            min_profit_gate=control.min_profit_gate,
        )
        report: Dict[str, Any] = {
            "initial_no_trade_ratio": initial["no_trade_ratio"],
            "initial_low_opportunity_ratio": initial["low_opportunity_ratio"],
            "final_no_trade_ratio": initial["no_trade_ratio"],
            "final_low_opportunity_ratio": initial["low_opportunity_ratio"],
            "low_opportunity_return_threshold": initial["low_opportunity_threshold"],
            "backfill_candidates": 0,
            "backfill_selected": 0,
            "replaced_opportunity": 0,
            "warnings": [],
        }
        target_count = len(train_horizons)
        if target_count == 0:
            return train_horizons, train_reject, report
        required_no_trade = int(np.ceil(target_count * control.min_no_trade_ratio))
        required_low = int(np.ceil(target_count * control.min_low_opportunity_ratio))
        current_no_trade = int(initial["no_trade_count"])
        current_low = int(initial["low_opportunity_count"])
        if (
            current_no_trade >= required_no_trade
            and current_low >= required_low
        ):
            return train_horizons, train_reject, report
        if not control.resample_when_below_min:
            report["warnings"].append("resample_when_below_min=false")
            return train_horizons, train_reject, report

        used_starts = {rec.start_index for rec in train_horizons}
        candidate_entries = [
            entry
            for entry in train_result.full_time_pool_entries
            if entry.window_start not in used_starts
        ]
        candidate_labels = [
            train_result.strata_by_start[entry.window_start]
            for entry in candidate_entries
        ]
        candidate_entries, candidate_labels = _filter_entries_away_from_starts(
            candidate_entries,
            candidate_labels,
            [rec.start_index for rec in train_horizons],
            min_gap=self.config.sampling_health.min_gap_between_samples,
        )
        report["backfill_candidates"] = len(candidate_entries)
        if not candidate_entries:
            report["warnings"].append("no_unused_full_time_candidates_available")
            self._logger.warning(
                "no_trade_low_opportunity_backfill_unavailable no_trade=%.6f/%.6f low=%.6f/%.6f",
                initial["no_trade_ratio"],
                control.min_no_trade_ratio,
                initial["low_opportunity_ratio"],
                control.min_low_opportunity_ratio,
            )
            return train_horizons, train_reject, report

        builder = HorizonBuilder(
            self.config.horizon, schema, self.config.dp.cost_config.reward_alignment
        )
        candidate_sampled = _sampled_from_entries(
            candidate_entries,
            candidate_labels,
            source="full_time",
        )
        candidate_horizons = builder.build(
            train_frame, candidate_sampled, pair=self.config.pair, split="train"
        )
        candidate_horizons, candidate_reject = _generate_demos(
            self.config, candidate_horizons
        )

        low_threshold = float(initial["low_opportunity_threshold"])
        selected: List[Any] = []
        selected_ids = set()

        def select_record(rec: Any) -> None:
            """记录已选回填样本并维护去重集合。"""
            selected.append(rec)
            selected_ids.add(rec.sample_id)

        missing_no_trade = max(0, required_no_trade - current_no_trade)
        for rec in candidate_horizons:
            if missing_no_trade <= 0:
                break
            if _is_no_trade_record(rec):
                select_record(rec)
                missing_no_trade -= 1

        selected_low_count = sum(
            1 for rec in selected if _is_low_opportunity_record(rec, low_threshold)
        )
        missing_low = max(0, required_low - current_low - selected_low_count)
        for rec in candidate_horizons:
            if missing_low <= 0:
                break
            if rec.sample_id in selected_ids:
                continue
            if _is_low_opportunity_record(rec, low_threshold):
                select_record(rec)
                missing_low -= 1

        if not selected:
            report["warnings"].append("full_time_candidates_did_not_improve_coverage")
            return train_horizons, train_reject, report

        removable = _rank_replacement_candidates(train_horizons, low_threshold)
        if len(removable) < len(selected):
            report["warnings"].append(
                f"insufficient_replacement_candidates={len(removable)}<{len(selected)}"
            )
            selected = selected[: len(removable)]
        if not selected:
            return train_horizons, train_reject, report

        remove_ids = {rec.sample_id for rec in removable[: len(selected)]}
        final_horizons = [
            rec for rec in train_horizons if rec.sample_id not in remove_ids
        ] + selected
        final_horizons.sort(key=lambda rec: rec.start_index)
        merged_reject = _reject_stats_for_final_records(
            final_horizons,
            train_horizons,
            train_reject,
            candidate_horizons,
            candidate_reject,
        )
        final = _coverage_stats(
            final_horizons,
            low_opportunity_return_quantile=control.low_opportunity_return_quantile,
            min_profit_gate=control.min_profit_gate,
        )
        report.update(
            {
                "final_no_trade_ratio": final["no_trade_ratio"],
                "final_low_opportunity_ratio": final["low_opportunity_ratio"],
                "low_opportunity_return_threshold": final["low_opportunity_threshold"],
                "backfill_selected": len(selected),
                "replaced_opportunity": len(remove_ids),
            }
        )
        if final["no_trade_ratio"] + 1e-12 < control.min_no_trade_ratio:
            report["warnings"].append("min_no_trade_ratio_not_met_after_backfill")
        if final["low_opportunity_ratio"] + 1e-12 < control.min_low_opportunity_ratio:
            report["warnings"].append(
                "min_low_opportunity_ratio_not_met_after_backfill"
            )
        self._logger.info(
            "no_trade_low_opportunity_backfill_done initial_no_trade=%.6f final_no_trade=%.6f initial_low=%.6f final_low=%.6f selected=%d replaced=%d warnings=%s",
            initial["no_trade_ratio"],
            final["no_trade_ratio"],
            initial["low_opportunity_ratio"],
            final["low_opportunity_ratio"],
            len(selected),
            len(remove_ids),
            report["warnings"],
        )
        return final_horizons, merged_reject, report


def _check_prospective_diagnostic(config: Phase1DataProcessConfig) -> None:
    """运行期校验 hindsight 实验是否绑定 prospective 对照。

    该检查保护实验解释性：后视分层只能作为离线 curation 手段，必须配套
    prospective_past 诊断或显式风险声明。
    """
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
    """根据 factor 配置创建输入 schema 校验器。

    有显式/默认 factor list 时使用固定特征清单；否则保留 legacy numeric
    自动识别模式，兼容旧数据处理入口。
    """
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
    """根据 reward 对齐模式选择 split 边界禁运长度。

    paper_formula 与 next_row_execution 使用的最后成交/markout 行不同，因此
    需要不同 embargo，避免窗口标注越过 split 文件边界。
    """
    if config.dp.cost_config.reward_alignment == "paper_formula":
        return config.sampling_health.split_boundary_embargo
    return config.sampling_health.next_row_split_boundary_embargo


def _filter_split_boundary_entries(
    entries: List[WindowIndexEntry],
    frame_height: int,
    embargo: int,
) -> Tuple[List[WindowIndexEntry], int]:
    """排除 markout 行过于接近 split 末尾的候选窗口。

    这样做是为了保证每个保留下来的窗口都有完整 markout 缓冲区，避免边界处
    的 label 依赖不存在或跨 split 的未来数据。
    """
    if embargo <= 0:
        return list(entries), 0
    max_markout_row = frame_height - 1 - embargo
    eligible = [e for e in entries if e.last_markout_row <= max_markout_row]
    return eligible, len(entries) - len(eligible)


def _num_samples_for_split(
    config: Phase1DataProcessConfig, split: str, num_entries: int
) -> int:
    """返回当前 split 的目标 horizon 数。

    train 受 ``num_demos`` 训练预算限制；val/test 默认按 horizon 步长评估，
    目标数量为全部边界合格窗口按 ``horizon`` 整包折算后的数量。
    """
    if split == "train":
        return min(config.num_demos, num_entries)
    mode = _eval_labeling_mode(config, split)
    if mode == "horizon_stride":
        return _horizon_stride_eval_count(config, num_entries)
    if mode == "all_eligible":
        return num_entries
    raise ValueError(f"非法 eval labeling mode: {mode}")


def _eval_labeling_mode(config: Phase1DataProcessConfig, split: str) -> str:
    """读取当前 eval split 的 label 生成模式。"""
    if split == "val":
        return config.eval_labeling.val_mode
    if split == "test":
        return config.eval_labeling.test_mode
    raise ValueError(f"非法 eval split: {split}")


def _horizon_stride_eval_count(config: Phase1DataProcessConfig, num_entries: int) -> int:
    """val/test 默认目标数量: eligible 起点数按 horizon 整包折算。"""
    if num_entries <= 0:
        return 0
    return num_entries // max(config.horizon, 1)


def _select_eval_label_entries(
    config: Phase1DataProcessConfig,
    split: str,
    entries: Sequence[WindowIndexEntry],
    labels: Sequence[str],
) -> Tuple[List[WindowIndexEntry], List[str], str, str]:
    """按 eval labeling 契约选择最终需要 DP label 的窗口。"""
    mode = _eval_labeling_mode(config, split)
    if mode == "all_eligible":
        return list(entries), list(labels), "all_eligible", "eval_all_eligible"
    if mode == "horizon_stride":
        step = max(config.horizon, 1)
        target_count = _horizon_stride_eval_count(config, len(entries))
        return (
            list(entries[::step])[:target_count],
            list(labels[::step])[:target_count],
            "horizon_stride",
            "eval_horizon_stride",
        )
    raise ValueError(f"非法 eval labeling mode: {mode}")


def _full_time_pool_entries(
    entries: Sequence[WindowIndexEntry],
    *,
    horizon: int,
    mode: str,
    stride: int,
) -> List[WindowIndexEntry]:
    """从 eligible 窗口中抽出用于时间覆盖的候选池。

    non_overlap 用 horizon 作为步长保证互不重叠；stride 模式允许更密集覆盖，
    用于在 ``num_demos`` 较大时仍满足 full-time 比例。
    """
    if mode == "non_overlap":
        step = horizon
    elif mode == "stride":
        step = max(1, int(stride))
    else:
        raise ValueError(f"非法 full_time_mode: {mode}")
    if len(entries) == 0:
        return []
    starts = np.fromiter(
        (int(entry.window_start) for entry in entries),
        dtype=np.int64,
        count=len(entries),
    )
    keep = starts % step == 0
    return [entry for entry, flag in zip(entries, keep) if bool(flag)]


def _take_evenly_spaced(items: Sequence[Any], count: int) -> List[Any]:
    """从有序候选中均匀取样，保留时间覆盖而不是只取头部。"""
    if count <= 0 or len(items) == 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    indices = np.unique(
        np.rint(np.linspace(0, len(items) - 1, num=count)).astype(np.int64)
    )
    if indices.size < count:
        extras = np.setdiff1d(
            np.arange(len(items), dtype=np.int64), indices, assume_unique=True
        )
        indices = np.sort(np.concatenate([indices, extras[: count - indices.size]]))
    return [items[int(idx)] for idx in indices[:count]]


def _sampled_from_entries(
    entries: Sequence[WindowIndexEntry],
    labels: Sequence[str],
    *,
    source: str,
) -> List[SampledHorizon]:
    """把窗口索引项转换成 HorizonBuilder 可消费的 SampledHorizon。

    采样分支和 all_eligible 分支都通过这个轻量结构传递窗口边界、
    strata label 和 sample_source，避免 HorizonBuilder 了解上游选择策略。
    """
    out: List[SampledHorizon] = []
    for entry, label in zip(entries, labels):
        out.append(
            SampledHorizon(
                sample_id=_sample_id_for_entry(entry),
                window_start=entry.window_start,
                window_end=entry.window_end,
                last_execution_row=entry.last_execution_row,
                last_markout_row=entry.last_markout_row,
                strata_label=label,
                sample_source=source,
            )
        )
    return out


def _sample_id_for_entry(entry: WindowIndexEntry) -> str:
    """为窗口生成稳定 sample_id，确保重复运行和跨产物 join 可对齐。"""
    return f"s_{entry.window_start:08d}_{entry.window_start:06d}"


def _filter_entries_away_from_starts(
    entries: Sequence[WindowIndexEntry],
    labels: Sequence[str],
    starts: Sequence[int],
    *,
    min_gap: int,
) -> Tuple[List[WindowIndexEntry], List[str]]:
    """过滤掉距离已选窗口过近的候选。

    train 采样和回填都会用它维护起点间隔，降低训练样本之间的时间自相关；
    val/test 的 all_eligible 路径不会调用这个过滤。
    """
    if len(starts) == 0 or min_gap <= 0:
        return list(entries), list(labels)
    if len(entries) == 0:
        return [], []
    anchor_starts = np.unique(np.asarray(starts, dtype=np.int64))
    entry_starts = np.fromiter(
        (int(entry.window_start) for entry in entries),
        dtype=np.int64,
        count=len(entries),
    )
    positions = np.searchsorted(anchor_starts, entry_starts)
    left_ok = np.ones(len(entries), dtype=np.bool_)
    has_left = positions > 0
    left_ok[has_left] = (
        entry_starts[has_left] - anchor_starts[positions[has_left] - 1] >= min_gap
    )
    right_ok = np.ones(len(entries), dtype=np.bool_)
    has_right = positions < anchor_starts.size
    right_ok[has_right] = (
        anchor_starts[positions[has_right]] - entry_starts[has_right] >= min_gap
    )
    keep = left_ok & right_ok
    kept_entries = [entry for entry, flag in zip(entries, keep) if bool(flag)]
    kept_labels = [label for label, flag in zip(labels, keep) if bool(flag)]
    return kept_entries, kept_labels


def _merge_sampled_sources(
    primary: Sequence[SampledHorizon],
    secondary: Sequence[SampledHorizon],
) -> List[SampledHorizon]:
    """合并多个采样来源，并在同一窗口重复出现时标记为 both。

    full-time 覆盖和 opportunity 采样可能选中同一个起点，保留单条样本能避免
    重复训练，同时通过 source 字段保留审计信息。
    """
    by_start: Dict[int, SampledHorizon] = {}
    for item in list(primary) + list(secondary):
        existing = by_start.get(item.window_start)
        if existing is None:
            by_start[item.window_start] = item
            continue
        source = (
            existing.sample_source
            if existing.sample_source == item.sample_source
            else "both"
        )
        by_start[item.window_start] = SampledHorizon(
            sample_id=existing.sample_id,
            window_start=existing.window_start,
            window_end=existing.window_end,
            last_execution_row=existing.last_execution_row,
            last_markout_row=existing.last_markout_row,
            strata_label=existing.strata_label,
            sample_source=source,
        )
    return [by_start[start] for start in sorted(by_start)]


def _sample_source_counts(records: Sequence[Any]) -> Dict[str, int]:
    """统计样本来源分布，用于 manifest 和覆盖审计。"""
    counts: Dict[str, int] = {}
    for rec in records:
        source = getattr(rec, "sample_source", "unknown") or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _coverage_stats(
    records: Sequence[Any],
    *,
    low_opportunity_return_quantile: float,
    min_profit_gate: float,
) -> Dict[str, Any]:
    """计算 no-trade 和低机会样本覆盖率。

    这些统计用于判断训练集是否只剩高收益/有交易样本；低机会阈值由 reward
    分位数和 profit gate 共同决定。
    """
    if len(records) > 0:
        returns = np.fromiter(
            (float(np.sum(rec.rewards or [], dtype=np.float64)) for rec in records),
            dtype=np.float64,
            count=len(records),
        )
        q = float(
            np.quantile(
                returns,
                min(max(float(low_opportunity_return_quantile), 0.0), 1.0),
            )
        )
        low_threshold = max(q, float(min_profit_gate))
    else:
        returns = np.asarray([], dtype=np.float64)
        low_threshold = float(min_profit_gate)
    no_trade = sum(1 for rec in records if _is_no_trade_record(rec))
    low = int(np.count_nonzero(returns <= low_threshold)) if returns.size else 0
    total = len(records)
    return {
        "total": total,
        "no_trade_count": no_trade,
        "no_trade_ratio": no_trade / max(total, 1),
        "low_opportunity_count": low,
        "low_opportunity_ratio": low / max(total, 1),
        "low_opportunity_threshold": low_threshold,
    }


def _is_no_trade_record(rec: Any) -> bool:
    """判断一个 DP 结果是否全程保持 no-trade 动作。"""
    actions = list(rec.actions or [])
    return bool(actions) and all(int(action) == 1 for action in actions)


def _is_low_opportunity_record(rec: Any, threshold: float) -> bool:
    """判断一个 horizon 的总 reward 是否落入低机会区间。"""
    return float(sum(rec.rewards or [])) <= threshold


def _rank_replacement_candidates(
    records: Sequence[Any],
    low_threshold: float,
) -> List[Any]:
    """为回填替换选择优先被移除的训练样本。

    替换时优先移除非 full-time、非低机会/非 no-trade 且收益更高的样本，
    尽量减少对时间覆盖和稀缺场景覆盖的破坏。
    """
    def priority(rec: Any) -> Tuple[int, int, float]:
        """返回排序键，数值越小越适合作为替换对象。"""
        source = getattr(rec, "sample_source", "")
        is_full_time = source in {"full_time", "both"}
        is_low = _is_low_opportunity_record(rec, low_threshold)
        is_no_trade = _is_no_trade_record(rec)
        return (
            1 if is_full_time else 0,
            1 if (is_low or is_no_trade) else 0,
            -float(sum(rec.rewards or [])),
        )

    return sorted(records, key=priority)


def _reject_stats_for_final_records(
    final_records: Sequence[Any],
    original_records: Sequence[Any],
    original_stats: RejectStats,
    candidate_records: Sequence[Any],
    candidate_stats: RejectStats,
) -> RejectStats:
    """重建回填替换后的 reject 统计。

    回填会把原训练样本和候选样本混合，因此需要按 sample_id 重新拼接每条
    horizon 的 reject count/rate，保证保存的 reject_stats 对应最终 records。
    """
    counts_by_id: Dict[str, int] = {}
    rates_by_id: Dict[str, float] = {}
    for records, stats in (
        (original_records, original_stats),
        (candidate_records, candidate_stats),
    ):
        counts = list(stats.per_horizon_reject_count or [])
        rates = list(stats.per_horizon_reject_rate or [])
        for idx, rec in enumerate(records):
            counts_by_id[rec.sample_id] = int(counts[idx]) if idx < len(counts) else 0
            rates_by_id[rec.sample_id] = float(rates[idx]) if idx < len(rates) else 0.0
    final_counts = [counts_by_id.get(rec.sample_id, 0) for rec in final_records]
    final_rates = [rates_by_id.get(rec.sample_id, 0.0) for rec in final_records]
    worst = sorted(
        (
            {
                "sample_id": rec.sample_id,
                "rate": rates_by_id.get(rec.sample_id, 0.0),
                "rejected": counts_by_id.get(rec.sample_id, 0),
                "window_start": rec.start_index,
                "strata": rec.strata_label,
            }
            for rec in final_records
        ),
        key=lambda item: item["rate"],
        reverse=True,
    )[:10]
    return RejectStats(
        dataset_reject_rate=sum(final_rates) / max(len(final_rates), 1),
        per_horizon_reject_count=final_counts,
        per_horizon_reject_rate=final_rates,
        worst_reject_horizons=worst,
        reject_by_action_pair={},
    )


def _check_overlap_health_feasibility(
    config: Phase1DataProcessConfig,
    num_samples: int,
    entries: List[WindowIndexEntry],
) -> None:
    """在采样前判断 overlap guardrail 是否理论可行。

    如果给定数据长度、horizon 和 ``num_demos`` 已经不可能满足最大重叠率，
    这里提前报错，避免采样器跑完后才得到不可解释的健康检查失败。
    """
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
    """为 horizon 集合运行 DP teacher 并返回可训练 demonstration。

    这里集中构造成本模型、reward alignment 和 planner，是为了保证 train/val/test
    使用完全一致的交易语义和并行生成参数。
    """
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
        max_workers=config.dp_workers,
        worker_chunksize=config.dp_worker_chunksize,
    )
    return gen.generate(horizons)


def _log_reward_distribution(logger, horizons, split: str) -> None:
    """输出 DP teacher reward 分布诊断。

    这些统计不改变产物，只用于快速发现 reward 全零、极端偏态或动作分布异常，
    方便在长批次运行时从日志中判断数据质量。
    """
    import math as _math

    all_rewards = [v for rec in horizons for v in (rec.rewards or [])]
    if not all_rewards:
        logger.warning("reward_distribution_diagnostic split=%s 说明=无 reward 数据", split)
        return

    n = len(all_rewards)
    horizon_returns = []
    for rec in horizons:
        if rec.rewards:
            horizon_returns.append(sum(rec.rewards))

    mean_val = sum(all_rewards) / n
    var_val = sum((v - mean_val) ** 2 for v in all_rewards) / max(n - 1, 1)
    std_val = _math.sqrt(max(var_val, 0.0))
    sorted_r = sorted(all_rewards)
    zero_count = sum(1 for v in all_rewards if abs(v) < 1e-9)
    positive_count = sum(1 for v in all_rewards if v > 1e-9)
    negative_count = sum(1 for v in all_rewards if v < -1e-9)

    def _pct(vals, p):
        """计算线性插值百分位，避免为日志诊断引入额外依赖。"""
        k = (len(vals) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= len(vals):
            return vals[-1]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    action_counter = {}
    for rec in horizons:
        for a in (rec.actions or []):
            action_counter[a] = action_counter.get(a, 0) + 1
    total_actions = sum(action_counter.values()) or 1

    if horizon_returns:
        hr_sorted = sorted(horizon_returns)
        hr_mean = sum(horizon_returns) / len(horizon_returns)
        hr_pos = sum(1 for v in horizon_returns if v > 1e-9)
    else:
        hr_sorted = []
        hr_mean = 0.0
        hr_pos = 0

    logger.warning(
        "reward_distribution_diagnostic 说明=DP teacher 原始 reward 分布 split=%s "
        "step_rewards_n=%d mean=%.10f std=%.10f "
        "min=%.10f p1=%.10f p5=%.10f p25=%.10f p50=%.10f p75=%.10f p95=%.10f p99=%.10f max=%.10f "
        "zero_count=%d(%.4f) positive_count=%d(%.4f) negative_count=%d(%.4f) "
        "horizon_n=%d horizon_return_mean=%.10f horizon_return_positive=%d(%.4f) "
        "horizon_return_p50=%.10f horizon_return_p95=%.10f horizon_return_max=%.10f "
        "action_dist=%s",
        split,
        n, mean_val, std_val,
        sorted_r[0], _pct(sorted_r, 1), _pct(sorted_r, 5), _pct(sorted_r, 25),
        _pct(sorted_r, 50), _pct(sorted_r, 75), _pct(sorted_r, 95), _pct(sorted_r, 99),
        sorted_r[-1],
        zero_count, zero_count / n,
        positive_count, positive_count / n,
        negative_count, negative_count / n,
        len(horizon_returns), hr_mean,
        hr_pos, hr_pos / max(len(horizon_returns), 1),
        _pct(hr_sorted, 50) if hr_sorted else 0.0,
        _pct(hr_sorted, 95) if hr_sorted else 0.0,
        hr_sorted[-1] if hr_sorted else 0.0,
        {str(k): f"{v / total_actions:.4f}" for k, v in sorted(action_counter.items())},
    )


def _file_audit(path: str) -> dict:
    """记录输入文件的尺寸和 mtime，用于数据处理 hash 与审计。"""
    target = Path(path)
    stat = target.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _relative_to_artifact(path: Path, artifact_dir: Path) -> str:
    """把产物路径尽量写成相对 artifact_dir 的形式。

    manifest 中使用相对路径便于整体移动 artifact 目录；不在目录内的路径则
    保留原值，避免路径转换失败中断运行。
    """
    try:
        return str(Path(path).relative_to(artifact_dir))
    except ValueError:
        return str(path)

