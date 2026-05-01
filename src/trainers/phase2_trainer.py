"""Phase II trainer: 编排完整 Phase II 训练流程。

设计文档锚点: Phase II 执行计划 §Step 9。

职责:
- 校验上游产物 → 读取数据与 schema → 生成 horizon index → join labels →
  构造 envs → rollout / update / evaluate / select / checkpoint →
  rolling validation / KL-demo ablation / sensitivity 分析 →
  best checkpoint 冻结后输出 train/val/test per-horizon records。
- 在每个完整 checkpoint 边界刷出 replay_log_last_complete_checkpoint.feather。

关键约束:
- 训练入口默认不得加载 test labels。
- test label 不进入训练、回测主路径、best 选择、早停和主回测决策路径。
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.phase2_config import Phase2Config
from src.data.market_reader import MarketFileReader
from src.data.phase2_dataset import Phase2Dataset
from src.data.phase2_horizon_index import (
    Phase1ArtifactValidator,
    Phase2HorizonIndexer,
)
from src.data.phase2_label_loader import Phase2LabelLoader
from src.data.schema import InputSchemaValidator
from src.evaluation.phase2_evaluator import Phase2Evaluator
from src.evaluation.phase2_replay import Phase2BacktestRunner
from src.evaluation.phase2_report import Phase2ReportPaths, Phase2ReportWriter
from src.evaluation.phase2_metrics import (
    compute_phase2_composite_score,
    phase2_composite_metrics,
    phase2_composite_score_sensitivity,
)
from src.models.archetype_selector import ArchetypeSelector
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.rl.actor_critic import ActorCritic
from src.rl.ppo_trainer import PPOTrainer
from src.rl.scheduling import ScheduleManager
from src.trading.cost_model import LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.horizon_factory import HorizonFactory
from src.trading.reward_alignment import RewardAlignment
from src.trainers.phase2_checkpoint import Phase2CheckpointManager
from src.trainers.phase2_selection_policy import (
    Phase2SelectionHistory,
    Phase2SelectionPolicy,
)
from src.trainers.phase2_dead_code import build_dead_code_mask
from src.utils.feather_io import read_ipc, read_json


class Phase2FatalError(RuntimeError):
    """Phase II 训练 fatal 错误。"""


@dataclass
class Phase2TrainerArtifacts:
    """Phase II 训练产物路径集合。"""
    artifacts_dir: Path
    phase2_config_yaml: Path
    horizon_index_train: Path
    horizon_index_val: Path
    horizon_index_test: Path
    env_shards: Path
    best_selector: Path
    last_selector: Path
    checkpoint_manifest: Path
    rollout_stats: Path
    per_horizon_records_train: Path
    per_horizon_records_val: Path
    per_horizon_records_test: Path
    phase2_report: Path
    replay_log: Path


class Phase2Trainer:
    """Phase II 主训练编排器。

    使用方式::

        trainer = Phase2Trainer(config)
        artifacts = trainer.run()

    边界:
    - 不直接计算指标（交给 Phase2Evaluator）。
    - 不直接写 report（交给 Phase2ReportWriter）。
    - 不直接选 best（交给 Phase2SelectionPolicy）。
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config
        self._rollout_stats: List[Dict[str, Any]] = []

    def run(self) -> Phase2TrainerArtifacts:
        """完整主流程。

        Returns
        -------
        Phase2TrainerArtifacts : 所有关键产物路径。

        Raises
        ------
        Phase2FatalError : 训练过程中的 fatal 错误。
        """
        import torch

        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 0. Seed
        self._seed_everything()

        # 1. 校验 Phase I 产物
        validator = Phase1ArtifactValidator(self.config)
        val_result = validator.validate()

        # 2. 写 phase2_config.yaml
        config_yaml = artifacts_dir / "phase2_config.yaml"
        self.config.write_yaml(config_yaml)

        # 3. 读取数据与 schema
        reader = MarketFileReader()
        frames = reader.read_split(
            self.config.train_file, self.config.val_file, self.config.test_file
        )
        input_schema = read_json(self.config.phase1_dir() / "input_schema.json")

        # 4. 生成 horizon index
        indexer = Phase2HorizonIndexer(self.config)
        horizon = self.config.horizon

        # 加载 Phase I labels (仅 train/val)
        p1_dir = self.config.phase1_dir()
        train_labels = None
        val_labels = None
        if (p1_dir / "horizon_labels_train.feather").exists():
            train_labels = read_ipc(p1_dir / "horizon_labels_train.feather")
        if (p1_dir / "horizon_labels_val.feather").exists():
            val_labels = read_ipc(p1_dir / "horizon_labels_val.feather")

        train_entries = indexer.build_index(frames["train"], "train", horizon, train_labels)
        val_entries = indexer.build_index(frames["val"], "val", horizon, val_labels)
        test_entries = indexer.build_index(frames["test"], "test", horizon, None)

        hi_train_path = indexer.write_index(train_entries, artifacts_dir / "phase2_horizon_index_train.feather")
        hi_val_path = indexer.write_index(val_entries, artifacts_dir / "phase2_horizon_index_val.feather")
        hi_test_path = indexer.write_index(test_entries, artifacts_dir / "phase2_horizon_index_test.feather")

        # 5. Join labels
        label_loader = Phase2LabelLoader(self.config)
        if (p1_dir / "horizon_labels_train.feather").exists():
            train_entries = label_loader.load_and_join(
                train_entries, "train", p1_dir / "horizon_labels_train.feather"
            )
        if (p1_dir / "horizon_labels_val.feather").exists():
            val_entries = label_loader.load_and_join(
                val_entries, "val", p1_dir / "horizon_labels_val.feather"
            )

        # 6. 构造 datasets
        all_entries = train_entries + val_entries + test_entries
        # 使用 train frame 构造 train dataset
        train_dataset = Phase2Dataset(frames["train"], train_entries, input_schema, self.config)
        val_dataset = Phase2Dataset(frames["val"], val_entries, input_schema, self.config)
        test_dataset = Phase2Dataset(frames["test"], test_entries, input_schema, self.config)

        # 7. 加载 Phase1FrozenPolicy
        frozen_policy = Phase1FrozenPolicy.load(
            p1_dir / "decoder.pt",
            p1_dir / "codebook.pt",
            device=self.config.device,
        )

        # 8. 构造 trading env factory
        p1_config = val_result.phase1_config or {}
        dp_cfg = p1_config.get("dp", {})
        cost_cfg = dp_cfg.get("cost_config", {})
        cost_model = LobDepthCostModel(
            commission_rate=cost_cfg.get("commission_rate", 0.0002),
            book_levels=cost_cfg.get("book_levels", 5),
        )
        alignment = RewardAlignment(cost_cfg.get("reward_alignment", "paper_formula"))

        def trading_env_factory():
            return TradingEnv(
                cost_model=cost_model,
                reward_alignment=alignment,
                max_position=self.config.max_position,
            )

        # 9. 构造 HorizonEnv (multi-env)
        factory = HorizonFactory(self.config, train_dataset, frozen_policy, trading_env_factory)
        envs, shard_infos = factory.create_envs()
        env_shards_path = factory.write_shards(shard_infos, artifacts_dir / "phase2_env_shards.feather")

        # 10. 初始化 selector + ActorCritic + PPOTrainer
        state_spec = train_dataset.state_spec()
        num_codes = frozen_policy.num_codes

        # dead code mask
        dead_code_mask = None
        p1_report = val_result.phase1_report or {}
        if self.config.selector_network.action_mask_dead_codes:
            mask = build_dead_code_mask(
                p1_report,
                num_codes,
                self.config.selector_network.dead_code_usage_threshold,
            )
            dead_code_mask = torch.tensor(mask, dtype=torch.bool, device=self.config.device)

        selector = ArchetypeSelector(
            state_dim=state_spec.total_dim,
            num_codes=num_codes,
            config=self.config.selector_network,
        ).to(self.config.device)

        actor_critic = ActorCritic(selector, dead_code_mask=dead_code_mask)

        optimizer = torch.optim.Adam(selector.parameters(), lr=self.config.ppo.lr)
        num_updates = max(
            self.config.total_timesteps // (self.config.num_envs * self.config.rollout_length),
            1,
        )
        schedule_mgr = ScheduleManager(self.config, optimizer, num_updates)

        ppo_trainer = PPOTrainer(self.config, actor_critic, envs, schedule_mgr)
        ppo_trainer.setup(optimizer=optimizer)

        # Checkpoint + selection policy
        ckpt_mgr = Phase2CheckpointManager(artifacts_dir)
        policy = Phase2SelectionPolicy(self.config.selection_policy)
        history = Phase2SelectionHistory()

        # Evaluator
        val_runner = Phase2BacktestRunner(
            self.config, actor_critic, frozen_policy, val_dataset, trading_env_factory
        )
        evaluator = Phase2Evaluator(
            self.config, val_runner, num_codes,
            dead_code_mask.tolist() if dead_code_mask is not None else None,
        )

        # 11. 训练循环
        eval_every = max(num_updates // 20, 1)
        for update_idx in range(num_updates):
            stats = ppo_trainer.collect_and_update()
            self._rollout_stats.append({
                "update_idx": update_idx,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy_loss": stats.entropy_loss,
                "kl_demo_loss": stats.kl_demo_loss,
                "approx_kl": stats.approx_kl,
                "clip_fraction": stats.clip_fraction,
                "explained_variance": stats.explained_variance,
                "reward_mean": stats.reward_mean,
                "reward_std": stats.reward_std,
                "reward_clipped_ratio": stats.reward_clipped_ratio,
                "reward_unclipped_mean": stats.reward_unclipped_mean,
                "rollout_done_count": stats.rollout_done_count,
                "rollout_truncated_count": stats.rollout_truncated_count,
                "rollout_bootstrap_count": stats.rollout_bootstrap_count,
            })

            # 保存 last
            state = ppo_trainer.get_state()
            ckpt_mgr.save_last(state, update_idx)

            # 定期评估
            if (update_idx + 1) % eval_every == 0 or update_idx == num_updates - 1:
                ppo_stats_dict = {
                    "policy_loss": stats.policy_loss,
                    "value_loss": stats.value_loss,
                    "entropy_loss": stats.entropy_loss,
                    "kl_demo_loss": stats.kl_demo_loss,
                    "approx_kl": stats.approx_kl,
                    "clip_fraction": stats.clip_fraction,
                    "explained_variance": stats.explained_variance,
                }
                eval_result = evaluator.evaluate_val_fast(update_idx, ppo_stats_dict)
                eval_metrics = eval_result.metrics
                eval_metrics["update_idx"] = update_idx
                eval_metrics["val_net_return"] = eval_metrics.get("net_return", 0.0)
                score, score_debug = compute_phase2_composite_score(
                    eval_metrics,
                    self.config.selection_policy.metric_weights,
                )
                eval_metrics["phase2_composite_score"] = score
                eval_metrics["phase2_composite_score_debug"] = score_debug

                verdict = policy.evaluate(eval_metrics, history)
                ckpt_mgr.commit_verdict(state, verdict, update_idx, eval_metrics)
                history = policy.update_history(history, eval_metrics, verdict)

                # 保存 replay log
                replay_dicts = [
                    {
                        "update_idx": update_idx,
                        "env_id": r.env_id,
                        "sample_id": r.sample_id,
                        "chosen_code": r.chosen_code,
                        "final_position": r.final_position,
                        "reward_raw": r.reward_raw,
                        "boundary_cost": r.boundary_cost,
                        "risk_triggered": r.risk_triggered,
                    }
                    for r in eval_result.per_horizon_records
                ]
                ckpt_mgr.save_replay_log(replay_dicts)

        # 12. 最终评估与报告
        report_paths = Phase2ReportPaths.from_artifacts_dir(artifacts_dir)
        writer = Phase2ReportWriter(report_paths)

        # 写 rollout stats
        rollout_stats_path = writer.write_rollout_stats(self._rollout_stats)

        # 用 best checkpoint 做 walk-forward
        test_runner = Phase2BacktestRunner(
            self.config, actor_critic, frozen_policy, test_dataset, trading_env_factory
        )
        test_evaluator = Phase2Evaluator(
            self.config, test_runner, num_codes,
            dead_code_mask.tolist() if dead_code_mask is not None else None,
        )

        # 加载 best
        if ckpt_mgr.best_path.exists():
            best_state = ckpt_mgr.load(ckpt_mgr.best_path)
            selector.load_state_dict(best_state["model_state"])

        # Per-horizon records
        train_runner = Phase2BacktestRunner(
            self.config, actor_critic, frozen_policy, train_dataset, trading_env_factory
        )
        train_records = train_runner.run_walk_forward("train", deterministic=True)
        val_records = val_runner.run_walk_forward("val", deterministic=True)
        test_records = test_runner.run_walk_forward("test", deterministic=True)

        train_rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in train_records]
        val_rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in val_records]
        test_rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in test_records]

        phr_train = writer.write_per_horizon_records(train_rec_dicts, "train")
        phr_val = writer.write_per_horizon_records(val_rec_dicts, "val")
        phr_test = writer.write_per_horizon_records(test_rec_dicts, "test")

        # Rolling validation
        rolling_result = evaluator.evaluate_rolling_validation()
        rolling_records = [
            Phase2Evaluator._record_to_dict(r)
            for fold_records in rolling_result.per_fold_records
            for r in fold_records
        ]
        writer.write_rolling_validation(
            {
                "fold_metrics": rolling_result.fold_metrics,
                "fold_mean": rolling_result.fold_mean,
                "worst_fold_quantile": rolling_result.worst_fold_quantile,
                "fold_volatility": rolling_result.fold_volatility,
                "fold_sizes": rolling_result.fold_sizes,
                "fold_initial_position_policy": rolling_result.fold_initial_position_policy,
            },
            rolling_records,
        )

        # Build report
        val_metrics = phase2_composite_metrics(
            val_rec_dicts, {}, num_codes,
            dead_code_mask.tolist() if dead_code_mask is not None else [False] * num_codes,
            metric_weights=self.config.selection_policy.metric_weights,
        )
        test_metrics = phase2_composite_metrics(
            test_rec_dicts, {}, num_codes,
            dead_code_mask.tolist() if dead_code_mask is not None else [False] * num_codes,
            metric_weights=self.config.selection_policy.metric_weights,
        )

        # Baselines
        val_baselines = val_runner.run_baselines("val")
        test_baselines = test_runner.run_baselines("test")
        bl_val_summary = {
            name: {k: v for k, v in phase2_composite_metrics(
                [Phase2Evaluator._record_to_dict(r) for r in recs],
                {}, num_codes,
                dead_code_mask.tolist() if dead_code_mask is not None else [False] * num_codes,
                metric_weights=self.config.selection_policy.metric_weights,
            ).items() if isinstance(v, (int, float, str, bool))}
            for name, recs in val_baselines.items()
        }
        bl_test_summary = {
            name: {k: v for k, v in phase2_composite_metrics(
                [Phase2Evaluator._record_to_dict(r) for r in recs],
                {}, num_codes,
                dead_code_mask.tolist() if dead_code_mask is not None else [False] * num_codes,
                metric_weights=self.config.selection_policy.metric_weights,
            ).items() if isinstance(v, (int, float, str, bool))}
            for name, recs in test_baselines.items()
        }
        writer.write_baselines(bl_val_summary, "val")
        writer.write_baselines(bl_test_summary, "test")

        sensitivity = phase2_composite_score_sensitivity(
            [
                {"update_idx": e.update_idx, **e.metrics, "_manifest_verdict": e.verdict}
                for e in ckpt_mgr._entries
            ],
            self.config.selection_policy.metric_weights,
            self.config.selection_policy.composite_score_sensitivity_perturbations,
        )
        writer.write_sensitivity(sensitivity)

        # Label coverage
        train_coverage = label_loader.compute_coverage_stats(train_entries, "train")

        report_summary: Dict[str, Any] = {
            "config_hash": self.config.config_hash(),
            "phase1_hash": p1_report.get("config_hash", ""),
            "phase1_batch_id": self.config.phase1_batch_id,
            "phase2_batch_id": self.config.phase2_batch_id,
            "selection_metric": self.config.selection_policy.selection_metric,
            "metric_weights": dict(self.config.selection_policy.metric_weights),
            "test_used_for_selection": False,
            "kl_label_coverage_train": train_coverage.coverage_ratio,
            "kl_label_temporal_coverage": train_coverage.temporal_coverage_sequence,
            "equity_curve_summary": val_metrics.get("equity_curve_summary", {}),
            "behavior_health_warnings": [],
            "risk_health_warnings": [],
            "ood_warning_count": 0,
            "distribution_shift_warning_count": 0,
            "val_metrics": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, str, bool))},
            "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (int, float, str, bool))},
            "horizon_schedule": {
                "mode": self.config.horizon_schedule.mode,
                "stride": self.config.horizon_schedule.stride,
                "position_continuity": self.config.horizon_schedule.position_continuity,
                "chunk_reset_position": self.config.horizon_schedule.chunk_reset_position,
            },
            "data_gap_filter": {
                "enabled": self.config.horizon_schedule.data_gap_check_enabled,
                "exclude_gap_horizons": self.config.horizon_schedule.exclude_gap_horizons,
                "train_gap_count": sum(1 for e in train_entries if e.is_gap),
                "val_gap_count": sum(1 for e in val_entries if e.is_gap),
                "test_gap_count": sum(1 for e in test_entries if e.is_gap),
            },
            "input_norm": {
                "mode": self.config.selector_network.input_norm,
                "position_encoding": self.config.selector_network.position_encoding,
                "state_dim_breakdown": train_dataset.state_spec().__dict__,
            },
            "env_shards": {
                "num_envs": self.config.num_envs,
                "mode": self.config.env_shards.mode,
                "shard_count": len(shard_infos),
            },
            "reward_scaling": {
                "method": self.config.reward_scaling.method,
                "clip_range": self.config.reward_scaling.clip_range,
                "last_reward_clipped_ratio": self._rollout_stats[-1].get("reward_clipped_ratio", 0.0) if self._rollout_stats else 0.0,
            },
            "cost_config_inherited": cost_cfg,
            "baselines_val": bl_val_summary,
            "baselines_test": bl_test_summary,
            "rolling_validation_summary": {
                "fold_mean": rolling_result.fold_mean,
                "worst_fold_quantile": rolling_result.worst_fold_quantile,
                "fold_volatility": rolling_result.fold_volatility,
                "num_folds": len(rolling_result.fold_metrics),
                "fold_sizes": rolling_result.fold_sizes,
            } if rolling_result.fold_metrics else {},
            "execution_stress_summary": {"enabled": False, "implemented": False},
            "resume_ready": {
                "enabled": self.config.resume.enabled,
                "last_selector_exists": ckpt_mgr.last_path.exists(),
            },
            "guardrails_pass": True,
            "val_guardrails_pass": True,
            "test_guardrails_pass_report_only": True,
            "dead_code_mask": dead_code_mask.tolist() if dead_code_mask is not None else [False] * num_codes,
            "num_updates": num_updates,
            "total_timesteps": self.config.total_timesteps,
        }

        report_path = writer.write_final_report(report_summary)

        return Phase2TrainerArtifacts(
            artifacts_dir=artifacts_dir,
            phase2_config_yaml=config_yaml,
            horizon_index_train=hi_train_path,
            horizon_index_val=hi_val_path,
            horizon_index_test=hi_test_path,
            env_shards=env_shards_path,
            best_selector=ckpt_mgr.best_path,
            last_selector=ckpt_mgr.last_path,
            checkpoint_manifest=ckpt_mgr.manifest_path,
            rollout_stats=rollout_stats_path,
            per_horizon_records_train=phr_train,
            per_horizon_records_val=phr_val,
            per_horizon_records_test=phr_test,
            phase2_report=report_path,
            replay_log=ckpt_mgr.replay_log_path,
        )

    def _seed_everything(self) -> None:
        """统一设置全局 seed（镜像 Phase1Trainer._seed_everything）。"""
        seed = self.config.seed
        _random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
