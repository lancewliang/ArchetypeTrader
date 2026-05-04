"""Phase II trainer: 编排完整 Phase II 训练流程。

设计文档锚点: Phase II 执行计划 §Step 9。

职责:
- 校验上游产物 → 读取数据与 schema → 生成 horizon index → join labels →
  构造 envs → rollout / update / evaluate / select / checkpoint →
  rolling validation / KL-demo ablation / sensitivity 分析 →
  best checkpoint 冻结后输出 train/val per-horizon records。
- 在每个完整 checkpoint 边界刷出 replay_log_last_complete_checkpoint.feather。

关键约束:
- 训练入口不得加载 test market data 或 test labels。
- test 评估只能走独立 backtest 入口，不进入训练、best 选择、早停和报告诊断路径。
"""
from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass, field, replace
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
from src.evaluation.phase2_distribution_shift import Phase2DistributionShiftMonitor
from src.evaluation.phase2_execution_stress import (
    ExecutionStressScenario,
    Phase2ExecutionStressRunner,
)
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
from src.utils.run_logging import configure_run_logger


class Phase2FatalError(RuntimeError):
    """Phase II 训练 fatal 错误。"""


@dataclass
class Phase2TrainerArtifacts:
    """Phase II 训练产物路径集合。"""
    artifacts_dir: Path
    phase2_config_yaml: Path
    horizon_index_train: Path
    horizon_index_val: Path
    env_shards: Path
    best_selector: Path
    last_selector: Path
    checkpoint_manifest: Path
    rollout_stats: Path
    per_horizon_records_train: Path
    per_horizon_records_val: Path
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
        self.config = replace(config, test_file="")
        self._rollout_stats: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("archetype.phase2")

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

        self._logger, log_path = configure_run_logger(
            phase="phase2",
            pair=self.config.pair,
            batch_id=self.config.phase2_batch_id,
        )
        self._logger.info(
            "Phase II 训练启动：交易对=%s，Phase I 批次=%s，Phase II 批次=%s，产物根目录=%s，日志路径=%s",
            self.config.pair,
            self.config.phase1_batch_id,
            self.config.phase2_batch_id,
            self.config.artifact_root,
            log_path,
        )
        self._validate_unsupported_features()
        self._logger.info("Phase II 未支持功能配置检查通过")

        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Phase II 产物目录已准备：路径=%s", artifacts_dir)

        # 0. Seed
        self._seed_everything()
        self._logger.info("Phase II 随机种子已设置：随机种子=%s", self.config.seed)

        # 1. 校验 Phase I 产物
        validator = Phase1ArtifactValidator(self.config)
        val_result = validator.validate()
        self._logger.info(
            "Phase I 产物校验完成：Phase I 目录=%s，奖励对齐方式=%s，警告数=%d",
            self.config.phase1_dir(),
            val_result.resolved_reward_alignment,
            len(val_result.warnings),
        )

        # 2. 写 phase2_config.yaml
        config_yaml = artifacts_dir / "phase2_config.yaml"
        self.config.write_yaml(config_yaml)
        self._logger.info(
            "Phase II 配置已写入：路径=%s，配置哈希=%s",
            config_yaml,
            self.config.config_hash(),
        )

        # 3. 读取数据与 schema。训练阶段只允许读取 train/val；
        # test 留给 scripts/backtest_phase2.py 的独立评估路径。
        reader = MarketFileReader()
        frames = {
            "train": reader.read(self.config.train_file),
            "val": reader.read(self.config.val_file),
        }
        input_schema = read_json(self.config.phase1_dir() / "input_schema.json")
        self._logger.info(
            "Phase II 市场数据已加载：训练行数=%s，验证行数=%s，输入特征数=%s",
            getattr(frames["train"], "height", None),
            getattr(frames["val"], "height", None),
            len(input_schema.get("feature_columns", [])) if isinstance(input_schema, dict) else None,
        )

        # 4. 生成 horizon index
        reward_alignment_name = val_result.resolved_reward_alignment
        indexer = Phase2HorizonIndexer(
            self.config,
            reward_alignment=reward_alignment_name,
        )
        horizon = self.config.horizon

        # 加载 Phase I labels (仅 train/val)
        p1_dir = self.config.phase1_dir()
        train_labels = None
        val_labels = None
        train_label_path = self._phase1_label_path(p1_dir, "train")
        val_label_path = self._phase1_label_path(p1_dir, "val")
        if train_label_path.exists():
            train_labels = read_ipc(train_label_path)
        if val_label_path.exists():
            val_labels = read_ipc(val_label_path)
        self._logger.info(
            "Phase I 标签已加载：source=%s 训练标签数=%s，验证标签数=%s train_path=%s val_path=%s",
            self.config.phase1_label_source,
            getattr(train_labels, "height", 0) if train_labels is not None else 0,
            getattr(val_labels, "height", 0) if val_labels is not None else 0,
            train_label_path,
            val_label_path,
        )

        train_entries = indexer.build_index(frames["train"], "train", horizon, train_labels)
        val_entries = indexer.build_index(frames["val"], "val", horizon, val_labels)
        self._logger.info(
            "Phase II 时间窗索引已生成：模式=%s，训练条目=%d，验证条目=%d，时间窗=%d",
             self.config.horizon_schedule.mode,
            len(train_entries),
            len(val_entries),
            horizon,
        )

        hi_train_path = indexer.write_index(train_entries, artifacts_dir / "phase2_horizon_index_train.feather")
        hi_val_path = indexer.write_index(val_entries, artifacts_dir / "phase2_horizon_index_val.feather")
        self._logger.info(
            "Phase II 时间窗索引已写入：训练路径=%s，验证路径=%s",
           
            hi_train_path,
            hi_val_path,
        )

        # 5. Join labels
        label_loader = Phase2LabelLoader(self.config)
        if train_label_path.exists():
            train_entries = label_loader.load_and_join(
                train_entries, "train", train_label_path
            )
        if val_label_path.exists():
            val_entries = label_loader.load_and_join(
                val_entries, "val", val_label_path
            )
        self._logger.info(
            "Phase II 标签已合并：训练已标注=%d/%d，验证已标注=%d/%d",
            sum(1 for entry in train_entries if entry.is_labeled),
            len(train_entries),
            sum(1 for entry in val_entries if entry.is_labeled),
            len(val_entries),
        )

        # 6. 构造 datasets
        # 使用 train frame 构造 train dataset
        train_dataset = Phase2Dataset(
            frames["train"], train_entries, input_schema, self.config,
            reward_alignment=reward_alignment_name,
        )
        val_dataset = Phase2Dataset(
            frames["val"], val_entries, input_schema, self.config,
            reward_alignment=reward_alignment_name,
        )
        self._logger.info(
            "Phase II 数据集已准备：训练=%d，验证=%d，奖励对齐方式=%s",
            len(train_dataset),
            len(val_dataset),
            reward_alignment_name,
        )

        # 7. 加载 Phase1FrozenPolicy
        frozen_policy = Phase1FrozenPolicy.load(
            p1_dir / "decoder.pt",
            p1_dir / "codebook.pt",
            device=self.config.device,
        )
        self._logger.info(
            "Phase I 冻结策略已加载：原型代码数=%d，设备=%s",
            frozen_policy.num_codes,
            self.config.device,
        )

        # 8. 构造 trading env factory
        p1_config = val_result.phase1_config or {}
        dp_cfg = p1_config.get("dp", {})
        cost_cfg = val_result.cost_config or dp_cfg.get("cost_config", {})
        cost_model = LobDepthCostModel(
            commission_rate=cost_cfg.get("commission_rate", 0.0002),
            book_levels=cost_cfg.get("book_levels", 5),
            insufficient_depth_policy=cost_cfg.get(
                "insufficient_depth_policy", "reject_transition"
            ),
        )
        alignment = RewardAlignment(reward_alignment_name)

        def trading_env_factory():
            return TradingEnv(
                cost_model=cost_model,
                reward_alignment=alignment,
                max_position=self.config.max_position,
            )

        # 9. 构造 HorizonEnv / process worker specs (multi-env)
        factory = HorizonFactory(self.config, train_dataset, frozen_policy, trading_env_factory)
        worker_specs = []
        if self.config.rollout_collection.mode == "process":
            envs = []
            worker_specs, shard_infos = factory.create_worker_specs(
                phase1_decoder_path=p1_dir / "decoder.pt",
                phase1_codebook_path=p1_dir / "codebook.pt",
                cost_config=cost_cfg,
                reward_alignment_name=reward_alignment_name,
            )
        else:
            envs, shard_infos = factory.create_envs()
        env_shards_path = factory.write_shards(shard_infos, artifacts_dir / "phase2_env_shards.feather")
        self._logger.info(
            "Phase II 训练环境已准备：环境数=%d，worker specs=%d，分片数=%d，分片文件=%s",
            len(envs) if envs else len(worker_specs),
            len(worker_specs),
            len(shard_infos),
            env_shards_path,
        )

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

        dead_code_mask_list = (
            dead_code_mask.tolist()
            if dead_code_mask is not None
            else [False] * num_codes
        )
        unmasked_probe = self._unmasked_diagnostic_probe(
            selector,
            train_dataset,
            dead_code_mask_list,
        )
        self._logger.info(
            "Phase II 选择器已准备：状态维度=%d，原型代码数=%d，死代码掩码数=%d，无掩码探针命中率=%.6f",
            state_spec.total_dim,
            num_codes,
            sum(1 for flag in dead_code_mask_list if flag),
            float(unmasked_probe.get("probe_pick_rate", 0.0)),
        )

        actor_critic = ActorCritic(selector, dead_code_mask=dead_code_mask)

        optimizer = torch.optim.Adam(selector.parameters(), lr=self.config.ppo.lr)
        num_updates = max(
            self.config.total_timesteps // (self.config.num_envs * self.config.rollout_length),
            1,
        )
        schedule_mgr = ScheduleManager(self.config, optimizer, num_updates)

        ppo_trainer = PPOTrainer(
            self.config,
            actor_critic,
            envs,
            schedule_mgr,
            worker_specs=worker_specs,
        )
        ppo_trainer.setup(optimizer=optimizer)
        rollout_collection_info = ppo_trainer.rollout_collection_info()
        self._logger.info(
            "Phase II rollout 采样器已准备：模式=%s，worker数=%s，启动方式=%s，worker设备=%s，数据共享=%s",
            rollout_collection_info.get("mode"),
            rollout_collection_info.get("max_workers"),
            rollout_collection_info.get("process_start_method"),
            rollout_collection_info.get("worker_device"),
            rollout_collection_info.get("shared_dataset_mode"),
        )

        # Checkpoint + selection policy
        ckpt_mgr = Phase2CheckpointManager(artifacts_dir)
        policy = Phase2SelectionPolicy(self.config.selection_policy)
        history = self._history_from_manifest(ckpt_mgr, policy)

        resume_audit = self._maybe_resume(ppo_trainer, ckpt_mgr)
        start_update = int(resume_audit.get("restored_update_count", 0))
        self._logger.info(
            "Phase II 训练准备完成：更新总数=%d，起始更新=%d，采样长度=%d，环境数=%d，恢复来源=%s",
            num_updates,
            start_update,
            self.config.rollout_length,
            self.config.num_envs,
            self.config.resume_from,
        )

        # Evaluator
        val_runner = Phase2BacktestRunner(
            self.config, actor_critic, frozen_policy, val_dataset, trading_env_factory
        )
        evaluator = Phase2Evaluator(
            self.config, val_runner, num_codes,
            dead_code_mask_list if dead_code_mask is not None else None,
        )

        # 11. 训练循环
        eval_every = max(num_updates // 20, 1)
        self._logger.info(
            "Phase II 训练循环开始：更新总数=%d，评估间隔=%d，总时间步=%d",
            num_updates,
            eval_every,
            self.config.total_timesteps,
        )
        early_stop_triggered = False
        early_stop_update_idx: Optional[int] = None
        early_stop_best: Optional[float] = None
        early_stop_wait = 0
        rolling_guardrail_pass = True
        rolling_guardrail_reasons: List[str] = []
        last_candidate_rolling_result = None

        for update_idx in range(start_update, num_updates):
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
                "rollout_collect_seconds": stats.rollout_collect_seconds,
                "rollout_policy_forward_seconds": stats.rollout_policy_forward_seconds,
                "rollout_env_step_seconds": stats.rollout_env_step_seconds,
                "rollout_ipc_wait_seconds": stats.rollout_ipc_wait_seconds,
                "rollout_worker_startup_seconds": stats.rollout_worker_startup_seconds,
                "rollout_samples_per_second": stats.rollout_samples_per_second,
                "kl_demo_dominance_ratio": stats.kl_demo_dominance_ratio,
            })

            # 保存 last
            state = ppo_trainer.get_state()
            ckpt_mgr.save_last(state, update_idx)
            if (
                self.config.checkpoint_every_updates
                and self.config.checkpoint_every_updates > 0
                and (update_idx + 1) % self.config.checkpoint_every_updates == 0
            ):
                ckpt_mgr.save_periodic(state, update_idx)

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
                    "kl_demo_dominance_ratio": stats.kl_demo_dominance_ratio,
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
                rolling_payload = None
                if verdict.decision == "promote_to_best" and self.config.rolling_validation.enabled:
                    candidate_rolling = evaluator.evaluate_rolling_validation()
                    last_candidate_rolling_result = candidate_rolling
                    rolling_payload = self._rolling_result_payload(candidate_rolling)
                    verdict = policy.evaluate(eval_metrics, history, rolling_payload)
                    rolling_guardrail_pass = verdict.decision != "reject"
                    rolling_guardrail_reasons = [
                        r for r in verdict.reasons if r.startswith("rolling_")
                    ]
                    eval_metrics["rolling_guardrail_pass"] = rolling_guardrail_pass
                    eval_metrics["rolling_guardrail_reasons_count"] = len(
                        rolling_guardrail_reasons
                    )
                ckpt_mgr.commit_verdict(state, verdict, update_idx, eval_metrics)
                history = policy.update_history(history, eval_metrics, verdict)
                self._logger.info(
                    "Phase II 评估结果：更新=%d，决策=%s，综合分=%.6f，验证净收益=%s，夏普=%s，近似 KL=%.6f，平均奖励=%.6f，原因=%s",
                    update_idx,
                    verdict.decision,
                    float(eval_metrics.get("phase2_composite_score", 0.0)),
                    eval_metrics.get("val_net_return"),
                    eval_metrics.get("sharpe_ratio"),
                    stats.approx_kl,
                    stats.reward_mean,
                    ",".join(verdict.reasons),
                )

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

                if self.config.early_stopping.enabled:
                    metric_name = self.config.early_stopping.metric
                    current = float(eval_metrics.get(metric_name, 0.0))
                    improved = self._is_metric_improved(
                        current,
                        early_stop_best,
                        self.config.early_stopping.min_delta,
                    )
                    if improved:
                        early_stop_best = current
                        early_stop_wait = 0
                    else:
                        early_stop_wait += 1
                    if early_stop_wait >= self.config.early_stopping.patience:
                        early_stop_triggered = True
                        early_stop_update_idx = update_idx
                        self._logger.info(
                            "Phase II 早停触发：更新=%d，指标=%s，最佳值=%s，等待轮数=%d",
                            update_idx,
                            self.config.early_stopping.metric,
                            early_stop_best,
                            early_stop_wait,
                        )
                        break

        # 12. 最终评估与报告
        self._logger.info(
            "Phase II 训练循环结束：已完成更新=%d，早停=%s",
            len(self._rollout_stats),
            early_stop_triggered,
        )
        report_paths = Phase2ReportPaths.from_artifacts_dir(artifacts_dir)
        writer = Phase2ReportWriter(report_paths)

        # 写 rollout stats
        rollout_stats_path = writer.write_rollout_stats(self._rollout_stats)
        self._logger.info(
            "Phase II 采样统计已写入：路径=%s，行数=%d",
            rollout_stats_path,
            len(self._rollout_stats),
        )

        # 加载 best
        if ckpt_mgr.best_path.exists():
            best_state = ckpt_mgr.load(ckpt_mgr.best_path)
            selector.load_state_dict(best_state["model_state"])
            self._logger.info("Phase II 最优检查点已加载：路径=%s", ckpt_mgr.best_path)
        else:
            self._logger.info("Phase II 最优检查点不存在：路径=%s", ckpt_mgr.best_path)

        # Per-horizon records
        train_runner = Phase2BacktestRunner(
            self.config, actor_critic, frozen_policy, train_dataset, trading_env_factory
        )
        train_records = train_runner.run_walk_forward("train", deterministic=True)
        val_records = val_runner.run_walk_forward("val", deterministic=True)
        self._logger.info(
            "Phase II 滚动前向回测完成：训练记录数=%d，验证记录数=%d",
            len(train_records),
            len(val_records),
        )

        train_rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in train_records]
        val_rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in val_records]

        phr_train = writer.write_per_horizon_records(train_rec_dicts, "train")
        phr_val = writer.write_per_horizon_records(val_rec_dicts, "val")
        self._logger.info(
            "Phase II 逐时间窗记录已写入：训练路径=%s，验证路径=%s",
            phr_train,
            phr_val,
        )

        # Rolling validation
        rolling_result = last_candidate_rolling_result or evaluator.evaluate_rolling_validation()
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
                "fold_initial_positions": rolling_result.fold_initial_positions,
                "fold_initial_position_policy": rolling_result.fold_initial_position_policy,
            },
            rolling_records,
        )
        self._logger.info(
            "Phase II 滚动验证已写入：折数=%d，记录数=%d，护栏通过=%s",
            len(rolling_result.fold_metrics),
            len(rolling_records),
            rolling_guardrail_pass,
        )

        # Build report
        val_metrics = phase2_composite_metrics(
            val_rec_dicts, {}, num_codes,
            dead_code_mask_list,
            metric_weights=self.config.selection_policy.metric_weights,
        )

        # Baselines
        val_baselines = val_runner.run_baselines("val")
        bl_val_summary = {
            name: {k: v for k, v in phase2_composite_metrics(
                [Phase2Evaluator._record_to_dict(r) for r in recs],
                {}, num_codes,
                dead_code_mask_list,
                metric_weights=self.config.selection_policy.metric_weights,
            ).items() if isinstance(v, (int, float, str, bool))}
            for name, recs in val_baselines.items()
        }
        writer.write_baselines(bl_val_summary, "val")
        self._logger.info(
            "Phase II 基线结果已写入：验证基线数=%d",
            len(bl_val_summary),
        )

        sensitivity = phase2_composite_score_sensitivity(
            [
                {"update_idx": e.update_idx, **e.metrics, "_manifest_verdict": e.verdict}
                for e in ckpt_mgr._entries
            ],
            self.config.selection_policy.metric_weights,
            self.config.selection_policy.composite_score_sensitivity_perturbations,
        )
        writer.write_sensitivity(sensitivity)
        self._logger.info(
            "Phase II 敏感性分析已写入：条目数=%d",
            len(sensitivity.get("results", [])) if isinstance(sensitivity, dict) else 0,
        )

        # Label coverage
        train_coverage = label_loader.compute_coverage_stats(train_entries, "train")
        distribution_summary = self._distribution_shift_summary(
            train_dataset, val_dataset
        )
        execution_stress_summary = {
            "enabled": False,
            "implemented": False,
            "reason": "test_not_loaded_in_phase2_training",
        }
        self._logger.info(
            "Phase II 诊断完成：标签覆盖率=%.6f，分布外警告数=%d，压力场景数=%d",
            train_coverage.coverage_ratio,
            distribution_summary["warning_count"],
            len(execution_stress_summary.get("scenarios", [])),
        )
        kl_dominance_values = [
            float(r.get("kl_demo_dominance_ratio", 0.0))
            for r in self._rollout_stats
        ]
        kl_dominance_summary = {
            "last": kl_dominance_values[-1] if kl_dominance_values else 0.0,
            "mean": (
                sum(kl_dominance_values) / len(kl_dominance_values)
                if kl_dominance_values
                else 0.0
            ),
            "max": max(kl_dominance_values) if kl_dominance_values else 0.0,
        }

        report_summary: Dict[str, Any] = {
            "config_hash": self.config.config_hash(),
            "phase1_hash": p1_report.get("config_hash", ""),
            "phase1_batch_id": self.config.phase1_batch_id,
            "feature_provenance_hash": val_result.feature_provenance_hash,
            "no_leakage_signoff": val_result.no_leakage_signoff,
            "no_leakage_signoff_blockers": val_result.no_leakage_signoff_blockers,
            "phase2_batch_id": self.config.phase2_batch_id,
            "selection_metric": self.config.selection_policy.selection_metric,
            "metric_weights": dict(self.config.selection_policy.metric_weights),
            "test_used_for_selection": False,
            "test_loaded_in_training": False,
            "training_splits": ["train", "val"],
            "backtest_required_for_test_metrics": True,
            "kl_label_coverage_train": train_coverage.coverage_ratio,
            "kl_label_temporal_coverage": train_coverage.temporal_coverage_sequence,
            "equity_curve_summary": val_metrics.get("equity_curve_summary", {}),
            "behavior_health_warnings": [],
            "risk_health_warnings": [],
            "ood_warning_count": distribution_summary["warning_count"],
            "distribution_shift_warning_count": distribution_summary["warning_count"],
            "val_metrics": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, str, bool))},
            "test_metrics": {},
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
                "test_gap_count": None,
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
            "rollout_collection": {
                "mode": self.config.rollout_collection.mode,
                "max_workers": rollout_collection_info.get("max_workers"),
                "process_start_method": rollout_collection_info.get("process_start_method"),
                "worker_device": rollout_collection_info.get("worker_device"),
                "shared_dataset_mode": rollout_collection_info.get("shared_dataset_mode"),
                "last_collect_seconds": (
                    self._rollout_stats[-1].get("rollout_collect_seconds", 0.0)
                    if self._rollout_stats else 0.0
                ),
                "last_ipc_wait_seconds": (
                    self._rollout_stats[-1].get("rollout_ipc_wait_seconds", 0.0)
                    if self._rollout_stats else 0.0
                ),
                "last_samples_per_second": (
                    self._rollout_stats[-1].get("rollout_samples_per_second", 0.0)
                    if self._rollout_stats else 0.0
                ),
            },
            "reward_scaling": {
                "method": self.config.reward_scaling.method,
                "clip_range": self.config.reward_scaling.clip_range,
                "last_reward_clipped_ratio": self._rollout_stats[-1].get("reward_clipped_ratio", 0.0) if self._rollout_stats else 0.0,
            },
            "reward_normalization": {
                "enabled": self.config.reward_normalization.enabled or self.config.ppo.reward_normalization,
                "implemented": False,
                "reward_normalization_rejected_for_signoff": (
                    self.config.reward_normalization.enabled
                    or self.config.ppo.reward_normalization
                ),
            },
            "cost_config_inherited": cost_cfg,
            "baselines_val": bl_val_summary,
            "baselines_test": {},
            "rolling_validation_summary": {
                "fold_mean": rolling_result.fold_mean,
                "worst_fold_quantile": rolling_result.worst_fold_quantile,
                "fold_volatility": rolling_result.fold_volatility,
                "num_folds": len(rolling_result.fold_metrics),
                "fold_sizes": rolling_result.fold_sizes,
                "fold_initial_positions": rolling_result.fold_initial_positions,
                "fold_initial_position_policy": rolling_result.fold_initial_position_policy,
            } if rolling_result.fold_metrics else {},
            "rolling_guardrail_pass": rolling_guardrail_pass,
            "rolling_guardrail_reasons": rolling_guardrail_reasons,
            "execution_stress_summary": execution_stress_summary,
            "distribution_shift_summary": distribution_summary,
            "resume_ready": {
                "enabled": self.config.resume.enabled,
                "last_selector_exists": ckpt_mgr.last_path.exists(),
                **resume_audit,
            },
            "early_stopping": {
                "enabled": self.config.early_stopping.enabled,
                "triggered": early_stop_triggered,
                "metric": self.config.early_stopping.metric,
                "update_idx": early_stop_update_idx,
                "hypothetical_early_stop_timestep": (
                    early_stop_update_idx
                    * self.config.num_envs
                    * self.config.rollout_length
                    if early_stop_update_idx is not None
                    else None
                ),
            },
            "kl_demo_dominance_ratio": kl_dominance_summary,
            "entropy_schedule": {
                "entropy_min_coef": self.config.ppo.entropy_min_coef,
                "last_entropy_coef": (
                    schedule_mgr.current_state().entropy_coef
                    if schedule_mgr is not None
                    else self.config.ppo.entropy_coef
                ),
            },
            "guardrails_pass": rolling_guardrail_pass and val_result.no_leakage_signoff,
            "val_guardrails_pass": rolling_guardrail_pass,
            "test_guardrails_pass_report_only": False,
            "dead_code_mask": dead_code_mask_list,
            "unmasked_diagnostic_probe": unmasked_probe,
            "probe_pick_rate": unmasked_probe["probe_pick_rate"],
            "num_updates": num_updates,
            "total_timesteps": self.config.total_timesteps,
        }

        report_path = writer.write_final_report(report_summary)
        self._logger.info(
            "Phase II 报告已写入：路径=%s，验证净收益=%s，测试集是否在训练中加载=%s，护栏通过=%s",
            report_path,
            report_summary["val_metrics"].get("net_return"),
            report_summary["test_loaded_in_training"],
            report_summary.get("guardrails_pass"),
        )

        self._logger.info("Phase II 训练完成：产物目录=%s", artifacts_dir)
        ppo_trainer.close()
        return Phase2TrainerArtifacts(
            artifacts_dir=artifacts_dir,
            phase2_config_yaml=config_yaml,
            horizon_index_train=hi_train_path,
            horizon_index_val=hi_val_path,
            env_shards=env_shards_path,
            best_selector=ckpt_mgr.best_path,
            last_selector=ckpt_mgr.last_path,
            checkpoint_manifest=ckpt_mgr.manifest_path,
            rollout_stats=rollout_stats_path,
            per_horizon_records_train=phr_train,
            per_horizon_records_val=phr_val,
            phase2_report=report_path,
            replay_log=ckpt_mgr.replay_log_path,
        )

    def _validate_unsupported_features(self) -> None:
        """对尚未实现但有配置入口的功能 fail-fast。"""
        if self.config.reward_normalization.enabled or self.config.ppo.reward_normalization:
            raise Phase2FatalError(
                "Phase II reward_normalization 尚未实现 running_mean_std；"
                "请关闭 reward_normalization 并使用 reward_scaling。"
            )

    def _phase1_label_path(self, p1_dir: Path, split: str) -> Path:
        if split == "train" and self.config.phase1_label_source == "full_time":
            path = p1_dir / "horizon_labels_full_time_train.feather"
            if not path.exists():
                raise Phase2FatalError(
                    "phase1_label_source=full_time 但缺少 "
                    f"{path}; 请使用支持 full-time label export 的 Phase I 批次。"
                )
            return path
        return p1_dir / f"horizon_labels_{split}.feather"

    def _unmasked_diagnostic_probe(
        self,
        selector: ArchetypeSelector,
        dataset: Phase2Dataset,
        dead_code_mask: List[bool],
        max_samples: int = 64,
    ) -> Dict[str, Any]:
        """Record initial selector picks with dead-code masking disabled."""
        sample_count = min(len(dataset.horizon_entries), max(int(max_samples), 0))
        dead_code_count = sum(1 for flag in dead_code_mask if flag)
        if sample_count == 0 or dead_code_count == 0:
            return {
                "enabled": dead_code_count > 0,
                "sample_count": sample_count,
                "dead_code_count": dead_code_count,
                "probe_pick_count": 0,
                "probe_pick_rate": 0.0,
                "picked_codes": [],
            }

        import torch

        was_training = selector.training
        selector.eval()
        obs = np.array(
            [dataset.get_selector_state(i, 0) for i in range(sample_count)],
            dtype=np.float32,
        )
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
        with torch.no_grad():
            logits, _ = selector(obs_tensor)
            actions = logits.argmax(dim=-1).cpu().tolist()
        if was_training:
            selector.train()

        dead_codes = {idx for idx, flag in enumerate(dead_code_mask) if flag}
        probe_pick_count = sum(1 for action in actions if int(action) in dead_codes)
        return {
            "enabled": True,
            "sample_count": sample_count,
            "dead_code_count": dead_code_count,
            "probe_pick_count": probe_pick_count,
            "probe_pick_rate": probe_pick_count / sample_count,
            "picked_codes": [int(action) for action in actions],
        }

    def _history_from_manifest(
        self,
        ckpt_mgr: Phase2CheckpointManager,
        policy: Phase2SelectionPolicy,
    ) -> Phase2SelectionHistory:
        """从 checkpoint manifest 恢复 best selection history。"""
        history = Phase2SelectionHistory()
        metric_name = self.config.selection_policy.selection_metric
        fallback_metric = self.config.selection_policy.primary_metric
        for entry in ckpt_mgr._entries:
            if not entry.is_best:
                continue
            metrics = entry.metrics or {}
            if metric_name in metrics:
                value = metrics[metric_name]
            else:
                value = metrics.get(fallback_metric)
            if value is None:
                continue
            history.best_metric = float(value)
            history.best_update_idx = int(entry.update_idx)
        return history

    def _maybe_resume(
        self,
        ppo_trainer: PPOTrainer,
        ckpt_mgr: Phase2CheckpointManager,
    ) -> Dict[str, Any]:
        """按 config.resume_from 恢复训练状态并返回审计信息。"""
        audit: Dict[str, Any] = {
            "source_checkpoint": self.config.resume_from,
            "restored_update_count": 0,
            "missing_fields": [],
            "optimizer_state_restored": False,
            "env_state_restored": False,
            "rng_state_restored": False,
        }
        if not self.config.resume_from:
            return audit

        path = Path(self.config.resume_from)
        state = ckpt_mgr.load(path)
        required = ["model_state", "schedule_state"]
        if self.config.resume.require_optimizer_state:
            required.append("optimizer_state")
        if self.config.resume.require_env_state:
            required.append("env_states")
        missing = [key for key in required if key not in state]
        audit["missing_fields"] = missing
        if missing:
            raise Phase2FatalError(
                f"resume checkpoint 缺失关键字段: {missing}"
            )

        ppo_trainer.load_state(state)
        audit["restored_update_count"] = int(state.get("update_count", 0))
        audit["optimizer_state_restored"] = "optimizer_state" in state
        audit["env_state_restored"] = "env_states" in state
        audit["rng_state_restored"] = "rng_state" in state
        return audit

    def _rolling_result_payload(self, rolling_result) -> Dict[str, Any]:
        """转换 rolling validation 结果为 selection policy 可读 payload。"""
        return {
            "enabled": self.config.rolling_validation.enabled,
            "fold_mean": rolling_result.fold_mean,
            "worst_fold_quantile": rolling_result.worst_fold_quantile,
            "fold_volatility": rolling_result.fold_volatility,
            "fold_sizes": rolling_result.fold_sizes,
            "max_fold_volatility": self.config.rolling_validation.max_fold_volatility,
            "min_worst_fold_score": self.config.rolling_validation.min_worst_fold_score,
        }

    def _is_metric_improved(
        self,
        current: float,
        best: Optional[float],
        min_delta: float,
    ) -> bool:
        """按 selection primary_mode 判断 early-stopping metric 是否改善。"""
        if best is None:
            return True
        if self.config.selection_policy.primary_mode == "min":
            return current < best - float(min_delta)
        return current > best + float(min_delta)

    def _distribution_shift_summary(
        self,
        train_dataset: Phase2Dataset,
        val_dataset: Phase2Dataset,
    ) -> Dict[str, Any]:
        """fit train selector-state stats，并只对 val 做 OOD score。"""
        if len(train_dataset) == 0:
            return {
                "enabled": True,
                "warning_count": 0,
                "max_score_val": 0.0,
                "max_score_test": None,
                "test_loaded_in_training": False,
                "dims": [],
                "fallback_action": self.config.distribution_shift.fallback_action,
            }
        feature_dim = train_dataset.state_spec().feature_dim
        dims = (
            list(range(feature_dim))
            if self.config.distribution_shift.use_market_features_only
            else list(range(train_dataset.state_spec().total_dim))
        )
        monitor = Phase2DistributionShiftMonitor(
            self.config.distribution_shift,
            dims=dims,
        )
        monitor.fit(
            train_dataset.get_selector_state(i, 0)
            for i in range(len(train_dataset))
        )

        def _scores(dataset: Phase2Dataset) -> List[float]:
            vals = []
            for i in range(len(dataset)):
                vals.append(monitor.score(dataset.get_selector_state(i, 0)).score)
            return vals

        val_scores = _scores(val_dataset)
        threshold = self.config.distribution_shift.threshold
        warning_count = sum(1 for s in val_scores if s > threshold)
        return {
            "enabled": True,
            "warning_count": warning_count,
            "max_score_val": max(val_scores) if val_scores else 0.0,
            "max_score_test": None,
            "test_loaded_in_training": False,
            "dims": dims,
            "fallback_action": self.config.distribution_shift.fallback_action,
        }

    def _execution_stress_summary(
        self,
        actor_critic: ActorCritic,
        frozen_policy: Phase1FrozenPolicy,
        test_dataset: Phase2Dataset,
        cost_cfg: Dict[str, Any],
        alignment: RewardAlignment,
        num_codes: int,
        dead_code_mask: List[bool],
    ) -> Dict[str, Any]:
        """运行 report-only execution stress scenarios。"""
        base_commission = cost_cfg.get("commission_rate", 0.0002)
        book_levels = cost_cfg.get("book_levels", 5)
        insufficient_depth_policy = cost_cfg.get(
            "insufficient_depth_policy", "reject_transition"
        )

        def _run_records(scenario: ExecutionStressScenario):
            def _factory():
                return TradingEnv(
                    cost_model=LobDepthCostModel(
                        commission_rate=base_commission * scenario.commission_multiplier,
                        book_levels=book_levels,
                        insufficient_depth_policy=insufficient_depth_policy,
                        slippage_multiplier=scenario.slippage_multiplier,
                    ),
                    reward_alignment=alignment,
                    max_position=self.config.max_position,
                )

            runner = Phase2BacktestRunner(
                self.config,
                actor_critic,
                frozen_policy,
                test_dataset,
                _factory,
            )
            return runner.run_walk_forward(
                "test",
                deterministic=True,
                execution_lag_offset=scenario.execution_lag_offset,
            )

        runner = Phase2ExecutionStressRunner(
            self.config.execution_stress,
            _run_records,
            num_codes,
            dead_code_mask,
        )
        result = runner.run()
        worst = None
        if result.scenarios:
            worst = min(result.scenarios, key=lambda s: s.get("net_return", 0.0))
        return {
            "enabled": True,
            "implemented": True,
            "scenarios": result.scenarios,
            "selector_latency": result.selector_latency,
            "worst_scenario": worst,
        }

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
