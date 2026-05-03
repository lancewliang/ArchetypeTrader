"""Phase I trainer: manifest-mode 训练编排.

设计文档锚点: §4.6 与 §7。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.phase1_config import (
    Phase1Config,
    apply_paper_strict_overrides,
)
from src.data.dataset import Phase1DemoDataset, collate_phase1
from src.data.demo_store import HorizonLabel, Phase1DemoStore
from src.data.phase1_processed_store import Phase1ProcessedStore
from src.data.state_normalizer import StateNormalizer
from src.evaluation.phase1_evaluator import Phase1Evaluator
from src.evaluation.phase1_metrics import (
    composite_score_sensitivity_across_epochs,
)
from src.evaluation.phase1_replay import Phase1ReplayEvaluator
from src.evaluation.phase1_report import Phase1ReportWriter, ReportPaths
from src.models.encoder_inputs import RewardNormalizer
from src.models.vq_archetype import VQArchetypeModel
from src.models.vq_losses import Phase1Loss
from src.planners.demo_generator import RejectStats
from src.trading.cost_model import LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.reward_alignment import RewardAlignment
from src.utils.feather_io import atomic_write_json
from src.utils.feather_io import read_json
from src.utils.run_logging import configure_run_logger

from .phase1_checkpoint import Phase1CheckpointManager, Phase1FatalCollapse
from .selection_policy import (
    Phase1SelectionPolicy,
    SelectionHistory,
    SelectionVerdict,
)


class Phase1FatalError(RuntimeError):
    """trainer 主流程里被 selection_policy 判定 fatal 后转抛出。"""


_AMP_SAFE_INPUT_ABS_MAX = 1.0e4


@dataclass
class TrainerArtifacts:
    artifacts_dir: Path
    phase1_config_yaml: Path
    input_schema_json: Path
    window_index_train: Path
    demos_train: Path
    horizon_labels_train: Path
    horizon_labels_val: Path
    horizon_labels_test: Path
    state_normalizer_json: Path
    reward_normalizer_json: Path
    best_vq_model: Path
    last_vq_model: Path
    encoder_pt: Path
    decoder_pt: Path
    codebook_pt: Path
    checkpoint_manifest: Path
    phase1_report: Path
    composite_score_sensitivity_json: Path
    sampling_leakage_diagnostics_json: Path


class Phase1Trainer:
    """主训练编排器。

    使用方式::

        trainer = Phase1Trainer(config)
        artifacts = trainer.run()  # 阻塞直到训练结束或 fatal

    边界
    ----
    - 不直接计算指标（交给 ``Phase1Evaluator``）。
    - 不直接写 report 文件（交给 ``Phase1ReportWriter``）。
    - 不直接选 best（交给 ``Phase1SelectionPolicy``）。
    - ``apply_paper_strict_overrides`` 在 ``__init__`` 时统一调用，
      保证后续所有读 ``self.config`` 的代码都看到 strict 模式下的稳定值。
    """

    def __init__(self, config: Phase1Config) -> None:
        # paper_strict_reproduction 时同步 codebook 与 encoder 的工程稳定项。
        self.config = apply_paper_strict_overrides(config)
        self._dead_code_restart_events: List[dict] = []
        self._best_epoch_diagnostics: Dict[str, Any] = {}
        self._processed_data_metadata: Dict[str, Any] = {
            "processed_data_mode": "legacy_inline",
            "data_process_manifest": "",
            "data_batch_id": "",
            "schema_hash": "",
            "data_process_hash": "",
            "dp_teacher_hash": "",
        }
        self._logger = logging.getLogger("archetype.phase1")
        self._device = self._resolve_device()
        self._scheduler = None

    def _resolve_device(self) -> str:
        try:
            import torch
            requested = self.config.training.device
            if requested == "cuda" and not torch.cuda.is_available():
                self._logger.warning(
                    "phase1_device_fallback 说明=cuda 不可用,回退到 cpu"
                )
                return "cpu"
            return requested
        except ImportError:
            return "cpu"

    # ---------- 入口 ----------

    def _seed_everything(self) -> None:
        """统一设置 random / numpy / torch 全局 seed。

        是设计 §6.9 / §9 reproducibility 的最低要求；在 ``run()`` 入口调用。
        ``torch.use_deterministic_algorithms`` 可选；为兼容 cuDNN 不强制开启，
        若需要严格复现可由调用方在 trainer 之外另设。
        """
        seed = self.config.training.seed
        import random as _random
        _random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def run(self) -> TrainerArtifacts:
        """完整主流程。

        Steps
        -----
        0. ``_seed_everything``: 设置 random/numpy/torch 全局 seed（设计 §9 可复现性）。
        1. ``_check_prospective_diagnostic``: 强制 prospective 对照检查。
        2. 写 ``phase1_config.yaml``，并计算 ``config_hash``。
        3. 读三个 split + schema 校验 → ``input_schema.json``。
        4. 滑窗枚举 + 分层采样 + 采样健康检查 → ``window_index_*.feather``。
        5. 数据增强（仅 train，按配置）。
        6. ``HorizonBuilder`` 切出 horizon。
        7. ``Phase1DemoGenerator`` 跑 DP，写 ``demos_train.feather``；同时收集
           reject_transition 统计，超阈值时直接抛错（fail_when_exceeded=True）。
        8. ``RewardNormalizer.fit_train`` → ``reward_normalizer.json``；
           train rewards 立即应用 transform，val/test 同理。
        9. 训练 ``VQArchetypeModel`` ``epochs`` 轮：
           - 每 epoch: forward + Phase1Loss + backward + EMA 更新。
           - 每若干 epoch 调 evaluator 跑 validation。
           - selection_policy → checkpoint manager (commit_verdict)。
           - 触发 ``fatal`` 时 trainer 退出。
        10. 训练结束: composite_score sensitivity → 报告写入。
        11. 用 best checkpoint 编码 horizon labels（train/val/test）。
        12. 导出 Phase II/III 产物 ``encoder.pt`` / ``decoder.pt`` / ``codebook.pt``。

        Returns
        -------
        TrainerArtifacts : 所有关键产物路径，集成测试可直接断言存在性。

        Raises
        ------
        Phase1FatalError : selection_policy 判定 fatal collapse。
        SamplingHealthError : 采样健康检查阻塞（默认 ``warn_only=False``）。
        RejectTransitionExceeded : reject_transition 超阈值且 ``fail_when_exceeded=True``。
        """
        self._logger, log_path = configure_run_logger(
            phase="phase1",
            pair=self.config.pair,
            batch_id=self.config.train_batch_id,
        )
        self._logger.info(
            "phase1_start 说明=Phase I 训练启动 pair=%s batch=%s artifact_root=%s log_path=%s",
            self.config.pair,
            self.config.train_batch_id,
            self.config.artifact_root,
            log_path,
        )
        artifacts_dir = self.config.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            "phase1_artifacts_dir_ready 说明=产物目录已准备 path=%s",
            artifacts_dir,
        )

        # 0a. 设置全局 seed（最早进行，覆盖后续所有 sampler/torch 操作）。
        self._seed_everything()
        self._logger.info(
            "phase1_seed_set 说明=随机种子已设置 seed=%s",
            self.config.training.seed,
        )

        self._logger.info(
            "phase1_manifest_mode_enabled 说明=训练将只读取离线数据处理产物 manifest=%s",
            self.config.data_process_manifest,
        )

        config_yaml = artifacts_dir / "phase1_config.yaml"
        self.config.write_yaml(config_yaml)
        config_hash = self.config.config_hash()
        training_hash = self.config.training_config_hash()
        self._logger.info(
            "phase1_config_written 说明=Phase I 配置已写入 path=%s config_hash=%s training_config_hash=%s",
            config_yaml,
            config_hash,
            training_hash,
        )

        processed_store = Phase1ProcessedStore(
            Path(self.config.data_process_manifest).parent
        )
        manifest = processed_store.load_manifest(self.config.data_process_manifest)
        if manifest.pair != self.config.pair:
            raise Phase1FatalError(
                f"data_process_manifest pair mismatch: config={self.config.pair} manifest={manifest.pair}"
            )
        schema = processed_store.load_schema(manifest)
        schema_hash = manifest.schema_hash
        schema_path = atomic_write_json(
            schema.to_dict(), artifacts_dir / "input_schema.json"
        )
        train_horizons = processed_store.load_records(manifest, "train")
        val_horizons = processed_store.load_records(manifest, "val")
        test_horizons = processed_store.load_records(manifest, "test")
        reject_stats = processed_store.load_reject_stats(manifest, "train")
        train_window_path = manifest.resolve(
            manifest.splits["train"].window_index_path
        )
        self._processed_data_metadata = {
            "processed_data_mode": "manifest",
            "data_process_manifest": str(self.config.data_process_manifest),
            "data_batch_id": manifest.data_batch_id,
            "schema_hash": manifest.schema_hash,
            "data_process_hash": manifest.data_process_hash,
            "dp_teacher_hash": manifest.dp_teacher_hash,
        }
        self._logger.info(
            "phase1_processed_data_loaded 说明=已从 manifest 加载固化训练数据 train=%d val=%d test=%d schema_hash=%s data_process_hash=%s dp_teacher_hash=%s",
            len(train_horizons),
            len(val_horizons),
            len(test_horizons),
            manifest.schema_hash,
            manifest.data_process_hash,
            manifest.dp_teacher_hash,
        )

        state_norm = StateNormalizer.fit_records(
            train_horizons,
            feature_columns=schema.feature_columns,
        )
        state_norm_path = atomic_write_json(
            state_norm.to_dict(), artifacts_dir / "state_normalizer.json"
        )
        state_train_diag = state_norm.transform_records(train_horizons)
        state_val_diag = state_norm.transform_records(val_horizons)
        state_test_diag = state_norm.transform_records(test_horizons)
        self._logger.info(
            "phase1_state_normalizer_fit 说明=状态特征归一化器已用训练集拟合 path=%s method=%s feature_dim=%d clip=%.2f train_max_abs_before=%.6e train_max_abs_after=%.6e val_max_abs_before=%.6e val_max_abs_after=%.6e test_max_abs_before=%.6e test_max_abs_after=%.6e log_transform_columns=%d fallback_to_standard_count=%d",
            state_norm_path,
            state_norm.stats.method,
            len(schema.feature_columns),
            state_norm.stats.clip_value,
            state_train_diag["max_abs_before"],
            state_train_diag["max_abs_after"],
            state_val_diag["max_abs_before"],
            state_val_diag["max_abs_after"],
            state_test_diag["max_abs_before"],
            state_test_diag["max_abs_after"],
            sum(1 for kind in state_norm.stats.transform_kinds if kind == "signed_log1p"),
            state_norm.stats.fallback_to_standard_count,
        )

        norm = RewardNormalizer(self.config.model.encoder_input)
        flat_rewards = [v for rec in train_horizons for v in (rec.rewards or [])]
        norm.fit_train(flat_rewards)
        norm_path = atomic_write_json(norm.to_dict(), artifacts_dir / "reward_normalizer.json")
        norm_stats = norm.to_dict()
        self._logger.info(
            "phase1_reward_normalizer_fit 说明=奖励归一化器已用训练集拟合 path=%s method=%s center=%.8f scale=%.8f clip_ratio=%.6f",
            norm_path,
            norm_stats.get("method"),
            float(norm_stats.get("center", 0.0)),
            float(norm_stats.get("scale", 0.0)),
            float(norm_stats.get("clip_ratio", 0.0)),
        )

        self._log_reward_composition(train_horizons, "train")
        self._log_reward_composition(val_horizons, "val")

        # 7. 保存 demos / labels（labels 在训练后由 best 模型回填 code_label）
        store = Phase1DemoStore(artifacts_dir, config_hash, schema_hash)
        demos_path = store.save_demos(train_horizons, split="train")
        self._logger.info(
            "phase1_demos_saved 说明=训练示范样本已保存 path=%s count=%d",
            demos_path,
            len(train_horizons),
        )

        # 8. 模型与训练
        feature_dim = schema.feature_dim()
        model, evaluator, loss_fn, optimizer = self._build_training_components(
            feature_dim, val_horizons, reward_normalizer=norm,
        )
        self._logger.info(
            "phase1_training_components_ready 说明=模型、损失、优化器与评估器已初始化 feature_dim=%d num_codes=%d hidden_dim=%d code_dim=%d",
            feature_dim,
            self.config.model.num_codes,
            self.config.model.hidden_dim,
            self.config.model.code_dim,
        )

        # codebook warmup: Phase A 开启时延后到预训练结束后执行，避免用随机 encoder 初始化。
        pretrain_epochs = self._effective_pretrain_epochs()
        if pretrain_epochs > 0:
            self._logger.info(
                "phase1_codebook_warmup_deferred 说明=Phase A 启用,codebook warmup 将在预训练结束后执行 pretrain_epochs=%d",
                pretrain_epochs,
            )
        else:
            self._warmup_codebook(model, train_horizons, norm)
            self._logger.info(
                "phase1_codebook_warmup_done 说明=codebook 预热完成 init_method=%s warmup_batches=%d",
                self.config.model.codebook.init_method,
                self.config.model.codebook.kmeans_warmup_batches,
            )

        ckpt = Phase1CheckpointManager(artifacts_dir)
        policy = Phase1SelectionPolicy(self.config.selection_policy)
        history = SelectionHistory()

        # 不再 in-place 修改 rec.rewards：保留原始 reward 用于 demo_return 计算；
        # encoder 输入的归一化由 dataset / evaluator 内部即时完成（同一 normalizer 实例）。
        train_dataset = Phase1DemoDataset(
            records=train_horizons,
            contrastive_pairs=None,
            reward_normalizer=norm,
        )

        try:
            history = self._train_loop(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_dataset=train_dataset,
                evaluator=evaluator,
                val_dataset=Phase1DemoDataset(
                    records=val_horizons, reward_normalizer=norm
                ),
                val_records=val_horizons,
                checkpoint=ckpt,
                policy=policy,
                history=history,
            )
        except Phase1FatalCollapse as exc:
            # 把 fatal 转换为 trainer 层异常，由入口转非零退出码。
            raise Phase1FatalError(str(exc)) from exc
        self._logger.info(
            "phase1_train_loop_done 说明=训练循环结束 best_epoch=%s manifest=%s",
            history.best_epoch,
            ckpt.manifest_path,
        )

        # 9. 用 best checkpoint 导出 horizon labels
        if not ckpt.best_path.exists():
            raise Phase1FatalError(
                "训练结束后没有可用 best_vq_model.pt；禁止导出 Phase II artifacts。"
            )
        best_state = ckpt.load(ckpt.best_path)
        self._reload_state(model, best_state)
        labels_paths = self._export_horizon_labels(
            model,
            store=store,
            horizons_by_split={"train": train_horizons, "val": val_horizons, "test": test_horizons},
            normalizer=norm,
        )
        self._logger.info(
            "phase1_horizon_labels_exported 说明=各 split 的 horizon code label 已导出 train=%s val=%s test=%s",
            labels_paths["train"],
            labels_paths["val"],
            labels_paths["test"],
        )

        # 10. 导出 Phase II/III 产物
        encoder_path, decoder_path, codebook_path = self._export_phase2_artifacts(
            best_state
        )
        self._logger.info(
            "phase1_phase2_artifacts_exported 说明=供 Phase II 使用的模型产物已导出 encoder=%s decoder=%s codebook=%s",
            encoder_path,
            decoder_path,
            codebook_path,
        )

        # 11. composite sensitivity + 报告
        best_metrics_payload = self._best_metrics(ckpt)
        sensitivity = composite_score_sensitivity_across_epochs(
            self._all_epoch_metrics(ckpt),
            base_weights=self.config.selection_policy.metric_weights,
            perturbations=self.config.selection_policy.composite_score_sensitivity_perturbations,
        )
        sensitivity_path = artifacts_dir / "composite_score_sensitivity.json"
        atomic_write_json(sensitivity, sensitivity_path)

        # 计算 train demo 上的真实 no_trade_ratio，进入最终报告。
        # 该指标在 §9.4 验收中要求小于 max_no_trade_ratio；不直接在训练循环中
        # 维护是因为它一次性可算（DP 在 step 5 已经标好 actions）。
        no_trade_count = sum(
            1 for rec in train_horizons
            if rec.actions is not None and all(a == 1 for a in rec.actions)
        )
        no_trade_ratio = no_trade_count / max(len(train_horizons), 1)

        report_paths = ReportPaths.from_artifacts_dir(artifacts_dir)
        writer = Phase1ReportWriter(report_paths)
        report_summary = self._build_final_summary(
            metrics=best_metrics_payload,
            reject_stats=reject_stats,
            normalizer=norm,
            state_normalizer=state_norm,
            best_epoch=history.best_epoch if history.best_epoch is not None else 0,
            no_trade_ratio=no_trade_ratio,
        )
        leakage_payload = self._build_sampling_leakage_diagnostics(report_summary)
        report_summary["hindsight_bias_warning"] = leakage_payload["hindsight_bias_warning"]
        report_summary["hindsight_vs_prospective_metric_delta"] = leakage_payload.get(
            "hindsight_vs_prospective_metric_delta", {}
        )
        report_summary["local_smoke_relaxed_guardrails"] = (
            self.config.local_smoke_relaxed_guardrails
        )
        report_summary["best_checkpoint_signoff"] = (
            leakage_payload["hindsight_bias_warning"]
            in {"ok", "not_required", "not_applicable"}
            and not self.config.local_smoke_relaxed_guardrails
        )
        signoff_blocked_reason = leakage_payload.get("signoff_blocked_reason", "")
        if self.config.local_smoke_relaxed_guardrails and not signoff_blocked_reason:
            signoff_blocked_reason = "local_smoke_relaxed_guardrails"
        report_summary["signoff_blocked_reason"] = signoff_blocked_reason
        report_summary["training_config_hash"] = training_hash
        writer.write_final_report(report_summary)
        self._logger.info(
            "phase1_report_written 说明=Phase I 最终报告已写入 path=%s best_epoch=%s no_trade_ratio=%.6f signoff=%s blocked_reason=%s training_config_hash=%s",
            report_paths.phase1_report,
            report_summary.get("best_epoch"),
            float(report_summary.get("no_trade_ratio", 0.0)),
            report_summary.get("best_checkpoint_signoff"),
            report_summary.get("signoff_blocked_reason"),
            training_hash,
        )
        if self._best_epoch_diagnostics:
            diagnostics_payload = dict(self._best_epoch_diagnostics)
            diagnostics_payload["sampling_leakage"] = leakage_payload
            diagnostics_payload["composite_score_sensitivity"] = sensitivity
            diagnostic_paths = writer.write_diagnostics(diagnostics_payload)
            self._logger.info(
                "phase1_diagnostics_written 说明=诊断文件已写入 count=%d paths=%s",
                len(diagnostic_paths),
                [str(path) for path in diagnostic_paths],
            )

        # 采样诊断 JSON: 主实验记录 hindsight bias warning 与 followup batch
        leakage_path = artifacts_dir / "sampling_leakage_diagnostics.json"
        atomic_write_json(leakage_payload, leakage_path)
        self._logger.info(
            "phase1_sampling_leakage_written 说明=采样泄漏/后视偏差诊断已写入 path=%s warning=%s",
            leakage_path,
            leakage_payload.get("hindsight_bias_warning"),
        )

        self._logger.info(
            "phase1_complete 说明=Phase I 流程完成 artifacts_dir=%s",
            artifacts_dir,
        )
        return TrainerArtifacts(
            artifacts_dir=artifacts_dir,
            phase1_config_yaml=config_yaml,
            input_schema_json=schema_path,
            window_index_train=train_window_path,
            demos_train=demos_path,
            horizon_labels_train=labels_paths["train"],
            horizon_labels_val=labels_paths["val"],
            horizon_labels_test=labels_paths["test"],
            state_normalizer_json=state_norm_path,
            reward_normalizer_json=norm_path,
            best_vq_model=ckpt.best_path,
            last_vq_model=ckpt.last_path,
            encoder_pt=encoder_path,
            decoder_pt=decoder_path,
            codebook_pt=codebook_path,
            checkpoint_manifest=ckpt.manifest_path,
            phase1_report=report_paths.phase1_report,
            composite_score_sensitivity_json=sensitivity_path,
            sampling_leakage_diagnostics_json=leakage_path,
        )

    # ---------- 子流程 ----------

    def _build_training_components(self, feature_dim, val_horizons, reward_normalizer=None):
        """实例化 model / loss / optimizer / evaluator。

        - ``Phase1Loss`` 的 ``usage_weight`` 在 ``paper_strict_reproduction=True``
          下被强制设 0，严格对齐论文公式 (4)。``num_codes`` 必须显式传入
          以避免 KL(uniform || p_code) 在 collapse 时低估真实 K。
        - ``contrastive_weight`` 仅在 ``temporal_contrastive.enabled=True`` 时启用。
        - ``Phase1ReplayEvaluator`` 通过 ``env_factory`` 闭包共享同一份 cost_model
          与 alignment，保证 teacher / student replay 完全可比。
        - ``reward_normalizer`` 透传到 evaluator，确保 val/test encoder 输入
          与 train 同分布。
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("Phase1Trainer 需要 torch")
        model = VQArchetypeModel(feature_dim, self.config.model)
        loss_fn = Phase1Loss(
            beta0=self.config.model.beta0,
            usage_weight=(
                self.config.model.codebook.health.usage_regularization_weight
                if not self.config.training.paper_strict_reproduction
                else 0.0
            ),
            contrastive_weight=self.config.data_augmentation.temporal_contrastive.contrastive_weight
            if self.config.data_augmentation.temporal_contrastive.enabled
            else 0.0,
            contrastive_temperature=self.config.data_augmentation.temporal_contrastive.temperature,
            num_codes=self.config.model.num_codes,
            usage_profit_alignment_weight=(
                self.config.model.codebook.health.usage_profit_alignment_weight
                if not self.config.training.paper_strict_reproduction
                else 0.0
            ),
            usage_profit_alignment_target_corr=(
                self.config.model.codebook.health.usage_profit_alignment_target_corr
            ),
        )
        cost_model = LobDepthCostModel(
            commission_rate=self.config.dp.cost_config.commission_rate,
            book_levels=self.config.dp.cost_config.book_levels,
        )
        alignment = RewardAlignment(self.config.dp.cost_config.reward_alignment)

        def env_factory():
            return TradingEnv(
                cost_model=cost_model,
                reward_alignment=alignment,
                max_position=self.config.dp.max_position,
            )

        replay_eval = Phase1ReplayEvaluator(env_factory=env_factory)
        evaluator = Phase1Evaluator(
            replay_evaluator=replay_eval,
            fast_probe_size=self.config.training.fast_val_probe_size,
            reward_normalizer=reward_normalizer,
        )
        model.to(self._device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.training.lr)
        return model, evaluator, loss_fn, optimizer

    def _effective_pretrain_epochs(self) -> int:
        """返回本次训练实际使用的 Phase A epoch 数。"""
        configured = int(self.config.training.pretrain_epochs)
        if configured < 0:
            raise Phase1FatalError("training.pretrain_epochs must be >= 0")
        if configured >= self.config.training.epochs:
            effective = max(self.config.training.epochs - 1, 0)
            if effective != configured:
                self._logger.warning(
                    "phase1_pretrain_epochs_clamped 说明=pretrain_epochs 必须小于 epochs,已自动收敛到可运行值 configured=%d epochs=%d effective=%d",
                    configured,
                    self.config.training.epochs,
                    effective,
                )
            return effective
        return configured

    @staticmethod
    def _set_optimizer_lr(optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

    def _batch_amp_enabled(
        self,
        *,
        base_amp_enabled: bool,
        states,
        rewards,
        epoch: int,
        batch_idx: int,
        logged: bool,
    ) -> tuple[bool, bool]:
        """根据 batch 输入量级决定是否安全启用 CUDA autocast。

        Phase I 的特征产物中可能包含未标准化的大额成交量/金额类字段。fp32
        前向仍然有限，但 fp16 线性层会在输入量级过大时溢出为 inf/nan。
        """
        import torch

        if not bool(torch.isfinite(states).all().item()):
            raise Phase1FatalError(
                f"phase1 batch contains non-finite states: epoch={epoch} batch={batch_idx}"
            )
        if not bool(torch.isfinite(rewards).all().item()):
            raise Phase1FatalError(
                f"phase1 batch contains non-finite rewards: epoch={epoch} batch={batch_idx}"
            )
        if not base_amp_enabled:
            return False, logged

        state_abs_max = float(states.detach().abs().max().item())
        reward_abs_max = float(rewards.detach().abs().max().item())
        max_abs = max(state_abs_max, reward_abs_max)
        if max_abs <= _AMP_SAFE_INPUT_ABS_MAX:
            return True, logged

        if not logged:
            self._logger.warning(
                "phase1_amp_disabled_for_input_scale 说明=batch 连续输入量级超过 fp16 安全阈值,本轮训练将对超阈值 batch 使用 fp32 前向以避免 NaN epoch=%d batch=%d max_abs=%.6e threshold=%.6e state_abs_max=%.6e reward_abs_max=%.6e",
                epoch,
                batch_idx,
                max_abs,
                _AMP_SAFE_INPUT_ABS_MAX,
                state_abs_max,
                reward_abs_max,
            )
            logged = True
        return False, logged

    def _log_reward_composition(self, horizons, split: str) -> None:
        import math as _math

        all_rewards = [v for rec in horizons for v in (rec.rewards or [])]
        if not all_rewards:
            return

        n = len(all_rewards)
        n_horizons = len(horizons)
        horizon_returns = [sum(rec.rewards) for rec in horizons if rec.rewards]

        action_counter = {}
        for rec in horizons:
            for a in (rec.actions or []):
                action_counter[a] = action_counter.get(a, 0) + 1
        total_actions = sum(action_counter.values()) or 1

        zero_step = sum(1 for v in all_rewards if abs(v) < 1e-9)
        pos_step = sum(1 for v in all_rewards if v > 1e-9)
        neg_step = sum(1 for v in all_rewards if v < -1e-9)

        if horizon_returns:
            hr_sorted = sorted(horizon_returns)
            hr_pos = sum(1 for v in horizon_returns if v > 1e-9)
            hr_neg = sum(1 for v in horizon_returns if v < -1e-9)
        else:
            hr_sorted = []
            hr_pos = 0
            hr_neg = 0

        def _pct(vals, p):
            k = (len(vals) - 1) * p / 100.0
            f = int(k)
            c = f + 1
            if c >= len(vals):
                return vals[-1]
            return vals[f] + (vals[c] - vals[f]) * (k - f)

        self._logger.warning(
            "phase1_reward_composition_diagnostic 说明=训练前 reward 组成诊断 split=%s "
            "step_rewards_n=%d zero=%d(%.4f) positive=%d(%.4f) negative=%d(%.4f) "
            "horizon_n=%d horizon_return_positive=%d(%.4f) horizon_return_negative=%d(%.4f) "
            "horizon_return_mean=%.10f horizon_return_p50=%.10f horizon_return_p95=%.10f "
            "action_dist=%s max_position=%d commission_rate=%.6f",
            split,
            n, zero_step, zero_step / n, pos_step, pos_step / n, neg_step, neg_step / n,
            n_horizons,
            hr_pos, hr_pos / max(n_horizons, 1),
            hr_neg, hr_neg / max(n_horizons, 1),
            sum(horizon_returns) / max(len(horizon_returns), 1),
            _pct(hr_sorted, 50) if hr_sorted else 0.0,
            _pct(hr_sorted, 95) if hr_sorted else 0.0,
            {str(k): f"{v / total_actions:.4f}" for k, v in sorted(action_counter.items())},
            self.config.dp.max_position,
            self.config.dp.cost_config.commission_rate,
        )

    def _warmup_codebook(self, model, train_horizons, normalizer):
        """用前若干 batch 的 ``z_e`` 跑 K-means 初始化 codebook。

        - ``init_method=random_normal`` 时直接跳过。
        - 其他模式下取前 ``kmeans_warmup_batches * batch_size`` 个 records 跑一次
          encoder forward，把 ``z_e`` 喂给 ``quantizer.warmup_initialize``。
        - 该步骤只用 train batches，绝不读 val/test。
        """
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            return
        if self.config.model.codebook.init_method == "random_normal":
            return
        warmup_records = train_horizons[: max(self.config.model.codebook.kmeans_warmup_batches, 1) * self.config.training.batch_size]
        if not warmup_records:
            return
        # warmup 必须用同一 normalizer，否则 z_e 分布与训练阶段不一致，
        # K-means 中心一开始就跑偏。
        dataset = Phase1DemoDataset(records=warmup_records, reward_normalizer=normalizer)
        loader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=False, collate_fn=collate_phase1)
        z_e_list = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(self._device, non_blocking=True)
                actions = batch["actions"].to(self._device, non_blocking=True)
                rewards = batch["rewards"].to(self._device, non_blocking=True)
                fused = model.input_adapter(states, actions, rewards)
                z_e = model.encoder(fused)
                z_e_list.append(z_e)
                if len(z_e_list) >= self.config.model.codebook.kmeans_warmup_batches:
                    break
        if z_e_list:
            samples = torch.cat(z_e_list, dim=0)
            model.quantizer.warmup_initialize(samples)
        if was_training:
            model.train()

    def _train_loop(
        self,
        *,
        model,
        loss_fn,
        optimizer,
        train_dataset,
        evaluator,
        val_dataset,
        val_records,
        checkpoint: Phase1CheckpointManager,
        policy: Phase1SelectionPolicy,
        history: SelectionHistory,
    ):
        """主训练循环。

        每 epoch 的步骤
        ----------------
        1. ``model.train()`` + DataLoader 迭代:
           - forward → Phase1Loss → backward。
           - 梯度裁剪（``gradient_clip_norm``）。
           - optimizer step。
           - ``quantizer.update_codebook(z_e.detach(), code_id)``: EMA 模式下更新
             codebook；gradient 模式 no-op。
        2. ``checkpoint.save_last`` + ``save_periodic``。
        3. 调 ``evaluator.evaluate_epoch`` 拿 metrics（按 ``full_validation_every_epochs``
           决定 fast probe 还是 full）。
        4. 注入 ``_consecutive_collapse_epochs`` / ``_consecutive_collapse_limit``
           供 selection_policy 判定 fatal。
        5. ``policy.evaluate(metrics, history)`` → verdict。
        6. ``checkpoint.commit_verdict``: ``promote_to_best`` 时 copy last → best；
           ``fatal`` 时抛 ``Phase1FatalCollapse``。
        7. ``policy.update_history`` 更新 best / collapse 计数。

        Notes
        -----
        - 该方法允许 ``Phase1FatalCollapse`` 自然向上抛出；``run()`` 会捕获并
          转抛 ``Phase1FatalError``。
        """
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:  # pragma: no cover
            raise RuntimeError("Phase1Trainer 需要 torch")

        _use_cuda = self._device != "cpu"
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_phase1,
            num_workers=2 if _use_cuda else 0,
            pin_memory=_use_cuda,
        )
        _amp_enabled = self.config.training.mixed_precision and _use_cuda
        _scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)
        effective_pretrain_epochs = self._effective_pretrain_epochs()
        phase_b_epochs = max(self.config.training.epochs - effective_pretrain_epochs, 1)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase_b_epochs, eta_min=1e-5
        )
        self._logger.info(
            "phase1_train_loop_start 说明=训练循环开始 epochs=%d pretrain_epochs=%d train_records=%d val_records=%d batches_per_epoch=%d batch_size=%d",
            self.config.training.epochs,
            effective_pretrain_epochs,
            len(train_dataset),
            len(val_dataset),
            len(train_loader),
            self.config.training.batch_size,
        )

        _last_batch_idx = len(train_loader) - 1
        _amp_disabled_for_input_scale_logged = False
        for epoch in range(self.config.training.epochs):
            is_phase_a = epoch < effective_pretrain_epochs
            if is_phase_a and self.config.training.pretrain_lr is not None:
                self._set_optimizer_lr(optimizer, self.config.training.pretrain_lr)
            elif (
                epoch == effective_pretrain_epochs
                and effective_pretrain_epochs > 0
                and self.config.training.pretrain_lr is not None
            ):
                self._set_optimizer_lr(optimizer, self.config.training.lr)
            model.train()
            epoch_batch_diagnostic: Optional[Dict[str, Any]] = None
            for batch_idx, batch in enumerate(train_loader):
                states = batch["states"].to(self._device, non_blocking=True)
                actions = batch["actions"].to(self._device, non_blocking=True)
                rewards = batch["rewards"].to(self._device, non_blocking=True)
                trajectory_returns = batch["trajectory_returns"].to(
                    self._device, non_blocking=True
                )
                batch_amp_enabled, _amp_disabled_for_input_scale_logged = (
                    self._batch_amp_enabled(
                        base_amp_enabled=_amp_enabled,
                        states=states,
                        rewards=rewards,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        logged=_amp_disabled_for_input_scale_logged,
                    )
                )
                with torch.amp.autocast("cuda", enabled=batch_amp_enabled):
                    if is_phase_a:
                        outputs = model.forward_pretrain(states, actions, rewards)
                        loss = loss_fn.forward_pretrain(
                            action_logits=outputs.action_logits,
                            target_actions=actions,
                        )
                    else:
                        outputs = model(states, actions, rewards)
                        loss = loss_fn(
                            action_logits=outputs.action_logits,
                            target_actions=actions,
                            z_e=outputs.z_e,
                            z_q_no_grad=outputs.z_q_no_grad,
                            code_id=outputs.code_id,
                            contrastive_pair_ids=batch["contrastive_pair_ids"],
                            trajectory_returns=trajectory_returns,
                            codebook=model.quantizer.codebook,
                            soft_assignment_temperature=(
                                self.config.model.codebook.health.usage_profit_alignment_temperature
                            ),
                        )
                optimizer.zero_grad(set_to_none=True)
                _scaler.scale(loss.total).backward()
                if self.config.training.gradient_clip_norm > 0:
                    _scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.training.gradient_clip_norm
                    )
                _scaler.step(optimizer)
                _scaler.update()
                if not is_phase_a and outputs.code_id is not None:
                    model.quantizer.update_codebook(
                        outputs.z_e.detach(), outputs.code_id.detach()
                    )

                if batch_idx == _last_batch_idx:
                    if outputs.code_id is not None:
                        _code_ids = outputs.code_id.detach()
                        _unique = torch.unique(_code_ids).numel()
                        _counts = torch.bincount(_code_ids.flatten(), minlength=self.config.model.num_codes).float()
                        _total = _counts.sum().clamp_min(1.0)
                        _p = _counts / _total
                        _entropy = -(_p * _p.clamp_min(1e-8).log()).sum().item()
                    else:
                        _unique = 0
                        _entropy = 0.0
                    _max_entropy = math.log(self.config.model.num_codes)
                    epoch_batch_diagnostic = {
                        "phase": "A" if is_phase_a else "B",
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch_count": len(train_loader),
                        "loss_total": loss.total.item(),
                        "loss_reconstruction": loss.reconstruction.item(),
                        "loss_commitment": loss.commitment.item(),
                        "loss_usage": (
                            f"{loss.usage.item():.6f}" if loss.usage is not None else "None"
                        ),
                        "loss_alignment": (
                            f"{loss.alignment.item():.6f}" if loss.alignment is not None else "None"
                        ),
                        "code_unique": _unique,
                        "num_codes": self.config.model.num_codes,
                        "code_entropy": _entropy,
                        "max_entropy": _max_entropy,
                        "z_e_norm": outputs.z_e.detach().norm(dim=-1).mean().item(),
                        "z_q_norm": outputs.z_q_no_grad.detach().norm(dim=-1).mean().item(),
                    }

            if is_phase_a and epoch + 1 == effective_pretrain_epochs:
                self._warmup_codebook(
                    model,
                    train_dataset.records,
                    train_dataset.reward_normalizer,
                )
                self._logger.info(
                    "phase1_codebook_warmup_after_pretrain_done 说明=Phase A 结束后已初始化 codebook epoch=%d init_method=%s warmup_batches=%d",
                    epoch,
                    self.config.model.codebook.init_method,
                    self.config.model.codebook.kmeans_warmup_batches,
                )

            if self._scheduler is not None and not is_phase_a:
                self._scheduler.step()

            # 每 5 轮进行一次评估、诊断和 checkpoint 保存；Phase A 不参与选择。
            evaluation_interval = 5
            should_evaluate = (not is_phase_a) and (
                (epoch + 1) % evaluation_interval == 0
                or epoch == self.config.training.epochs - 1
            )

            restarted_code_ids = []
            if should_evaluate:
                restarted_code_ids = self._maybe_restart_dead_codes(
                    model=model,
                    train_dataset=train_dataset,
                    epoch=epoch,
                )
            restart_events = [
                {"epoch": epoch, "code_id": int(cid)} for cid in restarted_code_ids
            ]
            self._dead_code_restart_events.extend(restart_events)

            state = {"model": model.state_dict(), "epoch": epoch}
            metrics_for_select: Dict[str, Any] = {"epoch": epoch}

            if should_evaluate:
                full_val = (epoch + 1) % max(self.config.training.full_validation_every_epochs, 1) == 0
                ep_metrics = evaluator.evaluate_epoch(
                    epoch=epoch,
                    model=model,
                    val_data=val_dataset,
                    val_records=val_records,
                    full_validation=full_val,
                )
                metrics_for_select.update(ep_metrics.metrics)
                if "val_weighted_reconstruction_accuracy" in metrics_for_select:
                    metrics_for_select["weighted_reconstruction_accuracy"] = metrics_for_select[
                        "val_weighted_reconstruction_accuracy"
                    ]

                metrics_for_select["_dead_code_restart_triggered"] = bool(restarted_code_ids)
                metrics_for_select["_dead_code_restart_cooldown_epochs"] = (
                    self.config.model.codebook.health.restart_cooldown_epochs
                )
                metrics_for_select["dead_code_restarts"] = len(restarted_code_ids)
                metrics_for_select["dead_code_restart_events"] = restart_events
                metrics_for_select["_consecutive_collapse_epochs"] = (
                    history.consecutive_collapse_epochs + (
                        1 if metrics_for_select.get("code_usage_ratio", 1.0) < self.config.selection_policy.min_code_usage_ratio else 0
                    )
                )
                metrics_for_select["_consecutive_collapse_limit"] = self.config.model.codebook.health.consecutive_collapse_epoch_limit

                verdict = policy.evaluate(metrics_for_select, history)
                metrics_for_select["phase1_composite_score"] = verdict.composite_score
                metrics_for_select["phase1_composite_score_debug"] = verdict.composite_score_debug
                checkpoint.save_last(state, metrics_for_select, epoch)
                checkpoint.save_periodic(state, metrics_for_select, epoch, self.config.training.save_every)
                checkpoint.commit_verdict(state, metrics_for_select, verdict, epoch)
                if verdict.decision == "promote_to_best":
                    self._best_epoch_diagnostics = dict(ep_metrics.diagnostics)
                history = policy.update_history(history, metrics_for_select, verdict)
            else:
                # 非评估轮次:不保存 checkpoint,不进行评估和诊断
                verdict = SelectionVerdict(decision="skipped", reasons=["skipped_evaluation"], composite_score=0.0, composite_score_debug={})

            if epoch_batch_diagnostic is not None:
                self._logger.info(
                    "phase1_batch_diagnostic 说明=训练 batch 诊断 phase=%s epoch=%d batch=%d/%d "
                    "loss_total=%.6f loss_rec=%.6f loss_commit=%.6f loss_usage=%s loss_alignment=%s "
                    "code_unique=%d/%d code_entropy=%.4f/%.4f "
                    "z_e_norm=%.4f z_q_norm=%.4f",
                    epoch_batch_diagnostic["phase"],
                    epoch_batch_diagnostic["epoch"],
                    epoch_batch_diagnostic["batch_idx"],
                    epoch_batch_diagnostic["batch_count"],
                    epoch_batch_diagnostic["loss_total"],
                    epoch_batch_diagnostic["loss_reconstruction"],
                    epoch_batch_diagnostic["loss_commitment"],
                    epoch_batch_diagnostic["loss_usage"],
                    epoch_batch_diagnostic["loss_alignment"],
                    epoch_batch_diagnostic["code_unique"],
                    epoch_batch_diagnostic["num_codes"],
                    epoch_batch_diagnostic["code_entropy"],
                    epoch_batch_diagnostic["max_entropy"],
                    epoch_batch_diagnostic["z_e_norm"],
                    epoch_batch_diagnostic["z_q_norm"],
                )
            if should_evaluate:
                self._logger.info(
                    "phase1_epoch_result 说明=单个 epoch 训练完成 epoch=%d full_validation=%s decision=%s score=%.6f code_usage_ratio=%s val_return_capture_ratio=%s val_sharpe_ratio=%s restarts=%d reasons=%s",
                    epoch,
                    full_val,
                    verdict.decision,
                    verdict.composite_score,
                    metrics_for_select.get("code_usage_ratio"),
                    metrics_for_select.get("val_return_capture_ratio"),
                    metrics_for_select.get("val_sharpe_ratio"),
                    len(restarted_code_ids),
                    ",".join(verdict.reasons),
                )
        return history

    def _maybe_restart_dead_codes(self, *, model, train_dataset, epoch: int) -> List[int]:
        """按配置把 dead-code restart 接入训练循环。"""
        health = self.config.model.codebook.health
        if not health.dead_code_restart:
            return []
        if epoch + 1 < max(health.dead_code_patience, 1):
            return []
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:  # pragma: no cover
            return []

        loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collate_phase1,
        )
        z_e_chunks = []
        err_chunks = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(self._device, non_blocking=True)
                actions = batch["actions"].to(self._device, non_blocking=True)
                rewards = batch["rewards"].to(self._device, non_blocking=True)
                outputs = model(states, actions, rewards)
                per_step = torch.nn.functional.cross_entropy(
                    outputs.action_logits.reshape(-1, outputs.action_logits.shape[-1]),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(actions.shape)
                z_e_chunks.append(outputs.z_e.detach())
                err_chunks.append(per_step.mean(dim=1).detach())
        if was_training:
            model.train()
        if not z_e_chunks:
            return []
        z_e = torch.cat(z_e_chunks, dim=0)
        errors = torch.cat(err_chunks, dim=0)
        return model.quantizer.restart_dead_codes(
            encoder_outputs=z_e,
            reconstruction_errors=errors,
            current_epoch=epoch,
        )

    def _export_horizon_labels(self, model, *, store, horizons_by_split, normalizer=None):
        """对 train/val/test horizon 跑 encoder，得到 ``code_label`` 并写 feather。

        - 用 best checkpoint（``run`` 在调用本方法前会 reload best state）。
        - 把每条 horizon 的 ``code_label / demo_return / num_switches / is_no_trade``
          打包成 ``HorizonLabel`` 写入 ``horizon_labels_{split}.feather``。
        - encoder 输入的 rewards 必须经 ``normalizer.transform``（保证与训练同分布）；
          ``demo_return`` 仍用原始 rec.rewards，反映真实 DP 收益。
        - 这是 Phase II selector 的 KL/demo regularization 标签来源。
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("export_horizon_labels 需要 torch")
        out: Dict[str, Path] = {}
        export_batch_size = max(1, min(int(self.config.training.batch_size), 1024))
        was_training = model.training
        model.eval()
        for split, recs in horizons_by_split.items():
            labels: List[HorizonLabel] = []
            if not recs:
                out[split] = store.save_labels(labels, split)
                continue
            code_id_values: List[int] = []
            with torch.no_grad():
                for start in range(0, len(recs), export_batch_size):
                    chunk = recs[start : start + export_batch_size]
                    states = torch.tensor(
                        [r.states for r in chunk],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    actions = torch.tensor(
                        [r.actions for r in chunk],
                        dtype=torch.long,
                        device=self._device,
                    )
                    if normalizer is not None:
                        normalized = [list(normalizer.transform(r.rewards)) for r in chunk]
                    else:
                        normalized = [list(r.rewards) for r in chunk]
                    rewards = torch.tensor(
                        normalized,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    code_ids, _ = model.encode(states, actions, rewards)
                    code_id_values.extend(int(v) for v in code_ids.detach().cpu().tolist())
            for rec, code_id in zip(recs, code_id_values):
                # demo_return 必须使用原始 rewards（actual return），而不是 normalizer 后的值。
                demo_return = sum(rec.rewards or [])
                action_seq = list(rec.actions or [])
                switch_seq = action_seq[:-1] if len(action_seq) > 1 else action_seq
                num_switches = sum(
                    1 for i in range(1, len(switch_seq))
                    if switch_seq[i] != switch_seq[i - 1]
                )
                is_no_trade = all(a == 1 for a in action_seq)
                labels.append(
                    HorizonLabel(
                        sample_id=rec.sample_id,
                        start_index=rec.start_index,
                        end_index=rec.end_index,
                        last_execution_row=(
                            rec.last_execution_row
                            if rec.last_execution_row is not None
                            else rec.start_index
                            + self.config.horizon
                            - 1
                            + (
                                0
                                if self.config.dp.cost_config.reward_alignment == "paper_formula"
                                else 1
                            )
                        ),
                        last_markout_row=(
                            rec.last_markout_row
                            if rec.last_markout_row is not None
                            else rec.start_index
                            + self.config.horizon
                            + (
                                0
                                if self.config.dp.cost_config.reward_alignment == "paper_formula"
                                else 1
                            )
                        ),
                        strata_label=rec.strata_label,
                        stratification_mode=self.config.stratification.mode,
                        is_augmented=rec.is_augmented,
                        augmentation_type=rec.augmentation_type,
                        code_label=int(code_id),
                        demo_return=float(demo_return),
                        num_switches=int(num_switches),
                        is_no_trade=bool(is_no_trade),
                    )
                )
            out[split] = store.save_labels(labels, split)
        if was_training:
            model.train()
        return out

    def _export_phase2_artifacts(self, state: Dict[str, Any]):
        """从 best checkpoint 抽 encoder / decoder / codebook 单独保存。

        - 通过 state_dict 前缀切分:
          * ``input_adapter.*`` + ``encoder.*`` → ``encoder.pt``
          * ``decoder.*`` → ``decoder.pt``
          * ``quantizer.*`` (含 codebook 与 EMA buffer) → ``codebook.pt``
        - Phase II/III 训练只需读取这三份文件 + ``horizon_labels_*.feather``，
          即可启动；不必重跑 Phase I。
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("export 需要 torch")
        artifacts_dir = self.config.artifacts_dir()
        encoder_path = artifacts_dir / "encoder.pt"
        decoder_path = artifacts_dir / "decoder.pt"
        codebook_path = artifacts_dir / "codebook.pt"
        # state["model"] 是 state_dict；按前缀切分。
        sd = state.get("model", state)
        encoder_sd = {k: v for k, v in sd.items() if k.startswith(("input_adapter.", "encoder."))}
        decoder_sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
        codebook_sd = {k: v for k, v in sd.items() if k.startswith("quantizer.")}
        torch.save(encoder_sd, encoder_path)
        torch.save(decoder_sd, decoder_path)
        torch.save(codebook_sd, codebook_path)
        return encoder_path, decoder_path, codebook_path

    def _reload_state(self, model, state):
        sd = state.get("model", state)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass

    def _snapshot_state(self, model) -> Dict[str, Any]:
        return {"model": model.state_dict(), "epoch": -1}

    def _best_metrics(self, ckpt: Phase1CheckpointManager) -> Dict[str, float]:
        """从 manifest 中读取 best epoch 对应的 metrics。"""
        entries = ckpt.load_manifest()
        if not entries:
            return {}
        best_entries = [e for e in entries if e.is_best]
        if not best_entries:
            return {}
        best = best_entries[-1]
        try:
            return read_json(best.metrics_path)
        except Exception:
            return {}

    def _all_epoch_metrics(self, ckpt: Phase1CheckpointManager) -> List[dict]:
        """从 manifest 读取每个 epoch 的 metrics，用于 sensitivity 重新选 best。"""
        out: List[dict] = []
        for entry in ckpt.load_manifest():
            try:
                payload = read_json(entry.metrics_path)
            except Exception:
                continue
            payload.setdefault("epoch", entry.epoch)
            payload.setdefault("_manifest_verdict", entry.verdict)
            payload.setdefault("_manifest_is_best", entry.is_best)
            out.append(payload)
        return out

    def _build_final_summary(
        self,
        *,
        metrics,
        reject_stats,
        normalizer,
        state_normalizer=None,
        best_epoch,
        no_trade_ratio: float = 0.0,
    ) -> dict:
        cost = self.config.dp.cost_config
        norm_dict = normalizer.to_dict() if normalizer.stats else {}
        state_norm_dict = state_normalizer.to_dict() if state_normalizer is not None else {}
        summary = dict(metrics)
        if "val_weighted_reconstruction_accuracy" in summary:
            summary.setdefault(
                "weighted_reconstruction_accuracy",
                summary["val_weighted_reconstruction_accuracy"],
            )
        summary.setdefault("reconstruction_accuracy", 0.0)
        summary.setdefault("weighted_reconstruction_accuracy", 0.0)
        summary.setdefault("non_flat_accuracy", 0.0)
        summary.setdefault("code_usage", {"used": 0, "K": self.config.model.num_codes})
        summary.setdefault("perplexity", 0.0)
        summary.setdefault("single_trade_consistency_rate", 0.0)
        # 用 trainer 在 demo 生成后实际统计的 no_trade_ratio 覆盖默认。
        summary["no_trade_ratio"] = float(no_trade_ratio)
        summary["reward_alignment"] = cost.reward_alignment
        summary["max_position"] = self.config.dp.max_position
        summary["factor_profile"] = self.config.factor_profile
        summary["factor_list_file"] = self.config.factor_list_file or ""
        summary["processed_data_mode"] = self._processed_data_metadata.get(
            "processed_data_mode", "legacy_inline"
        )
        summary["data_process_manifest"] = self._processed_data_metadata.get(
            "data_process_manifest", ""
        )
        summary["data_batch_id"] = self._processed_data_metadata.get(
            "data_batch_id", ""
        )
        summary["schema_hash"] = self._processed_data_metadata.get("schema_hash", "")
        summary["data_process_hash"] = self._processed_data_metadata.get(
            "data_process_hash", ""
        )
        summary["dp_teacher_hash"] = self._processed_data_metadata.get(
            "dp_teacher_hash", ""
        )
        summary["reward_normalization_resolved"] = norm_dict.get("method", "")
        summary["reward_norm_clip_ratio"] = norm_dict.get("clip_ratio", 0.0)
        summary["state_normalization_resolved"] = state_norm_dict.get("method", "")
        summary["state_norm_max_abs_before"] = state_norm_dict.get("max_abs_before", 0.0)
        summary["state_norm_max_abs_after_fit"] = state_norm_dict.get("max_abs_after_fit", 0.0)
        summary["dataset_reject_rate"] = float(reject_stats.dataset_reject_rate)
        summary["stratification_mode"] = self.config.stratification.mode
        summary["is_hindsight_stratification"] = self.config.stratification.mode == "hindsight_horizon"
        summary["prospective_diagnostic_required"] = self.config.stratification.require_prospective_diagnostic
        summary["diagnostic_pair_batch_id"] = self.config.stratification.diagnostic_pair_batch_id or ""
        summary.setdefault("phase1_composite_score", 0.0)
        summary["best_epoch"] = best_epoch
        summary["best_checkpoint_path"] = str(self.config.artifacts_dir() / "best_vq_model.pt")
        summary["selection_metric"] = self.config.selection_policy.selection_metric
        summary["dead_code_restarts"] = len(self._dead_code_restart_events)
        summary["dead_code_restart_events"] = list(self._dead_code_restart_events)
        summary["composite_score_sensitivity"] = "composite_score_sensitivity.json"
        return summary

    def _hindsight_warning_triggered(self, summary: dict) -> bool:
        return summary.get("hindsight_bias_warning") == "exceeded"

    def _build_sampling_leakage_diagnostics(self, report_summary: dict) -> dict:
        """读取 prospective 对照 report 并计算 hindsight/prospective 指标差异。"""
        payload = {
            "stratification_mode": self.config.stratification.mode,
            "is_hindsight_stratification": self.config.stratification.mode == "hindsight_horizon",
            "diagnostic_pair_batch_id": self.config.stratification.diagnostic_pair_batch_id,
            "allow_missing_prospective_diagnostic": self.config.allow_missing_prospective_diagnostic,
            "risk_acknowledged_by": self.config.risk_acknowledged_by,
            "expected_sign_off_followup_batch_id": self.config.expected_sign_off_followup_batch_id,
            "hindsight_vs_prospective_metric_delta": {},
            "missing_metrics": [],
            "hindsight_bias_warning": "ok",
            "signoff_blocked_reason": "",
        }
        if self.config.stratification.mode != "hindsight_horizon":
            payload["hindsight_bias_warning"] = "not_applicable"
            return payload
        if not self.config.stratification.require_prospective_diagnostic:
            payload["hindsight_bias_warning"] = "not_required"
            return payload

        diagnostic_id = self.config.stratification.diagnostic_pair_batch_id
        if not diagnostic_id:
            payload["hindsight_bias_warning"] = "missing_acknowledged"
            payload["signoff_blocked_reason"] = "missing_diagnostic_pair_batch_id"
            return payload

        prospective_report = (
            Path(self.config.artifact_root)
            / self.config.pair
            / diagnostic_id
            / "phase1"
            / "phase1_report.json"
        )
        payload["prospective_report_path"] = str(prospective_report)
        if not prospective_report.exists():
            if self.config.allow_missing_prospective_diagnostic:
                payload["hindsight_bias_warning"] = "missing_acknowledged"
                payload["signoff_blocked_reason"] = "missing_prospective_report"
                return payload
            raise Phase1FatalError(
                f"缺少 prospective 对照报告: {prospective_report}; "
                "主实验不可 sign-off。"
            )

        prospective = read_json(prospective_report)
        exceeded: List[str] = []
        for metric, max_delta in self.config.stratification.hindsight_vs_prospective_max_delta.items():
            if metric not in report_summary or metric not in prospective:
                payload["missing_metrics"].append(metric)
                continue
            current_value = float(report_summary[metric])
            prospective_value = float(prospective[metric])
            delta = current_value - prospective_value
            abs_delta = abs(delta)
            is_exceeded = abs_delta > float(max_delta)
            if is_exceeded:
                exceeded.append(metric)
            payload["hindsight_vs_prospective_metric_delta"][metric] = {
                "hindsight": current_value,
                "prospective": prospective_value,
                "delta": delta,
                "abs_delta": abs_delta,
                "max_delta": float(max_delta),
                "exceeded": is_exceeded,
            }

        if exceeded:
            payload["hindsight_bias_warning"] = "exceeded"
            payload["signoff_blocked_reason"] = "hindsight_vs_prospective_delta_exceeded"
            payload["exceeded_metrics"] = exceeded
        return payload
