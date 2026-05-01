"""Phase II walk-forward replay 与 test label 泄漏防护。

设计文档锚点: Phase II 执行计划 §Step 3 / §Step 6。

职责:
- 实现 train/val/test 的 walk-forward replay（按时间正序、仓位连续）。
- _guard_no_test_label_in_decision_path(): 防止 test label 进入决策路径。
- 支持 deterministic argmax 主路径和 stochastic seed pack 诊断。

关键约束:
- backtest 主路径固定为 deterministic argmax。
- stochastic seed pack 只作为诊断输出。
- test label 进入决策路径时立即抛 Phase2TestLabelLeakageError。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.config.phase2_config import Phase2Config
from src.data.phase2_dataset import Phase2Dataset
from src.evaluation.phase2_online_action_throttle import Phase2OnlineActionThrottle
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.rl.actor_critic import ActorCritic
from src.trading.env import HorizonInputs, TradingEnv


class Phase2TestLabelLeakageError(RuntimeError):
    """检测到 test label 进入 selector 决策路径。"""


@dataclass
class Phase2HorizonReplayRecord:
    """单个 horizon 的 replay 记录。"""
    sample_id: str
    env_id: int
    chosen_code: int
    final_position: int
    reward_raw: float
    reward_scaled: float
    boundary_cost: float
    cost_paid: float
    risk_triggered: bool = False
    risk_trigger_step: Optional[int] = None
    risk_reason: Optional[str] = None
    fold_id: Optional[int] = None
    timestamp_start: Optional[str] = None
    step_returns: List[float] = field(default_factory=list)
    selector_confidence: Optional[float] = None
    throttle_triggered: bool = False
    original_code: Optional[int] = None
    throttled_code: Optional[int] = None


class Phase2BacktestRunner:
    """Phase II walk-forward backtest runner。

    使用方式::

        runner = Phase2BacktestRunner(config, actor_critic, frozen_policy, dataset, env_factory)
        records = runner.run_walk_forward(split="val", deterministic=True)
    """

    def __init__(
        self,
        config: Phase2Config,
        actor_critic: ActorCritic,
        frozen_policy: Phase1FrozenPolicy,
        dataset: Phase2Dataset,
        trading_env_factory: Callable[[], TradingEnv],
    ) -> None:
        self.config = config
        self.actor_critic = actor_critic
        self.frozen_policy = frozen_policy
        self.dataset = dataset
        self.trading_env_factory = trading_env_factory
        self._test_label_guard_active: bool = False

    def run_walk_forward(
        self,
        split: str,
        deterministic: bool = True,
        stochastic_seeds: Optional[List[int]] = None,
        entry_indices: Optional[Sequence[int]] = None,
        initial_position: int = 0,
        fold_id: Optional[int] = None,
    ) -> List[Phase2HorizonReplayRecord]:
        """执行 walk-forward replay。

        Parameters
        ----------
        split : "train" / "val" / "test"。
        deterministic : True 时使用 argmax；False 时使用 stochastic。
        stochastic_seeds : stochastic 模式的 seed 列表（仅诊断）。

        Returns
        -------
        List[Phase2HorizonReplayRecord] : per-horizon replay 记录。
        """
        if split == "test":
            self._guard_no_test_label_in_decision_path(split)

        resolved_entries = self._resolve_walk_forward_entries(split, entry_indices)
        if not resolved_entries:
            return []

        records: List[Phase2HorizonReplayRecord] = []
        prev_terminal_position = int(initial_position)
        cumulative_loss = 0.0
        consecutive_losses = 0
        trading_env = self.trading_env_factory()
        self.actor_critic.selector.eval()
        device = next(self.actor_critic.selector.parameters()).device
        throttle = Phase2OnlineActionThrottle(self.config.online_action_throttle)

        for actual_idx, entry in resolved_entries:
            # 获取 selector state
            obs = self.dataset.get_selector_state(actual_idx, prev_terminal_position)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # test label 泄漏检查
            if split == "test" and entry.is_labeled:
                raise Phase2TestLabelLeakageError(
                    f"test horizon {entry.sample_id} 的 code_label 进入了决策路径"
                )

            with torch.no_grad():
                act_out = self.actor_critic.act(obs_tensor, deterministic=deterministic)
            chosen_code = int(act_out.action.item())
            confidence = None
            if hasattr(act_out, "log_prob"):
                try:
                    confidence = float(torch.exp(act_out.log_prob).item())
                except Exception:
                    confidence = None
            throttle_decision = throttle.apply(chosen_code, confidence=confidence)
            original_code = chosen_code
            chosen_code = throttle_decision.action

            # 使用 frozen policy streaming decode
            horizon_states = self.dataset.get_horizon_states(actual_idx)
            horizon_inputs = self.dataset.get_horizon_inputs(actual_idx)
            h = self.config.horizon

            self.frozen_policy.reset(code_id=chosen_code)
            base_actions: List[int] = []
            for t in range(min(h, len(horizon_states))):
                out = self.frozen_policy.decode_step(horizon_states[t])
                base_actions.append(out.action)

            risk_triggered = False
            risk_reason = None
            risk_trigger_step = None
            lrc = self.config.live_risk_controls
            if lrc.mid_horizon_emergency_flatten and lrc.flatten_on_trigger:
                if lrc.daily_loss_limit is not None and abs(cumulative_loss) > lrc.daily_loss_limit:
                    risk_triggered = True
                    risk_reason = "daily_loss_limit"
                if (
                    lrc.consecutive_loss_limit is not None
                    and consecutive_losses >= lrc.consecutive_loss_limit
                ):
                    risk_triggered = True
                    risk_reason = risk_reason or "consecutive_loss_limit"
                if risk_triggered:
                    risk_trigger_step = 0
                    base_actions = [1] * len(base_actions)

            # TradingEnv replay
            init_pos = prev_terminal_position
            if not self.config.horizon_schedule.position_continuity:
                init_pos = 0

            trading_env.reset(horizon_inputs, initial_position=init_pos)
            rewards, infos = trading_env.replay(base_actions)

            horizon_reward = sum(rewards)
            cost_paid = sum(info.fee + info.slippage for info in infos)
            boundary_cost = (infos[0].fee + infos[0].slippage) if infos else 0.0
            final_position = infos[-1].filled_position if infos else init_pos

            prev_terminal_position = final_position
            cumulative_loss += min(horizon_reward, 0.0)
            if horizon_reward < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            records.append(Phase2HorizonReplayRecord(
                sample_id=entry.sample_id,
                env_id=0,
                chosen_code=chosen_code,
                final_position=final_position,
                reward_raw=horizon_reward,
                reward_scaled=horizon_reward / max(h, 1),
                boundary_cost=boundary_cost,
                cost_paid=cost_paid,
                risk_triggered=risk_triggered,
                risk_trigger_step=risk_trigger_step,
                risk_reason=risk_reason,
                fold_id=fold_id,
                timestamp_start=getattr(entry, "timestamp_start", None),
                step_returns=list(rewards),
                selector_confidence=confidence,
                throttle_triggered=throttle_decision.triggered,
                original_code=original_code,
                throttled_code=chosen_code if throttle_decision.triggered else None,
            ))

        return records

    def _resolve_walk_forward_entries(
        self,
        split: str,
        entry_indices: Optional[Sequence[int]],
    ) -> List[Tuple[int, Any]]:
        """解析 walk-forward entry 子集，返回 dataset index + entry。"""
        all_pairs = [
            (idx, e)
            for idx, e in enumerate(self.dataset.horizon_entries)
            if e.split == split
        ]
        if entry_indices is None:
            return all_pairs
        wanted = {int(i) for i in entry_indices}
        by_idx = {idx: (idx, e) for idx, e in all_pairs}
        matched = [by_idx[i] for i in sorted(wanted) if i in by_idx]
        if matched:
            return matched
        # Fallback: allow split-relative fold indices for callers that do not
        # know the dataset-global entry positions.
        return [
            pair
            for rel_idx, pair in enumerate(all_pairs)
            if rel_idx in wanted
        ]

    def _guard_no_test_label_in_decision_path(self, split: str) -> None:
        """在 selector 决策前后检查 test label 是否被加载。

        Raises
        ------
        Phase2TestLabelLeakageError : 检测到 test label 进入决策路径。
        """
        if split != "test":
            return
        self._test_label_guard_active = True
        # 检查 dataset 中 test entries 是否有 code_label
        for entry in self.dataset.horizon_entries:
            if entry.split == "test" and entry.code_label is not None:
                raise Phase2TestLabelLeakageError(
                    f"test horizon {entry.sample_id} 的 code_label 已被加载到 dataset 中"
                )

    def run_baselines(
        self,
        split: str,
    ) -> Dict[str, List[Phase2HorizonReplayRecord]]:
        """运行所有 baseline（random / single_archetype_k / buy_and_hold）。

        phase1_demo_label 仅 posthoc baseline，不能进入主 checkpoint 选择。

        Returns
        -------
        dict : baseline_name → replay records。
        """
        results: Dict[str, List[Phase2HorizonReplayRecord]] = {}

        entries = [e for e in self.dataset.horizon_entries if e.split == split]
        if not entries:
            return results

        num_codes = self.frozen_policy.num_codes

        # Random baseline
        results["random_selector"] = self._run_fixed_strategy(
            entries, lambda _idx, _e: random.randint(0, num_codes - 1), split
        )

        # Single archetype baselines
        for k in range(num_codes):
            results[f"single_archetype_{k}"] = self._run_fixed_strategy(
                entries, lambda _idx, _e, _k=k: _k, split
            )

        # Buy and hold (always flat = action 1 in TradingEnv)
        results["buy_and_hold"] = self._run_fixed_strategy(
            entries, lambda _idx, _e: 1, split  # flat
        )

        return results

    def _run_fixed_strategy(
        self,
        entries,
        strategy_fn,
        split: str,
    ) -> List[Phase2HorizonReplayRecord]:
        """运行固定策略 baseline。"""
        # 预建 entry → dataset index 映射
        _entry_to_idx = {id(e): i for i, e in enumerate(self.dataset.horizon_entries)}

        records: List[Phase2HorizonReplayRecord] = []
        prev_pos = 0
        trading_env = self.trading_env_factory()

        for idx_in_list, entry in enumerate(entries):
            actual_idx = _entry_to_idx[id(entry)]
            chosen_code = strategy_fn(idx_in_list, entry)
            horizon_states = self.dataset.get_horizon_states(actual_idx)
            horizon_inputs = self.dataset.get_horizon_inputs(actual_idx)
            h = self.config.horizon

            self.frozen_policy.reset(code_id=min(chosen_code, self.frozen_policy.num_codes - 1))
            base_actions: List[int] = []
            for t in range(min(h, len(horizon_states))):
                out = self.frozen_policy.decode_step(horizon_states[t])
                base_actions.append(out.action)

            init_pos = prev_pos if self.config.horizon_schedule.position_continuity else 0
            trading_env.reset(horizon_inputs, initial_position=init_pos)
            rewards, infos = trading_env.replay(base_actions)

            horizon_reward = sum(rewards)
            cost_paid = sum(info.fee + info.slippage for info in infos)
            final_position = infos[-1].filled_position if infos else init_pos
            prev_pos = final_position

            records.append(Phase2HorizonReplayRecord(
                sample_id=entry.sample_id,
                env_id=0,
                chosen_code=chosen_code,
                final_position=final_position,
                reward_raw=horizon_reward,
                reward_scaled=horizon_reward / max(h, 1),
                boundary_cost=(infos[0].fee + infos[0].slippage) if infos else 0.0,
                cost_paid=cost_paid,
                step_returns=list(rewards),
            ))

        return records
