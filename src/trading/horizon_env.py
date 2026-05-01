"""Horizon-level 交易环境: selector 每步选一个 archetype，执行完整 horizon。

设计文档锚点: Phase II 执行计划 §Step 3。

职责:
- reset() 返回当前 horizon 的 s^sel。
- step(action) 执行一个完整 horizon（decode_step 循环 h 次 + TradingEnv replay）。
- 维护 cursor / prev_terminal_position / recurrent_state。
- 支持 gap 裁切后的仓位处理（carry / force_flatten / warmup_only）。
- 支持 mid-horizon emergency flatten。

关键约束:
- done=True 仅在时间分片末端 / split 末端 / 独立 horizon episode 终止时出现。
- truncated=True 仅在 rollout_length 到达时出现；此时 done=False。
- 正式 replay 只能使用 decode_step()，不允许批量 decode()。
- gap 裁切后禁止静默 flat reset。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config.phase2_config import Phase2Config
from src.data.phase2_dataset import Phase2Dataset
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.trading.env import HorizonInputs, TradingEnv


@dataclass
class HorizonStepInfo:
    """HorizonEnv.step() 返回的诊断信息。"""
    chosen_code: int
    horizon_reward: float
    horizon_steps: int
    boundary_cost: float
    prev_terminal_position: int
    final_position: int
    cost_paid: float
    risk_triggered: bool = False
    risk_trigger_step: Optional[int] = None
    gap_bars: int = 0
    gap_mode_applied: Optional[str] = None


class HorizonEnv:
    """Horizon-level 环境。

    生命周期:
    1. reset() → 返回 s^sel（当前 horizon 的 selector 状态）。
    2. step(action) → 执行完整 horizon，返回 (next_obs, reward, done, truncated, info)。
    3. 重复直到 done=True。

    action 空间: {0, 1, ..., K-1}，对应 K 个 archetype。

    边界:
    - 内部使用 Phase1FrozenPolicy.decode_step() 循环 h 次。
    - 内部使用 TradingEnv 执行 step-wise replay。
    - 不直接实现 reward/cost 逻辑（复用 src/trading/）。
    """

    def __init__(
        self,
        env_id: int,
        dataset: Phase2Dataset,
        frozen_policy: Phase1FrozenPolicy,
        trading_env: TradingEnv,
        config: Phase2Config,
        horizon_indices: List[int],
    ) -> None:
        self.env_id = env_id
        self.dataset = dataset
        self.frozen_policy = frozen_policy
        self.trading_env = trading_env
        self.config = config
        self.horizon_indices = horizon_indices
        self._cursor: int = 0
        self._prev_terminal_position: int = 0
        self._done: bool = False
        self._cumulative_loss: float = 0.0
        self._consecutive_losses: int = 0

    def reset(
        self,
        prev_terminal_position: int = 0,
        cursor: int = 0,
        reset_risk_state: bool = True,
    ) -> np.ndarray:
        """重置环境，返回第一个 horizon 的 s^sel。

        Returns
        -------
        obs : selector 状态向量。
        """
        self._cursor = max(min(int(cursor), len(self.horizon_indices)), 0)
        self._prev_terminal_position = int(prev_terminal_position)
        self._done = False
        if reset_risk_state:
            self._cumulative_loss = 0.0
            self._consecutive_losses = 0

        if not self.horizon_indices or self._cursor >= len(self.horizon_indices):
            self._done = True
            return np.zeros(1, dtype=np.float32)

        idx = self.horizon_indices[self._cursor]
        obs = self.dataset.get_selector_state(idx, self._prev_terminal_position)
        return obs

    def restore_state(
        self,
        cursor: int,
        prev_terminal_position: int,
        cumulative_loss: float = 0.0,
        consecutive_losses: int = 0,
    ) -> np.ndarray:
        """恢复公开环境状态并返回当前 obs。

        该接口供 checkpoint resume 使用，避免外部直接写私有属性。
        """
        self._cumulative_loss = float(cumulative_loss)
        self._consecutive_losses = int(consecutive_losses)
        return self.reset(
            prev_terminal_position=prev_terminal_position,
            cursor=cursor,
            reset_risk_state=False,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, HorizonStepInfo]:
        """执行一个完整 horizon。

        Parameters
        ----------
        action : archetype code_id（0 到 K-1）。

        Returns
        -------
        next_obs : 下一个 horizon 的 s^sel（done=True 时为 dummy）。
        reward : r^sel，horizon 累计净收益（经 reward scaling）。
        done : 是否到达时间分片末端。
        truncated : 是否因 rollout_length 截断（此时 done=False）。
        info : HorizonStepInfo 诊断信息。

        实现步骤:
        1. frozen_policy.reset(code_id=action)。
        2. 循环 h 次 decode_step() 生成 base_actions。
        3. trading_env.reset(initial_position=prev_terminal_position)。
        4. trading_env.replay(base_actions) 累加 r^sel。
        5. 更新 cursor / prev_terminal_position。
        6. 检查 done / truncated 条件。
        """
        if self._done:
            dummy_obs = np.zeros(1, dtype=np.float32)
            info = HorizonStepInfo(
                chosen_code=action, horizon_reward=0.0, horizon_steps=0,
                boundary_cost=0.0, prev_terminal_position=self._prev_terminal_position,
                final_position=self._prev_terminal_position, cost_paid=0.0,
            )
            return dummy_obs, 0.0, True, False, info

        idx = self.horizon_indices[self._cursor]
        horizon_states = self.dataset.get_horizon_states(idx)
        horizon_inputs = self.dataset.get_horizon_inputs(idx)
        h = self.config.horizon

        # 1. 使用 decode_step streaming 接口生成 base_actions
        self.frozen_policy.reset(code_id=action)
        base_actions: List[int] = []
        for t in range(min(h, len(horizon_states))):
            state_t = horizon_states[t]
            out = self.frozen_policy.decode_step(state_t)
            base_actions.append(out.action)

        # 2. 检查 mid-horizon risk flatten
        risk_triggered = False
        risk_trigger_step = None
        if self.config.live_risk_controls.mid_horizon_emergency_flatten:
            base_actions, risk_triggered = self._handle_mid_horizon_flatten(
                0, base_actions
            )
            if risk_triggered:
                risk_trigger_step = 0

        # 3. TradingEnv replay
        prev_pos = self._prev_terminal_position
        gap_mode_applied = None
        entry = self.dataset.horizon_entries[idx]
        gap_bars = int(getattr(entry, "gap_bars", 0) or 0)
        if gap_bars > 0:
            gap_mode_applied = self._handle_gap(gap_bars)
            if gap_mode_applied in {"force_flatten", "warmup_only"}:
                prev_pos = 0
        if not self.config.horizon_schedule.position_continuity:
            prev_pos = 0

        self.trading_env.reset(horizon_inputs, initial_position=prev_pos)
        rewards, infos = self.trading_env.replay(base_actions)

        horizon_reward = sum(rewards)
        cost_paid = sum(info.fee + info.slippage for info in infos)
        boundary_cost = 0.0
        if infos:
            # 第一步的换仓成本即为边界成本
            boundary_cost = infos[0].fee + infos[0].slippage

        # 获取最终仓位
        final_position = infos[-1].filled_position if infos else prev_pos

        # 4. 更新状态
        self._prev_terminal_position = final_position
        self._cursor += 1

        # 更新风控统计
        self._cumulative_loss += min(horizon_reward, 0)
        if horizon_reward < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # 5. 检查 done
        done = self._cursor >= len(self.horizon_indices)
        truncated = False
        self._done = done

        # 6. 获取 next_obs
        if done:
            next_obs = np.zeros_like(
                self.dataset.get_selector_state(
                    self.horizon_indices[0], 0
                )
            )
        else:
            next_idx = self.horizon_indices[self._cursor]
            next_obs = self.dataset.get_selector_state(
                next_idx, self._prev_terminal_position
            )

        step_info = HorizonStepInfo(
            chosen_code=action,
            horizon_reward=horizon_reward,
            horizon_steps=len(base_actions),
            boundary_cost=boundary_cost,
            prev_terminal_position=prev_pos,
            final_position=final_position,
            cost_paid=cost_paid,
            risk_triggered=risk_triggered,
            risk_trigger_step=risk_trigger_step,
            gap_bars=gap_bars,
            gap_mode_applied=gap_mode_applied,
        )

        return next_obs, horizon_reward, done, truncated, step_info

    def _handle_gap(self, gap_bars: int) -> str:
        """处理 gap horizon 的仓位。

        根据配置返回 carry / force_flatten / warmup_only。
        """
        threshold = self.config.horizon_schedule.gap_threshold_bars
        mode = self.config.horizon_schedule.gap_mode
        if gap_bars <= threshold:
            return "carry"
        return mode

    def _handle_mid_horizon_flatten(
        self,
        step_idx: int,
        base_actions: List[int],
    ) -> Tuple[List[int], bool]:
        """处理 mid-horizon emergency flatten。

        触发时立即结算 liquidation action 及其 cost，
        之后以 flat 状态推进到 horizon 末尾。

        Returns
        -------
        modified_actions : 修改后的 action 序列。
        risk_triggered : 是否触发了风控。
        """
        lrc = self.config.live_risk_controls
        triggered = False

        # 检查 daily loss limit
        if lrc.daily_loss_limit is not None:
            if abs(self._cumulative_loss) > lrc.daily_loss_limit:
                triggered = True

        # 检查 consecutive loss limit
        if lrc.consecutive_loss_limit is not None:
            if self._consecutive_losses >= lrc.consecutive_loss_limit:
                triggered = True

        if triggered and lrc.flatten_on_trigger:
            # 将所有 action 设为 flat (1)
            return [1] * len(base_actions), True

        return base_actions, False

    @property
    def prev_terminal_position(self) -> int:
        """上一个 horizon 的终端仓位。"""
        return self._prev_terminal_position

    @property
    def cursor(self) -> int:
        """当前 horizon index 在 horizon_indices 中的位置。"""
        return self._cursor

    @property
    def cumulative_loss(self) -> float:
        """累计亏损风控状态。"""
        return self._cumulative_loss

    @property
    def consecutive_losses(self) -> int:
        """连续亏损次数风控状态。"""
        return self._consecutive_losses
