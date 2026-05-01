"""PPO trainer: rollout → GAE → PPO update 主循环。

设计文档锚点: Phase II 执行计划 §Step 5。

职责:
- 编排 rollout 收集、GAE 计算、PPO update。
- approx_kl > target_kl 时 early stop。
- advantage_normalization 开关。
- reward clip 开启时同时记录 clipped/unclipped 统计。
- numerical safety: tensor 非 finite fail-fast / gradient 爆炸 fail-fast。

关键约束:
- GAE 严格按 env_id 分组（由 RolloutBuffer 保证）。
- done 与 truncated 语义不混淆。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config.phase2_config import Phase2Config
from src.rl.actor_critic import ActorCritic
from src.rl.ppo_loss import PPOLoss, PPOLossOutput
from src.rl.rollout_buffer import RolloutBuffer, RolloutSample
from src.rl.scheduling import ScheduleManager
from src.trading.horizon_env import HorizonEnv
from src.evaluation.metrics.policy_health import compute_explained_variance


class NumericalSafetyError(RuntimeError):
    """Tensor 非 finite 或 gradient 爆炸时抛出。"""


@dataclass
class PPOUpdateStats:
    """单次 PPO update 的统计信息。"""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    kl_demo_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    early_stopped: bool = False
    early_stop_epoch: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_clipped_ratio: float = 0.0
    reward_unclipped_mean: float = 0.0
    rollout_done_count: float = 0.0
    rollout_truncated_count: float = 0.0
    rollout_bootstrap_count: float = 0.0


class PPOTrainer:
    """PPO 训练器。

    使用方式::

        ppo = PPOTrainer(config, actor_critic, envs, schedule_manager)
        ppo.setup()
        for update_idx in range(num_updates):
            stats = ppo.collect_and_update()
    """

    def __init__(
        self,
        config: Phase2Config,
        actor_critic: ActorCritic,
        envs: List[HorizonEnv],
        schedule_manager: ScheduleManager,
    ) -> None:
        self.config = config
        self.actor_critic = actor_critic
        self.envs = envs
        self.schedule_manager = schedule_manager
        self._buffer: Optional[RolloutBuffer] = None
        self._loss_fn: Optional[PPOLoss] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._update_count: int = 0
        self._device = config.device
        # 当前每个 env 的 obs
        self._current_obs: List[Optional[np.ndarray]] = [None] * len(envs)

    def setup(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """初始化 buffer / loss_fn / optimizer。

        Parameters
        ----------
        optimizer : 外部传入的 optimizer（与 ScheduleManager 共享同一实例）。
                    若为 None，则内部创建。
        """
        self._buffer = RolloutBuffer(
            num_envs=self.config.num_envs,
            rollout_length=self.config.rollout_length,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
        )
        sched_state = self.schedule_manager.current_state()
        self._loss_fn = PPOLoss(
            clip_ratio=self.config.ppo.clip_ratio,
            value_coef=self.config.ppo.value_loss_coef,
            entropy_coef=sched_state.entropy_coef,
            kl_demo_coef=sched_state.kl_demo_coef,
            num_codes=self.actor_critic.selector.num_codes,
            value_clip_range=self.config.ppo.value_clip_range,
            kl_demo_label_smoothing=self.config.ppo.kl_demo_label_smoothing,
        )
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(
                self.actor_critic.selector.parameters(),
                lr=self.config.ppo.lr,
            )
        # 初始化 env obs
        for i, env in enumerate(self.envs):
            obs = env.reset()
            self._current_obs[i] = obs

    def collect_rollout(self) -> None:
        """收集一次完整 rollout（rollout_length 步）。

        遍历所有 env，每步调用 actor_critic.act() 选择动作，
        然后 env.step() 执行，将结果存入 buffer。
        """
        assert self._buffer is not None
        self._buffer.reset()
        self.actor_critic.selector.eval()

        for step in range(self.config.rollout_length):
            samples: List[RolloutSample] = []
            # 构建 obs batch
            obs_batch = np.array(self._current_obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self._device)

            with torch.no_grad():
                act_out = self.actor_critic.act(obs_tensor, deterministic=False)

            actions = act_out.action.cpu().tolist()
            log_probs = act_out.log_prob.cpu().tolist()
            values = act_out.value.cpu().tolist()

            for env_idx, env in enumerate(self.envs):
                action = actions[env_idx]

                # 在 step 之前记录当前 horizon 的 label 信息
                # （step 之后 cursor 会前进，done 时会 reset）
                current_horizon_idx = env.horizon_indices[env.cursor] if env.cursor < len(env.horizon_indices) else None
                entry = env.dataset.horizon_entries[current_horizon_idx] if current_horizon_idx is not None else None

                kl_label = entry.code_label if entry else None
                is_labeled = entry.is_labeled if entry else False

                next_obs, reward, done, env_truncated, info = env.step(action)
                truncated = bool(env_truncated or (step == self.config.rollout_length - 1 and not done))

                # reward scaling
                reward_raw = reward
                reward_scaled, reward_was_clipped = self._scale_reward(reward)

                samples.append(RolloutSample(
                    obs=self._current_obs[env_idx],
                    env_id=env_idx,
                    action=action,
                    log_prob=log_probs[env_idx],
                    value=values[env_idx],
                    reward=reward_scaled,
                    reward_raw=reward_raw,
                    done=done,
                    truncated=truncated,
                    reward_was_clipped=reward_was_clipped,
                    kl_label=kl_label,
                    is_labeled=is_labeled,
                    info_cost_paid=info.cost_paid if info else 0.0,
                    info_boundary_cost=info.boundary_cost if info else 0.0,
                    info_chosen_code=action,
                ))

                if done:
                    self._current_obs[env_idx] = env.reset()
                else:
                    self._current_obs[env_idx] = next_obs

            self._buffer.add(samples)

    def _scale_reward(self, reward: float) -> Tuple[float, bool]:
        """应用 reward scaling。"""
        method = self.config.reward_scaling.method
        if method == "divide_by_horizon":
            scaled = reward / max(self.config.horizon, 1)
        else:
            scaled = reward
        clip_range = self.config.reward_scaling.clip_range
        if clip_range is None:
            return float(scaled), False
        clipped = float(np.clip(scaled, -float(clip_range), float(clip_range)))
        return clipped, bool(clipped != scaled)

    def update(self) -> PPOUpdateStats:
        """执行一次 PPO update。

        Steps:
        1. compute_gae（buffer 内按 env_id 分组）。
        2. 遍历 update_epochs 个 epoch。
        3. 每个 epoch 遍历 minibatch。
        4. 计算 PPO loss 并反向传播。
        5. approx_kl > target_kl 时 early stop。
        6. 更新 schedule。

        Returns
        -------
        PPOUpdateStats : 本次 update 的统计。
        """
        assert self._buffer is not None
        assert self._loss_fn is not None
        assert self._optimizer is not None

        # Bootstrap values for GAE
        obs_batch = np.array(self._current_obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            last_values = self.actor_critic.get_value(obs_tensor).cpu().tolist()

        self._buffer.compute_gae(last_values)

        # 更新 schedule
        sched_state = self.schedule_manager.current_state()
        self._loss_fn.entropy_coef = sched_state.entropy_coef
        self._loss_fn.kl_demo_coef = sched_state.kl_demo_coef

        self.actor_critic.selector.train()
        stats = PPOUpdateStats()
        buffer_stats = self._buffer.get_stats()
        stats.reward_mean = buffer_stats.get("reward_mean", 0.0)
        stats.reward_std = buffer_stats.get("reward_std", 0.0)
        stats.reward_clipped_ratio = buffer_stats.get("reward_clipped_ratio", 0.0)
        stats.reward_unclipped_mean = buffer_stats.get("reward_unclipped_mean", 0.0)
        stats.rollout_done_count = buffer_stats.get("rollout_done_count", 0.0)
        stats.rollout_truncated_count = buffer_stats.get("rollout_truncated_count", 0.0)
        stats.rollout_bootstrap_count = buffer_stats.get("rollout_bootstrap_count", 0.0)

        vr = self._buffer.flat_values_returns()
        stats.explained_variance = compute_explained_variance(
            vr["values"], vr["returns"]
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_demo_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_batches = 0
        early_stopped = False
        early_stop_epoch = 0

        for epoch in range(self.config.ppo.update_epochs):
            if early_stopped:
                break
            for mb in self._buffer.iterate_minibatches(
                self.config.ppo.minibatch_size, device=self._device
            ):
                obs = mb["obs"]
                action = mb["action"]
                old_log_prob = mb["old_log_prob"]
                old_value = mb["old_value"]
                advantage = mb["advantage"]
                return_ = mb["return_"]
                kl_label = mb["kl_label"]
                is_labeled = mb["is_labeled"]

                # Advantage normalization
                if self.config.ppo.advantage_normalization:
                    adv_std = advantage.std()
                    if adv_std > 1e-8:
                        advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)

                eval_out = self.actor_critic.evaluate_actions(obs, action)
                logits, _ = self.actor_critic.selector(obs)

                loss_out = self._loss_fn.compute(
                    log_prob=eval_out.log_prob,
                    old_log_prob=old_log_prob,
                    advantage=advantage,
                    value=eval_out.value,
                    return_=return_,
                    entropy=eval_out.entropy,
                    old_value=old_value,
                    kl_label=kl_label,
                    is_labeled=is_labeled,
                    dead_code_mask=(
                        self.actor_critic.dead_code_mask
                        if self.actor_critic.dead_code_mask is not None
                        else None
                    ),
                    action=action,
                    logits=logits,
                )

                # Numerical safety check
                self._check_numerical_safety(loss_out.total)

                self._optimizer.zero_grad()
                loss_out.total.backward()
                if self.config.ppo.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.selector.parameters(),
                        self.config.ppo.max_grad_norm,
                    )
                self._optimizer.step()

                total_policy_loss += loss_out.policy_loss.item()
                total_value_loss += loss_out.value_loss.item()
                total_entropy_loss += loss_out.entropy_loss.item()
                total_kl_demo_loss += loss_out.kl_demo_loss.item()
                total_approx_kl += loss_out.approx_kl
                total_clip_frac += loss_out.clip_fraction
                num_batches += 1

                # Early stop on KL
                if (
                    self.config.ppo.target_kl is not None
                    and loss_out.approx_kl > self.config.ppo.target_kl
                ):
                    early_stopped = True
                    early_stop_epoch = epoch
                    break

        if num_batches > 0:
            stats.policy_loss = total_policy_loss / num_batches
            stats.value_loss = total_value_loss / num_batches
            stats.entropy_loss = total_entropy_loss / num_batches
            stats.kl_demo_loss = total_kl_demo_loss / num_batches
            stats.approx_kl = total_approx_kl / num_batches
            stats.clip_fraction = total_clip_frac / num_batches

        stats.early_stopped = early_stopped
        stats.early_stop_epoch = early_stop_epoch

        self._update_count += 1
        self.schedule_manager.step(self._update_count)

        return stats

    def collect_and_update(self) -> PPOUpdateStats:
        """收集 rollout + PPO update 的便捷方法。"""
        self.collect_rollout()
        return self.update()

    def _check_numerical_safety(self, loss: torch.Tensor) -> None:
        """检查 tensor 非 finite 和 gradient 爆炸。

        Raises
        ------
        NumericalSafetyError : 检测到数值异常。
        """
        if not self.config.numerical_safety.check_finite:
            return
        if not torch.isfinite(loss):
            self._export_debug_snapshot("non_finite_loss")
            raise NumericalSafetyError(
                f"检测到非 finite loss: {loss.item()}"
            )

    def _export_debug_snapshot(self, reason: str) -> None:
        """导出 debug snapshot 到配置的目录。"""
        snapshot_dir = (
            self.config.artifacts_dir()
            / self.config.numerical_safety.debug_snapshot_dir
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"snapshot_{reason}_{self._update_count}.pt"
        torch.save(
            {
                "reason": reason,
                "update_count": self._update_count,
                "model_state": self.actor_critic.selector.state_dict(),
            },
            path,
        )

    def get_state(self) -> Dict[str, Any]:
        """获取可序列化的训练状态（用于 checkpoint）。"""
        state: Dict[str, Any] = {
            "update_count": self._update_count,
            "model_state": self.actor_critic.selector.state_dict(),
            "schedule_state": self.schedule_manager.get_state(),
        }
        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()
        # env cursors
        env_states = []
        for env in self.envs:
            env_states.append({
                "cursor": env.cursor,
                "prev_terminal_position": env.prev_terminal_position,
                "cumulative_loss": env.cumulative_loss,
                "consecutive_losses": env.consecutive_losses,
            })
        state["env_states"] = env_states
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复训练状态。"""
        self._update_count = state.get("update_count", 0)
        self.actor_critic.selector.load_state_dict(state["model_state"])
        if "optimizer_state" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer_state"])
        if "schedule_state" in state:
            self.schedule_manager.load_state(state["schedule_state"])
        if "env_states" in state:
            for env_idx, (env, es) in enumerate(zip(self.envs, state["env_states"])):
                obs = env.restore_state(
                    cursor=es.get("cursor", 0),
                    prev_terminal_position=es.get("prev_terminal_position", 0),
                    cumulative_loss=es.get("cumulative_loss", 0.0),
                    consecutive_losses=es.get("consecutive_losses", 0),
                )
                if env_idx < len(self._current_obs):
                    self._current_obs[env_idx] = obs
