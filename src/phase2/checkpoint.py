"""检查点管理模块

功能说明:
    负责保存和加载训练检查点，包括模型参数、
    优化器状态、训练历史等。
"""

from __future__ import annotations

from typing import Any

import torch

from src.phase2.selection_agent import SelectionAgent


def save_checkpoint(
    save_path: str,
    agent: SelectionAgent,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    reward_history: list[float],
    best_val_return: float,
    step_count: int,
    config: Any,
    ppo_hparams: dict[str, Any],
) -> None:
    """统一保存 checkpoint。

    功能说明:
        保存当前 SelectionAgent、优化器状态、训练奖励历史、最佳验证表现，
        以及 Phase II 所需的关键超参数，便于恢复训练和对照实验。

    论文相关:
        保存的核心对象仍然围绕论文 Section 4.2：
        高层 selector 参数 + 训练时的 archetype selection 配置。
        这里额外保存 PPO 风格超参数，是为了复现实验时可追溯优化器设定。
    """
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "training_rewards": reward_history,
            "best_validation_return": best_val_return,
            "step": step_count,
            "config": {
                "state_dim": config.state_dim,
                "num_archetypes": config.num_archetypes,
                "selection_alpha": config.selection_alpha,
                "phase2_alpha_schedule": ppo_hparams["alpha_schedule"],
                "phase2_alpha_final_ratio": ppo_hparams["alpha_final_ratio"],
                "phase2_imitation_min_raw_return": ppo_hparams["imitation_min_raw_return"],
                "phase2_total_steps": config.phase2_total_steps,
                "learning_rate": config.learning_rate,
                "discount_factor": config.discount_factor,
                "phase2_val_interval_multiplier": ppo_hparams["val_interval_multiplier"],
                "phase2_rollout_batch_size": ppo_hparams["rollout_batch_size"],
                "phase2_ppo_epochs": ppo_hparams["ppo_epochs"],
                "phase2_minibatch_size": ppo_hparams["minibatch_size"],
                "phase2_clip_eps": ppo_hparams["clip_eps"],
                "phase2_vf_coef": ppo_hparams["vf_coef"],
                "phase2_ent_coef": ppo_hparams["ent_coef"],
                "phase2_max_grad_norm": ppo_hparams["max_grad_norm"],
                "phase2_diagnostic_horizons": ppo_hparams["diagnostic_horizons"],
            },
        },
        save_path,
    )
