"""Phase II 配置管理模块

功能说明:
    负责读取和验证 Phase II 训练所需的所有超参数，
    包括 PPO 相关参数、alpha 调度策略等。
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def cfg(config: Any, name: str, default: Any) -> Any:
    """安全读取配置项；若不存在则回退到默认值。

    功能说明:
        为 PPO 新增超参数提供向后兼容能力；即使 src.config.parse_args
        尚未加入这些字段，本脚本也可以直接运行。

    论文相关:
        论文本身定义了 Phase II 的高层 MDP 和目标函数 Eq.(5)，
        但未强制规定 PPO 的工程超参数；因此这里把 rollout/minibatch/
        clip/entropy 等都做成可选配置，属于训练器层面的实现细节。
    """
    return getattr(config, name, default)


def get_phase2_hparams(config: Any) -> dict[str, Any]:
    """读取 PPO 相关超参数。

    功能说明:
        从 config 中读取 Phase II 的 PPO 风格训练参数，若外部配置未定义，
        则使用安全默认值。
    ppo 参数说明：
        rollout_batch_size	每轮收集的样本数（horizon 数量），用于构建经验池
        ppo_epochs	对同一批数据重复训练的轮数（通常 3-10）
        minibatch_size	每个 epoch 内切分成的小批量大小
        clip_eps	策略裁剪范围（如 0.2 表示新旧策略概率比限制在 [0.8, 1.2]），防止策略突变
        vf_coef	价值函数损失的权重系数（总 loss = policy_loss + vf_coef × value_loss）
        ent_coef	熵正则化系数，鼓励探索（越大越倾向于均匀分布）
        max_grad_norm	梯度裁剪阈值，防止梯度爆炸
        log_interval	每 N 步输出一次日志
        eval_max_horizons	验证集评估时最多评估的 horizon 数量（None 表示全部）
        diagnostic_horizons	训练子集诊断时抽样的 horizon 数量
    论文相关:
        论文的核心是 Section 4.2 的 horizon-level selector 与 Eq.(5) 的目标，
        这里的 clip_eps / ppo_epochs / minibatch_size / ent_coef / vf_coef
        是为了把原先的单步 Actor-Critic 升级为更稳定的 PPO 风格优化器。

    Returns:
        dict[str, Any]: 统一整理后的 PPO 超参数字典。
    """
    rollout_batch_size = int(cfg(config, "phase2_rollout_batch_size", 1024))
    ppo_epochs = int(cfg(config, "phase2_ppo_epochs", 8))
    minibatch_size = int(cfg(config, "phase2_minibatch_size", 256))
    clip_eps = float(cfg(config, "phase2_clip_eps", 0.2))
    vf_coef = float(cfg(config, "phase2_vf_coef", 0.001))
    ent_coef = float(cfg(config, "phase2_ent_coef", 0.1))
    max_grad_norm = float(cfg(config, "phase2_max_grad_norm", 1.0))
    log_interval = int(cfg(config, "phase2_log_interval", 1000000))
    eval_max_horizons = cfg(config, "phase2_eval_max_horizons", None)
    diagnostic_horizons = int(cfg(config, "phase2_diagnostic_horizons", 128))
    alpha_schedule = str(cfg(config, "phase2_alpha_schedule", "linear"))
    alpha_final_ratio = float(cfg(config, "phase2_alpha_final_ratio", 0.0))
    imitation_min_raw_return = float(cfg(config, "phase2_imitation_min_raw_return", 0.0))
    val_interval_multiplier = int(cfg(config, "phase2_val_interval_multiplier", 10))

    rollout_batch_size = max(1, rollout_batch_size)
    ppo_epochs = max(1, ppo_epochs)
    minibatch_size = max(1, minibatch_size)
    log_interval = max(1, log_interval)
    diagnostic_horizons = max(1, diagnostic_horizons)
    val_interval_multiplier = max(1, val_interval_multiplier)
    alpha_final_ratio = max(0.0, alpha_final_ratio)

    if alpha_schedule not in {"constant", "linear"}:
        logger.warning(
            "未知的 phase2_alpha_schedule=%s，回退为 constant。",
            alpha_schedule,
        )
        alpha_schedule = "constant"

    # PPO 关键保护：minibatch 必须小于 rollout_batch，否则第一轮 full-batch
    # 更新在 advantage 零均值归一化后很容易导致 policy loss 接近 0。
    if rollout_batch_size > 1 and minibatch_size >= rollout_batch_size:
        adjusted_minibatch = max(1, rollout_batch_size // 4)
        logger.warning(
            "检测到 minibatch_size(%d) >= rollout_batch_size(%d)，自动调整为 %d，避免 full-batch PPO 导致 actor 更新退化。",
            minibatch_size,
            rollout_batch_size,
            adjusted_minibatch,
        )
        minibatch_size = adjusted_minibatch

    return {
        "rollout_batch_size": rollout_batch_size,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "clip_eps": clip_eps,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "log_interval": log_interval,
        "eval_max_horizons": eval_max_horizons,
        "diagnostic_horizons": diagnostic_horizons,
        "alpha_schedule": alpha_schedule,
        "alpha_final_ratio": alpha_final_ratio,
        "imitation_min_raw_return": imitation_min_raw_return,
        "val_interval_multiplier": val_interval_multiplier,
    }


def get_current_selection_alpha(
    initial_alpha: float,
    schedule: str,
    final_ratio: float,
    step_count: int,
    total_steps: int,
) -> float:
    """按训练进度计算当前 selection_alpha。"""
    initial_alpha = max(0.0, float(initial_alpha))
    final_ratio = max(0.0, float(final_ratio))
    if schedule != "linear" or total_steps <= 0:
        return initial_alpha

    progress = min(max(float(step_count) / float(total_steps), 0.0), 1.0)
    end_alpha = initial_alpha * final_ratio
    return initial_alpha + (end_alpha - initial_alpha) * progress
