"""Phase II archetype selection 配置定义。

本文件只放 Phase II selector 训练需要的配置骨架，不放模型、数据构建、
reward 计算、校验、序列化或文件读写逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..model.data_types import TSize


@dataclass(frozen=True)
class Phase2DatasetConfig:
    """Phase II selector dataset 构建配置。"""

    # selector 可见的当前分片状态窗口长度。
    tsize: TSize = 1


@dataclass(frozen=True)
class Phase2ModelConfig:
    """Phase II selector Q-network 配置。"""

    # 原始 state stream 的特征维度。
    state_dim: int

    # relative state stream 的特征维度。
    relative_state_dim: int

    # trend state stream 的特征维度。
    trend_state_dim: int

    # Phase I codebook 中可选 archetype 数量，也是 Q value 输出维度。
    num_archetypes: int

    # Q-network 隐层宽度。
    hidden_dim: int = 128

    # Q-network 隐层层数。
    num_layers: int = 2

    # Q-network dropout。
    dropout: float = 0.1


@dataclass(frozen=True)
class Phase2RewardConfig:
    """Phase II reward 和 imitation regularization 配置。"""

    # Double DQN bootstrap 折扣因子。
    gamma: float = 0.99

    # 交易手续费率。
    fee_rate: float = 0.0002

    # assigned-label imitation regularization 的全局权重。
    imitation_alpha: float = 1.0

    # 可选 reward 裁剪阈值。
    reward_clip: float | None = None

    # 是否对 replay batch reward 做标准化。
    normalize_rewards: bool = True


@dataclass(frozen=True)
class Phase2TrainConfig:
    """Phase II Double DQN 训练配置。"""

    # Double DQN 总训练轮数。
    epochs: int = 100

    # 每次从 replay buffer 采样的 transition 数。
    batch_size: int = 256

    # online Q-network optimizer 学习率。
    learning_rate: float = 3e-4

    # replay buffer 最大 transition 数。
    replay_capacity: int = 200_000

    # 从第几轮开始更新 Q-network。
    learning_start_epoch: int = 1

    # 每轮执行多少次 Q-network update。
    updates_per_epoch: int = 1

    # 每多少轮将 online network 同步到 target network。
    target_update_interval_epochs: int = 5

    # epsilon-greedy 初始探索率。
    epsilon_start: float = 1.0

    # epsilon-greedy 最低探索率。
    epsilon_end: float = 0.05

    # epsilon 从 start 衰减到 end 的轮数。
    epsilon_decay_epochs: int = 20

    # TD loss 权重。
    td_loss_beta: float = 1.0

    # imitation loss 权重。
    imitation_loss_beta: float = 1.0

    # 梯度裁剪阈值。
    max_grad_norm: float = 0.5

    # 随机种子。
    seed: int = 42


@dataclass(frozen=True)
class Phase2MainConfig:
    """Phase II 主流程入口配置。"""

    # 交易标的或数据域名称。
    pair: str

    # 当前训练批次 ID。
    train_batch_id: str

    # Phase I best checkpoint 路径。
    phase1_checkpoint_path: Path

    # 全阶段产物根目录。
    artifacts_root: Path = Path("artifacts")

    # 训练和推理设备。
    device: str = "cuda"
