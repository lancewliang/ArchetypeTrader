"""Phase II Double DQN replay buffer 骨架。

文件功能说明:
    本文件定义 Phase II horizon-level transition 和 replay buffer 入口。Phase II
    环境的一步对应一个完整 horizon，因此 replay buffer 保存的是 selector
    observation、archetype action、horizon reward、next observation、done mask
    和 Phase I assigned label 监督信号。

设计边界:
    - 只定义 transition schema、buffer 初始化参数、写入和采样接口；
    - 不实现实际环形数组、随机采样、tensor 拼接或设备搬运；
    - 不计算 reward，不调用 environment 或 Q-network；
    - 不计算 Double DQN loss，也不处理 target network 同步；
    - 不修改 Phase I assigned labels，只把它们作为 imitation regularization target 保留。

使用场景:
    ``Phase2DoubleDqnTrainer`` 从 ``ArchetypeSelectionEnv`` 收集
    ``Phase2ReplayTransition``，写入本 buffer；训练更新时调用 ``sample()``
    得到 ``Phase2SelectionTransitionBatch``，再传给 Double DQN loss。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...model.tensor_data_types import (
    DemonstrationHorizonLabelTensorBatch,
    VisibleStatesTensorBatch,
)

from ...model.data_types import DemonstrationHorizonLabel,  VisibleStates


@dataclass(frozen=True)
class Phase2ReplayTransition:
    """Phase II 单条 horizon-level replay transition。

    功能说明:
        保存 Double DQN 更新所需的一条环境交互结果。这里的 action 是 selector
        选择的 archetype id，reward 是执行该 archetype 对应 decoder 动作后的
        horizon-level return。

    设计边界:
        本类只承载 transition 数据，不负责校验 shape、计算 reward 或转换 tensor。

    使用场景:
        trainer 调用 env 后构造该对象，并传入 ``Phase2ReplayBuffer.add()``。
    """

    # 当前 selector observation，结构为 previous/current 各三路 states。
    visible_states: VisibleStates

    # selector 选择的 archetype id。
    action: int

    # 当前 horizon-level reward。
    reward: float

    # 下一条可训练 horizon 样本的 selector observation。
    next_visible_states: VisibleStates

    # horizon/episode 是否结束。
    done: bool

    # Phase I assigned label 数据，用于 imitation regularization 和样本追踪。
    demonstration_horizon_label: DemonstrationHorizonLabel



@dataclass(frozen=True)
class Phase2SelectionTransitionTensorBatch:
    """Phase II Double DQN replay transition batch schema。

    适用场景:
        作为 ``Phase2ReplayBuffer.sample()`` 的输出，以及
        ``compute_double_dqn_loss()`` 的输入。

    字段解释:
        保存 Double DQN 更新所需的当前 observation、action、reward、下一
        observation 和 done mask，同时保留 assigned label 作为 imitation
        regularization target。
    """

    # selector observation，结构为 previous/current 各三路 states。
    visible_states: VisibleStatesTensorBatch

    # selector 选择的 archetype id，形状 [batch]。
    actions: torch.Tensor

    # horizon-level reward，形状 [batch]。
    rewards: torch.Tensor

    # selector observation，结构为 previous/current 各三路 states。
    next_visible_states: VisibleStatesTensorBatch
 
    # episode/horizon 结束标记，形状 [batch]。
    dones: torch.Tensor

    # Phase I assigned label 数据，结构为 (sample_ids, code_labels)。
    demonstration_horizon_label_batch: DemonstrationHorizonLabelTensorBatch

class Phase2ReplayBuffer:
    """Phase II fixed-capacity replay buffer 骨架。

    功能说明:
        管理 horizon-level transition 的固定容量缓存。完整实现应使用环形写入策略，
        支持按 seed 可复现随机采样，并把 numpy transition 组装为
        ``Phase2SelectionTransitionBatch``。

    设计边界:
        本类只负责 replay buffer 的接口边界，不理解 Q-network、decoder policy 或
        reward 的内部计算。

    使用场景:
        ``Phase2DoubleDqnTrainer`` 在采样阶段调用 ``add()``，在更新阶段调用
        ``sample()``。
    """

    def __init__(
        self,
        capacity: int,
        visible_state_shapes: tuple[tuple[int, ...], ...],
        seed: int,
    ) -> None:
        """初始化固定容量 replay buffer。

        功能说明:
            保存 buffer 容量、visible state shape 和随机种子。完整实现应在这里
            初始化环形存储数组、写指针、当前大小和随机数生成器。

        使用场景:
            ``Phase2MainFlow`` 或 trainer 根据 ``Phase2TrainConfig.replay_capacity``
            创建 replay buffer。

        参数:
            capacity: replay buffer 最大 transition 数。
            visible_state_shapes: ``VisibleStatesDataset`` 六个数组各自的 shape，
                不包含 batch 维度。
            seed: 随机采样 seed，保证训练和测试可复现。
        """

        self.capacity = capacity
        self.visible_state_shapes = visible_state_shapes
        self.seed = seed

    def add(self, transition: Phase2ReplayTransition) -> None:
        """写入一个 horizon-level transition。

        功能说明:
            完整实现应将 transition 写入环形 buffer；容量满后覆盖最旧 transition。

        使用场景:
            trainer 每次 env step 后调用本方法，把新 transition 放入 replay buffer。

        参数:
            transition: 待写入的 Phase II horizon-level transition。
        """

        raise NotImplementedError("Phase2 replay buffer add is not implemented yet.")

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> Phase2SelectionTransitionTensorBatch:
        """随机采样 Double DQN 训练 batch。

        功能说明:
            完整实现应从当前可用 transition 中随机采样 ``batch_size`` 条，组装为
            ``Phase2SelectionTransitionBatch``，并把 tensor 搬到指定 device。

        使用场景:
            ``Phase2DoubleDqnTrainer.update_q_network()`` 调用该方法获取 TD loss 和
            imitation loss 的输入 batch。

        参数:
            batch_size: 采样 transition 数量。
            device: 输出 tensor 所在设备。

        返回:
            ``Phase2SelectionTransitionBatch``。
        """

        raise NotImplementedError("Phase2 replay buffer sampling is not implemented yet.")

    def __len__(self) -> int:
        """返回当前 buffer 中可采样 transition 数量。

        功能说明:
            完整实现应返回当前已写入且可采样的 transition 数，而不是固定 capacity。

        使用场景:
            trainer 用它判断 replay buffer 是否达到 ``learning_start_epoch`` 或最小
            batch size 要求。
        """

        raise NotImplementedError("Phase2 replay buffer length is not implemented yet.")
