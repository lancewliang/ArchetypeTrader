"""Phase II Double DQN 单轮训练器骨架。

文件功能说明:
    本文件定义 Phase II archetype selector 的 Double DQN 训练器接口。该训练器的
    职责粒度是一轮 epoch：接收外部创建好的 online/target Q-network、环境、
    replay buffer、optimizer 和训练配置，完成本轮 transition 采集、Q-network
    更新、target network 同步，并返回本轮训练完成后的 ``Phase2Checkpoint``
    payload。

设计边界:
    - 不创建 Q-network，不决定模型结构；
    - 不计算 horizon reward，reward 由 ``ArchetypeSelectionEnv.step()`` 产生；
    - 不实现 Double DQN target、TD loss 或 imitation loss，loss 逻辑属于
      ``phase2_double_dqn_loss.py``；
    - 不执行 validation，不选择 best checkpoint；
    - 不写文件、不调用 ``torch.save``，checkpoint 的落盘由 main flow /
      artifact store 负责。

使用场景:
    ``Phase2MainFlow`` 在外层 ``for epoch in range(...)`` 循环中调用
    ``train_one_epoch()``。每次调用返回一个模型 checkpoint payload；主流程随后
    可调用 evaluator 生成 validation checkpoint，并交给 checkpoint selector
    做 best model 选择。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from ...model.data_types import VisibleStatesDataset
    from ..checkpoint.phase2_checkpoint import Phase2Checkpoint
    from ..model.phase2_q_network import Phase2QNetwork
    from ..phase2_config import Phase2RewardConfig, Phase2TrainConfig
    from ..phase2_env import ArchetypeSelectionEnv
    from ..phase2_selection_data_schema import Phase2SelectionTransitionBatch
    from .phase2_replay_buffer import Phase2ReplayBuffer, Phase2ReplayTransition


class Phase2DoubleDqnTrainer:
    """Phase II Double DQN 单轮训练器。

    功能说明:
        封装 Phase II selector 的单 epoch 训练流程。它负责训练过程编排，而不是
        负责模型定义、reward 计算、loss 细节或 checkpoint 文件保存。

    输入依赖:
        online Q-network、target Q-network、环境、replay buffer、optimizer、
        训练配置、reward 配置和运行设备均由外部注入。

    输出结果:
        ``train_one_epoch()`` 返回本轮训练完成后的 ``Phase2Checkpoint`` payload，
        供 main flow 统一保存和后续验证。

    使用场景:
        在 ``Phase2MainFlow._train_double_dqn()`` 中创建一次 trainer，然后每轮调用
        ``train_one_epoch(epoch, sample_indices)``。main flow 负责控制总 epoch 数、
        validation、best checkpoint selection 和 report。
    """

    def __init__(
        self,
        online_q_network: Phase2QNetwork,
        target_q_network: Phase2QNetwork,
        env: ArchetypeSelectionEnv,
        replay_buffer: Phase2ReplayBuffer,
        train_config: Phase2TrainConfig,
        reward_config: Phase2RewardConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
    ) -> None:
        """初始化单轮训练器依赖。

        功能说明:
            保存训练一轮所需的外部依赖。完整实现中应把 online/target network
            移动到指定 device，保存 optimizer、env、replay buffer 和配置，并在
            必要时进行初始 target network 同步。

        输入参数:
            online_q_network: 当前训练的 selector Q-network，用于动作选择和梯度更新。
            target_q_network: Double DQN target network，用于稳定 TD target 估计。
            env: Phase II horizon-level 环境，负责根据 selected archetype 产生 reward。
            replay_buffer: 存储 horizon-level transition 的经验回放缓冲区。
            train_config: 训练超参数，例如 batch size、update 次数、epsilon 衰减等。
            reward_config: reward/loss 相关配置，例如 gamma、fee rate、imitation 权重等。
            optimizer: 外部创建的 optimizer，便于恢复训练和替换优化器。
            device: 训练设备，例如 ``"cuda"`` 或 ``"cpu"``。

        输出:
            无返回值。该方法只保存依赖，不启动训练。

        使用场景:
            由 ``Phase2MainFlow`` 在训练开始前构造一次；后续每个 epoch 复用同一实例。
        """

    def train_one_epoch(
        self,
        epoch: int,
        sample_indices: Sequence[int] | None = None,
    ) -> Phase2Checkpoint:
        """训练一个 epoch 并返回模型 checkpoint payload。

        功能说明:
            执行一轮 Double DQN 训练：根据 ``sample_indices`` 遍历或采样 horizon
            样本，调用 ``collect_transition()`` 收集经验，写入 replay buffer，在满足
            update 条件时调用 ``update_q_network()``，并按配置同步 target network。
            一轮结束后构造并返回 ``Phase2Checkpoint``。

        输入参数:
            epoch: 当前训练轮次，从主流程传入，用于 epsilon 调度、update gate、
                target network 同步和 checkpoint 元数据。
            sample_indices: 当前 epoch 要训练的样本索引。为 ``None`` 时，完整实现可由
                trainer 或 env 使用默认顺序/随机策略选择样本。

        输出:
            ``Phase2Checkpoint``。该对象只表示本轮训练后的模型/optimizer 状态 payload，
            不包含 validation metrics，也不表达 best checkpoint 语义。

        使用场景:
            ``Phase2MainFlow`` 的外层训练循环每轮调用一次。返回值交给 artifact store
            保存；随后 main flow 可调用 evaluator 生成对应 validation checkpoint。
        """

    def select_action(
        self,
        visible_states: VisibleStatesDataset,
        epsilon: float,
        deterministic: bool = False,
    ) -> int:
        """根据当前可见状态选择 archetype action。

        功能说明:
            训练阶段执行 epsilon-greedy：以 ``epsilon`` 概率随机探索，否则选择
            online Q-network 预测 Q value 最大的 archetype。评估或测试阶段可设置
            ``deterministic=True``，直接使用 greedy action。

        输入参数:
            visible_states: selector 在线可见状态，结构为
                ``(previous_t_states, current_t_states)``，不能包含未来 horizon 信息。
            epsilon: 探索率。训练时由 epoch 调度产生；deterministic 模式下应被忽略。
            deterministic: 是否强制使用 greedy action。

        输出:
            ``int``，selector 选择的 archetype id。

        使用场景:
            ``collect_transition()`` 在 env reset 后调用本方法生成 action；evaluator
            也可复用同一策略接口并传入 ``deterministic=True``。
        """

    def collect_transition(
        self,
        sample_index: int | None,
        epsilon: float,
    ) -> Phase2ReplayTransition:
        """采集一条 horizon-level replay transition。

        功能说明:
            重置环境到指定样本，读取 selector 可见状态，调用 ``select_action()``
            选择 archetype，然后执行 ``env.step(action)`` 获得 reward、下一观察和
            done 标记，最后组装为 ``Phase2ReplayTransition``。

        输入参数:
            sample_index: 要训练的 horizon 样本索引。为 ``None`` 时由环境或采样策略决定。
            epsilon: 当前 epoch 的探索率，传给 ``select_action()``。

        输出:
            ``Phase2ReplayTransition``，包含当前 visible states、action、reward、
            next visible states、done 和 assigned-label imitation target。

        使用场景:
            ``train_one_epoch()`` 针对每个训练样本调用本方法，并将返回 transition
            写入 ``Phase2ReplayBuffer``。
        """

    def should_update(self, epoch: int) -> bool:
        """判断当前是否应该执行 Q-network 更新。

        功能说明:
            检查训练轮次和 replay buffer 状态。完整实现通常需要同时满足：
            当前 ``epoch`` 已达到 ``learning_start_epoch``，并且 replay buffer
            可采样样本数不少于 ``batch_size``。

        输入参数:
            epoch: 当前训练轮次。

        输出:
            ``bool``。为 True 时，``train_one_epoch()`` 可以从 replay buffer 采样并
            调用 ``update_q_network()``。

        使用场景:
            每次收集 transition 后调用，避免 replay 样本不足时提前更新网络。
        """

    def update_q_network(
        self,
        batch: Phase2SelectionTransitionBatch,
    ):
        """使用一个 replay batch 更新 online Q-network。

        功能说明:
            调用独立 loss 模块计算 Double DQN loss，并执行反向传播、梯度裁剪和
            ``optimizer.step()``。trainer 只负责编排参数更新，不在本方法中实现
            TD target、TD loss 或 imitation loss 的具体公式。

        输入参数:
            batch: 从 ``Phase2ReplayBuffer.sample()`` 得到的
                ``Phase2SelectionTransitionBatch``，包含当前观察、action、reward、
                next observation、done 和 assigned label。

        输出:
            第一版可返回训练日志字典或后续定义的 step stats payload，例如
            ``td_loss``、``imitation_loss``、``total_loss``、``mean_q_selected``、
            ``mean_td_target`` 和 ``grad_norm``。

        使用场景:
            ``train_one_epoch()`` 在 ``should_update(epoch)`` 为 True 时调用本方法，
            每轮可按 ``updates_per_epoch`` 执行多次。
        """

    def sync_target_network(self) -> None:
        """同步 target Q-network 参数。

        功能说明:
            将 online Q-network 的参数复制到 target Q-network。第一版设计采用硬同步，
            触发条件由 ``train_config.target_update_interval_epochs`` 控制。后续如需
            MacroHFT 风格软更新，可新增独立方法，不应放入 loss/reward 模块。

        输入参数:
            无。

        输出:
            无返回值。该方法只修改 target network 参数。

        使用场景:
            ``train_one_epoch()`` 在 epoch 结束或达到同步间隔时调用，用于稳定 Double
            DQN bootstrap target。
        """
 
