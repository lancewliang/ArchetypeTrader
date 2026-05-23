"""Phase II Double DQN 单轮训练器骨架。

本模块负责 Phase II archetype selector 的 epoch 级训练编排。模型结构、
reward 计算、replay 采样和 Double DQN loss 细节分别保留在各自模块中。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from ...model.data_types import VisibleStates
from ..checkpoint.phase2_checkpoint import Phase2Checkpoint
from ..model.phase2_q_network import Phase2QNetwork
from ..phase2_config import Phase2RewardConfig, Phase2TrainConfig
from ..phase2_env import ArchetypeSelectionEnv, Phase2SelectionStepResult
from .phase2_double_dqn_loss import (
    Phase2DoubleDqnLossOutput,
    compute_double_dqn_loss,
)
from .phase2_replay_buffer import (
    Phase2ReplayBuffer,
    Phase2ReplayTransition,
    Phase2SelectionTransitionTensorBatch,
)

def build_epsilon_by_epoch(
    epoch: int,
    train_config: Phase2TrainConfig,
) -> float:
    """根据 epoch 计算 epsilon-greedy 探索率。

    功能说明:
        按 ``Phase2TrainConfig.epsilon_start``、``epsilon_end`` 和
        ``epsilon_decay_epochs`` 做线性退火，返回当前训练轮次使用的 epsilon。

    输入参数:
        epoch: 当前训练轮次，通常由 ``Phase2MainFlow`` 或 trainer 外层循环传入。
        train_config: Phase II 训练配置，包含 epsilon 起点、终点和衰减长度。

    输出:
        ``float``，当前 epoch 的探索率。

    使用场景:
        ``Phase2DoubleDqnTrainer.train_one_epoch()`` 在收集 transition 前调用，
        并传给 ``select_action()``。

    论文算法:
        这是训练稳定性策略，不直接改变目标函数；用于 epsilon-greedy 行为策略采样。
    """

    raise NotImplementedError("Phase2 epsilon schedule is not implemented yet.")


class Phase2DoubleDqnTrainer:
    """Phase II Double DQN 单轮训练器。

    本训练器串起 Double DQN 的训练主干流程：

    1. 使用 epsilon-greedy 行为策略从 ``ArchetypeSelectionEnv`` 采集
       horizon-level transition；
    2. 将 transition 写入 ``Phase2ReplayBuffer``；
    3. replay buffer 预热完成后采样 batch，并通过 ``compute_double_dqn_loss``
       更新 online Q-network；
    4. 按配置周期性硬同步 target Q-network；
    5. 返回 checkpoint payload，交给外层 main flow 负责持久化。
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
        """初始化 Phase II Double DQN trainer 的外部依赖。

        功能说明:
            保存 online/target Q-network、环境、replay buffer、optimizer、配置和
            运行设备，并将两套 Q-network 移动到目标设备。初始化结束时执行一次
            target network 硬同步，保证训练开始前 target 参数与 online 参数一致。

        输入参数:
            online_q_network: 当前被训练的 Q-network，用于动作选择和梯度更新。
            target_q_network: Double DQN target network，用于 bootstrap target 估计。
            env: Phase II horizon-level 交互环境，负责执行 archetype action 并返回
                reward、next observation 和 done。
            replay_buffer: 存储 horizon-level transition 的经验回放缓存。
            train_config: Phase II 训练超参数，例如 batch size、epsilon、target
                同步间隔和梯度裁剪阈值。
            reward_config: reward/loss 相关配置，例如 gamma、reward 标准化和
                imitation 权重。
            optimizer: 外部创建的 optimizer，用于更新 online Q-network。
            device: 训练设备，例如 ``"cuda"`` 或 ``"cpu"``。

        输出:
            无返回值。初始化后 trainer 持有训练一轮所需的全部依赖。
        """

        self.device = torch.device(device)
        self.online_q_network = online_q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.env = env
        self.replay_buffer = replay_buffer
        self.train_config = train_config
        self.reward_config = reward_config
        self.optimizer = optimizer
        self.rng = np.random.default_rng(train_config.seed)

        self.sync_target_network()

    def train_one_epoch(
        self,
        epoch: int
    ) -> Phase2Checkpoint:
        """执行一个 Double DQN 训练 epoch 并返回 checkpoint payload。

        功能说明:
            本方法是 Phase II 单轮训练的主干编排入口。它先根据 epoch 计算
            epsilon，再逐样本采集 transition 并写入 replay buffer；当 buffer 达到
            warmup 条件后，从 replay 中采样 batch，调用 ``update_q_network()``
            执行 Double DQN 参数更新；epoch 结束时按配置同步 target network，并返回
            当前 online Q-network 的 checkpoint payload。

        输入参数:
            epoch: 当前训练轮次，用于 epsilon 调度、learning start 判断、target
                network 同步判断和 checkpoint 元数据。
            sample_indices: 当前 epoch 要采集的样本索引。为 ``None`` 时，默认遍历
                env dataset 中全部可见状态样本。

        输出:
            ``Phase2Checkpoint``，包含 epoch、训练配置、online Q-network state dict
            和 optimizer state dict。不包含 validation 指标，也不负责写盘。

        当 ``sample_indices=None`` 时，表示遍历 env dataset 中已知的全部训练样本。
        这样可以让外层 ``Phase2MainFlow`` 保持简单，并把采样策略变化收敛在
        trainer 内部。
        """

        self.online_q_network.train()
        self.target_q_network.eval()

        epsilon = build_epsilon_by_epoch(epoch, self.train_config)
        step_result = self.env.reset()

        while step_result.done is not True:
            visible_states = step_result.observation
            action = self.select_action(
                visible_states=visible_states,
                epsilon=epsilon,
                deterministic=False,
            )
            step_result = self.env.step(action)

            transition = Phase2ReplayTransition(
                visible_states=visible_states,
                action=action,
                reward=float(step_result.reward),
                next_visible_states=step_result.observation,
                done=bool(step_result.done),
                demonstration_horizon_label=tuple[step_result.info.sample_id,
                                                step_result.info.selected_code_id]
            )
            self.replay_buffer.add(transition)

            if not self.should_update(epoch):
                continue

            for _ in range(self.train_config.updates_per_epoch):
                batch = self.replay_buffer.sample(
                    batch_size=self.train_config.batch_size,
                    device=self.device,
                )
                loss_output = self.update_q_network(batch)

            if self._should_sync_target(epoch):
                self.sync_target_network()

        # 消费的 checkpoint payload。

        return self.build_checkpoint(epoch)

    def select_action(
        self,
        visible_states: VisibleStates,
        epsilon: float,
        deterministic: bool = False,
    ) -> int:
        """使用 greedy 或 epsilon-greedy 策略选择 archetype id。

        功能说明:
            训练阶段在 ``deterministic=False`` 时执行 epsilon-greedy 行为策略：
            以 ``epsilon`` 概率随机选择 archetype，否则使用 online Q-network 对
            当前 visible states 计算 Q values，并选择 Q value 最大的 action。
            当 ``deterministic=True`` 时忽略 epsilon，直接走 greedy action。

        输入参数:
            visible_states: 单个 selector observation，包含 previous/current 各三路
                visible states，不包含未来 horizon 信息。
            epsilon: 当前 epoch 的探索率。
            deterministic: 是否强制使用 greedy 策略，通常用于评估或测试。

        输出:
            ``int``，被选择的 archetype id，取值范围为
            ``[0, num_archetypes)``。
        """

        if not deterministic and self.rng.random() < epsilon:
            return int(self.rng.integers(self.online_q_network.config.num_archetypes))

        visible_state_batch = self._visible_states_to_tensor_batch(visible_states)
        with torch.no_grad():
            q_values = self.online_q_network(visible_state_batch)
        return int(torch.argmax(q_values, dim=-1).item())
 

    def should_update(self, epoch: int) -> bool:
        """判断 replay buffer 是否已满足 Q-network 更新条件。

        功能说明:
            同时检查训练轮次和 replay buffer 样本数量。只有当前 epoch 达到
            ``learning_start_epoch``，且 replay buffer 中可采样 transition 数量不少于
            ``batch_size`` 时，才允许执行 Q-network 更新。

        输入参数:
            epoch: 当前训练轮次。

        输出:
            ``bool``。为 True 表示当前可以从 replay buffer 采样并更新 online
            Q-network；为 False 表示继续采集 transition。
        """

        return (
            epoch >= self.train_config.learning_start_epoch
            and len(self.replay_buffer) >= self.train_config.batch_size
        )

    def update_q_network(
        self,
        batch: Phase2SelectionTransitionTensorBatch,
    ) :
        """使用一个 replay batch 执行一次 online Q-network 参数更新。

        功能说明:
            调用 ``compute_double_dqn_loss()`` 计算 total loss、TD loss 和 imitation
            loss，然后执行 optimizer 清梯度、反向传播、梯度裁剪和参数更新。本方法
            不实现 Double DQN target 或 TD loss 细节，只负责编排更新步骤。

        输入参数:
            batch: 从 ``Phase2ReplayBuffer.sample()`` 返回的 tensor batch，包含当前
                visible states、action、reward、next visible states、done 和
                assigned label。

        输出:
            ``dict[str, float]``，包含 total loss、TD loss、imitation loss、
            selected Q 均值、TD target 均值和梯度范数等训练诊断指标。
        """

        self.online_q_network.train()
        loss_output = compute_double_dqn_loss(
            online_q_network=self.online_q_network,
            target_q_network=self.target_q_network,
            batch=batch,
            reward_config=self.reward_config,
            train_config=self.train_config,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss_output.total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_q_network.parameters(),
            max_norm=self.train_config.max_grad_norm,
        )
        self.optimizer.step()     
        return loss_output

    def sync_target_network(self) -> None:
        """将 online Q-network 参数硬同步到 target Q-network。

        功能说明:
            使用 online Q-network 的 ``state_dict`` 覆盖 target Q-network，并将
            target network 切到 eval 模式，保证 Double DQN bootstrap target 估计不受
            dropout 等训练态行为影响。

        输入参数:
            无。

        输出:
            无返回值。该方法会原地修改 target Q-network 参数。
        """

        self.target_q_network.load_state_dict(self.online_q_network.state_dict())
        self.target_q_network.eval()

    def build_checkpoint(self, epoch: int) -> Phase2Checkpoint:
        """构造 checkpoint payload，但不负责写入磁盘。

        功能说明:
            从当前 online Q-network 和 optimizer 提取可恢复训练的状态，构造成
            ``Phase2Checkpoint``。Q-network 权重会先搬到 CPU 并 clone，降低后续保存
            或传递 payload 时对训练中 GPU tensor 的影响。

        输入参数:
            epoch: checkpoint 对应的训练轮次。

        输出:
            ``Phase2Checkpoint``，包含 epoch、训练配置、Q-network state dict 和
            optimizer state dict。
        """

        return Phase2Checkpoint(
            epoch=epoch,
            config=self.train_config,
            q_network_state_dict= self.online_q_network.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        ) 
    def _should_sync_target(self, epoch: int) -> bool:
        """判断当前 epoch 结束后是否需要同步 target network。

        功能说明:
            读取 ``target_update_interval_epochs``。当同步间隔大于 0，且当前 epoch
            可以被同步间隔整除时，返回 True；否则返回 False。

        输入参数:
            epoch: 当前训练轮次。

        输出:
            ``bool``。为 True 表示应调用 ``sync_target_network()``。
        """

        interval = self.train_config.target_update_interval_epochs
        return interval > 0 and epoch % interval == 0

    def _visible_states_to_tensor_batch(
        self,
        visible_states: VisibleStates,
    ) -> tuple[torch.Tensor, ...]:
        """把单样本 visible states 转为 Q-network 可消费的 tensor batch。

        功能说明:
            将六路 numpy visible state 转成 ``torch.float32`` tensor，并移动到
            trainer 的目标设备。如果输入是单样本 ``[time, feature]``，则补 batch
            维度为 ``[1, time, feature]``；如果输入已经是
            ``[batch, time, feature]``，则保持 batch 维度不变。

        输入参数:
            visible_states: 六路 visible states，结构为 previous/current 各三路状态。

        输出:
            ``tuple[torch.Tensor, ...]``，六路 tensor batch，形状均为
            ``[batch, time, feature]``，可直接传给 ``Phase2QNetwork.forward()``。

        异常:
            当任一路 visible state 不是二维或三维 tensor/array 时抛出
            ``ValueError``。
        """

        tensors: list[torch.Tensor] = []
        for state in visible_states:
            tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise ValueError(
                    "每路 visible state 必须是 [time, feature] "
                    "或 [batch, time, feature] 形状"
                )
            tensors.append(tensor)
        return tuple(tensors)

__all__ = ["Phase2DoubleDqnTrainer"]
