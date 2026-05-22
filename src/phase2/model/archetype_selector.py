"""Phase II archetype selector decision wrapper.

文件功能说明:
    本文件定义位于 ``Phase2QNetwork`` 之上的 selector 决策包装类。Q-network 只
    负责输出 raw Q value；本类负责把 Q value 转成单步 greedy action、
    epsilon-greedy action，以及 report/diagnostics 使用的 softmax 伪概率。

设计边界:
    - 不定义 Q-network 结构，不持有 optimizer，不保存 checkpoint；
    - 不计算 Double DQN TD target、TD loss 或 imitation loss；
    - 不访问 replay buffer，不调用 frozen decoder，不计算交易 reward；
    - 不读取 Phase I assigned label，也不把 label、price 或完整 horizon 拼入输入；
    - 只消费 ``VisibleStatesTensor`` / ``VisibleStatesTensorBatch``，让 Q-network
      的 visible state 校验继续阻止未来信息泄漏。

使用场景:
    ``Phase2DoubleDqnTrainer`` 采集 transition 时可用 ``select_action()`` 做探索；
    ``Phase2Evaluator`` 可用 ``greedy_action()`` 做 deterministic validation/test；
    report 或 diagnostics 可用 ``predict_proba()`` / ``predict_proba_batch()``
    查看 selector 对 archetype 的相对偏好。
"""

from __future__ import annotations

import torch

from ...model.tensor_data_types import VisibleStatesTensor, VisibleStatesTensorBatch
from .phase2_q_network import Phase2QNetwork


class ArchetypeSelector:
    """Phase II archetype selector 的预测与决策包装类。

    功能说明:
        包装一个 ``Phase2QNetwork``，基于模型输出的 Q value 实现 deterministic
        greedy 决策、epsilon-greedy 探索决策和 softmax 伪概率解释。该类不改变
        Q-network 的训练语义；训练 loss 仍应直接使用 Q-network 的 ``forward()``
        输出 raw Q value。

    设计边界:
        本类只负责 Q-network 之上的预测解释和动作选择。不更新参数，不计算 reward，
        不处理 replay 采样，也不接受包含未来信息的 horizon/price/label batch。

    使用场景:
        训练采样阶段从单个 env observation 选择 action；validation/test 阶段做
        greedy selection；报告阶段批量生成 archetype 偏好分布。
    """

    def __init__(self, q_network: Phase2QNetwork) -> None:
        """初始化 selector 决策包装类。

        功能说明:
            保存外部创建和管理的 Q-network。Q-network 的 device、eval/train 模式、
            checkpoint 恢复和参数更新仍由主流程或 trainer 控制。

        输入参数:
            q_network: 已创建的 Phase II selector Q-network。

        输出:
            无返回值。初始化后本类通过 ``self.q_network`` 读取 raw Q value。

        使用场景:
            ``Phase2MainFlow`` 或 trainer/evaluator 在已有 Q-network 后创建本类，
            用于动作选择和评估诊断。
        """

        self.q_network = q_network

    @torch.no_grad()
    def greedy_action(
        self,
        visible_state: VisibleStatesTensor,
    ) -> int:
        """返回单个决策下 Q value 最大的 archetype id。

        功能说明:
            将单样本 visible state 转成 batch，然后调用 ``q_network.forward()``
            计算 Q value，并取 archetype 维度 ``argmax``。该方法不做随机探索，输出
            完全由当前网络参数决定。

        输入参数:
            visible_state: 单个 selector 决策可见状态六元组，每路为
                ``[time, feature]``。

        输出:
            ``int``，当前单个样本的 greedy archetype id。

        使用场景:
            validation/test 阶段逐 horizon deterministic selection；调试时快速查看
            单个 horizon 当前最偏好的 archetype。
        """

        q_values = self._predict_single_q_values(visible_state)
        return self._greedy_action_from_q_values(q_values)

    @torch.no_grad()
    def select_action(
        self,
        visible_state: VisibleStatesTensor,
        epsilon: float,
        deterministic: bool = False,
    ) -> int:
        """按 epsilon-greedy 或 greedy 策略选择单个 archetype id。

        功能说明:
            先计算单个样本的 greedy action，再以 ``epsilon`` 概率替换为随机
            archetype id。``deterministic=True`` 或 ``epsilon == 0`` 时退化为
            greedy。该方法不改变模型状态，不接受 batch。

        输入参数:
            visible_state: 单个 selector 决策可见状态六元组，每路为
                ``[time, feature]``。
            epsilon: 探索率，必须位于 ``[0, 1]``。
            deterministic: 是否强制使用 greedy action；为 True 时忽略 ``epsilon``。

        输出:
            ``int``，取值范围为 ``[0, num_archetypes)``。

        使用场景:
            训练采样阶段对 env 当前 observation 做 epsilon-greedy 探索；测试评估时
            传入 ``deterministic=True`` 做 greedy 选择。
        """

        epsilon = self._validate_epsilon(epsilon)
        q_values = self._predict_single_q_values(visible_state)
        greedy_action = self._greedy_action_from_q_values(q_values)
        if deterministic or epsilon == 0.0:
            return greedy_action

        explore = torch.rand((), device=q_values.device).item() < epsilon
        if not explore:
            return greedy_action
        return self._sample_random_action(q_values.device)

    @torch.no_grad()
    def predict_proba(
        self,
        visible_state: VisibleStatesTensor,
    ) -> torch.Tensor:
        """对单个决策的 Q value 做 softmax，返回伪概率。

        功能说明:
            将单样本 visible state 转成 batch，然后调用 ``q_network.forward()``
            计算 Q value，并沿 archetype 维度应用 softmax。该结果不是策略训练目标
            中的真实概率，只用于解释性展示。

        输入参数:
            visible_state: 单个 selector 决策可见状态六元组。

        输出:
            ``torch.Tensor``，形状为 ``[num_archetypes]``。元素和为 1，可视作各
            archetype 的相对偏好分布。

        使用场景:
            report 展示单个 horizon 的 selector 偏好；人工排查 checkpoint 行为时
            查看 soft assignment。
        """

        q_values = self._predict_single_q_values(visible_state)
        return self._softmax_q_values(q_values)

    @torch.no_grad()
    def predict_proba_batch(
        self,
        visible_states: VisibleStatesTensorBatch,
    ) -> torch.Tensor:
        """对批量 Q value 做 softmax，返回仅供评估/report 使用的伪概率。

        功能说明:
            调用 ``q_network.forward()`` 计算 batch Q value，并沿 archetype 维度
            应用 softmax。该方法只接受 batch visible states，不参与训练目标计算。

        输入参数:
            visible_states: ``VisibleStatesTensorBatch``，六路 tensor 均为
                ``[batch, time, feature]``。

        输出:
            ``torch.Tensor``，形状为 ``[batch, num_archetypes]``。

        使用场景:
            批量 diagnostics、report 中的 code preference 分布、离线分析
            checkpoint 是否出现 code collapse。
        """

        q_values = self.q_network(visible_states).q_values
        return self._softmax_q_values(q_values)

    @torch.no_grad()
    def _predict_single_q_values(
        self,
        visible_state: VisibleStatesTensor,
    ) -> torch.Tensor:
        """预测单个决策的 raw Q value。

        功能说明:
            将 ``VisibleStatesTensor`` 的六路 ``[time, feature]`` 输入转换为
            ``VisibleStatesTensorBatch`` 的六路 ``[1, time, feature]``，再调用
            Q-network 唯一的 ``forward()``。该方法不执行动作选择或概率解释。

        输入参数:
            visible_state: 单个 selector 决策可见状态六元组。

        输出:
            ``torch.Tensor``，形状为 ``[num_archetypes]``。

        使用场景:
            ``greedy_action()``、``select_action()`` 和 ``predict_proba()`` 内部复用。
        """

        visible_state_batch = self._single_visible_state_to_batch(visible_state)
        return self.q_network(visible_state_batch).q_values.squeeze(0)

    def _single_visible_state_to_batch(
        self,
        visible_state: VisibleStatesTensor,
    ) -> VisibleStatesTensorBatch:
        """把单个决策 visible state 转为 batch visible state。

        功能说明:
            检查单样本 visible state 数量和形状。每一路输入必须是
            ``[time, feature]``，方法会补成 ``[1, time, feature]``。转换后的 batch
            仍只包含六路在线可见状态，随后由 Q-network 的 ``forward()`` 继续执行
            batch 形状和防泄漏校验。

        输入参数:
            visible_state: 单个 selector 决策的六路 visible state。

        输出:
            ``VisibleStatesTensorBatch``，六路 tensor 均为 ``[1, time, feature]``。

        使用场景:
            单步动作选择和单步概率解释调用 Q-network 前的输入适配。
        """

        if len(visible_state) != self.q_network.VISIBLE_STATE_COUNT:
            raise ValueError(
                "visible state must contain exactly six tensors: "
                "previous/current states, relative states and trend states"
            )
        return tuple(self._single_sequence_to_batch(value) for value in visible_state)

    @staticmethod
    def _single_sequence_to_batch(value: torch.Tensor) -> torch.Tensor:
        """把一路单样本 visible state 补成 batch tensor。

        功能说明:
            接收 ``[time, feature]`` 的单路 tensor，并返回
            ``[1, time, feature]``。非 tensor 或非二维输入会被拒绝。

        输入参数:
            value: 单路单样本 visible state。

        输出:
            ``torch.Tensor``，形状为 ``[1, time, feature]``。

        使用场景:
            ``_single_visible_state_to_batch()`` 对六路单样本输入逐个调用。
        """

        if not isinstance(value, torch.Tensor):
            raise TypeError("visible state values must be torch.Tensor")
        if value.ndim != 2:
            raise ValueError(
                "single-decision visible state tensors must have shape [time, feature]"
            )
        if value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError("visible state time and feature dimensions must be non-empty")
        return value.unsqueeze(0)

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        """校验并标准化 epsilon-greedy 探索率。

        功能说明:
            将外部传入的 epsilon 转为 ``float``，并确保其位于 ``[0, 1]``。

        输入参数:
            epsilon: 探索率。

        输出:
            ``float``，合法的探索率。

        使用场景:
            ``select_action()`` 在采样探索动作前调用。
        """

        epsilon = float(epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        return epsilon

    @staticmethod
    def _greedy_action_from_q_values(q_values: torch.Tensor) -> int:
        """从单样本 Q value 中读取 greedy action。

        功能说明:
            对 ``[num_archetypes]`` 的 Q value 取 ``argmax`` 并转成 Python ``int``。

        输入参数:
            q_values: 单样本 Q value，形状为 ``[num_archetypes]``。

        输出:
            ``int``，Q value 最大的 archetype id。

        使用场景:
            ``greedy_action()`` 和 ``select_action()`` 内部复用。
        """

        if q_values.ndim != 1:
            raise ValueError("single-decision q_values must have shape [num_archetypes]")
        return int(torch.argmax(q_values, dim=-1).item())

    def _sample_random_action(self, device: torch.device) -> int:
        """随机采样一个合法 archetype id。

        功能说明:
            根据 Q-network 配置中的 ``num_archetypes`` 在 ``[0, num_archetypes)``
            范围内均匀采样一个整数动作。

        输入参数:
            device: 随机 tensor 所在设备，通常与 Q value 所在设备一致。

        输出:
            ``int``，随机 archetype id。

        使用场景:
            ``select_action()`` 命中 epsilon 探索分支时调用。
        """

        return int(
            torch.randint(
                low=0,
                high=self.q_network.config.num_archetypes,
                size=(),
                device=device,
                dtype=torch.long,
            ).item()
        )

    @staticmethod
    def _softmax_q_values(q_values: torch.Tensor) -> torch.Tensor:
        """把 Q value 转成解释用 softmax 伪概率。

        功能说明:
            沿最后一维应用 softmax。输入可以是单样本 ``[num_archetypes]``，也可以是
            batch ``[batch, num_archetypes]``。

        输入参数:
            q_values: Q-network 输出的 raw Q value。

        输出:
            ``torch.Tensor``，形状与输入一致，最后一维和为 1。

        使用场景:
            ``predict_proba()`` 和 ``predict_proba_batch()`` 内部复用。
        """

        return torch.softmax(q_values, dim=-1)


__all__ = ["ArchetypeSelector"]
