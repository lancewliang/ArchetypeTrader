"""Phase II horizon-level archetype selection 环境骨架。

文件功能说明:
    本文件定义 Phase II selector 训练和评估使用的 horizon-level MDP 环境入口。
    selector 每次输入一个 online 可见状态，输出一个 archetype id；环境使用冻结的
    Phase I decoder policy 将 archetype 解码成当前 horizon 的基础动作序列，并
    通过统一交易执行口径计算 reward。

设计边界:
    - 只定义环境依赖、方法签名、输入输出契约和职责说明；
    - 不实现 decoder 推理、动作执行、reward 聚合或 dataset sampler；
    - 不训练 selector，不更新 replay buffer，不执行 optimizer step；
    - 不读取未来状态作为 selector observation；
    - 不调用 Phase I encoder，也不使用 DP teacher 在线推理。

使用场景:
    ``Phase2DoubleDqnTrainer`` 使用本环境采集 horizon-level transition；
    ``Phase2Evaluator`` 可调用 ``run_horizon()`` 对指定样本做无状态评估；
    reward 口径通过 ``ActionExecutionCalculator`` 与 Phase I validation 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..model.data_types import VisibleStates
from ..utils import ActionExecutionCalculator
from .phase2_config import Phase2RewardConfig
from .model.phase2_decoder_policy import FrozenArchetypeDecoderPolicy

if TYPE_CHECKING:
    from .phase2_selection_dataset import Phase2SelectionDataset


@dataclass(frozen=True)
class Phase2SelectionStepResult:
    """Phase II horizon-level env step 结果 schema。

    适用场景:
        作为 ``ArchetypeSelectionEnv.step()`` 和 ``run_horizon()`` 的返回对象。

    字段解释:
        ``observation`` 是下一次 selector 决策可见状态；``reward`` 是当前
        horizon-level action 的交易收益；``done`` 表示当前 horizon 是否结束；
        ``info`` 承载训练、评估和报告所需的诊断字段。
    """

    # 下一 observation，结构为 previous/current 各三路 states。
    observation: VisibleStates

    # 当前 horizon 执行 selected archetype 后得到的 scalar reward。
    reward: float

    # Phase II 环境一步对应一个 horizon，通常为 True。
    done: bool

    # 诊断信息，例如 sample_id、selected_code_id、assigned_label、return、fee、turnover。
    info: dict[str, Any]


class ArchetypeSelectionEnv:
    """Phase II horizon-level archetype selection MDP 骨架。

    功能说明:
        将 ``Phase2SelectionDataset`` 包装成 selector 可交互的环境。一次 ``step()``
        对应一个完整 horizon：selector 选择一个 archetype，冻结 decoder 生成动作，
        交易执行器计算该 horizon 的收益，环境返回 reward、done 和诊断信息。

    设计边界:
        本类只负责 Phase II env 的接口边界和依赖组织。第一版骨架不实现真实
        sampler、decoder 调用或 reward 计算，避免在核心模型和 reward 模块未落地前
        固化行为。

    使用场景:
        训练时由 trainer 顺序或随机 reset 到某个 horizon 样本；评估时由 evaluator
        使用 ``run_horizon()`` 对指定样本和 code id 批量计算结果。
    """

    def __init__(
        self,
        dataset: Phase2SelectionDataset,
        decoder_policy: FrozenArchetypeDecoderPolicy,
        reward_config: Phase2RewardConfig,
    ) -> None:
        """构建 horizon-level selection MDP。

        功能说明:
            保存 Phase II dataset、冻结 decoder policy、reward 配置和统一交易执行器。

        使用场景:
            ``Phase2MainFlow`` 组装训练组件时创建；trainer 复用同一个环境进行
            transition 采样。

        参数:
            dataset: Phase II selection dataset，包含 visible states、完整 horizon
                数据和 Phase I assigned labels。
            decoder_policy: 冻结的 Phase I decoder policy，用 selected code 生成动作。
            reward_config: reward、手续费和 reward 后处理配置。
        """

        self.dataset = dataset
        self.decoder_policy = decoder_policy
        self.reward_config = reward_config
        self.execution_calculator = ActionExecutionCalculator(
            fee_rate=reward_config.fee_rate
        )
        self.current_index: int | None = None

    def reset(self, index: int | None = None) -> VisibleStates:
        """重置到一个 horizon 样本并返回 visible states。

        功能说明:
            后续实现应根据 ``index`` 或内部 sampler 选择当前样本，并返回该样本的
            previous/current 各三路 states。返回值只能包含 selector
            在线可见状态，不能包含当前 horizon 未来状态、价格或 label。

        使用场景:
            trainer 在采集每个 horizon transition 前调用；当 ``index`` 为 None 时，
            后续实现可采用顺序、随机或外部 sampler 策略。

        参数:
            index: 可选样本索引。指定时重置到该样本；为空时由环境自行选择。

        返回:
            当前样本的 ``VisibleStates``。
        """

        raise NotImplementedError("Phase2 env reset is not implemented yet.")

    def step(self, selected_code_id: int) -> Phase2SelectionStepResult:
        """执行一个 horizon-level archetype action。

        功能说明:
            使用当前样本和 ``selected_code_id`` 运行一个完整 horizon，返回 reward、
            ``done=True`` 和诊断信息。Phase II 中一步即完成一个 horizon。

        使用场景:
            trainer 在 ``reset()`` 后调用本方法采集 transition；结果可直接写入
            replay buffer。

        参数:
            selected_code_id: selector 选择的 archetype id。

        返回:
            ``Phase2SelectionStepResult``，包含下一 observation、reward、done 和 info。
        """

        raise NotImplementedError("Phase2 env step is not implemented yet.")
