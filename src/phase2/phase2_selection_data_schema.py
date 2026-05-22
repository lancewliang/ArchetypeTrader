"""Phase II selection 数据流 schema 骨架。

本文件只定义 Phase II horizon-level selector 在 dataset、env、replay buffer
和 Double DQN loss 之间传递的数据对象。不放数据构建、校验、tensor 转换、
reward 计算、采样或训练逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..model.data_types import (
    DemonstrationHorizonLabelDataset,
    HorizonDataset,
    VisibleStatesDataset,
)
from ..model.tensor_data_types import ArchetypeLabelTensor


@dataclass(frozen=True)
class Phase2SelectionDataset:
    """Phase II selector 的 numpy dataset schema。

    适用场景:
        作为 dataset builder 的输出、dataset cache 的落盘结构，以及
        ``ArchetypeSelectionEnv`` / evaluator 的 numpy 输入。

    字段解释:
        ``visible_states`` 是 selector 在线推理可见的状态；
        ``horizon_dataset`` 是 frozen decoder 和环境 reward 计算需要的完整当前分片数据；
        ``demonstration_horizon_label_dataset`` 是 Phase I 离线导出的 imitation
        target 和诊断标签。
    """

    # selector observation，结构为 previous/current 各三路 states。
    visible_states: VisibleStatesDataset

    # 当前可训练 horizon 数据，结构为
    # (horizon_states, relative_states, trend_states, prices, depthprices)。
    horizon_dataset: HorizonDataset

    # Phase I assigned label 数据，结构为 (sample_ids, code_labels)。
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset


@dataclass(frozen=True)
class Phase2SelectionTensorBatch:
    """Phase II selector 的 tensor batch schema。

    适用场景:
        作为 ``DataLoader``、trainer、Q-network、decoder policy 和 evaluator
        之间传递的小批量 tensor 输入。

    字段解释:
        前六列是 selector 可见状态；中间五列是环境模拟和 decoder 推理需要的
        当前 horizon 数据；最后两列是 imitation label 和样本追踪信息。
    """

    # selector observation，结构为 previous/current 各三路 states。
    visible_states: VisibleStatesDataset

    # 当前可训练 horizon 数据，结构为
    # (horizon_states, relative_states, trend_states, prices, depthprices)。
    horizon_dataset: HorizonDataset

    # Phase I assigned label 数据，结构为 (sample_ids, code_labels)。
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset


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
    observation: VisibleStatesDataset

    # 当前 horizon 执行 selected archetype 后得到的 scalar reward。
    reward: float

    # Phase II 环境一步对应一个 horizon，通常为 True。
    done: bool

    # 诊断信息，例如 sample_id、selected_code_id、assigned_label、return、fee、turnover。
    info: dict[str, Any]


@dataclass(frozen=True)
class Phase2SelectionTransitionBatch:
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
    visible_states: VisibleStatesDataset

    # selector 选择的 archetype id，形状 [batch]。
    actions: ArchetypeLabelTensor

    # horizon-level reward，形状 [batch]。
    rewards: torch.Tensor

    # selector observation，结构为 previous/current 各三路 states。
    next_visible_states: VisibleStatesDataset
 
    # episode/horizon 结束标记，形状 [batch]。
    dones: torch.Tensor

    # Phase I assigned label 数据，结构为 (sample_ids, code_labels)。
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset
