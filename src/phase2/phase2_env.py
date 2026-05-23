"""Phase II horizon-level archetype selection environment.

文件功能说明:
    本文件定义 Phase II selector 训练和评估使用的 horizon-level MDP 环境入口。
    selector 每次输入一个 online 可见状态，输出一个 archetype id；环境使用冻结的
    Phase I decoder policy 将 archetype 解码成当前 horizon 的基础动作序列，并
    通过统一交易执行口径计算 reward。

设计边界:
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

import numpy as np
import torch

from ..model.data_types import VisibleStates
from ..utils import ActionExecutionCalculator, ActionExecutionResult
from .model.phase2_decoder_policy import FrozenArchetypeDecoderPolicy
from .phase2_config import Phase2RewardConfig
from .phase2_selection_dataset import Phase2SelectionDataset


@dataclass(frozen=True)
class Phase2SelectionStepResultInfo:
    """Phase II horizon-level env step 结果诊断信息 schema。"""

    demo_sample_id: int
    selected_code_id: int
    demo_assigned_code_label: int
    reward: float = 0.0
    gross_return: float = 0.0
    fee: float = 0.0
    turnover: float = 0.0


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

    # 当前 dataset 遍历是否结束；run_horizon() 返回值固定为 True。
    done: bool

    # 诊断信息，例如 demo_sample_id、selected_code_id、demo_assigned_label、return、fee、turnover。
    info: Phase2SelectionStepResultInfo


class ArchetypeSelectionEnv:
    """Phase II horizon-level archetype selection MDP.

    功能说明:
        将 ``Phase2SelectionDataset`` 包装成 selector 可交互的环境。一次 ``step()``
        对应一个完整 horizon：selector 选择一个 archetype，冻结 decoder 生成动作，
        交易执行器计算该 horizon 的收益，环境返回 reward、done 和诊断信息。

    设计边界:
        本类只负责 Phase II env 的样本游标、decoder 推理、动作执行和 reward
        聚合。不负责 Q-network 动作选择、replay buffer 或 optimizer 更新。

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
        self._validate_dataset()
        self.current_index: int | None = None

    def reset(self) -> Phase2SelectionStepResult:
        """重置到一个 horizon 样本并返回 visible states。

        功能说明:            
            previous/current 各三路 states。返回值只包含 selector 在线可见状态，
            不包含当前 horizon 未来状态、价格或 label。

        使用场景:
            trainer 在每个 epoch 采集 transition 前调用；
            从第 0 个样本开始顺序遍历整个 dataset。

        返回:
            ``Phase2SelectionStepResult``，其中 ``reward=0``、``done=False``，
            ``observation`` 为当前样本的 selector 可见状态。
        """

        self.current_index = 0
        sample_id, assigned_code_label = self._sample_label_at(self.current_index)
        return Phase2SelectionStepResult(
            observation=self._visible_states_at(self.current_index),
            reward=0.0,
            done=False,
            info=Phase2SelectionStepResultInfo(
                demo_sample_id=sample_id,
                selected_code_id=-1,
                demo_assigned_code_label=assigned_code_label,
            ),
        )

    def step(self, selected_code_id: int) -> Phase2SelectionStepResult:
        """执行一个 horizon-level archetype action。

        功能说明:
            使用当前样本和 ``selected_code_id`` 运行一个完整 horizon，返回 reward、
            下一条 observation 和诊断信息；当 dataset 被遍历完时返回 ``done=True``。

        使用场景:
            trainer 在 ``reset()`` 后调用本方法采集 transition；结果可直接写入
            replay buffer。

        参数:
            selected_code_id: selector 选择的 archetype id。

        返回:
            ``Phase2SelectionStepResult``，包含下一 observation、reward、done 和 info。
        """

        if self.current_index is None:
            raise RuntimeError("reset() must be called before step()")

        current_index = self.current_index
        horizon_result = self.run_horizon(
            index=current_index,
            selected_code_id=selected_code_id,
        )
        next_index = current_index + 1
        done = next_index >= self.sample_count
        if done:
            next_observation = horizon_result.observation
            self.current_index = None
        else:
            next_observation = self._visible_states_at(next_index)
            self.current_index = next_index

        return Phase2SelectionStepResult(
            observation=next_observation,
            reward=horizon_result.reward,
            done=done,
            info=horizon_result.info,
        )

    def run_horizon(
        self,
        index: int,
        selected_code_id: int,
    ) -> Phase2SelectionStepResult:
        """对指定样本和 archetype id 执行一个完整 horizon。

        功能说明:
            本方法是无状态评估入口，不依赖也不修改 ``current_index``。它读取指定
            样本的完整 horizon states，调用冻结 decoder 得到基础动作序列，然后用
            ``ActionExecutionCalculator`` 计算收益、手续费和换手。

        参数:
            index: ``Phase2SelectionDataset`` 内部样本行号。
            selected_code_id: selector 选择的 archetype id。

        返回:
            当前样本的 execution 结果。``done=True`` 表示单个 horizon 已执行完毕；
            ``observation`` 仍返回该样本的可见状态，方便调用方保留上下文。
        """

        self._validate_index(index)
        selected_code_id = self._validate_selected_code_id(selected_code_id)
        actions = self._decode_actions(index, selected_code_id)
        execution = self._execute_actions(index, actions)
        reward = self._postprocess_reward(float(execution.returns[0]))
        sample_id, assigned_code_label = self._sample_label_at(index)
        return Phase2SelectionStepResult(
            observation=self._visible_states_at(index),
            reward=reward,
            done=True,
            info=Phase2SelectionStepResultInfo(
                demo_sample_id=sample_id,
                selected_code_id=selected_code_id,
                demo_assigned_code_label=assigned_code_label,
                reward=reward,
                gross_return=float(execution.gross_returns[0]),
                fee=float(execution.fees[0]),
                turnover=float(execution.turnover[0]),
            ),
        )

    @property
    def sample_count(self) -> int:
        """返回 Phase II selection dataset 的样本数量。"""

        return int(self.dataset.visible_states[0].shape[0])

    def __len__(self) -> int:
        """返回环境可遍历的 horizon 样本数量。"""

        return self.sample_count

    def _visible_states_at(self, index: int) -> VisibleStates:
        """读取单样本 selector 可见状态，并复制以避免外部原地修改 dataset。"""

        self._validate_index(index)
        return tuple(
            np.asarray(visible_state[index]).copy()
            for visible_state in self.dataset.visible_states
        )

    def _decode_actions(self, index: int, selected_code_id: int) -> np.ndarray:
        """使用 frozen decoder 将 selected code 转成 ``[1, horizon]`` 动作序列。"""

        horizon_states, relative_states, trend_states, _, _ = self.dataset.horizon_dataset
        horizon_states_tensor = torch.as_tensor(
            horizon_states[index : index + 1],
            dtype=torch.float32,
        )
        relative_states_tensor = torch.as_tensor(
            relative_states[index : index + 1],
            dtype=torch.float32,
        )
        trend_states_tensor = torch.as_tensor(
            trend_states[index : index + 1],
            dtype=torch.float32,
        )
        selected_code_ids = torch.as_tensor([selected_code_id], dtype=torch.long)

        decoded_actions = self.decoder_policy.decode_actions(
            horizon_states=horizon_states_tensor,
            horizon_relative_states=relative_states_tensor,
            horizon_trend_states=trend_states_tensor,
            selected_code_ids=selected_code_ids,
        )
        if isinstance(decoded_actions, torch.Tensor):
            action_values = decoded_actions.detach().cpu().numpy()
        else:
            action_values = np.asarray(decoded_actions)
        if action_values.ndim == 1:
            action_values = action_values.reshape(1, -1)
        if action_values.ndim != 2 or action_values.shape[0] != 1:
            raise ValueError(
                "decoder actions must have shape [1, horizon], "
                f"got {tuple(action_values.shape)}"
            )
        expected_horizon = int(horizon_states.shape[1])
        if action_values.shape[1] != expected_horizon:
            raise ValueError(
                "decoder actions horizon length must match horizon states: "
                f"{action_values.shape[1]} != {expected_horizon}"
            )
        action_values = np.asarray(action_values, dtype=np.int64)
        if np.any(action_values < 0) or np.any(action_values > 2):
            min_action = int(np.min(action_values))
            max_action = int(np.max(action_values))
            raise ValueError(
                "decoder actions must use ids 0=short, 1=flat, 2=long; "
                f"got min={min_action}, max={max_action}"
            )
        return action_values

    def _execute_actions(
        self,
        index: int,
        actions: np.ndarray,
    ) -> ActionExecutionResult:
        """按统一交易执行口径计算单样本 horizon 收益。"""

        _, _, _, prices, depthprices = self.dataset.horizon_dataset
        depthprice_slice = None
        if depthprices is not None:
            depthprice_slice = depthprices[index : index + 1]
        return self.execution_calculator.execute(
            prices=prices[index : index + 1],
            actions=actions,
            depthprices=depthprice_slice,
        )

    def _postprocess_reward(self, reward: float) -> float:
        """应用 Phase II reward 后处理。"""

        if self.reward_config.reward_clip is None:
            return float(reward)
        clip_value = float(self.reward_config.reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be positive, got {clip_value}")
        return float(np.clip(reward, -clip_value, clip_value))

    def _sample_label_at(self, index: int) -> tuple[int, int]:
        """读取 dataset 中与样本行号对应的原始 sample_id 和 assigned code label。"""

        sample_ids, code_labels = self.dataset.demonstration_horizon_label_dataset
        return int(sample_ids[index]), int(code_labels[index])

    def _validate_dataset(self) -> None:
        """校验 env 依赖的 dataset 形状契约。"""

        if len(self.dataset.visible_states) != 6:
            raise ValueError("dataset.visible_states must contain six arrays")
        sample_count = int(self.dataset.visible_states[0].shape[0])
        if sample_count <= 0:
            raise ValueError("Phase2SelectionDataset must contain at least one sample")
        for visible_state in self.dataset.visible_states:
            if visible_state.ndim != 3:
                raise ValueError("each visible state array must be 3D")
            if int(visible_state.shape[0]) != sample_count:
                raise ValueError("all visible state arrays must share sample count")

        if len(self.dataset.horizon_dataset) != 5:
            raise ValueError(
                "dataset.horizon_dataset must be "
                "(states, relative_states, trend_states, prices, depthprices)"
            )
        horizon_states, relative_states, trend_states, prices, depthprices = (
            self.dataset.horizon_dataset
        )
        horizon_arrays = (horizon_states, relative_states, trend_states, prices)
        for horizon_array in horizon_arrays:
            if horizon_array.ndim != 3:
                raise ValueError("horizon dataset arrays must be 3D")
            if int(horizon_array.shape[0]) != sample_count:
                raise ValueError("horizon dataset arrays must share sample count")
            if horizon_array.shape[1] != horizon_states.shape[1]:
                raise ValueError("horizon dataset arrays must share horizon length")
        if depthprices is not None:
            if depthprices.ndim != 3:
                raise ValueError("depthprices must be 3D when provided")
            if depthprices.shape[:2] != horizon_states.shape[:2]:
                raise ValueError(
                    "depthprices must share sample and horizon dimensions"
                )

        sample_ids, code_labels = self.dataset.demonstration_horizon_label_dataset
        if sample_ids.shape != (sample_count,):
            raise ValueError("sample_ids must have shape [sample]")
        if code_labels.shape != (sample_count,):
            raise ValueError("code_labels must have shape [sample]")

    def _validate_index(self, index: int) -> None:
        """校验 dataset 行号。"""

        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        if int(index) < 0 or int(index) >= self.sample_count:
            raise IndexError(
                f"index must be in [0, {self.sample_count}), got {index}"
            )

    def _validate_selected_code_id(self, selected_code_id: int) -> int:
        """校验 selected archetype id 的基础类型和值域下界。"""

        if not isinstance(selected_code_id, (int, np.integer)):
            raise TypeError("selected_code_id must be an integer")
        selected_code_id = int(selected_code_id)
        if selected_code_id < 0:
            raise ValueError("selected_code_id must be non-negative")
        num_archetypes = self._num_archetypes()
        if num_archetypes is not None and selected_code_id >= num_archetypes:
            raise ValueError(
                f"selected_code_id must be in [0, {num_archetypes}), "
                f"got {selected_code_id}"
            )
        return selected_code_id

    def _num_archetypes(self) -> int | None:
        """从 decoder policy 尽力读取 codebook 大小；fake policy 可不提供。"""

        if hasattr(self.decoder_policy, "num_archetypes"):
            return int(getattr(self.decoder_policy, "num_archetypes"))
        phase1_model = getattr(self.decoder_policy, "phase1_model", None)
        if phase1_model is not None and hasattr(phase1_model, "num_archetypes"):
            return int(getattr(phase1_model, "num_archetypes"))
        return None


__all__ = [
    "ArchetypeSelectionEnv",
    "Phase2SelectionStepResult",
    "Phase2SelectionStepResultInfo",
]
