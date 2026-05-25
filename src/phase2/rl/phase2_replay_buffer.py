"""Phase II Double DQN replay buffer。

文件功能说明:
    本文件定义 Phase II horizon-level transition 和 replay buffer 入口。Phase II
    环境的一步对应一个完整 horizon，因此 replay buffer 保存的是 selector
    observation、archetype action、horizon reward、next observation、done mask
    和 Phase I assigned label 监督信号。

设计边界:
    - 实现 horizon-level transition 的环形存储、随机采样和 tensor batch 组装；
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
from numbers import Integral, Real

import numpy as np
import torch

from ...model.tensor_data_types import (
    DemonstrationHorizonLabelTensorBatch,
    VisibleStatesTensorBatch,
)

from ...model.data_types import DemonstrationHorizonLabel, VisibleStates


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
    """Phase II fixed-capacity replay buffer。

    功能说明:
        管理 horizon-level transition 的固定容量缓存。使用环形写入策略，支持按
        seed 可复现随机采样，并把 numpy transition 组装为
        ``Phase2SelectionTransitionBatch``。同一个 ``(sample_id, action)`` 只保留
        一条 active transition。

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
            保存 buffer 容量、visible state shape 和随机种子，并初始化环形存储
            数组、写指针、当前大小和随机数生成器。

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
        self.visible_state_shapes = _validate_visible_state_shapes(
            visible_state_shapes
        )
        self.seed = int(seed)

        self._visible_state_buffers = [
            np.empty((self.capacity, *shape), dtype=np.float32)
            for shape in self.visible_state_shapes
        ]
        self._next_visible_state_buffers = [
            np.empty((self.capacity, *shape), dtype=np.float32)
            for shape in self.visible_state_shapes
        ]
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.bool_)
        self._sample_ids = np.empty(self.capacity, dtype=np.int64)
        self._code_labels = np.empty(self.capacity, dtype=np.int64)
        self._transition_index_by_key: dict[tuple[int, int], int] = {}
        self._transition_keys_by_index: list[tuple[int, int] | None] = [
            None
            for _ in range(self.capacity)
        ]

        self._write_index = 0
        self._size = 0
        self._rng = np.random.default_rng(self.seed)

    def add(self, transition: Phase2ReplayTransition) -> None:
        """写入一个 horizon-level transition。

        功能说明:
            将 transition 写入环形 buffer；容量满后覆盖最旧 transition。若当前
            buffer 中已存在相同 ``(sample_id, action)``，则跳过写入。

        使用场景:
            trainer 每次 env step 后调用本方法，把新 transition 放入 replay buffer。

        参数:
            transition: 待写入的 Phase II horizon-level transition。
        """

        visible_states = self._prepare_visible_states(
            transition.visible_states,
            name="visible_states",
        )
        next_visible_states = self._prepare_visible_states(
            transition.next_visible_states,
            name="next_visible_states",
        )
        action = _validate_non_negative_int(transition.action, name="action")
        reward = _validate_reward(transition.reward)
        done = bool(transition.done)
        sample_id, code_label = _validate_demonstration_horizon_label(
            transition.demonstration_horizon_label
        )

        transition_key = (sample_id, action)
        if transition_key in self._transition_index_by_key:
            return

        self._write_transition(
            visible_states=visible_states,
            action=action,
            reward=reward,
            next_visible_states=next_visible_states,
            done=done,
            sample_id=sample_id,
            code_label=code_label,
            transition_key=transition_key,
        )

    def _write_transition(
        self,
        *,
        visible_states: tuple[np.ndarray, ...],
        action: int,
        reward: float,
        next_visible_states: tuple[np.ndarray, ...],
        done: bool,
        sample_id: int,
        code_label: int,
        transition_key: tuple[int, int],
    ) -> None:
        index = self._write_index
        old_key = self._transition_keys_by_index[index]
        if old_key is not None:
            self._transition_index_by_key.pop(old_key, None)

        for stream_index, state in enumerate(visible_states):
            self._visible_state_buffers[stream_index][index] = state
        for stream_index, state in enumerate(next_visible_states):
            self._next_visible_state_buffers[stream_index][index] = state
        self._actions[index] = action
        self._rewards[index] = reward
        self._dones[index] = done
        self._sample_ids[index] = sample_id
        self._code_labels[index] = code_label
        self._transition_keys_by_index[index] = transition_key
        self._transition_index_by_key[transition_key] = index

        self._write_index = (self._write_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        *,
        visible_states: VisibleStates,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_visible_states: VisibleStates,
        dones: np.ndarray,
        demonstration_horizon_label_batch: DemonstrationHorizonLabelTensorBatch,
    ) -> None:
        """批量写入 horizon-level transitions。

        语义与连续调用 ``add()`` 一致；重复 ``(sample_id, action)`` 会被跳过。
        当输入批量大于 capacity 时，仅保留最后 ``capacity`` 条 active transition。
        """

        visible_state_batch = self._prepare_visible_state_batch(
            visible_states,
            name="visible_states",
        )
        next_visible_state_batch = self._prepare_visible_state_batch(
            next_visible_states,
            name="next_visible_states",
        )
        batch_size = visible_state_batch[0].shape[0]
        action_values = _validate_int_array(actions, name="actions", size=batch_size)
        reward_values = _validate_float_array(rewards, name="rewards", size=batch_size)
        done_values = _validate_bool_array(dones, name="dones", size=batch_size)
        sample_ids, code_labels = demonstration_horizon_label_batch
        sample_id_values = _validate_int_array(
            sample_ids,
            name="sample_ids",
            size=batch_size,
        )
        code_label_values = _validate_int_array(
            code_labels,
            name="code_labels",
            size=batch_size,
        )

        for batch_index in range(batch_size):
            action = int(action_values[batch_index])
            sample_id = int(sample_id_values[batch_index])
            transition_key = (sample_id, action)
            if transition_key in self._transition_index_by_key:
                continue

            self._write_transition(
                visible_states=tuple(
                    state[batch_index]
                    for state in visible_state_batch
                ),
                action=action,
                reward=float(reward_values[batch_index]),
                next_visible_states=tuple(
                    state[batch_index]
                    for state in next_visible_state_batch
                ),
                done=bool(done_values[batch_index]),
                sample_id=sample_id,
                code_label=int(code_label_values[batch_index]),
                transition_key=transition_key,
            )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> Phase2SelectionTransitionTensorBatch:
        """随机采样 Double DQN 训练 batch。

        功能说明:
            从当前可用 transition 中随机采样 ``batch_size`` 条，组装为
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

       
        if batch_size > self._size:
            raise ValueError(
                "batch_size must be less than or equal to current replay size, "
                f"got batch_size={batch_size}, size={self._size}"
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        target_device = torch.device(device)

        visible_states = tuple(
            torch.as_tensor(buffer[indices], dtype=torch.float32).to(target_device)
            for buffer in self._visible_state_buffers
        )
        next_visible_states = tuple(
            torch.as_tensor(buffer[indices], dtype=torch.float32).to(target_device)
            for buffer in self._next_visible_state_buffers
        )
        actions = torch.as_tensor(
            self._actions[indices],
            dtype=torch.long,
        ).to(target_device)
        rewards = torch.as_tensor(
            self._rewards[indices],
            dtype=torch.float32,
        ).to(target_device)
        dones = torch.as_tensor(
            self._dones[indices],
            dtype=torch.float32,
        ).to(target_device)
        demonstration_horizon_label_batch = (
            torch.as_tensor(self._sample_ids[indices], dtype=torch.long).to(
                target_device
            ),
            torch.as_tensor(self._code_labels[indices], dtype=torch.long).to(
                target_device
            ),
        )

        return Phase2SelectionTransitionTensorBatch(
            visible_states=visible_states,
            actions=actions,
            rewards=rewards,
            next_visible_states=next_visible_states,
            dones=dones,
            demonstration_horizon_label_batch=demonstration_horizon_label_batch,
        )

    def __len__(self) -> int:
        """返回当前 buffer 中可采样 transition 数量。

        功能说明:
            返回当前已写入且可采样的 transition 数，而不是固定 capacity。

        使用场景:
            trainer 用它判断 replay buffer 是否达到 ``learning_start_epoch`` 或最小
            batch size 要求。
        """

        return self._size

    def _prepare_visible_states(
        self,
        visible_states: VisibleStates,
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        if len(visible_states) != len(self.visible_state_shapes):
            raise ValueError(
                f"{name} must contain {len(self.visible_state_shapes)} arrays, "
                f"got {len(visible_states)}"
            )

        prepared_states: list[np.ndarray] = []
        for index, (state, expected_shape) in enumerate(
            zip(visible_states, self.visible_state_shapes, strict=True)
        ):
            state_array = np.asarray(state, dtype=np.float32)
            if state_array.shape != expected_shape:
                raise ValueError(
                    f"{name}[{index}] must have shape {expected_shape}, "
                    f"got {state_array.shape}"
                )
            prepared_states.append(state_array)
        return tuple(prepared_states)

    def _prepare_visible_state_batch(
        self,
        visible_states: VisibleStates,
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        if len(visible_states) != len(self.visible_state_shapes):
            raise ValueError(
                f"{name} must contain {len(self.visible_state_shapes)} arrays, "
                f"got {len(visible_states)}"
            )

        prepared_states: list[np.ndarray] = []
        batch_size: int | None = None
        for index, (state, expected_shape) in enumerate(
            zip(visible_states, self.visible_state_shapes, strict=True)
        ):
            state_array = np.asarray(state, dtype=np.float32)
            expected_ndim = len(expected_shape) + 1
            if state_array.ndim != expected_ndim:
                raise ValueError(
                    f"{name}[{index}] must have shape [batch, *{expected_shape}], "
                    f"got {state_array.shape}"
                )
            if tuple(state_array.shape[1:]) != expected_shape:
                raise ValueError(
                    f"{name}[{index}] sample shape must be {expected_shape}, "
                    f"got {state_array.shape[1:]}"
                )
            if batch_size is None:
                batch_size = int(state_array.shape[0])
                if batch_size <= 0:
                    raise ValueError(f"{name} batch size must be positive")
            elif int(state_array.shape[0]) != batch_size:
                raise ValueError(f"all {name} streams must share batch size")
            prepared_states.append(state_array)
        return tuple(prepared_states)

def _validate_visible_state_shapes(
    visible_state_shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if len(visible_state_shapes) != 6:
        raise ValueError(
            "visible_state_shapes must contain six stream shapes, "
            f"got {len(visible_state_shapes)}"
        )
    normalized_shapes: list[tuple[int, ...]] = []
    for index, shape in enumerate(visible_state_shapes):
        if len(shape) == 0:
            raise ValueError(f"visible_state_shapes[{index}] must not be empty")
        normalized_shape: list[int] = []
        for dimension in shape:
            if not isinstance(dimension, Integral):
                raise TypeError(
                    f"visible_state_shapes[{index}] dimensions must be integers"
                )
            dimension = int(dimension)
            if dimension <= 0:
                raise ValueError(
                    f"visible_state_shapes[{index}] dimensions must be positive, "
                    f"got {shape}"
                )
            normalized_shape.append(dimension)
        normalized_shapes.append(tuple(normalized_shape))
    return tuple(normalized_shapes)



def _validate_non_negative_int(value: int, *, name: str) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_reward(reward: float) -> float:
    if not isinstance(reward, Real):
        raise TypeError("reward must be a real number")
    reward = float(reward)
    if not np.isfinite(reward):
        raise ValueError(f"reward must be finite, got {reward}")
    return reward


def _as_numpy_array(value: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc


def _validate_int_array(
    value: np.ndarray | torch.Tensor,
    *,
    name: str,
    size: int,
) -> np.ndarray:
    values = np.asarray(_as_numpy_array(value, name=name), dtype=np.int64)
    if values.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}], got {values.shape}")
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative")
    return values


def _validate_float_array(
    value: np.ndarray | torch.Tensor,
    *,
    name: str,
    size: int,
) -> np.ndarray:
    values = np.asarray(_as_numpy_array(value, name=name), dtype=np.float32)
    if values.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _validate_bool_array(
    value: np.ndarray | torch.Tensor,
    *,
    name: str,
    size: int,
) -> np.ndarray:
    values = np.asarray(_as_numpy_array(value, name=name), dtype=np.bool_)
    if values.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}], got {values.shape}")
    return values


def _validate_demonstration_horizon_label(
    demonstration_horizon_label: DemonstrationHorizonLabel,
) -> tuple[int, int]:
    if len(demonstration_horizon_label) != 2:
        raise ValueError(
            "demonstration_horizon_label must be (sample_id, code_label)"
        )
    sample_id, code_label = demonstration_horizon_label
    return (
        _validate_non_negative_int(sample_id, name="sample_id"),
        _validate_non_negative_int(code_label, name="code_label"),
    )


__all__ = [
    "Phase2ReplayBuffer",
    "Phase2ReplayTransition",
    "Phase2SelectionTransitionTensorBatch",
]
