"""Phase II selector Q-network。

文件功能说明:
    本文件定义 Phase II horizon-level archetype selector 的 Q-network。
    Q-network 输入在线可见的 previous/current 三路状态，输出每个
    archetype 的 Q value，供 Double DQN trainer、evaluator 和 checkpoint 保存复用。

设计边界:
    - 只定义 Q-network 输出 schema 和 Q value 计算入口；
    - 不实现 greedy/epsilon-greedy action selection；
    - 不把 Q value 转成概率或报告用解释性分布；
    - 不计算 Double DQN TD target、不访问 replay buffer；
    - 不调用 frozen decoder，也不计算交易 reward；
    - 不读取 Phase I assigned label，label regularization 属于 loss/trainer 层。

使用场景:
    ``Phase2DoubleDqnTrainer`` 使用 ``forward()`` 计算 online/target Q value；
    ``ArchetypeSelector`` 包装本模型做单步动作选择和概率解释；
    checkpoint 保存本网络 ``state_dict`` 和恢复所需配置。
"""

from __future__ import annotations
import torch
from torch import nn
from ...model.tensor_data_types import VisibleStatesTensorBatch
from ..phase2_config import Phase2ModelConfig

class Phase2QNetwork(nn.Module):
    """Phase II horizon-level archetype selector Q-network.

    功能说明:
        接收 selector 在线可见状态，并输出每个 archetype 的 Q value。当前接口
        显式接收 previous/current 各三路状态，和
        ``Phase2SelectionDatasetBuilder.to_tensor_dataset()`` 的输出保持一致。

    设计边界:
        本类只负责 Q-network 的模型接口。它使用六路状态各自的 temporal encoder
        处理不同 feature 维度，再把 pooled 表征交给 MLP Q head。

    使用场景:
        trainer 创建 online network 和 target network；checkpoint 只保存本网络的
        ``state_dict``；``ArchetypeSelector`` 使用本模型的 Q value 做动作选择。
    """

    VISIBLE_STATE_COUNT = 6
    STREAM_POOL_COUNT = 3

    _VISIBLE_STATE_NAMES = (
        "previous_t_states",
        "previous_t_relative_states",
        "previous_t_trend_states",
        "current_t_states",
        "current_t_relative_states",
        "current_t_trend_states",
    )

    def __init__(self, config: Phase2ModelConfig) -> None:
        """构建 horizon-level archetype selector Q-network。

        功能说明:
            保存 Q-network 配置，创建六路 visible state temporal encoder 和
            ``num_archetypes`` 维 Q-value head。六路 encoder 的输入维度由
            ``state_dim``、``relative_state_dim``、``trend_state_dim`` 显式确定，
            不依赖第一次 forward 的 lazy 初始化。

        输入参数:
            config: Phase II selector 模型配置，包含三路输入维度、
                num_archetypes、hidden_dim、num_layers 和 dropout。

        输出:
            无返回值。初始化后对象持有 ``stream_encoders``、``q_head`` 和
            可序列化的 ``config``。

        使用场景:
            ``Phase2MainFlow._create_q_network()`` 根据 ``Phase2ModelConfig`` 创建
            online Q-network；训练编排中用同一份 config 创建 target Q-network；
            checkpoint 恢复时依赖该 config 重建相同结构。
        """

        super().__init__()
        self.config = config
        self.visible_state_feature_dims = self._visible_state_feature_dims(config)
        self.stream_encoders = nn.ModuleList(
            self._build_stream_encoder(config, input_dim)
            for input_dim in self.visible_state_feature_dims
        )
        self.q_head = self._build_q_head(config)

    def forward(
        self,
        visible_states: VisibleStatesTensorBatch,
    ) -> torch.Tensor:
        """批量输入 selector 可见状态，输出每个 archetype 的 Q value。

        功能说明:
            这是本模型唯一的 Q value 计算入口，只处理 batch visible state。
            编码上一分片完整三路状态序列和当前分片前 ``TSize`` 个三路状态。
            每路状态经过 temporal encoder 后做 mean/max/last pooling，六路 pooled
            表征拼接后进入 MLP head，输出 ``[batch, num_archetypes]`` 的 Q value。
            本方法只接受六路可见状态 tensor，不接受完整 horizon、prices、
            depthprices、assigned label 或 reward，从接口边界上避免训练 batch 混入
            未来信息。单个在线决策由 ``ArchetypeSelector`` 转成 batch 后调用本方法。

        输入参数:
            visible_states: 六元组，顺序为
                ``previous_t_states, previous_t_relative_states,
                previous_t_trend_states, current_t_states,
                current_t_relative_states, current_t_trend_states``。每个元素均为
                ``torch.Tensor``，形状必须为 ``[batch, time, feature]``。

        输出:
            ``Phase2QNetworkOutput``，其中 ``q_values`` 形状为
            ``[batch, num_archetypes]``，表示每个样本选择各 archetype 的 Q value。

        使用场景:
            Double DQN loss 计算 online/target Q value；批量 diagnostics 读取全部
            Q value 分布；trainer 从 replay buffer 取 batch 后更新 Q-network。
        """ 
        stream_features = [
            self._encode_stream(encoder, stream_state)
            for encoder, stream_state in zip(self.stream_encoders, visible_states)
        ]
        fused_features = torch.cat(stream_features, dim=-1)
        q_values = self.q_head(fused_features)
        # fused_features: [batch, hidden_dim * 3 * 6]
        # q_values:       [batch, num_archetypes]
        return q_values

    @staticmethod
    def _build_q_head(config: Phase2ModelConfig) -> nn.Sequential:
        """构建从六路 pooled 表征到 Q value 的 MLP head。

        功能说明:
            根据 ``hidden_dim``、``num_layers`` 和 ``dropout`` 生成 Q-value head。
            当 ``num_layers == 0`` 时直接使用线性层从 fused feature 投影到
            ``num_archetypes``；否则使用 Linear + LayerNorm + ReLU + Dropout
            的 hidden block。

        输入参数:
            config: Phase II selector 模型配置，决定 head 宽度、深度、dropout
                和输出维度。

        输出:
            ``nn.Sequential``，输入形状为
            ``[batch, hidden_dim * STREAM_POOL_COUNT * 6]``，
            输出形状为 ``[batch, num_archetypes]``。

        使用场景:
            ``Phase2QNetwork.__init__()`` 创建模型结构；target network 用相同 config
            创建一致的 head，保证 state_dict 可加载。
        """

        pooled_dim_per_stream = config.hidden_dim * Phase2QNetwork.STREAM_POOL_COUNT
        input_dim = pooled_dim_per_stream * Phase2QNetwork.VISIBLE_STATE_COUNT
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(config.num_layers):
            layers.extend(
                (
                    nn.Linear(current_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
            )
            current_dim = config.hidden_dim
        layers.append(nn.Linear(current_dim, config.num_archetypes))
        return nn.Sequential(*layers)

    @classmethod
    def _visible_state_feature_dims(
        cls,
        config: Phase2ModelConfig,
    ) -> tuple[int, int, int, int, int, int]:
        """返回六路 visible state 的显式 feature 维度。"""

        dims = (
            config.state_dim,
            config.relative_state_dim,
            config.trend_state_dim,
            config.state_dim,
            config.relative_state_dim,
            config.trend_state_dim,
        )
        for name, dim in zip(cls._VISIBLE_STATE_NAMES, dims):
            if dim <= 0:
                raise ValueError(f"{name} feature dim must be positive, got {dim}")
        return dims

    @staticmethod
    def _build_stream_encoder(
        config: Phase2ModelConfig,
        input_dim: int,
    ) -> nn.ModuleDict:
        """构建单路 visible state encoder 的子模块集合。

        功能说明:
            创建 feature projection、LayerNorm、ReLU 和 dropout。输入 feature
            维度由 ``Phase2ModelConfig`` 显式提供，避免 state dict 中出现 lazy
            参数，也让 checkpoint 恢复前后结构一致。

        输入参数:
            config: Phase II selector 模型配置，提供 hidden_dim 和 dropout。
            input_dim: 该路 visible state 的 feature 维度。

        输出:
            ``nn.ModuleDict``，包含 ``projection``、``norm``、``activation`` 和
            ``dropout`` 四个子模块。

        使用场景:
            ``__init__()`` 为 six visible state streams 分别创建一套 encoder 子模块。
        """

        return nn.ModuleDict(
            {
                "projection": nn.Linear(input_dim, config.hidden_dim),
                "norm": nn.LayerNorm(config.hidden_dim),
                "activation": nn.ReLU(),
                "dropout": nn.Dropout(config.dropout),
            }
        )   

    def _encode_stream(
        self,
        encoder: nn.ModuleDict,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """编码单路状态序列并返回时间池化表征。

        功能说明:
            将 ``states`` 投影到 ``hidden_dim``，经过归一化、激活和 dropout 后，
            在时间维上提取 mean/max/last 三个 summary 并拼接。

        输入参数:
            encoder: ``_build_stream_encoder()`` 创建的单路 encoder 子模块集合。
            states: 单路 visible state tensor，形状为 ``[batch, time, feature]``。

        输出:
            ``torch.Tensor``，形状为 ``[batch, hidden_dim * STREAM_POOL_COUNT]``。

        使用场景:
            ``forward()`` 对六路状态逐路调用本方法，得到可融合的 temporal feature。
        """

        encoded = encoder["projection"](states)
        encoded = encoder["norm"](encoded)
        encoded = encoder["activation"](encoded)
        encoded = encoder["dropout"](encoded)

        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        last_step = encoded[:, -1, :]
        return torch.cat((mean_pool, max_pool, last_step), dim=-1)



__all__ = [
    "Phase2QNetwork",
]
