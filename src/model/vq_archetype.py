"""Phase I VQ archetype 模型。

这个文件实现 ArchetypeTrader 第一阶段使用的 VQ encoder-decoder。

类的功能:
    模型接收一条或一批 DP demonstration trajectory：
    ``(states, relative_states, trend_states, actions, rewards)``，当前 encoder
    使用主 ``states/actions/rewards`` 压缩成连续 latent ``z_e``，再通过
    Vector Quantizer 把 ``z_e`` 分配到有限 codebook 中的一个离散 archetype，
    最后由 decoder 根据市场状态和 archetype 向量重构每个时间步的 teacher
    action。

为什么设计这个类:
    交易策略不应该只是一条固定动作序列。Phase I 的目标是从 DP teacher
    生成的大量轨迹中蒸馏出一组可复用的高层交易行为，例如趋势持有、
    均值回复、反转捕捉等。VQ codebook 提供了一个离散瓶颈，迫使模型把
    相似轨迹归入同一个 archetype，而不是记住每条轨迹的细节。

用途:
    1. Phase I 训练时，用 ``forward`` 学习 action reconstruction 和 VQ loss。
    2. Phase I 预训练时，用 ``forward_pretrain`` 跳过量化，先稳定
       encoder/decoder 的基本重构能力。
    3. Phase II 训练前，用 ``encode`` 给每个 horizon 生成 archetype label。
    4. Phase II/III 推理时，用 ``decode`` 根据 selector 选中的 code
       生成基础动作序列，再交给后续策略选择或细化模块。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .codebook import (
    CodebookInitResult,
    QuantizeOutput,
    VectorQuantizer,
    classify_trajectory_directions,
)
from .tensor_data_types import (
    ActionLogitTensor,
    ArchetypeLabelTensor,
    LatentTensor,
    TrajectoryTensorBatch,
)


@dataclass(frozen=True)
class VqModelOutputs:
    """VQ archetype 模型的一次前向输出。

    ``action_logits`` 用于重构 DP teacher action；
    ``code_id`` 是每条 trajectory 被分配到的 archetype label；
    ``total_loss`` 是基础训练损失，包含 action reconstruction 和 VQ loss。
    更复杂的 usage regularization、contrastive loss、alignment loss 应该在
    trainer/loss 模块中额外组合，避免模型类承担训练策略职责。
    """

    action_logits: ActionLogitTensor
    z_e: LatentTensor
    z_q: LatentTensor
    z_q_no_grad: LatentTensor
    code_id: ArchetypeLabelTensor
    code_indices: ArchetypeLabelTensor
    reconstruction_loss: torch.Tensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    total_loss: torch.Tensor


class ArchetypeTrajectoryEncoder(nn.Module):
    """把一条 demonstration trajectory 编码成连续 latent ``z_e``。

    设计原因:
        archetype 不是单个时点的市场状态，而是一整段 horizon 内
        ``状态-动作-reward`` 的联合行为模式。encoder 同时读取三路输入，
        让 latent 能表达 DP teacher 在该 horizon 中为什么这样交易。

    网络层说明:
        1. ``state_adapter``:
           把每个时间步的市场特征 ``[state_dim]`` 投影到统一的
           ``hidden_dim`` 表示空间。``LayerNorm`` 稳定不同特征尺度，
           ``GELU`` 提供非线性，使原始技术指标/市场状态可以组合成更
           适合时序建模的局部状态 embedding。
        2. ``action_embedding`` + ``action_norm``:
           把离散 teacher action id 映射成 ``hidden_dim`` 向量。这样
           short/flat/long 不再只是整数类别，而是可学习的动作语义向量；
           ``LayerNorm`` 让动作分支的数值尺度和状态、reward 分支更接近。
        3. ``reward_adapter``:
           把每个时间步的一维 reward 投影到 ``hidden_dim``。reward 分支
           让 encoder 看到 teacher 动作带来的即时收益/代价，帮助区分
           表面动作相似但盈亏路径不同的 trajectory。
        4. ``fusion``:
           在每个时间步拼接 ``state/action/reward`` 三路 embedding 后，
           压回 ``hidden_dim``。这一层学习“在该市场状态下采取该动作并
           获得该 reward”的局部行为含义。
        5. ``lstm``:
           沿 horizon 聚合局部行为 embedding，建模持仓延续、反转、止盈
           等跨时间步模式。这里使用单向 LSTM，encoder 编码整条 teacher
           trajectory；decoder 的因果性由 decoder 自身保证。
        6. ``projection``:
           把 LSTM 的最后隐状态压缩成 ``latent_dim`` 连续 latent ``z_e``。
           这个低维瓶颈迫使模型保留整条 trajectory 的核心交易风格，而
           不是保存每个时间步的细节。
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        action_dim: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # 状态分支：连续市场特征 -> hidden_dim 局部状态表示。
        self.state_adapter = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 动作分支：离散 teacher action -> 可学习动作语义向量。
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        # Reward 分支：一维逐步收益 -> hidden_dim 收益/代价表示。
        self.reward_adapter = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 三路信息融合：每个时间步形成统一的行为 token。
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 时序聚合：把 horizon 内的行为 token 汇总成 trajectory-level 表示。
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Latent 投影：trajectory-level hidden -> VQ 使用的连续 latent z_e。
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, batch: TrajectoryTensorBatch) -> LatentTensor:
        """返回每条 trajectory 的连续 latent ``z_e``，形状为 ``[batch, latent_dim]``。"""

        states, _, _, actions, rewards = _normalize_trajectory_batch(batch)

        # 三路输入先独立对齐到 hidden_dim，避免不同 dtype/尺度直接混合。
        state_emb = self.state_adapter(states.float())
        action_emb = self.action_norm(self.action_embedding(actions.long()))
        reward_emb = self.reward_adapter(rewards.float())

        # 在时间步维度保持不变，只在最后一维融合 state/action/reward 信息。
        fused = self.fusion(torch.cat([state_emb, action_emb, reward_emb], dim=-1))
        _, (hidden, _) = self.lstm(fused)
        # hidden: [num_layers, batch, hidden_dim]；取最后一层作为整条轨迹摘要。
        last_hidden = hidden[-1]
        return self.projection(last_hidden)


class ArchetypeActionDecoder(nn.Module):
    """根据市场状态序列和 archetype latent 重构逐步动作 logits。

    设计原因:
        decoder 是 Phase II/III 使用 archetype 的执行入口。它必须是因果的：
        第 ``tau`` 步动作只能依赖 ``s_0...s_tau`` 和选中的 archetype，
        不能偷看 horizon 后面的未来状态。

    网络层说明:
        1. ``state_adapter``:
           把每个时间步的市场状态投影到 ``hidden_dim``，与 encoder 的状态
           分支保持同一表示宽度。decoder 只接收 states，不接收未来动作或
           reward，因此推理阶段可以直接使用。
        2. ``z_q_seq`` 扩展:
           ``z_q`` 是每条 trajectory 一个 archetype 向量。forward 中会把它
           从 ``[batch, latent_dim]`` 扩展为 ``[batch, horizon, latent_dim]``，
           让每个时间步都知道当前要执行哪类交易原型。
        3. ``decoder_input`` 拼接:
           每个时间步输入为 ``[state_emb_t, z_q]``。状态提供当前市场上下文，
           archetype 提供全局交易风格/计划。
        4. ``lstm``:
           单向 LSTM 逐步处理 ``decoder_input``，第 ``tau`` 步 hidden 只包含
           到 ``tau`` 为止的信息，因此满足因果解码要求。
        5. ``action_head``:
           把每个时间步的 LSTM hidden 映射成 ``action_dim`` logits。训练时
           logits 用于 ``cross_entropy`` 重构 teacher action；推理时用
           ``argmax`` 得到基础动作序列。
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        action_dim: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # 状态编码：把实时市场状态对齐到 decoder 的 hidden 表示空间。
        self.state_adapter = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 因果时序解码：每步输入为 state embedding + archetype latent。
        self.lstm = nn.LSTM(
            input_size=hidden_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # 动作分类头：每个时间步 hidden -> short/flat/long logits。
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, z_q: LatentTensor) -> ActionLogitTensor:
        """返回动作 logits。

        输入形状:
            ``states``: ``[batch, horizon, state_dim]``
                一批市场状态序列。``batch`` 是样本数，``horizon`` 是每条
                trajectory 的时间步长度，``state_dim`` 是单步状态特征数。

            ``z_q``: ``[batch, latent_dim]``
                每条 trajectory 对应的 archetype latent/codebook 向量。
                ``z_q.shape[0]`` 必须和 ``states.shape[0]`` 一致。

        内部形状:
            ``state_emb``: ``[batch, horizon, hidden_dim]``
            ``z_q_seq``: ``[batch, horizon, latent_dim]``
            ``decoder_input``: ``[batch, horizon, hidden_dim + latent_dim]``

        输出形状:
            ``action_logits``: ``[batch, horizon, action_dim]``
        """

        if states.ndim != 3:
            raise ValueError("states must have shape [batch, horizon, state_dim]")
        if z_q.ndim != 2:
            raise ValueError("z_q must have shape [batch, latent_dim]")
        if states.shape[0] != z_q.shape[0]:
            raise ValueError("states and z_q must have the same batch size")

        batch_size, horizon, _ = states.shape
        state_emb = self.state_adapter(states.float())
        # 同一个 archetype 条件向量复制到该 trajectory 的所有时间步。
        z_q_seq = z_q.float().unsqueeze(1).expand(batch_size, horizon, self.latent_dim)
        decoder_input = torch.cat([state_emb, z_q_seq], dim=-1)
        hidden_seq, _ = self.lstm(decoder_input)
        return self.action_head(hidden_seq)


class ArchetypeVQModel(nn.Module):
    """Phase I VQ encoder-decoder 总模型。

    这个类把 trajectory encoder、VQ codebook 和 causal decoder 串起来。
    它既能作为 Phase I 训练模型，也能在训练完成后作为 label generator
    和 base action generator 被 Phase II/III 复用。

    模块说明:
        1. ``encoder``:
           输入 ``(states, relative_states, trend_states, actions, rewards)``，
           当前使用主 ``states/actions/rewards`` 输出连续 latent ``z_e``。
           它负责从 DP teacher trajectory 中抽取整段交易行为模式。
        2. ``quantizer``:
           把 ``z_e`` 映射到最近的 codebook 向量 ``z_q``，并输出离散
           ``code_indices``。这是模型从连续表示变成 archetype label 的核心。
        3. ``decoder``:
           输入 ``states`` 和 ``z_q``，重构逐步动作 logits。它验证 code
           是否真的包含足够信息来指导 action generation。
        4. ``_build_outputs``:
           统一计算 action reconstruction loss 和 VQ loss，并把训练/诊断
           需要的中间 Tensor 打包返回。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_archetypes: int = 10,
        commitment_cost: float = 0.25,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """初始化 Phase I VQ archetype 模型。

        为什么需要:
            顶层模型需要把 encoder、quantizer、decoder 用同一组维度参数
            组装起来，保证 ``latent_dim``、``hidden_dim``、``action_dim``
            在三个子模块之间一致，避免训练时出现隐式 shape mismatch。

        功能说明:
            保存模型超参数，并创建三个核心子模块：
            ``encoder`` 负责把 demonstration trajectory 压缩成 ``z_e``；
            ``quantizer`` 负责把 ``z_e`` 离散化为 archetype code；
            ``decoder`` 负责根据 states 和 archetype code 重构动作 logits。

        使用场景:
            在 Phase I 训练脚本、checkpoint 加载、离线 label 生成工具中创建
            模型实例时调用。调用方通常只需要提供 ``state_dim``，其余参数
            可来自实验配置。
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_archetypes = num_archetypes

        # Trajectory -> 连续 latent：学习 DP teacher 的整段行为摘要。
        self.encoder = ArchetypeTrajectoryEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        # 连续 latent -> 离散 archetype：形成 Phase II 可监督的 code label。
        self.quantizer = VectorQuantizer(
            num_archetypes=num_archetypes,
            latent_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        # states + archetype -> action logits：检验并复用离散原型的执行能力。
        self.decoder = ArchetypeActionDecoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, batch: TrajectoryTensorBatch) -> VqModelOutputs:
        """完整 VQ 训练前向。

        为什么需要:
            Phase I 的核心目标是同时学习“如何把 teacher trajectory 聚类成
            离散 archetype”和“每个 archetype 是否能重构 teacher action”。
            ``forward`` 把这两个目标放在同一次前向里，方便 trainer 直接
            使用 ``total_loss`` 做标准 VQ-VAE 风格训练。

        功能说明:
            1. 标准化 ``batch`` 的形状。
            2. 用 ``encoder`` 从 trajectory batch 得到连续 latent ``z_e``。
            3. 用 ``quantizer`` 把 ``z_e`` 映射成离散 code 和 STE 后的
               ``z_q``。
            4. 用 ``decoder`` 根据 ``states`` 和 ``z_q`` 重构
               ``action_logits``。
            5. 调用 ``_build_outputs`` 计算 reconstruction loss、VQ loss，
               并打包所有训练和诊断需要的 Tensor。

        使用场景:
            Phase I 正式 VQ 训练阶段调用。此时 ``actions`` 同时作为 encoder
            输入和 reconstruction target，模型会学习 codebook 分配和动作
            重构。
        """

        batch = _normalize_trajectory_batch(batch)
        states, _, _, actions, _ = batch
        z_e = self.encoder(batch)
        quantize_output = self.quantizer(z_e)
        action_logits = self.decoder(states, quantize_output.quantized)
        return self._build_outputs(
            action_logits=action_logits,
            actions=actions,
            z_e=z_e,
            z_q=quantize_output.quantized,
            z_q_no_grad=quantize_output.z_q_no_grad,
            code_indices=quantize_output.code_indices,
            vq_loss=quantize_output.vq_loss,
            codebook_loss=quantize_output.codebook_loss,
            commitment_loss=quantize_output.commitment_loss,
        )

    def forward_pretrain(self, batch: TrajectoryTensorBatch) -> VqModelOutputs:
        """Phase A 预训练前向。

        为什么需要:
            VQ 训练一开始如果 encoder/decoder 还没有基本重构能力，codebook
            很容易被噪声 latent 牵引，导致 code 使用不稳定或塌缩。预训练
            先跳过离散量化，让连续 ``z_e`` 直接条件化 decoder，降低正式
            VQ 阶段的优化难度。

        功能说明:
            1. 用 ``encoder`` 得到连续 latent ``z_e``。
            2. 直接把 ``z_e`` 传给 ``decoder``，不经过 ``quantizer``。
            3. VQ 相关 loss 返回 0，``code_indices`` 用占位 0 填充。
            4. 仍然复用 ``_build_outputs``，保证返回结构和正式
               ``forward`` 一致。

        使用场景:
            Phase I 的 warmup / Phase A 预训练阶段调用。训练目标只关注
            action reconstruction，用来稳定 encoder/decoder 的基础表示。
        """

        batch = _normalize_trajectory_batch(batch)
        states, _, _, actions, _ = batch
        z_e = self.encoder(batch)
        action_logits = self.decoder(states, z_e)
        zero = z_e.new_zeros(())
        code_indices = z_e.new_zeros((z_e.shape[0],), dtype=torch.long)
        return self._build_outputs(
            action_logits=action_logits,
            actions=actions,
            z_e=z_e,
            z_q=z_e,
            z_q_no_grad=z_e.detach(),
            code_indices=code_indices,
            vq_loss=zero,
            codebook_loss=zero,
            commitment_loss=zero,
        )

    @torch.no_grad()
    def encode(
        self,
        batch: TrajectoryTensorBatch,
    ) -> tuple[ArchetypeLabelTensor, LatentTensor]:
        """离线生成 archetype label。

        为什么需要:
            Phase II 的 archetype selector 需要监督信号或参考标签，而这些
            标签来自 Phase I 训练好的 VQ codebook。``encode`` 提供一个只做
            编码和量化的入口，不执行 decoder，也不计算训练 loss。

        功能说明:
            1. 标准化 trajectory batch。
            2. 用 ``encoder`` 得到连续 latent ``z_e``。
            3. 用 ``quantizer`` 找到最近的 archetype code。
            4. 返回 ``code_indices`` 和 ``z_e``。方法带 ``torch.no_grad``，
               适合批量离线处理。

        使用场景:
            Phase II 训练前，为每个 horizon 生成 archetype label；也可用于
            诊断 encoder latent 分布、codebook 使用率和 trajectory 聚类效果。

        返回:
            ``(code_id, z_e)``。``code_id`` 可作为 Phase II selector 的
            demonstration label，``z_e`` 可用于诊断 latent 分布。
        """

        batch = _normalize_trajectory_batch(batch)
        z_e = self.encoder(batch)
        quantize_output = self.quantizer(z_e)
        return quantize_output.code_indices, z_e

    @torch.no_grad()
    def decode(
        self,
        states_seq: torch.Tensor,
        code_id: ArchetypeLabelTensor,
    ) -> tuple[torch.Tensor, ActionLogitTensor]:
        """在线生成，根据指定 archetype 生成基础动作。

        为什么需要:
            Phase II/III 中 selector 选择的是离散 archetype id，而真正下单
            或继续细化策略时需要逐时间步动作。``decode`` 是从 archetype id
            回到单个 horizon 内可执行基础动作序列的推理入口。

        功能说明:
            1. 接收一个 horizon 内的部分市场状态序列 ``states_seq`` 和一个
               指定 ``code_id``。
            2. 从 codebook 中取出对应 embedding ``z_q``。
            3. 内部临时补一个 batch 维度，复用 batch-first 的 ``decoder``。
            4. 对 logits 做 ``argmax``，得到基础动作 ``base_actions``。
               方法带 ``torch.no_grad``，默认用于推理/离线生成。

        输入形状:
            ``states_seq``: ``[partial_horizon, state_dim]``
                单个 horizon 内已经观察到或准备解码的状态片段。
                ``partial_horizon`` 可以小于完整 horizon 长度；``state_dim``
                必须和模型初始化时的 ``state_dim`` 一致。

            ``code_id``: 标量或单元素 Tensor
                该状态片段要使用的 archetype id。这个接口只解码一个
                horizon 片段，因此不接受 ``[batch]`` 形式的多个 code。

        内部形状:
            ``states_batch``: ``[1, partial_horizon, state_dim]``
            ``z_q``: ``[1, latent_dim]``
            ``decode_logits_batch``: ``[1, partial_horizon, action_dim]``

        使用场景:
            Phase II/III 推理时，根据 selector 选中的 archetype 生成基础交易
            动作；也可在一个 horizon 还没走完时，用当前已经有的
            ``partial_horizon`` 状态片段生成基础动作。

        返回:
            ``(base_actions, decode_logits)``，其中 ``base_actions`` 是
            ``argmax`` 后的动作 id，形状为 ``[partial_horizon]``；
            ``decode_logits`` 形状为 ``[partial_horizon, action_dim]``。
        """

        if states_seq.ndim != 2:
            raise ValueError("states_seq must have shape [partial_horizon, state_dim]")
        if states_seq.shape[-1] != self.state_dim:
            raise ValueError(
                f"states_seq last dim must be {self.state_dim}, got {states_seq.shape[-1]}"
            )

        if not torch.is_tensor(code_id):
            code_id = torch.as_tensor(code_id, dtype=torch.long, device=states_seq.device)
        else:
            code_id = code_id.to(states_seq.device)
        if code_id.numel() != 1:
            raise ValueError("code_id must be a scalar or single-element tensor")
        code_id = code_id.reshape(1).long()

        states_batch = states_seq.unsqueeze(0)
        z_q = self.quantizer.embedding_from_code(code_id.to(states_seq.device))
        decode_logits = self.decoder(states_batch, z_q).squeeze(0)
        base_actions = decode_logits.argmax(dim=-1)
        return base_actions, decode_logits

    def _build_outputs(
        self,
        *,
        action_logits: ActionLogitTensor,
        actions: torch.Tensor,
        z_e: LatentTensor,
        z_q: LatentTensor,
        z_q_no_grad: LatentTensor,
        code_indices: ArchetypeLabelTensor,
        vq_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
    ) -> VqModelOutputs:
        """统一计算 loss 并打包模型输出。

        为什么需要:
            ``forward`` 和 ``forward_pretrain`` 的前向路径不同，但输出结构
            应该完全一致。把 loss 计算和 ``VqModelOutputs`` 构造集中到一个
            私有方法，可以减少重复代码，也能避免两个训练入口返回字段不一致。

        功能说明:
            1. 把 ``action_logits`` 和 ``actions`` 展平成
               ``[batch * horizon, action_dim]`` 与 ``[batch * horizon]``。
            2. 使用 ``cross_entropy`` 计算 teacher action reconstruction loss。
            3. 将 reconstruction loss 与传入的 ``vq_loss`` 相加得到
               ``total_loss``。
            4. 返回包含 logits、latent、code id、分项 loss 和总 loss 的
               ``VqModelOutputs``。

        使用场景:
            仅供本类内部调用。正式 VQ 训练和预训练都复用它，trainer 可以
            依赖稳定的返回字段读取 loss、code label 和诊断信息。
        """

        reconstruction_loss = F.cross_entropy(
            action_logits.reshape(-1, self.action_dim),
            actions.reshape(-1).long(),
        )
        total_loss = reconstruction_loss + vq_loss
        return VqModelOutputs(
            action_logits=action_logits,
            z_e=z_e,
            z_q=z_q,
            z_q_no_grad=z_q_no_grad,
            code_id=code_indices,
            code_indices=code_indices,
            reconstruction_loss=reconstruction_loss,
            vq_loss=vq_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            total_loss=total_loss,
        )


def _normalize_trajectory_batch(batch: TrajectoryTensorBatch) -> TrajectoryTensorBatch:
    """统一模型输入形状，减少训练代码里的样板转换。"""

    if len(batch) >= 5:
        states, relative_states, trend_states, actions, rewards = batch[:5]    
    else:
        raise ValueError(
            "trajectory batch must be "
            "(states, relative_states, trend_states, actions, rewards)"
        )
    if states.ndim != 3:
        raise ValueError("states must have shape [batch, horizon, state_dim]")
    if relative_states.ndim != 3:
        raise ValueError(
            "relative_states must have shape [batch, horizon, relative_feature_dim]"
        )
    if trend_states.ndim != 3:
        raise ValueError("trend_states must have shape [batch, horizon, trend_feature_dim]")
    if actions.ndim != 2:
        raise ValueError("actions must have shape [batch, horizon]")
    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    if rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError("rewards must have shape [batch, horizon] or [batch, horizon, 1]")
    if states.shape[:2] != actions.shape:
        raise ValueError("states and actions must share [batch, horizon]")
    if relative_states.shape[:2] != states.shape[:2]:
        raise ValueError("relative_states and states must share [batch, horizon]")
    if trend_states.shape[:2] != states.shape[:2]:
        raise ValueError("trend_states and states must share [batch, horizon]")
    if states.shape[:2] != rewards.shape[:2]:
        raise ValueError("states and rewards must share [batch, horizon]")
    return states, relative_states, trend_states, actions, rewards


# 设计文档里的新名字。
ArchetypeEncoder = ArchetypeTrajectoryEncoder
ArchetypeDecoder = ArchetypeActionDecoder
VQArchetypeModel = ArchetypeVQModel


__all__ = [
    "ArchetypeActionDecoder",
    "ArchetypeDecoder",
    "ArchetypeEncoder",
    "ArchetypeTrajectoryEncoder",
    "ArchetypeVQModel",
    "classify_trajectory_directions",
    "CodebookInitResult",
    "VqModelOutputs",
    "QuantizeOutput",
    "VQArchetypeModel",
    "VectorQuantizer",
]
