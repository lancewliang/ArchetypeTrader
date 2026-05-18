"""Phase I VQ training auxiliary losses.

这个模块承载 Phase I VQ 训练中的“训练策略”部分。

设计背景:
    ``ArchetypeVQModel`` 自身只负责基础前向路径：
    trajectory encoder 生成 ``z_e``，VectorQuantizer 生成 ``z_q`` 和 code id，
    decoder 重构 teacher action。基础模型内部的 ``outputs.total_loss`` 仍然是
    ``action reconstruction + VQ loss``，这样模型类可以保持为通用 VQ
    encoder-decoder，不绑定某一种实验目标。

为什么辅助 loss 放在这里:
    weak_pair_code_ratio 的问题不是模型结构无法解码动作，而是训练目标没有明确
    要求 latent/code 绑定 market morphology 与 action motif。因此这里在 trainer
    边界额外组合:

    1. morphology auxiliary classification:
       要求 encoder latent ``z_e`` 可预测市场形态，防止 code 只记动作而忽略
       market structure。
    2. coarse morphology-motif pair auxiliary classification:
       直接优化 weak_pair_code_ratio 对应的监督信号，让 latent 表示同时包含
       “什么市场结构”和“采取什么行为模式”。训练目标使用 coarse pair，而不是
       validation report 中的完整 motif 字符串，避免 K=10 的 codebook 被几十个
       细粒度 pair 类别拉扯。
    3. codebook diversity regularization:
       约束 codebook embedding 不要过度相似，减少多个 code 吸收同一类 latent。
    4. prototype diversity regularization:
       约束不同 code 通过 decoder 表达出的 action prototype 不要过度相似，直接
       对应 duplicate_code_pair_count 的行为重复风险。

数据边界:
    辅助标签从 train split 的原始 horizon prices 和 DP teacher actions 预计算。
    训练 batch 通过 ``TensorDataset`` 第四列 stable ``sample_ids`` 回查标签，避免
    因 dataloader 顺序或 batch 切分导致 price/action/label 错配。

梯度边界:
    - morphology/pair CE loss 通过 ``outputs.*_logits`` 回传到 auxiliary heads 和
      encoder latent ``z_e``。
    - codebook diversity 直接作用于 ``model.quantizer.embedding.weight``。
    - prototype diversity 通过 ``model.decoder`` 和 codebook embedding 计算不同
      code 的 decoded action distribution，相比单纯 embedding 距离更贴近行为语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ...model.data_types import HorizonDataset, TrajectoryDataset
from ...model.tensor_data_types import TrajectoryTensorBatch
from ...model.vq_archetype import ArchetypeVQModel, VqModelOutputs
from ..evaluators.phase1_validation_layers.layer2_behavior_quality import (
    classify_action_motif,
    classify_market_morphology,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class Phase1LossConfig:
    """训练期辅助目标权重。

    字段说明:
        morphology_aux_weight:
            ``z_e -> morphology`` 分类 loss 权重。它提供较粗粒度的市场结构监督。
        pair_aux_weight:
            ``z_e -> coarse morphology-motif pair`` 分类 loss 权重。该项直接针对
            weak_pair_code_ratio，默认高于 morphology 权重。这里的 pair 是训练用
            粗粒度 pair，不是 report 中的完整 morphology-motif 诊断字符串。
        codebook_diversity_weight:
            codebook embedding 相似度惩罚权重。该项较轻，主要防止 embedding
            层面的重复和塌缩。
        prototype_diversity_weight:
            decoded action prototype 相似度惩罚权重。它比 codebook diversity 更贴近
            duplicate_code_pair_count，因为它比较的是 code 解码后的行为分布。
        codebook_diversity_margin / prototype_diversity_margin:
            cosine similarity 超过 margin 后才惩罚，避免强行把本来合理接近的 code
            过度推开。
        prototype_diversity_ref_samples:
            每个 batch 中用于构造 code prototype 的 state 样本数。该计算需要把
            ``ref_samples * K`` 个组合送入 decoder，数值过大会增加训练开销。
    """

    morphology_aux_weight: float = 0.10
    pair_aux_weight: float = 0.20
    codebook_diversity_weight: float = 0.001
    prototype_diversity_weight: float = 0.01
    codebook_diversity_margin: float = 0.80
    prototype_diversity_margin: float = 0.85
    prototype_diversity_ref_samples: int = 64


@dataclass(frozen=True)
class Phase1AuxiliaryLabels:
    """一个 batch 的辅助监督标签。

    ``morphology`` 和 ``pair`` 都是 ``LongTensor[batch]``。``pair`` 使用 coarse
    morphology + coarse motif 标签。若某个标签不在当前 vocab 中，会被编码为
    ``IGNORE_INDEX``，对应样本在 CE loss 中被跳过。
    """

    morphology: torch.Tensor
    pair: torch.Tensor

    def to(self, device: torch.device | str) -> "Phase1AuxiliaryLabels":
        return Phase1AuxiliaryLabels(
            morphology=self.morphology.to(device),
            pair=self.pair.to(device),
        )


@dataclass(frozen=True)
class Phase1AuxiliaryLabelStore:
    """按 stable sample id 索引的 Phase I 辅助标签。

    训练 dataloader 来自 ``build_trajectory_tensor_dataset()``，第四列是稳定的
    ``sample_ids``。本 store 保存完整 train split 的 label tensor，并在每个
    batch 中按 ``sample_ids`` 回查对应标签。这样可以保持标签生成逻辑和模型训练
    逻辑解耦，同时避免把 prices 塞进模型 batch。
    """

    morphology_labels: torch.Tensor
    pair_labels: torch.Tensor
    morphology_vocab: tuple[str, ...]
    pair_vocab: tuple[str, ...]

    def get(
        self,
        sample_ids: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> Phase1AuxiliaryLabels:
        indices = sample_ids.detach().cpu().long()
        return Phase1AuxiliaryLabels(
            morphology=self.morphology_labels.index_select(0, indices).to(device),
            pair=self.pair_labels.index_select(0, indices).to(device),
        )


@dataclass(frozen=True)
class Phase1VQLossBreakdown:
    """VQ 训练目标的分项 loss。

    ``base_loss`` 对应模型原始 ``outputs.total_loss``，即 reconstruction + VQ。
    ``total_loss`` 是 trainer 实际反向传播的组合目标。保留完整 breakdown 是为了
    训练日志能观察每个辅助项是否处于合理量级，避免 pair loss 或 diversity loss
    无声主导训练。
    """

    base_loss: torch.Tensor
    morphology_loss: torch.Tensor
    pair_loss: torch.Tensor
    codebook_diversity_loss: torch.Tensor
    prototype_diversity_loss: torch.Tensor
    total_loss: torch.Tensor


def build_phase1_auxiliary_label_store(
    *,
    trajectory_dataset: TrajectoryDataset,
    horizon_dataset: HorizonDataset,
    morphology_vocab: Sequence[str] | None = None,
    pair_vocab: Sequence[str] | None = None,
) -> Phase1AuxiliaryLabelStore:
    """从 train split 构建 morphology 和 coarse morphology-motif pair 标签。

    输入:
        trajectory_dataset:
            DP teacher 生成的 ``(states, actions, rewards)`` 序列。这里读取 teacher
            actions 来生成 motif 标签，再粗化成训练用 coarse motif；不读取模型
            decoded actions，保证监督目标是固定的。
        horizon_dataset:
            对应的 ``(states, prices, depthprices)``。这里读取 prices 来生成
            market morphology，并用 horizon states 与 trajectory states 做严格对齐。
        morphology_vocab / pair_vocab:
            可选固定词表。``pair_vocab`` 是 coarse pair vocab。训练恢复或复现实验
            时可以传入 checkpoint config 中保存的 vocab，保证 auxiliary head 输出
            维度和 label id 稳定。

    输出:
        ``Phase1AuxiliaryLabelStore``，包含完整 train split 的 label tensor 与 vocab。

    失败策略:
        如果 trajectory 与 horizon 的样本数、shape 或 states 数值对不上，直接
        抛出 ``ValueError``。辅助标签错位会污染训练目标，不能静默容错。
    """

    if not trajectory_dataset:
        raise ValueError("trajectory_dataset must not be empty")
    horizon_states, prices, *_ = horizon_dataset
    if len(trajectory_dataset) != int(np.asarray(prices).shape[0]):
        raise ValueError("trajectory_dataset and horizon_dataset must have same length")
    trajectory_states = np.stack([trajectory[0] for trajectory in trajectory_dataset])
    horizon_state_values = np.asarray(horizon_states)
    if trajectory_states.shape != horizon_state_values.shape:
        raise ValueError(
            "trajectory states and horizon states are not shape-aligned, "
            f"got {trajectory_states.shape} and {horizon_state_values.shape}"
        )
    if not np.allclose(trajectory_states, horizon_state_values, rtol=1e-5, atol=1e-6):
        raise ValueError("trajectory states do not align with horizon states")

    actions = np.stack([trajectory[1] for trajectory in trajectory_dataset])
    morphologies = classify_market_morphology(prices)
    if morphologies.shape != (len(trajectory_dataset),):
        raise ValueError("morphology labels must match trajectory count")
    motifs = classify_action_motif(actions, prices)
    pairs = np.asarray(
        [
            f"{_coarse_morphology(str(morphology))}:{_coarse_motif(str(motif))}"
            for morphology, motif in zip(morphologies, motifs, strict=False)
        ],
        dtype=object,
    )

    morphology_vocab_tuple = _build_vocab(morphologies, morphology_vocab)
    pair_vocab_tuple = _build_vocab(pairs, pair_vocab)
    return Phase1AuxiliaryLabelStore(
        morphology_labels=_encode_labels(morphologies, morphology_vocab_tuple),
        pair_labels=_encode_labels(pairs, pair_vocab_tuple),
        morphology_vocab=morphology_vocab_tuple,
        pair_vocab=pair_vocab_tuple,
    )


def compute_phase1_vq_training_loss(
    *,
    model: ArchetypeVQModel,
    outputs: VqModelOutputs,
    batch: TrajectoryTensorBatch,
    aux_labels: Phase1AuxiliaryLabels | None,
    config: Phase1LossConfig,
) -> Phase1VQLossBreakdown:
    """组合基础 VQ loss、辅助监督 loss 和 diversity regularization。

    组合公式:
        ``total = base + w_morph * L_morph + w_pair * L_pair
        + w_cb_div * L_cb_div + w_proto_div * L_proto_div``

    其中:
        ``base``:
            模型原始 loss，包含 action reconstruction 和 VQ codebook/commitment。
        ``L_morph``:
            ``outputs.morphology_logits`` 对 train morphology label 的 CE loss。
        ``L_pair``:
            ``outputs.pair_logits`` 对 train coarse morphology-motif pair label 的
            CE loss。
        ``L_cb_div``:
            codebook embedding 的 off-diagonal cosine similarity margin penalty。
        ``L_proto_div``:
            不同 code 在同一批参考 states 上解码得到的 action distribution
            prototype 相似度惩罚。

    设计细节:
        diversity loss 只有在对应权重大于 0 时才计算，避免预训练或 ablation 中
        产生不必要的 decoder/codebook 额外前向开销。
    """

    base_loss = outputs.total_loss
    zero = base_loss.new_zeros(())
    morphology_loss = zero
    pair_loss = zero
    if aux_labels is not None:
        morphology_loss = _masked_cross_entropy(
            outputs.morphology_logits,
            aux_labels.morphology,
            fallback=zero,
        )
        pair_loss = _masked_cross_entropy(
            outputs.pair_logits,
            aux_labels.pair,
            fallback=zero,
        )

    codebook_diversity = (
        _codebook_diversity_loss(
            model,
            margin=config.codebook_diversity_margin,
        )
        if config.codebook_diversity_weight > 0.0
        else zero
    )
    prototype_diversity = (
        _prototype_diversity_loss(
            model,
            batch=batch,
            margin=config.prototype_diversity_margin,
            ref_samples=config.prototype_diversity_ref_samples,
        )
        if config.prototype_diversity_weight > 0.0
        else zero
    )
    total_loss = (
        base_loss
        + config.morphology_aux_weight * morphology_loss
        + config.pair_aux_weight * pair_loss
        + config.codebook_diversity_weight * codebook_diversity
        + config.prototype_diversity_weight * prototype_diversity
    )
    return Phase1VQLossBreakdown(
        base_loss=base_loss,
        morphology_loss=morphology_loss,
        pair_loss=pair_loss,
        codebook_diversity_loss=codebook_diversity,
        prototype_diversity_loss=prototype_diversity,
        total_loss=total_loss,
    )


def _build_vocab(
    labels: np.ndarray,
    vocab: Sequence[str] | None,
) -> tuple[str, ...]:
    """返回稳定排序的标签词表，或复用调用方提供的固定词表。"""

    if vocab is not None and len(vocab) > 0:
        return tuple(str(label) for label in vocab)
    return tuple(sorted(str(label) for label in np.unique(labels)))


def _encode_labels(labels: np.ndarray, vocab: Sequence[str]) -> torch.Tensor:
    """把字符串标签编码为 CE target id。

    不在 vocab 内的标签编码为 ``IGNORE_INDEX``。这允许使用旧 checkpoint vocab
    恢复训练时跳过新出现的低频标签，而不是改变 head 维度。
    """

    label_to_id = {str(label): index for index, label in enumerate(vocab)}
    encoded = [label_to_id.get(str(label), IGNORE_INDEX) for label in labels]
    return torch.as_tensor(encoded, dtype=torch.long)


def _coarse_morphology(label: str) -> str:
    """把 validation morphology 标签映射成训练用粗粒度市场结构。

    粗化目标:
        降低 pair auxiliary head 的类别数，让 K=10 的 codebook 先学习稳定的大类
        绑定。完整 morphology 仍会在 validation report 中保留，不影响最终诊断。
    """

    if label in {"uptrend", "downtrend"}:
        return "trend"
    if label in {"reversal-up", "reversal-down"}:
        return "reversal"
    if label in {"range-high-vol", "range-low-vol"}:
        return "range"
    if label == "volatile-mixed":
        return "volatile"
    return "neutral"


def _coarse_motif(label: str) -> str:
    """把完整 action motif 字符串映射成训练用粗粒度行为模式。

    完整 motif 形如 ``long + middle + delayed-hold + against-recent-move``。
    训练辅助目标只保留方向和主要持仓风格，丢弃 entry bucket 和 recent-move
    细节，避免 pair vocab 过碎。
    """

    parts = [part.strip() for part in label.split("+")]
    direction = parts[0] if parts else "mixed"
    entry = parts[1] if len(parts) > 1 else "none"
    holding = parts[2] if len(parts) > 2 else "unknown"

    if direction == "flat":
        return "flat"
    if holding == "switching":
        return f"{direction}-switch"
    if holding == "brief-trade":
        return f"{direction}-brief"
    if holding == "delayed-hold":
        return f"{direction}-delayed"
    if holding == "hold":
        return f"{direction}-hold"
    return f"{direction}-{entry}"


def _masked_cross_entropy(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """带 ``IGNORE_INDEX`` 支持的分类 loss。"""

    if logits is None:
        return fallback
    mask = targets != IGNORE_INDEX
    if not bool(torch.any(mask)):
        return fallback
    return F.cross_entropy(logits[mask], targets[mask].long())


def _codebook_diversity_loss(
    model: ArchetypeVQModel,
    *,
    margin: float,
) -> torch.Tensor:
    """计算 codebook embedding 层面的相似度惩罚。"""

    embeddings = model.quantizer.embedding.weight
    return _off_diagonal_similarity_penalty(embeddings, margin=margin)


def _prototype_diversity_loss(
    model: ArchetypeVQModel,
    *,
    batch: TrajectoryTensorBatch,
    margin: float,
    ref_samples: int,
) -> torch.Tensor:
    """计算 decoder 行为原型层面的相似度惩罚。

    步骤:
        1. 从当前 batch 取前 ``ref_samples`` 条 state sequence。
        2. 将每条 state sequence 与每个 codebook embedding 组合，形成
           ``ref_samples * K`` 个 decoder 输入。
        3. 对 decoder logits 做 softmax，得到每个 code 在参考 states 上的
           action distribution。
        4. 对参考样本维度求均值，得到 ``prototype[K, H * action_dim]``。
        5. 对 prototype 的 off-diagonal cosine similarity 做 margin penalty。

    这个 loss 惩罚的是“两个 code 解码出来的动作分布太像”，因此比单纯拉开
    embedding 更直接地缓解 duplicate_code_pair_count。
    """

    states, _, _ = batch
    if ref_samples <= 0 or states.shape[0] == 0:
        return states.new_zeros(())
    codebook = model.quantizer.embedding.weight
    num_codes = int(codebook.shape[0])
    if num_codes < 2:
        return codebook.new_zeros(())

    sample_count = min(int(ref_samples), int(states.shape[0]))
    states_ref = states[:sample_count].float()
    _, horizon, state_dim = states_ref.shape
    latent_dim = int(codebook.shape[-1])
    states_tiled = (
        states_ref.unsqueeze(1)
        .expand(sample_count, num_codes, horizon, state_dim)
        .reshape(sample_count * num_codes, horizon, state_dim)
    )
    codes_tiled = (
        codebook.unsqueeze(0)
        .expand(sample_count, num_codes, latent_dim)
        .reshape(sample_count * num_codes, latent_dim)
    )
    logits = model.decoder(states_tiled, codes_tiled)
    action_probs = F.softmax(logits, dim=-1)
    prototypes = action_probs.reshape(
        sample_count,
        num_codes,
        horizon,
        model.action_dim,
    ).mean(dim=0)
    prototypes = prototypes.flatten(start_dim=1)
    return _off_diagonal_similarity_penalty(prototypes, margin=margin)


def _off_diagonal_similarity_penalty(
    values: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """对矩阵行向量的非对角 cosine similarity 做 margin penalty。

    若两个 code/prototype 的相似度低于 ``margin``，不产生惩罚；超过 margin 的
    部分按平方惩罚。这样可以减少重复 code，同时保留相近但仍有细微区别的有效
    archetype。
    """

    if values.shape[0] < 2:
        return values.new_zeros(())
    normalized = F.normalize(values, dim=-1)
    similarities = normalized @ normalized.T
    eye = torch.eye(
        similarities.shape[0],
        dtype=torch.bool,
        device=similarities.device,
    )
    off_diagonal = similarities.masked_select(~eye)
    return F.relu(off_diagonal - margin).pow(2).mean()


__all__ = [
    "Phase1AuxiliaryLabelStore",
    "Phase1AuxiliaryLabels",
    "Phase1LossConfig",
    "Phase1VQLossBreakdown",
    "build_phase1_auxiliary_label_store",
    "compute_phase1_vq_training_loss",
]
