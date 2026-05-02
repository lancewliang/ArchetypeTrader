"""向量量化模块.

设计文档锚点: §6.2 与 §6.5。

支持:
- init: ``random_normal`` / ``sample_encoder_outputs`` / ``kmeans_warmup``
- update: ``gradient`` (论文公式) / ``ema`` (工程稳定项)
- dead-code restart: 默认开启，从高重构误差样本拉取 encoder 输出重置 dead code
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from src.config.phase1_config import CodebookConfig

_EPS = 1e-6


def _no_grad():
    if torch is None:  # pragma: no cover
        def decorator(fn):
            return fn
        return decorator
    return torch.no_grad()


@dataclass
class CodeUsageStats:
    counts: List[int] = field(default_factory=list)
    code_usage_ratio: float = 0.0
    perplexity: float = 0.0
    dominant_code_ratio: float = 0.0
    dead_codes: List[int] = field(default_factory=list)
    dead_code_count: int = 0


@dataclass
class QuantizeOutput:
    z_q: "torch.Tensor"           # 含 STE
    code_id: "torch.Tensor"
    z_q_no_grad: "torch.Tensor"   # 不含 STE，便于 EMA / loss
    encodings_one_hot: "torch.Tensor"


class VectorQuantizer(nn.Module if nn is not None else object):  # type: ignore[misc]
    """VQ-VAE 风格 codebook。

    生命周期
    --------
    1. ``__init__`` 时按 ``init_method`` 初始化 codebook；``kmeans_warmup`` /
       ``sample_encoder_outputs`` 等到首次 ``warmup_initialize`` 才完成初始化。
    2. ``quantize(z_e)`` 返回最近邻 code 与 STE 张量。
    3. ``update_codebook(z_e, code_id)``: ``ema`` 模式下更新内部 buffer；
       ``gradient`` 模式直接由 loss 反传，update 仅记录 usage。
    4. ``restart_dead_codes(...)``: 从高重构误差样本重置长期未使用的 code。
    5. ``usage_stats(code_id)`` 用于汇总 epoch metrics。

    实现注意
    --------
    - codebook 始终保留为 ``nn.Parameter``：``gradient`` 模式 ``requires_grad=True``，
      ``ema`` 模式 ``requires_grad=False`` 但仍参与 quantize 矩阵乘法。
    - EMA buffer (``_ema_count`` / ``_ema_weight``) 由 ``register_buffer`` 注册，
      跟随模型一起 ``state_dict()``。
    """

    def __init__(self, num_codes: int, code_dim: int, config: CodebookConfig) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("VectorQuantizer 需要 torch")
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.config = config
        # codebook 既可作为 Parameter（gradient 模式）也可作为 buffer（ema 模式）。
        # 我们统一用 Parameter；ema 模式下在 update_codebook 中手动 in-place 更新。
        self.codebook = nn.Parameter(
            torch.randn(num_codes, code_dim) * 0.1,
            requires_grad=(config.update_method == "gradient"),
        )
        # EMA buffer：当 update_method=ema 时使用。
        self.register_buffer("_ema_count", torch.zeros(num_codes))
        self.register_buffer("_ema_weight", torch.zeros(num_codes, code_dim))
        self.register_buffer("_warmup_initialized", torch.tensor(False))

    # ---------- 初始化 ----------

    @_no_grad()
    def warmup_initialize(self, encoder_outputs: "torch.Tensor") -> None:
        """``kmeans_warmup`` / ``sample_encoder_outputs`` 初始化路径。

        - ``random_normal``: 直接标记已初始化、不做任何处理。
        - ``sample_encoder_outputs``: 从 batch 中随机抽 K 条 ``z_e`` 作为初始 code。
        - ``kmeans_warmup``: 跑简化 K-means++ 得到 K 个聚类中心。

        EMA 模式下，初始化结束后必须把 ``_ema_weight`` / ``_ema_count`` 同步到
        新的 codebook，避免后续 EMA 更新被旧 buffer 拉回。

        Parameters
        ----------
        encoder_outputs : ``[N, code_dim]``，来自首批 train batches；样本量
                          ``< num_codes`` 时回退到随机初始化以避免崩溃。
        """
        if self.config.init_method == "random_normal":
            self._warmup_initialized.fill_(True)
            return
        if encoder_outputs.shape[0] < self.num_codes:
            # 不够 K 个样本时回退到随机
            self._warmup_initialized.fill_(True)
            return
        if self.config.init_method == "sample_encoder_outputs":
            idx = torch.randperm(encoder_outputs.shape[0], device=encoder_outputs.device)[
                : self.num_codes
            ]
            self.codebook.data.copy_(encoder_outputs[idx])
        elif self.config.init_method == "kmeans_warmup":
            self.codebook.data.copy_(self._kmeans_init(encoder_outputs))
        else:
            raise ValueError(f"未知 init_method={self.config.init_method}")
        # 同步 EMA buffer 到当前 codebook
        if self.config.update_method == "ema":
            self._ema_weight.copy_(self.codebook.data)
            self._ema_count.fill_(1.0)
        self._warmup_initialized.fill_(True)

    def _kmeans_init(self, samples: "torch.Tensor", iters: int = 10) -> "torch.Tensor":
        """简化版 kmeans++ 初始化，适合少量 batch warmup。"""
        n, d = samples.shape
        device = samples.device
        # k-means++ 选 K 个中心
        centers = torch.empty(self.num_codes, d, device=device)
        first = torch.randint(0, n, (1,), device=device).item()
        centers[0] = samples[first]
        for k in range(1, self.num_codes):
            d2 = ((samples.unsqueeze(1) - centers[:k].unsqueeze(0)) ** 2).sum(-1).min(dim=1).values
            probs = d2 / max(d2.sum().item(), _EPS)
            idx = torch.multinomial(probs, 1).item()
            centers[k] = samples[idx]
        # 标准 kmeans 迭代
        for _ in range(iters):
            dists = ((samples.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
            assign = dists.argmin(dim=1)
            for k in range(self.num_codes):
                mask = assign == k
                if mask.any():
                    centers[k] = samples[mask].mean(dim=0)
        return centers

    # ---------- 量化 ----------

    def quantize(self, z_e: "torch.Tensor") -> QuantizeOutput:
        """对 ``z_e`` 做最近邻量化并应用 straight-through estimator。

        实现要点
        --------
        - 距离用展开式 ``||z_e||^2 - 2 z_e·e + ||e||^2``，矩阵化避免 Python for-loop。
        - STE: ``z_q_st = z_e + (z_q - z_e).detach()``，让 encoder 拿到 ``z_q`` 的
          梯度等价于直接对 ``z_e`` 求导。
        - 同时返回 ``z_q_no_grad``（不带 STE）与 ``encodings_one_hot``，
          供 commitment loss / EMA 更新 / KL 使用。

        Parameters
        ----------
        z_e : ``[B, code_dim]``。

        Returns
        -------
        QuantizeOutput
        """
        # 计算 ||z_e - e_i||^2 = ||z_e||^2 - 2 z_e · e + ||e||^2
        # 形状: [B, K]
        codebook = self.codebook
        z_norm = (z_e ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
        c_norm = (codebook ** 2).sum(dim=-1, keepdim=True).t()  # [1, K]
        cross = z_e @ codebook.t()  # [B, K]
        dists = z_norm + c_norm - 2 * cross
        code_id = dists.argmin(dim=-1)
        z_q_no_grad = codebook[code_id]
        # straight-through: 让 encoder 拿到的梯度等价于直接对 z_e 求导
        z_q = z_e + (z_q_no_grad - z_e).detach()
        encodings = torch.nn.functional.one_hot(code_id, num_classes=self.num_codes).float()
        return QuantizeOutput(
            z_q=z_q,
            code_id=code_id,
            z_q_no_grad=z_q_no_grad,
            encodings_one_hot=encodings,
        )

    # ---------- EMA 更新 ----------

    @_no_grad()
    def update_codebook(self, z_e: "torch.Tensor", code_id: "torch.Tensor") -> None:
        """``ema`` 模式更新 codebook embedding；``gradient`` 模式 no-op。

        EMA 公式:
        - ``N_i ← λ N_i + (1-λ) n_i``
        - ``m_i ← λ m_i + (1-λ) Σ z_e``
        - ``e_i ← m_i / (N_i + ε)``，含 Laplace 平滑避免 dead code 立即归零。

        实现注意
        --------
        - ``self.training=False`` 时直接返回，inference 不更新 codebook。
        - ``z_e`` 必须 ``detach`` 后传入（否则 EMA 会建反向图）；
          调用方在 trainer 主循环中已经确保。
        """
        if self.config.update_method != "ema":
            return  # gradient 模式由 optimizer 反传
        if not self.training:
            return  # eval 时不更新
        decay = self.config.ema_decay
        eps = self.config.ema_epsilon

        encodings = torch.nn.functional.one_hot(code_id, num_classes=self.num_codes).float()
        n_per_code = encodings.sum(dim=0)  # [K]
        z_e_f32 = z_e.float()
        weight_sum = encodings.t() @ z_e_f32

        self._ema_count.mul_(decay).add_(n_per_code, alpha=1 - decay)
        self._ema_weight.mul_(decay).add_(weight_sum, alpha=1 - decay)

        # Laplace smoothing
        n_total = self._ema_count.sum()
        smoothed = (self._ema_count + eps) / (n_total + self.num_codes * eps) * n_total
        new_codebook = self._ema_weight / smoothed.unsqueeze(-1).clamp_min(_EPS)
        self.codebook.data.copy_(new_codebook)

    # ---------- dead code restart ----------

    @_no_grad()
    def restart_dead_codes(
        self,
        encoder_outputs: "torch.Tensor",
        reconstruction_errors: "torch.Tensor",
        current_epoch: int,
    ) -> List[int]:
        """重启长期未使用的 code。

        - 当 ``health.dead_code_restart=False`` 时直接返回空列表。
        - 通过 ``_ema_count < 阈值`` 找出 dead；按 dead 数量从 ``reconstruction_errors``
          最高的样本中取 ``z_e`` 重置 codebook 与 EMA buffer。

        Returns
        -------
        list[int] : 本次被重启的 code id 列表；空列表表示无 dead 或样本量不足。

        Notes
        -----
        重启完成后，trainer 应在该 epoch 把 ``_dead_code_restart_triggered=True``
        放进 metrics，由 ``selection_policy`` 进入 cooldown，避免立即把刚扰动的
        checkpoint 选为 best。
        """
        if not self.config.health.dead_code_restart:
            return []
        # 找出 EMA count 低的 code（dead）。
        threshold = 0.1  # 经验阈值；可移到配置
        dead_mask = self._ema_count < threshold
        dead_ids = dead_mask.nonzero(as_tuple=False).flatten().tolist()
        if not dead_ids:
            return []
        # 在 reconstruction error 最高的样本中按 dead 数量取 top-K
        if encoder_outputs.shape[0] < len(dead_ids):
            return []
        topk = torch.topk(reconstruction_errors, k=len(dead_ids), largest=True).indices
        for code_id, sample_idx in zip(dead_ids, topk.tolist()):
            self.codebook.data[code_id].copy_(encoder_outputs[sample_idx])
            self._ema_weight[code_id].copy_(encoder_outputs[sample_idx])
            self._ema_count[code_id].fill_(1.0)
        return dead_ids

    # ---------- 统计 ----------

    @_no_grad()
    def usage_stats(self, code_id: "torch.Tensor") -> CodeUsageStats:
        """单 epoch 结束后统计 usage / perplexity / dominant ratio / dead codes。

        指标语义
        --------
        - ``code_usage_ratio`` : 出现过的 code 数 / K；selection_policy 的 codebook
          guardrail 默认要求 ≥ 0.7。
        - ``perplexity`` : ``exp(-Σ p log p)``；接近 1 表示崩塌到单 code。
        - ``dominant_code_ratio`` : 单 code 占比上限；> 0.5 视为占主导。
        - ``dead_codes`` : 0 次出现的 code id 列表，进入 dead-code restart 候选。
        """
        counts_tensor = torch.bincount(code_id.flatten(), minlength=self.num_codes)
        counts = counts_tensor.tolist()
        total = sum(counts) or 1
        probs = [c / total for c in counts]
        # perplexity = exp(-Σ p log p)，仅算非零项
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log(p)
        perplexity = math.exp(h)
        used = sum(1 for c in counts if c > 0)
        usage_ratio = used / self.num_codes
        dominant = max(probs)
        dead_codes = [i for i, c in enumerate(counts) if c == 0]
        return CodeUsageStats(
            counts=counts,
            code_usage_ratio=usage_ratio,
            perplexity=perplexity,
            dominant_code_ratio=dominant,
            dead_codes=dead_codes,
            dead_code_count=len(dead_codes),
        )
