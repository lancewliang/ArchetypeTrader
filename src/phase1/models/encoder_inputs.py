"""Encoder 输入适配 + reward normalization.

设计文档锚点: §6.1。
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from src.phase1.config import EncoderInputConfig

_EPS = 1e-6

_logger = logging.getLogger(__name__)


def _percentile(values: Sequence[float], p: float) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    d0 = sorted_vals[f]
    d1 = sorted_vals[c]
    return d0 + (d1 - d0) * (k - f)


@dataclass
class RewardNormalizerStats:
    method: Literal["train_reward_robust", "train_reward_standard"]
    center: float
    scale: float
    clip_value: float
    kurtosis: Optional[float]
    skew: Optional[float]
    fallback_reason: Optional[str] = None
    clip_ratio: Optional[float] = None  # fit 完毕后才赋值

    def to_dict(self) -> dict:
        return asdict(self)


def _kurtosis(values: Sequence[float]) -> float:
    """超额峰度（Pearson 定义减 3）；分母用 n-1 修正。"""
    n = len(values)
    if n < 4:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    if m2 < _EPS:
        return 0.0
    return m4 / (m2 ** 2) - 3.0


def _skew(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 < _EPS:
        return 0.0
    return m3 / (m2 ** 1.5)


def _median(values: Sequence[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2:
        return sorted_vals[n // 2]
    return 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])


def _mad(values: Sequence[float], center: float) -> float:
    return _median([abs(v - center) for v in values])


class RewardNormalizer:
    """train-only reward 归一化。

    用法::

        norm = RewardNormalizer(config)
        stats = norm.fit_train(train_rewards)        # 仅 train
        train_norm = norm.transform(train_rewards)
        val_norm = norm.transform(val_rewards)       # 复用 stats

    设计理由 (设计 §6.1):
    - 默认 ``train_reward_robust`` (median/MAD) + ``clip=8.0``。加密分钟级 reward
      在大行情段呈重尾分布，``standard`` 模式 σ 易被极端值拉大，把核心区间压扁；
      robust 对 outlier 不敏感，配合较宽 clip 只截断真正异常 tick。
    - 当 train reward 通过 kurtosis 检验接近正态时（``< fallback_to_standard_kurtosis_below``），
      自动回退 standard，并在 ``stats.fallback_reason`` 标注原因，便于审计。
    """

    def __init__(self, config: EncoderInputConfig) -> None:
        self.config = config
        self.stats: Optional[RewardNormalizerStats] = None
        self._fitted = False

    def fit_train(self, rewards) -> RewardNormalizerStats:
        """根据 train demonstration rewards 拟合。

        Steps
        -----
        1. 计算超额峰度 ``kurtosis``；若 ``< fallback_to_standard_kurtosis_below``
           且当前请求的是 robust → 自动回退 standard，记录 ``fallback_reason``。
        2. ``train_reward_robust``: ``center=median``, ``scale=1.4826*MAD``。
        3. ``train_reward_standard``: ``center=mean``, ``scale=std``。
        4. ``scale`` 过小时使用 ``epsilon`` 防除零。
        5. 计算 ``clip_ratio``: 多少 train 样本会被 ``clip_value`` 截断，写入 stats。

        Returns
        -------
        RewardNormalizerStats : 含 method/center/scale/clip_value/kurtosis/skew/
                                fallback_reason/clip_ratio。

        Raises
        ------
        ValueError : ``rewards`` 为空。
        """
        flat = [float(v) for v in rewards]
        if not flat:
            raise ValueError("train rewards 为空，无法拟合 normalizer")

        n = len(flat)
        mean_val = sum(flat) / n
        var_val = sum((v - mean_val) ** 2 for v in flat) / max(n - 1, 1)
        std_val = math.sqrt(max(var_val, 0.0))
        median_val = _median(flat)
        mad_val = _mad(flat, median_val)
        zero_count = sum(1 for v in flat if abs(v) < _EPS)
        positive_count = sum(1 for v in flat if v > _EPS)
        negative_count = sum(1 for v in flat if v < -_EPS)
        horizon_returns = []
        _curr = 0.0
        for v in flat:
            _curr += v
        horizon_returns.append(_curr)

        _logger.warning(
            "reward_normalizer_diagnostic 说明=原始 reward 分布诊断 "
            "n=%d mean=%.10f std=%.10f median=%.10f mad=%.10f "
            "min=%.10f max=%.10f "
            "p1=%.10f p5=%.10f p25=%.10f p75=%.10f p95=%.10f p99=%.10f "
            "zero_count=%d(%.4f) positive_count=%d(%.4f) negative_count=%d(%.4f) "
            "sum=%.10f "
            "scale_before_eps=%.10f eps=%.1e scale_hit_eps=%s",
            n, mean_val, std_val, median_val, mad_val,
            min(flat), max(flat),
            _percentile(flat, 1), _percentile(flat, 5), _percentile(flat, 25),
            _percentile(flat, 75), _percentile(flat, 95), _percentile(flat, 99),
            zero_count, zero_count / n,
            positive_count, positive_count / n,
            negative_count, negative_count / n,
            sum(flat),
            1.4826 * mad_val if median_val != 0 or mad_val > 0 else 0.0,
            _EPS,
            (1.4826 * mad_val) < _EPS if mad_val >= 0 else True,
        )

        kurt = _kurtosis(flat)
        skew = _skew(flat)
        method = self.config.reward_normalization
        fallback_reason: Optional[str] = None

        if (
            method == "train_reward_robust"
            and kurt < self.config.fallback_to_standard_kurtosis_below
        ):
            method = "train_reward_standard"
            fallback_reason = "kurtosis_below_threshold"

        if method == "train_reward_robust":
            center = _median(flat)
            mad = _mad(flat, center)
            scale = max(1.4826 * mad, _EPS)
        else:
            mean = sum(flat) / len(flat)
            var = sum((v - mean) ** 2 for v in flat) / max(len(flat) - 1, 1)
            std = math.sqrt(max(var, 0.0))
            center = mean
            scale = max(std, _EPS)

        clip_v = self.config.reward_clip_value
        # 计算 clip_ratio: 多少 train 样本被 clip
        clipped = sum(1 for v in flat if abs((v - center) / scale) > clip_v)
        clip_ratio = clipped / len(flat)

        self.stats = RewardNormalizerStats(
            method=method,
            center=center,
            scale=scale,
            clip_value=clip_v,
            kurtosis=kurt,
            skew=skew,
            fallback_reason=fallback_reason,
            clip_ratio=clip_ratio,
        )
        self._fitted = True
        return self.stats

    def transform(self, rewards):
        """对 rewards 执行 ``(x - center) / scale`` 后 clip 到 ``[-c, c]``。

        ``fit_train`` 必须先被调用过；否则抛 ``RuntimeError``。
        支持 ``torch.Tensor`` 与 Python iterable 两种输入。

        Notes
        -----
        val/test 严禁重新 fit；只能 transform。该约束由调用方保证
        （Trainer 在 train demonstration 上调一次 fit_train，然后所有 split 都用 transform）。
        """
        if not self._fitted or self.stats is None:
            raise RuntimeError("RewardNormalizer 未拟合 train rewards; 请先调用 fit_train")

        center = self.stats.center
        scale = self.stats.scale
        c = self.stats.clip_value

        if torch is not None and isinstance(rewards, torch.Tensor):
            normed = (rewards - center) / scale
            return normed.clamp(min=-c, max=c)
        # python list / iterable
        out = []
        for v in rewards:
            x = (float(v) - center) / scale
            out.append(max(-c, min(c, x)))
        return out

    def to_dict(self) -> dict:
        if self.stats is None:
            raise RuntimeError("尚未 fit_train")
        return self.stats.to_dict()


class EncoderInputAdapter(nn.Module if nn is not None else object):  # type: ignore[misc]
    """三路输入适配 + fusion。

    结构::

        state_t        -> Linear(d_state, state_adapter_dim)   -> LayerNorm -> GELU
        action_t       -> Embedding(3, action_embedding_dim)   -> LayerNorm
        reward_t       -> Linear(1, reward_embedding_dim)      -> LayerNorm -> GELU
        concat -> Linear(sum -> fusion_dim) -> LayerNorm

    输出 ``[batch, h, fusion_dim]``，作为 LSTM encoder 输入。

    为什么不能 raw concat: state/action/reward 量级差异巨大（state 已标准化、
    action 是 long、reward 已 robust normalize），各自走 adapter + LayerNorm
    避免某一模态主导 LSTM。
    """

    def __init__(self, feature_dim: int, config: EncoderInputConfig) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("EncoderInputAdapter 需要 torch")
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config
        # state adapter
        self.state_adapter = nn.Sequential(
            nn.Linear(feature_dim, config.state_adapter_dim),
            nn.LayerNorm(config.state_adapter_dim),
            nn.GELU(),
        )
        # action embedding (3 个动作: short/flat/long)
        self.action_emb = nn.Embedding(3, config.action_embedding_dim)
        self.action_norm = nn.LayerNorm(config.action_embedding_dim)
        # reward adapter
        self.reward_adapter = nn.Sequential(
            nn.Linear(1, config.reward_embedding_dim),
            nn.LayerNorm(config.reward_embedding_dim),
            nn.GELU(),
        )
        merged_dim = (
            config.state_adapter_dim
            + config.action_embedding_dim
            + config.reward_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(merged_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )

    def forward(self, states, actions, rewards):
        """前向计算。

        Parameters
        ----------
        states : Tensor[batch, h, feature_dim]
        actions : Tensor[batch, h] (long)
        rewards : Tensor[batch, h]，已经 RewardNormalizer.transform + clip。

        Returns
        -------
        Tensor[batch, h, fusion_dim]
        """
        state_emb = self.state_adapter(states)
        action_emb = self.action_norm(self.action_emb(actions))
        reward_emb = self.reward_adapter(rewards.unsqueeze(-1))
        merged = torch.cat([state_emb, action_emb, reward_emb], dim=-1)
        return self.fusion(merged)
