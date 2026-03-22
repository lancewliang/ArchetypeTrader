# ArchetypeTrader

基于 PyTorch 的三阶段强化学习加密货币交易框架，源自提交至 **AAAI 2026** 的论文。系统通过动态规划和向量量化从历史数据中发现可复用的交易原型，再通过分层 RL 智能体部署到实时交易中。

> ⚠️ **声明：** 本代码尚未经过真实交易数据的验证测试，无法确保实现的正确性。请关注后续更新。

> **论文：** *ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading* [[PDF]](AAAI26_ArchetypeTrader.pdf)
> Chuqiao Zong, Molei Qin, Haochong Xia, Bo An — Nanyang Technological University, Singapore
>
> 本代码库由上述研究论文（AAAI 2026）生成。代码注释中引用了论文的具体章节、公式和算法，便于追溯。对于论文描述模糊或缺少实现细节的部分，代码中标注了 `[NOTE]` 注释。

## 概述

ArchetypeTrader 处理 10 分钟级别的加密货币数据（BTC/ETH/DOT/BNB vs USDT），包含 25 级限价订单簿（LOB）信息。系统遵循三阶段流水线：

1. **Phase I — 原型发现**：动态规划规划器（Algorithm 1）在单次交易约束下生成最优示范轨迹。VQ 编码器-解码器将这些轨迹压缩为 K=10 个离散交易原型，存储在可学习码本中。

2. **Phase II — 原型选择**：Horizon 级别 RL 智能体在每个 72 步交易周期开始时选择最佳原型。冻结的解码器根据选定的原型码逐步生成微动作。

3. **Phase III — 原型精炼**：Step 级别 RL 智能体使用遗憾感知奖励信号对选定原型的动作进行微调，每个 horizon 最多一次调整。自适应层归一化（AdaLN）根据市场上下文条件化智能体。

```
历史数据 → 特征管道 → DP 规划器 → 3 万条轨迹
    → VQ 编码器-解码器 (Phase I) → 码本 (K=10 个原型)
    → 选择智能体 (Phase II) → 冻结解码器 → 微动作
    → 精炼智能体 (Phase III) → 最终交易动作
    → 评估引擎 (TR, Sharpe, Calmar, Sortino, MDD, Volatility)
```

## 项目结构

```
ArchetypeTrader/
├── data/feature_list/          # 市场特征数据 (.npy)
│   ├── single_features.npy     # 36 维单步特征（LOB、OHLCV、技术指标）
│   └── trend_features.npy      # 9 维 60 期趋势指标
├── src/
│   ├── config.py               # 全局超参数（dataclass + 命令行覆盖）
│   ├── data/                   # 特征管道 & PyTorch Dataset
│   ├── env/                    # MDP 交易环境
│   ├── phase1/                 # DP 规划器、VQ 编码器、解码器、码本
│   ├── phase2/                 # 选择智能体（horizon 级别 RL）
│   ├── phase3/                 # 精炼智能体、策略适配器、AdaLN
│   ├── evaluation/             # 评估指标引擎（TR/AVOL/MDD/ASR/ACR/ASoR）
│   └── utils/                  # 日志工具
├── scripts/
│   ├── train_phase1.py         # Phase I：DP 轨迹生成 + VQ 训练
│   ├── train_phase2.py         # Phase II：选择智能体训练
│   ├── train_phase3.py         # Phase III：精炼智能体训练
│   └── evaluate.py             # 三阶段完整评估
├── tests/                      # 单元测试 + 属性测试（共 278 个测试）
└── result/                     # 产物：轨迹、检查点、评估结果
```

## 环境配置

```bash
conda create -n ArchetypeTrade python=3.12
conda activate ArchetypeTrade
pip install torch numpy pytest hypothesis
```

## 使用方法

训练按顺序执行，每个阶段依赖前一阶段的产物：

```bash
# Phase I：生成 DP 轨迹 + 训练 VQ 编码器-解码器
python scripts/train_phase1.py --pair BTC

# Phase II：训练原型选择智能体
python scripts/train_phase2.py --pair BTC

# Phase III：训练精炼智能体（遗憾感知奖励）
python scripts/train_phase3.py --pair BTC --beta1 0.5

# 在测试集上评估（2024-01-01 至 2024-09-01）
python scripts/evaluate.py --pair BTC
```

主要命令行参数（均为可选，默认值见 `src/config.py`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pair` | 全部 4 个交易对 | 交易对（BTC/ETH/DOT/BNB） |
| `--horizon` | 72 | 每个交易周期的步数 |
| `--num-trajectories` | 30000 | DP 示范轨迹数量 |
| `--phase1-epochs` | 100 | VQ 编码器-解码器训练轮数 |
| `--phase2-total-steps` | 3000000 | 选择智能体训练步数 |
| `--phase3-total-steps` | 1000000 | 精炼智能体训练步数 |
| `--beta1` | 0.5 | 遗憾系数 β₁ ∈ {0.3, 0.5, 0.7} |
| `--lr` | 3e-4 | 学习率 |
| `--batch-size` | 256 | 批量大小 |

## 测试

测试套件包含 278 个测试：覆盖所有组件的单元测试，以及 23 个使用 [Hypothesis](https://hypothesis.readthedocs.io/) 的属性测试（每个属性 100 次迭代）。

```bash
python -m pytest tests/ -v
```

属性测试验证的形式化正确性属性包括：
- 特征维度不变量与拼接内容保持
- 持仓状态不变量（P_t ∈ {-m, 0, m}）
- 奖励计算公式正确性（Eq. 1）
- DP 单次交易约束与最优性（小规模暴力枚举验证）
- VQ 最近邻量化正确性
- 每 horizon 最多一次精炼调整
- 评估指标公式（TR、AVOL、MDD、ASR、ACR、ASoR）

## 核心超参数

| 参数 | 值 | 论文引用 |
|---|---|---|
| 状态维度 | 45（36 单步 + 9 趋势） | Section 3.1 |
| 动作空间 | {short, flat, long} | Section 3.1 |
| Horizon h | 72 步 | Section 3.1 |
| 佣金率 δ | 0.02% | Section 3.1 |
| 码本大小 K | 10 个原型 | Section 4.1 |
| 潜在维度 | 16 | Section 4.1 |
| LSTM 隐藏维度 | 128 | Section 4.1 |
| VQ 承诺系数 β₀ | 0.25 | Section 4.1 |
| KL 惩罚 α | 1.0 | Section 4.2 |
| 遗憾系数 β₁ | {0.3, 0.5, 0.7} | Section 4.3 |
| 年化因子 m | 52560 | Section 5 |

## 支持的交易对

| 交易对 | 最大持仓量 (m) |
|---|---|
| BTC/USDT | 8 |
| ETH/USDT | 100 |
| DOT/USDT | 2500 |
| BNB/USDT | 200 |

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{zong2026archetypetrader,
  title     = {ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading},
  author    = {Zong, Chuqiao and Qin, Molei and Xia, Haochong and An, Bo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

<!-- TODO: 论文正式发表后在此添加 arXiv / 会议链接 -->

## 许可

本项目为学术研究用途的实现。
