# ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading

# ArchetypeTrader：用于在量化交易中选择与细化可学习策略原型的强化学习

Chuqiao Zong, Molei Qin, Haochong Xia, Bo An
Nanyang Technological University, Singapore
{zong0005, molei001, haochong001}@e.ntu.edu.sg, boan@ntu.edu.sg

LaTeX+markdown format

*排版说明：每一节先英文原文，后中文译文；LaTeX 数学公式保持原样，便于学习与逐段校对。*

## Abstract / 摘要

**English**

Quantitative trading using mathematical models and automated execution to generate trading decisions has been
widely applied across financial markets. Recently, reinforcement learning (RL) has emerged as a promising approach for
developing profitable trading strategies, especially in highly volatile markets like cryptocurrency. However, existing RL methods for cryptocurrency trading face two critical drawbacks: 
1) Prior RL algorithms segment markets using handcrafted indicators (e.g., trend or volatility) to train specialized sub-policies. However, these coarse labels oversimplify market dynamics into rigid categories, biasing policies toward obvious patterns like trend-following and neglecting nuanced but lucrative opportunities. 
2) Current RL methods fail to systematically use demonstration data. While some approaches ignore demonstrations altogether, others rely on “optimal” yet overly granular trajectories or human-crafted strategies, both of which can overwhelm learning and introduce significant bias, resulting in high variance and large profit drawdowns.
To address these problems, we propose ArchetypeTrader, a novel reinforcement learning framework that automatically selects and refines data-driven trading archetypes distilled from demonstrations. The framework operates in three phases: 
1) We use dynamic programming (DP) to generate representative expert trajectories and train a vector-quantized (VQ) encoder-decoder architecture to distill these demonstrations into discrete, reusable strategic archetypes through self-supervised learning, capturing nuanced market-behavior patterns without human heuristics. 
2) We then train an RL agent to select contextually appropriate archetypes from the learned codebook and reconstruct action sequences for the upcoming horizons, effectively performing demonstrationguided strategy reuse. 
3) We finally train a policy adapter that leverages hindsight-informed rewards to dynamically refine the archetype actions based on real-time market observations and performance, enabling more fine-grained decisionmaking and yielding profitable and robust trading strategies. 
Extensive experiments on four popular cryptocurrency trading pairs demonstrate that ArchetypeTrader significantly outperforms state-of-the-art approaches in both profit generation and risk management.

**中文**

量化交易通过数学模型和自动化执行来生成交易决策，已被广泛应用于金融市场。近年来，强化学习（RL）已成为开发盈利性交易策略的一种有前景的方法，尤其适用于加密货币这类高波动市场。然而，现有面向加密货币交易的强化学习方法存在两个关键缺陷：  
1）以往的强化学习算法通常使用手工设计的指标（如趋势或波动率）对市场进行划分，并据此训练专门的子策略。然而，这类粗粒度标签将市场动态过度简化为僵硬的类别，使策略偏向于诸如趋势跟随等显性模式，而忽视了更细微但同样可获利的机会。  
2）现有方法未能系统性地利用示范数据。有些方法完全忽略示范数据，另一些则依赖“最优”但过于细粒度的轨迹，或依赖人工设计的策略；这两类做法都会干扰学习过程并引入显著偏差，导致较高方差和较大的收益回撤。  

为解决这些问题，我们提出 **ArchetypeTrader**，一个能够自动选择并细化由示范数据蒸馏得到的数据驱动型交易原型的全新强化学习框架。该框架分为三个阶段运行：  
1）我们使用动态规划（DP）生成具有代表性的专家轨迹，并训练一个向量量化（VQ）编码器—解码器架构，通过自监督学习将这些示范压缩为离散、可复用的策略原型，从而在无需人工启发式规则的情况下捕捉细致的市场行为模式。  
2）随后，我们训练一个强化学习智能体，使其能够从已学习的码本中选择与当前情境最匹配的原型，并重构未来若干时间步的动作序列，从而实现受示范引导的策略复用。  
3）最后，我们训练一个策略适配器，利用事后信息引导的奖励，根据实时市场观测和策略表现动态细化原型动作，从而实现更细粒度的决策，并获得更具盈利性且更稳健的交易策略。  

在四个主流加密货币交易对上的大量实验表明，ArchetypeTrader 在收益生成与风险控制两个方面都显著优于当前最先进的方法。

## Introduction / 引言

**English**

With a market capacity exceeding 90 trillion dollars, the global financial sector continues to draw a vast range of participants seeking steady gains and opportunistic profits.Over the past decade, quantitative trading has gained particular traction by leveraging increasingly sophisticated tools for data analysis and decision-making. Among these, reinforcement learning (RL) is promising because of its ability to process high-dimensional financial data and to address complex sequential decision-making problems (Deng et al.2016; Zhang, Zohren, and Stephen 2020; Liu et al. 2020a).In contrast to rule-based approaches that rely on human experts’ insights, RL allows the trading agent to learn adaptive policies directly from interactions with the market environment, which is especially effective in volatile markets.

Although RL has achieved great success in quantitative trading, existing algorithms exhibit several critical shortcomings, mainly including: 
1) Many existing methods treat the financial market as a homogeneous and stationary process (Briola et al. 2021; Jia et al. 2019). Consequently, the learned policies perform poorly on volatile instruments like cryptocurrencies, which experience frequent dynamic changes. Another common approach segments market conditions using human-designed indicators, such as market trends (e.g., bullish, bearish) or volatility, and trains specialized sub-policies for each regime (Qin et al. 2023; Zong et al. 2024). However, these coarse, human-engineered labels tend to oversimplify market dynamics and bias the sub-policies toward superficial trading behaviors like ”trendfollowing” while neglecting nuanced yet profitable trading opportunities, thereby limiting overall strategy effectiveness. 
2) Many RL-based trading methods fail to leverage demonstration data effectively. Some ignore demonstrations altogether (Zhu and Zhu 2022; Zou et al. 2024), while others adopt fully “optimal” yet overly granular trajectories (Qinet al. 2023; Zong et al. 2024) or rely on human-devised heuristic strategies (Liu et al. 2020b), both of which can overwhelm learning with noise or introduce bias. As a result, these approaches often exhibit high variance and suboptimal performance, having missed the opportunity to incorporate proven, high-return trading behaviors from the outset.

To address these challenges, we propose ArchetypeTrader, a novel RL framework learning to dynamically select and refine strategic archetypes, which are discrete and reusable trading strategies derived from demonstrations. Unlike existing approaches that rely on monolithic policies or human-designed market segmentation, ArchetypeTrader operates
through three tightly integrated phases: 
1) In the first phase, a dynamic programming (DP) planner generates demonstration trajectories over fixed horizons without predefined human heuristics. A vector-quantized encoder-decoder is then trained to compress these trajectories into a compact codebook of strategically meaningful trading archetypes. 
2) In the second phase, ArchetypeTrader trains an RL-driven archetype selector to pick the most suitable archetype for the current market regime at the start of each horizon. 
3) In the third phase, a step-wise policy adapter refines the selected archetype’s actions using the latest market observations and intra-horizon performance. Optimized not only for increasing total profit but also for minimizing regret that measures the gap to top hindsight opportunities within each horizon, our adapter is able to provides fine-grained, impactful adjustments without abandoning the archetype’s overall intent.

Our key contributions are three-fold: 
1) we introduce a self-supervised method to discover discrete trading archetypes from DP-generated demonstrations without human heuristics, which can be reused to provide reasonable trading actions; 
2) we design a two-layer control scheme: an RL selector that activates the right archetype per horizon, and a regret-aware adapter that performs high-impact, step-level refinements, which provides a profitable and robust trading strategy; 3) comprehensive experiments on 4 popular cryp-tocurrencies demonstrate that ArchetypeTrader significantly outperforms state-of-the-art baselines in both profit and risk management metrics.

**中文**

全球金融市场规模已超过 90 万亿美元，并持续吸引着大量参与者寻求稳定收益和机会性利润。在过去十年中，量化交易因能够借助日益复杂的数据分析与决策工具而获得了尤为广泛的关注。其中，强化学习（RL）之所以备受看好，是因为它能够处理高维金融数据，并解决复杂的序列决策问题（Deng et al. 2016; Zhang, Zohren, and Stephen 2020; Liu et al. 2020a）。与依赖人类专家经验的规则型方法不同，强化学习允许交易智能体通过与市场环境的交互直接学习自适应策略，这一点在高波动市场中尤为有效。

尽管强化学习在量化交易中取得了显著进展，现有算法仍存在若干关键缺陷，主要包括：  
1）许多现有方法将金融市场视为同质且平稳的过程（Briola et al. 2021; Jia et al. 2019）。因此，这些方法在加密货币等经常发生动态变化的高波动资产上表现较差。另一类常见方法是利用人工设计的指标（如牛市/熊市等市场趋势或波动率）对市场状态进行划分，并针对每一类状态训练专门的子策略（Qin et al. 2023; Zong et al. 2024）。然而，这类粗粒度、人工构造的标签往往会过度简化市场动态，并使子策略偏向于“趋势跟随”这类表层交易行为，而忽略更细腻但更有利可图的机会，从而限制整体策略效果。  
2）许多基于强化学习的交易方法未能有效利用示范数据。有些方法完全忽视示范数据（Zhu and Zhu 2022; Zou et al. 2024），另一些则采用完全“最优”但过于细粒度的轨迹（Qin et al. 2023; Zong et al. 2024），或依赖人工设计的启发式策略（Liu et al. 2020b）；这两类方式都会以噪声淹没学习过程或引入偏差。因此，这些方法常常表现出高方差和次优性能，错失了从一开始就纳入已被验证的高收益交易行为的机会。

为应对这些挑战，我们提出 **ArchetypeTrader**，一个学习如何动态选择并细化策略原型的全新强化学习框架。所谓策略原型，是从示范数据中提炼出的离散且可复用的交易策略。不同于依赖单一整体策略或人工市场划分的现有方法，ArchetypeTrader 由三个紧密耦合的阶段组成：  
1）在第一阶段，一个动态规划（DP）规划器在固定时间跨度上生成示范轨迹；随后，一个向量量化编码器—解码器被训练用来将这些轨迹压缩成一个紧凑的、具有策略意义的交易原型码本。  
2）在第二阶段，ArchetypeTrader 训练一个由强化学习驱动的原型选择器，使其在每个时间跨度开始时为当前市场状态选择最合适的原型。  
3）在第三阶段，一个逐步决策的策略适配器利用最新的市场观测和跨度内表现来细化所选原型的动作。该适配器不仅以提升总收益为目标，还以最小化后见信息下与最佳机会之间的遗憾为目标，因此能够在不偏离原型整体意图的前提下做出细粒度且有效的调整。

我们的主要贡献有三点：  
1）我们提出了一种无需人工启发式规则的自监督方法，可从 DP 生成的示范中发现离散的交易原型，并将其复用于提供合理的交易动作；  
2）我们设计了一个双层控制机制：在每个时间跨度上激活合适原型的强化学习选择器，以及一个进行高影响逐步细化的遗憾感知型适配器，从而形成兼具盈利性与稳健性的交易策略；  
3）在 4 个主流加密货币市场上的综合实验表明，ArchetypeTrader 在收益和风险管理指标上均显著优于最先进的基线方法。

## Related Work / 相关工作

**English**

In this section, we briefly introduce the existing quantitative trading methods, which are based on traditional financial technical analysis or reinforcement learning algorithms.

**中文**

本节简要介绍现有量化交易方法，包括基于传统金融技术分析的方法以及基于强化学习算法的方法。

### Traditional Financial Methods / 传统金融方法

**English**

Classic technical analysis assumes that recurring price-volume patterns foreshadow future moves (Mur-phy 1999) and has spawned a vast family of handcrafted indicators (Kakushadze 2016). Examples range from order-flow imbalance (Chordia, Roll, and Subrahmanyam 2002) capturing short-term market direction to momentum measures such as MACD (Hung 2016; Krug, Dobaj, and Macher 2022) reflecting potential trends. In highly non-stationary markets like cryptocurrency, however, such signals often become noisy and misleading (Liu et al.2020b; Qin et al. 2023; Li, Zheng, and Zheng 2019)

**中文**

经典技术分析认为，反复出现的价量模式能够预示未来走势（Murphy 1999），并由此催生了大量手工设计的指标（Kakushadze 2016）。例如，订单流失衡（Chordia, Roll, and Subrahmanyam 2002）用于刻画短期市场方向，MACD 等动量指标（Hung 2016; Krug, Dobaj, and Macher 2022）则用于反映潜在趋势。然而，在加密货币这类高度非平稳市场中，这些信号往往会变得嘈杂且具有误导性（Liu et al. 2020b; Qin et al. 2023; Li, Zheng, and Zheng 2019）。

### RL for Quantitative Trading / 面向量化交易的强化学习

**English**

Early work ports off-the-shelf algorithms such as DQN (Mnih et al. 2015) and PPO (Schulman et al. 2017) to financial markets, while subsequent variants improve stability or representation power. For example, CDQNRP (Zhu and Zhu 2022) adds random perturbations to steady DQN training, and CLSTM-PPO (Zou et al. 2024) augments PPO with an LSTM state encoder for high-frequency stock trading. However, these stationary policies struggle in the regime-switching, high-volatility markets (e.g., crypto).To capture longer-horizon dynamics, recent methods separate data with handcrafted trend/volatility indicators and apply hierarchical reinforcement learning (HRL) to achieve stable performance. EarnHFT (Qin et al. 2023) trains trend-specific agents and a router for agent selection in high-frequency cryptocurrency trading. MacroHFT (Zong et al. 2024) builds context-aware sub-policies and fuses them via a memory-augmented hyper-agent. Nevertheless, reliance on human labels biases behavior toward superficial “trend-following” and ignores latent opportunities, while the inability to leverage hindsight demonstrations often yields unstable, sub-optimal returns.

**中文**

早期工作将 DQN（Mnih et al. 2015）和 PPO（Schulman et al. 2017）等通用算法直接移植到金融市场，后续变体则进一步提升了稳定性或表示能力。例如，CDQNRP（Zhu and Zhu 2022）通过加入随机扰动来稳定 DQN 训练，CLSTM-PPO（Zou et al. 2024）则在 PPO 中加入 LSTM 状态编码器，以适应高频股票交易。然而，这类平稳策略难以适应会发生状态切换、且波动剧烈的市场（如加密货币）。为了捕捉更长时段的动态，近期方法使用手工构造的趋势/波动率指标对数据进行划分，并采用层次化强化学习（HRL）来获得更稳定的表现。EarnHFT（Qin et al. 2023）训练面向不同趋势的智能体，并借助路由器在高频加密货币交易中选择合适智能体。MacroHFT（Zong et al. 2024）则构建上下文感知子策略，并通过带记忆的超智能体进行融合。然而，对人工标签的依赖会使策略偏向表层的“趋势跟随”，并忽视潜在机会；同时，缺乏对事后示范的有效利用，也常常导致收益不稳定且非最优。

## Problem Formulation / 问题形式化

**English**

In this section, we first introduce the essential financial concepts and fundamental MDP formulation of cryptocurrency trading. We then position our task against current RL trading methods, pinpoint their core limitations, and motivate the pursuit of a more robust solution.

**中文**

本节首先介绍加密货币交易中的基本金融概念以及基础的 MDP 形式化定义。随后，我们将本文任务与现有强化学习交易方法进行对比，指出其核心局限，并说明为何需要寻求更稳健的解决方案。

### Financial Background & MDP Formulation / 金融背景与 MDP 形式化

##### Market observations: / 市场观测：

**English**

An $M$-level limit-order book: $b_t = \{(p_i^b, q_i^b), (p_i^a, q_i^a)\}_{i=1}^{M}$
The OHLCV bar: $x_t = (p_t^o, p_t^h, p_t^l, p_t^c, v_t)$
A set of technical indicators:$y_t = \psi(x_{t-w+1:t}, b_{t-w+1:t})$
computed over backward window $w$.

**中文**

一个 $M$ 档限价订单簿：$b_t = \{(p_i^b, q_i^b), (p_i^a, q_i^a)\}_{i=1}^{M}$  
OHLCV K 线：$x_t = (p_t^o, p_t^h, p_t^l, p_t^c, v_t)$  
一组技术指标：$y_t = \psi(x_{t-w+1:t}, b_{t-w+1:t})$  
其中指标基于回看窗口 $w$ 计算。

##### Position: / 持仓：

**English**

Asset amount:$P_t \in \{-m, 0, m\}$(short, flat, long).

**中文**

资产持有量：$P_t \in \{-m, 0, m\}$（分别对应做空、空仓、做多）。

##### Execution Loss: / 执行损失：

**English**

Executing position change:$\Delta P_t = P_{t+1} - P_t$
incurs:$O_t = C(|\Delta P_t|) - |\Delta P_t| \, p_t^{\text{mark}} + \delta |\Delta P_t| \, p_t^{\text{mark}}$

where:
- $C(\cdot)$ is the LOB fill cost  
- $p_t^{\text{mark}}$ is the mark price  
- $\delta$ is the commission rate

**中文**

执行持仓变化：$\Delta P_t = P_{t+1} - P_t$  
会产生：
$O_t = C(|\Delta P_t|) - |\Delta P_t| \, p_t^{\text{mark}} + \delta |\Delta P_t| \, p_t^{\text{mark}}$

其中：
- $C(\cdot)$ 表示订单簿成交成本  
- $p_t^{\text{mark}}$ 表示标记价格  
- $\delta$ 表示手续费率

##### Net Value: / 净值：

**English**

the sum of cash and the market value of cryptocurrency, calculated as $V_t = V_{ct} + P_t \cdot p_t^{\text{mark}}$ where $V_{ct}$ is the cash value. The purpose of quantitative trading is to maximize the final net value $V_t$ after trading a single asset for a time period.

We formulate cryptocurrency trading as an MDP $\langle S, A, T, R, \gamma \rangle$ where State $s_t \in S$  is the market observations, action $a_t \in A = \{0, 1, 2\}$ sets the target position $P_t = m(a_t - 1)$, transition $T$ follows the streamed market,reward $R$ is measured by net value difference $r_t^{\text{step}} = V_t - V_{t-1} = P_t \left(p_{t+1}^{\text{mark}} - p_t^{\text{mark}}\right) - O_t$ and $\gamma \in [0, 1)$ is the discount factor for future returns. A trading policy $\pi(a|s)$ specifies target positions given states. The objective is to find the optimal policy $\pi^*$ maximizing expected discounted return $J = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{+\infty} \gamma^t r_t^{\text{step}} \right]$

**中文**

净值是现金与加密货币市值之和，定义为 $V_t = V_{ct} + P_t \cdot p_t^{\text{mark}}$，其中 $V_{ct}$ 为现金价值。量化交易的目标是在单一资产的一段交易周期结束后，使最终净值 $V_t$ 最大化。

我们将加密货币交易建模为一个 MDP $\langle S, A, T, R, \gamma \rangle$，其中状态 $s_t \in S$ 表示市场观测，动作 $a_t \in A = \{0, 1, 2\}$ 用于设定目标持仓 $P_t = m(a_t - 1)$，转移函数 $T$ 由连续到来的市场流驱动，奖励 $R$ 由净值变化衡量，即 $r_t^{\text{step}} = V_t - V_{t-1} = P_t \left(p_{t+1}^{\text{mark}} - p_t^{\text{mark}}\right) - O_t$，$\gamma \in [0, 1)$ 为未来回报的折扣因子。交易策略 $\pi(a|s)$ 用于根据状态指定目标持仓。优化目标是找到最优策略 $\pi^*$，使期望折扣回报 $J = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{+\infty} \gamma^t r_t^{\text{step}} \right]$ 最大。

### Problem Statement / 问题陈述

**English**

Existing approaches (Qin et al. 2023; Zong et al. 2024) attempt to handle the non-stationary and high-volatility cryptocurrency market by assigning a high-level policy to select regime-specific sub-policies. However, they suffer from two hurdles: (i) human-engineered regime labels (e.g.,bullish/bearish) oversimplify dynamics and bias agents toward rigid trend-following; (ii) demonstrations are either ignored or used in ways that inject noise—fully “optimal” DP traces or handcrafted rules. A systematic, noise-filtered way to exploit demonstrations without human intervention therefore remains an open problem.

**中文**

现有方法（Qin et al. 2023; Zong et al. 2024）试图通过让高层策略选择面向不同市场状态的子策略，来应对非平稳且高波动的加密货币市场。然而，这些方法面临两个障碍：  
(i) 人工设计的状态标签（如牛市/熊市）会过度简化市场动态，并使智能体偏向僵化的趋势跟随；  
(ii) 示范数据要么被忽视，要么以会引入噪声的方式被使用——例如完整“最优”的 DP 轨迹或人工规则。  
因此，如何在无需人工干预的前提下，以系统化、去噪的方式利用示范数据，仍是一个开放问题。

## ArchetypeTrader / ArchetypeTrader

**English**

To overcome the pitfalls of human-engineered segmentation and underutilized demonstrations, we propose ArchetypeTrader, a hierarchical framework with three training phases illustrated in Fig. 1: 
1) In phase one, we design a DP planner to generate demonstration trajectories. A VQ-based encoder–decoder then compresses these trajectories into a discrete set of archetypes capturing key trading behaviors. 
2) In phase two, a horizon-level RL agent selects the most suitable archetype whose expert insights best match current market features. 
3) In phase three, a step-level policy adapter refines the archetype’s base actions in response to real-time market observations and interim performance, ensuring robustness and agility under volatile conditions.

**中文**

为克服人工市场划分与示范数据利用不足的问题，我们提出 **ArchetypeTrader**，一个分为三个训练阶段的层次化框架，如图 1 所示：  
1）在第一阶段，我们设计一个 DP 规划器来生成示范轨迹。随后，一个基于 VQ 的编码器—解码器将这些轨迹压缩为一组能够捕捉关键交易行为的离散原型。  
2）在第二阶段，一个时间跨度级别的强化学习智能体选择最适合当前市场特征、且最能体现专家经验的原型。  
3）在第三阶段，一个逐步决策的策略适配器根据实时市场观测与中间表现对原型的基础动作进行细化，从而在高波动环境下确保策略的稳健性与灵活性。

### Archetype Discovery / 原型发现

**English**

We propose a self-supervised learning pipeline that extracts compact and reusable trading archetypes directly from high-quality trading trajectories, eliminating reliance on human design or heuristics. Concretely, we first sample n data chunks of fixed length h from the training dataset and apply a DP planner (Algorithm 1) to identify profitable trading actions for each chunk. Unlike prior approaches that attempt to capitalize on every profitable fluctuation (Qin et al. 2023; Zong et al. 2024), we deliberately limit each data chunk to a single trade to emphasize the most significant and impactful opportunities within each horizon. By capturing only the primary movements, our demonstration trajectories filter out small, short-lived fluctuations, thereby reducing noise, sim-plifying the subsequent learning, and providing a clean foundation for coherent and reusable trading archetypes.

**中文**

我们提出一个自监督学习流程，直接从高质量交易轨迹中提取紧凑且可复用的交易原型，从而摆脱对人工设计或启发式规则的依赖。具体而言，我们首先从训练集采样 $n$ 个固定长度为 $h$ 的数据片段，并对每个片段应用一个 DP 规划器（算法 1），以识别可盈利的交易动作。不同于以往尝试捕捉每一次可获利波动的方法（Qin et al. 2023; Zong et al. 2024），我们刻意将每个数据片段限制为只包含一次交易，以突出每个时间跨度内最重要、最具影响力的机会。通过只保留主要运动，我们得到的示范轨迹过滤掉了细小且短暂的波动，从而降低噪声、简化后续学习过程，并为形成连贯且可复用的交易原型提供清晰基础。

#### Algorithm 1: Single-trade DP planner / 算法 1：单次交易 DP 规划器

**English**

---

**Input:** price series $P$ of length $N$, action set $\mathcal{A}$  
**Output:** $\{\hat{a}_t\}_{0}^{N-1}$

--- 

1:  $V[N+1, |\mathcal{A}|, 2] \leftarrow 0, \quad \Pi[N, |\mathcal{A}|, 2] \leftarrow -1 $
2:  for $t$ = $N$-1 downto 0 do
3:      for $i \in \mathcal{A}, c \in \{0,1\}$ do
4:          $\Pi[t,i,c] = \arg\max_{j \in \mathcal{A}:\; c + \mathbf{1}[i \neq j] \leq 1} \left( r_t(i \rightarrow j) + \gamma V[t+1, j, c + \mathbf{1}[i \neq j]] \right)$
5:          $V[t,i,c] = r_t(i \rightarrow \Pi[t,i,c]) + \gamma V[t+1, \Pi[t,i,c], c + \mathbf{1}[i \neq \Pi[t,i,c]]]$
6:      end for
7:  end for
8:  $i \leftarrow 1,\quad c \leftarrow 0$
9:  for $t$ = 0 to $N$-2 do
10:     $i' \leftarrow \Pi[t,i,c]$;$\hat{a}_t \leftarrow \mathcal{A}[i']$
11:     $c \leftarrow \min\{1, c + \mathbf{1}[i \neq i']\}, \quad i \leftarrow i'$
12: end for
13: $\hat{a}_{N-1} \leftarrow \hat{a}_{N-2}$ return $\{\hat{a}_t\}_{0}^{N-1}$

---

For each sampled horizon represented by market observations $\mathbf{s}_{\mathrm{demo}} = (s^{\mathrm{demo}}_{0:h-1})$, the DP planner outputs a demonstration action sequence $\mathbf{a}_{\mathrm{demo}} = (a^{\mathrm{demo}}_{0:h-1})$, where $a^{\mathrm{demo}} \in \{0,1,2\}$ denotes short, flat, or long positions respectively. By construction, exactly one timestep $t \in [0, h-1]$ satisfies $a_t^{\mathrm{demo}} - a_{t-1}^{\mathrm{demo}} \neq 0$, guaranteeing a single trade per horizon. The resulting reward sequence is $\mathbf{r}_{\mathrm{demo}} = (r^{\mathrm{demo}}_{0:h-1})$ by executing the demonstration actions over the horizon. We collect the resulting demonstration trajectories into tuples $\tau = (\mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}})$ and compile them into a dataset $\mathcal{D} = \{\tau_i\}_{i=0}^{n-1}$, which forms the training base for archetype extraction.

We now seek to distill the demonstration trajectories into epresentative trading archetypes. Inspired by skill-based RL approaches in robotics (Pertsch, Lee, and Lim 2021), we adopt an encoder-decoder framework: the encoder ingests each demonstration chunk and projects it to a continuous latent vector, while the decoder reconstructs the original action sequences given the state inputs and the learned representation. However, a purely continuous representation can be challenging for subsequent RL-based trading, primarily for two reasons: 
1) it expands the search space, making it harder for the RL selector in the second phase to effectively explore and select an optimal archetype from a continuous manifold; 
2) it complicates strategy clustering, resulting in fragmented archetypes and inconsistent policy behavior.
To address these limitations, we incorporate a VQ module (Van Den Oord, Vinyals et al. 2017) to learn more meaningful and generalizable archetype embeddings. By discretizing the encoder’s continuous latent outputs into a finite codebook, each demonstration chunk is abstracted to one of a limited set of discrete codes, forming a concise and highly reusable set of archetypes.

In practice, each demonstration trajectory is passed through an LSTM-based encoder,

$$
q_{\theta_e}\!\left(z_e \mid \mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}}\right)
\tag{1}
$$

which produces a continuous embedding $z_e$. Next, this embedding is quantized by selecting the nearest entry from a learnable codebook $\epsilon = \{e_0, \ldots, e_{K-1}\}$, i.e.,

$$
k=\arg\min_{0<=i<=K-1}\left\|z_e-e_i\right\|^2,\qquad z_q=e_k
\tag{2}
$$

yielding a discrete latent representation $z_q$. Finally, a decoder

$$
p_{\theta_d}\!\left(\hat{\mathbf{a}}_{\mathrm{demo}} \mid \mathbf{s}_{\mathrm{demo}}, z_q\right)
\tag{3}
$$

reconstructs the original actions $\mathbf{a}_{\mathrm{demo}}$ from states $\mathbf{s}_{\mathrm{demo}}$ and the quantized embedding $z_q$. We train this model by minimizing the combined loss function:

$$
L = L_{\mathrm{rec}} + \left\|\operatorname{sg}[z_e]-z_q\right\|^2 + \beta_0 \left\|z_e-\operatorname{sg}[z_q]\right\|^2
\tag{4}
$$

where $L_{\mathrm{rec}}$measures the reconstruction loss of demonstration actions, the rest terms enforce commitment to the chosen code and keep the codebook entries close to the encoder output, and sg[·] denotes the stop-gradient operation. By optimizing this loss, the codebook vectors {ei} learn to effectively capture reusable trading patterns, i.e., the archetypes,that effectively summarize the demonstration trajectories. In later phases, selecting an archetype based on current market observations and decoding its latent code into a concrete action sequence allows us to deploy these archetypes as coherent, data-driven trading strategies.

**中文**

---

**输入：** 长度为 $N$ 的价格序列 $P$，动作集合 $\mathcal{A}$  
**输出：** $\{\hat{a}_t\}_{0}^{N-1}$

--- 

1:  $V[N+1, |\mathcal{A}|, 2] \leftarrow 0, \quad \Pi[N, |\mathcal{A}|, 2] \leftarrow -1 $  
2:  for $t$ = $N$-1 downto 0 do  
3:      for $i \in \mathcal{A}, c \in \{0,1\}$ do  
4:          $\Pi[t,i,c] = \arg\max_{j \in \mathcal{A}:\; c + \mathbf{1}[i \neq j] \leq 1} \left( r_t(i \rightarrow j) + \gamma V[t+1, j, c + \mathbf{1}[i \neq j]] \right)$  
5:          $V[t,i,c] = r_t(i \rightarrow \Pi[t,i,c]) + \gamma V[t+1, \Pi[t,i,c], c + \mathbf{1}[i \neq \Pi[t,i,c]]]$  
6:      end for  
7:  end for  
8:  $i \leftarrow 1,\quad c \leftarrow 0$  
9:  for $t$ = 0 to $N$-2 do  
10:     $i' \leftarrow \Pi[t,i,c]$;$\hat{a}_t \leftarrow \mathcal{A}[i']$  
11:     $c \leftarrow \min\{1, c + \mathbf{1}[i \neq i']\}, \quad i \leftarrow i'$  
12: end for  
13: $\hat{a}_{N-1} \leftarrow \hat{a}_{N-2}$ return $\{\hat{a}_t\}_{0}^{N-1}$

---

对于由市场观测 $\mathbf{s}_{\mathrm{demo}} = (s^{\mathrm{demo}}_{0:h-1})$ 表示的每个采样时间跨度，DP 规划器会输出一个示范动作序列 $\mathbf{a}_{\mathrm{demo}} = (a^{\mathrm{demo}}_{0:h-1})$，其中 $a^{\mathrm{demo}} \in \{0,1,2\}$ 分别表示做空、空仓和做多。按照构造方式，恰有一个时间步 $t \in [0, h-1]$ 满足 $a_t^{\mathrm{demo}} - a_{t-1}^{\mathrm{demo}} \neq 0$，从而保证每个时间跨度内仅发生一次交易。通过在该时间跨度上执行示范动作，可得到相应的奖励序列 $\mathbf{r}_{\mathrm{demo}} = (r^{\mathrm{demo}}_{0:h-1})$。我们将这些示范轨迹组织为元组 $\tau = (\mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}})$，并汇总为数据集 $\mathcal{D} = \{\tau_i\}_{i=0}^{n-1}$，作为提取原型的训练基础。

接下来，我们希望将这些示范轨迹蒸馏为具有代表性的交易原型。受机器人领域基于技能的强化学习方法启发（Pertsch, Lee, and Lim 2021），我们采用编码器—解码器框架：编码器接收每个示范片段并将其映射为连续潜向量，解码器则在给定状态输入和已学习表示的条件下重建原始动作序列。然而，纯连续表示会给后续基于强化学习的交易带来困难，主要体现在两个方面：  
1）它扩大了搜索空间，使第二阶段中的强化学习选择器更难在连续流形上有效探索并选择最优原型；  
2）它增加了策略聚类的难度，导致原型碎片化以及策略行为不一致。  

为解决这些问题，我们引入一个 VQ 模块（Van Den Oord, Vinyals et al. 2017）来学习更有意义且更具泛化性的原型嵌入。通过将编码器输出的连续潜表示离散化到一个有限码本中，每个示范片段都会被抽象为有限个离散代码之一，从而形成一组简洁且高度可复用的原型。

在具体实现中，每条示范轨迹会先通过一个基于 LSTM 的编码器：

$$
q_{\theta_e}\!\left(z_e \mid \mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}}\right)
\tag{1}
$$

该编码器产生一个连续嵌入 $z_e$。接着，通过从可学习码本 $\epsilon = \{e_0, \ldots, e_{K-1}\}$ 中选择与之最近的条目，对该嵌入进行量化，即：

$$
k=\arg\min_{0<=i<=K-1}\left\|z_e-e_i\right\|^2,\qquad z_q=e_k
\tag{2}
$$

从而得到离散潜表示 $z_q$。最后，一个解码器

$$
p_{\theta_d}\!\left(\hat{\mathbf{a}}_{\mathrm{demo}} \mid \mathbf{s}_{\mathrm{demo}}, z_q\right)
\tag{3}
$$

根据状态 $\mathbf{s}_{\mathrm{demo}}$ 和量化嵌入 $z_q$ 重建原始动作 $\mathbf{a}_{\mathrm{demo}}$。我们通过最小化如下联合损失函数来训练该模型：

$$
L = L_{\mathrm{rec}} + \left\|\operatorname{sg}[z_e]-z_q\right\|^2 + \beta_0 \left\|z_e-\operatorname{sg}[z_q]\right\|^2
\tag{4}
$$

其中，$L_{\mathrm{rec}}$ 表示示范动作的重建损失，其余两项用于约束编码器输出向所选代码靠拢，并使码本条目接近编码器输出；sg[·] 表示 stop-gradient 操作。通过优化该损失，码本向量 $\{e_i\}$ 能够有效捕捉可复用的交易模式，即交易原型，从而对示范轨迹进行有效总结。在后续阶段中，系统可根据当前市场观测选择一个原型，并将其潜代码解码为具体动作序列，从而将这些原型作为连贯、数据驱动的交易策略进行部署。

### Archetype Selection / 原型选择

**English**

To effectively apply the learned strategic archetypes to trading, we train an RL-based archetype selector to choose the optimal archetype based on the current market observations and execute the micro-actions reconstructed by the previously trained decoder. Specifically, we lift the basic MDP defined in the previous section to a horizon-level

MDP $\mathcal{M}_{\mathrm{sel}} = \langle S_{\mathrm{sel}}, A_{\mathrm{sel}}, R_{\mathrm{sel}}, \gamma \rangle$. For a fixed horizon $H = [t, t + h - 1]$, state $s^{\mathrm{sel}}$ is defined as the state vector $s_t$ defined in the previous section, captured at the first bar of the horizon. Action $a^{\mathrm{sel}} \in \{0,1,\ldots,K-1\}$ denotes the selected archetype from previously learned archetype set $\epsilon = \{e_{0:k-1}\}$ via the selection policy $\pi^{\mathrm{sel}}_{\phi}(a^{\mathrm{sel}} \mid s^{\mathrm{sel}})$. Once an archetype is chosen, its archetype code $e_{a^{\mathrm{sel}}}$ is fed into the frozen decoder $p_{\theta_d}(a^{\mathrm{base}} \mid s, e_{a^{\mathrm{sel}}})$, which emits step-wise micro actions $(a^{\mathrm{base}}_{t:t+h-1})$ based on the upcoming states $(s_{t:t+h-1})$ for the next $h$ steps. We use the sum of step-wise reward over the whole archetype horizon $H$ as the reward function for archetype selection, which is calculated as $r_t^{\mathrm{sel}} = \sum_{\tau=t}^{t+h-1} r_{\tau}^{\mathrm{step}}$. The policy $\pi_{\phi}$ is optimized to maximize the expected return while also encouraging consistency with demonstration trajectories. Concretely, we define the following objective:
$$
J = \mathbb{E}_{\pi^{\mathrm{sel}}_{\phi}} \left[ \sum_{t=0}^{\infty} \left( \gamma^t r_t^{\mathrm{sel}} - \alpha \, KL\!\left(\hat{a}_t^{\mathrm{sel}} \,\|\, \pi^{\mathrm{sel}}_{\phi}\!\left(a_t^{\mathrm{sel}} \mid s_t^{\mathrm{sel}}\right)\right) \right) \right]
\tag{5}
$$

where $r_t^{\mathrm{sel}}$ is the cumulative return obtained over the horizon, $\hat{a}_t^{\mathrm{sel}}$ is the ground-truth archetype label assigned by the VQ encoder to the demonstration chunk of this horizon, and $\alpha$ is a hyperparameter balancing the environment reward against the KL-divergence penalty. This penalty encourages the selection policy to remain near the demonstrated archetype choices, yet still allows it to adapt as it gains experience in the live environment.

**中文**

为了将已学习到的策略原型有效应用于交易，我们训练一个基于强化学习的原型选择器，使其根据当前市场观测选择最优原型，并执行由先前训练好的解码器重构出的微观动作。具体而言，我们将上一节定义的基础 MDP 提升为一个时间跨度级别的

MDP $\mathcal{M}_{\mathrm{sel}} = \langle S_{\mathrm{sel}}, A_{\mathrm{sel}}, R_{\mathrm{sel}}, \gamma \rangle$。对于固定时间跨度 $H = [t, t + h - 1]$，状态 $s^{\mathrm{sel}}$ 定义为该跨度首个 bar 时刻的状态向量 $s_t$。动作 $a^{\mathrm{sel}} \in \{0,1,\ldots,K-1\}$ 表示通过选择策略 $\pi^{\mathrm{sel}}_{\phi}(a^{\mathrm{sel}} \mid s^{\mathrm{sel}})$ 从先前学习得到的原型集合 $\epsilon = \{e_{0:k-1}\}$ 中选出的原型。一旦原型被选定，其对应的原型代码 $e_{a^{\mathrm{sel}}}$ 会被输入到冻结的解码器 $p_{\theta_d}(a^{\mathrm{base}} \mid s, e_{a^{\mathrm{sel}}})$ 中，解码器据此根据未来 $h$ 个时间步的状态 $(s_{t:t+h-1})$ 生成逐步微动作 $(a^{\mathrm{base}}_{t:t+h-1})$。我们将整个原型时间跨度 $H$ 内逐步奖励之和作为原型选择的奖励函数，即 $r_t^{\mathrm{sel}} = \sum_{\tau=t}^{t+h-1} r_{\tau}^{\mathrm{step}}$。策略 $\pi_{\phi}$ 的优化目标是在最大化期望回报的同时，鼓励其与示范轨迹保持一致。具体地，我们定义如下目标函数：

$$
J = \mathbb{E}_{\pi^{\mathrm{sel}}_{\phi}} \left[ \sum_{t=0}^{\infty} \left( \gamma^t r_t^{\mathrm{sel}} - \alpha \, KL\!\left(\hat{a}_t^{\mathrm{sel}} \,\|\, \pi^{\mathrm{sel}}_{\phi}\!\left(a_t^{\mathrm{sel}} \mid s_t^{\mathrm{sel}}\right)\right) \right) \right]
\tag{5}
$$

其中，$r_t^{\mathrm{sel}}$ 是该时间跨度内获得的累计回报，$\hat{a}_t^{\mathrm{sel}}$ 是由 VQ 编码器为该时间跨度的示范片段分配的真实原型标签，$\alpha$ 是平衡环境奖励与 KL 散度惩罚的超参数。该惩罚项鼓励选择策略保持接近示范中的原型选择，同时又允许其在实时环境中积累经验后进行自适应调整。

### Archetype Refinement / 原型细化

**English**

A horizon-level choice of archetype already delivers a coherent trading plan. However, two key limitations arise when market conditions shift or the chosen archetype is suboptimal. First, a single decision at the beginning of the horizon cannot react to rapid intra-chunk changes, potentially missing short-lived trading opportunities or failing to perform timely stop-loss actions. Second, if the archetype selection policy chooses an ill-suited archetype at the horizon’s start,the agent is mostly limited to suboptimal actions for the entire horizon. Yet, in many cases, on-the-fly performance monitoring can quickly reveal that the chosen archetype is misaligned with ongoing market dynamics. Without an effective adaptation method, the agent endures unnecessary losses or foregoes profits until the horizon ends. To handle these issues, we propose a step-level archetype refine-
ment method by employing a policy adapter, which leverages ongoing market observations and partial archetype performance to fine-tune the archetype’s base actions.

For each horizon $H$, we freeze the horizon-level selection policy $\pi^{\mathrm{sel}}_{\phi}$ and decode the corresponding micro action sequence $\mathbf{a}_{\mathrm{base}} = (a^{\mathrm{base}}_{t:t+h-1})$ by the chosen archetype. The archetype refinement process is formulated as a step-level MDP $\mathcal{M}_{\mathrm{ref}} = \{S_{\mathrm{ref}}, A_{\mathrm{ref}}, R_{\mathrm{ref}}\}$. The archetype adaptation agent observes the real-time state $s^{\mathrm{ref}}_{\tau}$ at time $\tau \in [t, t+h-1]$, which consists of two parts: step-wise market observations $s^{\mathrm{ref1}}_{\tau} = s_{\tau}$ and archetype information $s^{\mathrm{ref2}}_{\tau} = [e_{a_t^{\mathrm{sel}}}, a^{\mathrm{base}}_{\tau}, R^{\mathrm{arche}}_{\tau}, \tau_{\mathrm{remain}}]$, where $e_{a_t^{\mathrm{sel}}}$ is the selected archetype embedding, $a_{\tau}^{\mathrm{base}}$ is the current micro action constructed by the archetype, $R_{\tau}^{\mathrm{arche}} = \sum_{i=t}^{\tau} r_{i}^{\mathrm{step}}$ is the cumulative reward under the archetype’s base policy and $\tau_{\mathrm{remain}} = t+h-\tau$ is the number of steps left in the horizon. The adapter generates a refinement signal $a_{\tau}^{\mathrm{ref}} \in \{-1,0,1\}$ that shifts the archetype’s base action $a_{\tau}^{\mathrm{base}}$ while ensuring the original trades are never overridden:

$$
a_{\tau}^{\mathrm{final}}=
\begin{cases}
a_{\tau}^{\mathrm{base}}, & \text{if } a_{\tau}^{\mathrm{base}} \neq a_{\tau-1}^{\mathrm{base}} \text{ or } a_{\tau}^{\mathrm{ref}} = 0, \\
0, & \text{if } a_{\tau}^{\mathrm{ref}} = -1, \\
2, & \text{if } a_{\tau}^{\mathrm{ref}} = 1,
\end{cases}
\tag{6}
$$

Executing $a_{\tau}^{\mathrm{final}}$ yields the realized step-wise profit $r_{\tau}^{\mathrm{step}}$. The refinement policy $\pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})$ is trained to issue an override at most once within each horizon, preventing the adjustment from deviating excessively from the selected archetype’s intended strategy. Formally, $a_{\tau}^{\mathrm{ref}} \neq 0$ for a single $\tau \in [t, t+h-1]$ per horizon and $a_{\tau}^{\mathrm{ref}} = 0$ for other timesteps. To enable the refinement policy to jointly consider real-time market signals and the archetype’s evolving performance, we apply adaptive layer normalization (Peebles and Xie 2023; Zong et al. 2024) to condition market state $s_{\tau}^{\mathrm{ref1}}$ on archetype context $s_{\tau}^{\mathrm{ref2}}$ in practice.

However, because only a single-step adaptation is allowed per horizon, the policy must carefully identify the most impactful moment to intervene rather than squandering its opportunity on minor market fluctuations or prematurely overriding an archetype strategy that may yield greater returns later. To focus the refinement policy on meaningful adjustments, we compute the top-5 hindsight-optimal adaptations via DP and introduce a novel regret-aware reward function that penalizes the agent not only for failing to outperform the base strategy but also for missing out on the best possible adaptation. Concretely, the DP solver returns:
$$
O_{\mathrm{top5}}=\left\{(\tau_{\mathrm{opt}}^{n}, a_{\mathrm{opt}}^{n}, R_{\mathrm{opt}}^{n})\right\}_{n=1}^{5}
\tag{7}
$$

where $\tau_{\mathrm{opt}}^{n}$ is the adaptation timestep, $a_{\mathrm{opt}}^{n} \in \{-1,1\}$ is the adaptation action, and $R_{\mathrm{opt}}^{n}$ is the horizon’s cumulative return of executing the adaptation. Instead of directly using the immediate step-wise reward $r_{\tau}^{\mathrm{step}}$, our regret-augmented reward function for the policy adaptation agent is formulated as follows:

$$
r_{\tau}^{\mathrm{ref}}=
\begin{cases}
(R-R_{\mathrm{base}})+\beta_{1}(R-R_{\mathrm{opt}}^{1}), & \text{if } a_{\tau}^{\mathrm{ref}} \neq 0, \\
0, & \text{otherwise}
\end{cases}
\tag{8}
$$
where $R=\sum_{\tau=0}^{h-1} r_{\tau}^{\mathrm{step}}$ is the horizon’s cumulative return of taking action $a_{\tau}^{\mathrm{ref}}$, $R_{\mathrm{base}}$ is the horizon’s return under the base archetype policy, $R_{\mathrm{opt}}^{1}$ is the horizon’s maximum return from the top DP-identified adaptation and $\beta_{1}$ is the hyper-parameter controlling the tolerance for suboptimality. This reward structure encourages profitable adjustments relative to the base strategy while penalizing the gap from the best possible adaptation. By striking this balance, the policy avoids wasting its single adaptation opportunity on negligible improvements and instead aims to deploy its intervention at the most valuable moment. Because only one adaptation is allowed per horizon, the RL episode terminates as soon as the adapter chooses a non-zero action. Finally, the objective function for training the refinement policy is given by
$$
J' = \mathbb{E}_{\pi_{\omega}^{\mathrm{ref}}} \left[ \sum_{\tau=0}^{h-1} \left( \gamma^{\tau} r_{\tau}^{\mathrm{ref}} - \beta_2 L\!\left(\hat{a}_{\tau}^{\mathrm{ref}}, \pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})\right) \right) \right]
\tag{9}
$$

where $\hat{a}_{\tau}^{\mathrm{ref}}$ denotes the optimal adaptation action ($\hat{a}_{\tau}^{\mathrm{ref}} = a_{\mathrm{opt}}^{n}$ if $\tau = \tau_{\mathrm{opt}}^{n}$, and $0$ otherwise), and $L$ is the cross-entropy loss guiding the refinement policy toward optimal behavior. By optimizing this objective, we encourage the policy adapter to perform high-impact adjustments to the base archetype actions, ultimately yielding a more profitable and robust trading strategy.

**中文**

在时间跨度级别做出一次原型选择，已经能够给出一个连贯的交易计划。然而，当市场条件变化或所选原型并不理想时，仍会出现两个关键问题。第一，在时间跨度起点做出的单次决策无法对跨度内部快速变化的市场做出反应，因此可能错失短暂的交易机会，或无法及时止损。第二，如果原型选择策略在时间跨度开始时选错了原型，智能体在整个跨度内大多只能执行次优动作。但很多情况下，通过持续监控策略表现，可以很快发现该原型与当前市场动态并不匹配。若缺乏有效的适应机制，智能体就必须在整个跨度结束前承受不必要的损失或错失盈利机会。为解决这些问题，我们提出一种逐步级别的原型细化方法，通过引入策略适配器，利用持续到来的市场观测和部分原型表现，对原型的基础动作进行微调。

对于每个时间跨度 $H$，我们冻结时间跨度级别的选择策略 $\pi^{\mathrm{sel}}_{\phi}$，并通过所选原型解码得到对应的微动作序列 $\mathbf{a}_{\mathrm{base}} = (a^{\mathrm{base}}_{t:t+h-1})$。原型适配过程被建模为一个逐步级别的 MDP $\mathcal{M}_{\mathrm{ref}} = \{S_{\mathrm{ref}}, A_{\mathrm{ref}}, R_{\mathrm{ref}}\}$。原型适配智能体在时间 $\tau \in [t, t+h-1]$ 观察实时状态 $s^{\mathrm{ref}}_{\tau}$，该状态包含两部分：逐步市场观测 $s^{\mathrm{ref1}}_{\tau} = s_{\tau}$ 与原型信息 $s^{\mathrm{ref2}}_{\tau} = [e_{a_t^{\mathrm{sel}}}, a^{\mathrm{base}}_{\tau}, R^{\mathrm{arche}}_{\tau}, \tau_{\mathrm{remain}}]$，其中 $e_{a_t^{\mathrm{sel}}}$ 是所选原型的嵌入，$a_{\tau}^{\mathrm{base}}$ 是原型当前生成的微动作，$R_{\tau}^{\mathrm{arche}} = \sum_{i=t}^{\tau} r_{i}^{\mathrm{step}}$ 是在原型基础策略下累计获得的奖励，$\tau_{\mathrm{remain}} = t+h-\tau$ 是该时间跨度剩余的步数。适配器输出一个细化信号 $a_{\tau}^{\mathrm{ref}} \in \{-1,0,1\}$，用于偏移原型的基础动作 $a_{\tau}^{\mathrm{base}}$，同时保证原始交易决策不会被覆盖：

$$
a_{\tau}^{\mathrm{final}}=
\begin{cases}
a_{\tau}^{\mathrm{base}}, & \text{if } a_{\tau}^{\mathrm{base}} \neq a_{\tau-1}^{\mathrm{base}} \text{ or } a_{\tau}^{\mathrm{ref}} = 0, \\
0, & \text{if } a_{\tau}^{\mathrm{ref}} = -1, \\
2, & \text{if } a_{\tau}^{\mathrm{ref}} = 1,
\end{cases}
\tag{6}
$$

执行 $a_{\tau}^{\mathrm{final}}$ 可得到实际的逐步收益 $r_{\tau}^{\mathrm{step}}$。细化策略 $\pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})$ 被训练为在每个时间跨度内至多发出一次覆盖信号，从而防止调整过度偏离所选原型的原始意图。形式化地说，在每个时间跨度中，仅允许某一个 $\tau \in [t, t+h-1]$ 满足 $a_{\tau}^{\mathrm{ref}} \neq 0$，其余时间步均有 $a_{\tau}^{\mathrm{ref}} = 0$。为了使细化策略能够同时考虑实时市场信号与原型逐步演化的表现，我们在实践中采用自适应层归一化（Peebles and Xie 2023; Zong et al. 2024），用原型上下文 $s_{\tau}^{\mathrm{ref2}}$ 对市场状态 $s_{\tau}^{\mathrm{ref1}}$ 进行条件化。

然而，由于每个时间跨度内只允许一次单步适配，策略必须仔细识别最具影响力的干预时刻，而不能把唯一一次机会浪费在轻微波动上，也不能过早覆盖一个可能在后续带来更高收益的原型策略。为了让细化策略聚焦于真正有价值的调整，我们通过 DP 计算事后最优的前 5 个适配动作，并提出一种全新的遗憾感知奖励函数。该奖励不仅惩罚策略未能超越基础策略的情况，也惩罚其错失最佳可能适配的情况。具体而言，DP 求解器返回：

$$
O_{\mathrm{top5}}=\left\{(\tau_{\mathrm{opt}}^{n}, a_{\mathrm{opt}}^{n}, R_{\mathrm{opt}}^{n})\right\}_{n=1}^{5}
\tag{7}
$$

其中，$\tau_{\mathrm{opt}}^{n}$ 表示适配发生的时间步，$a_{\mathrm{opt}}^{n} \in \{-1,1\}$ 表示适配动作，$R_{\mathrm{opt}}^{n}$ 表示执行该适配后整个时间跨度的累计回报。我们不直接使用即时逐步奖励 $r_{\tau}^{\mathrm{step}}$，而是将策略适配智能体的遗憾增强奖励定义为：

$$
r_{\tau}^{\mathrm{ref}}=
\begin{cases}
(R-R_{\mathrm{base}})+\beta_{1}(R-R_{\mathrm{opt}}^{1}), & \text{if } a_{\tau}^{\mathrm{ref}} \neq 0, \\
0, & \text{otherwise}
\end{cases}
\tag{8}
$$

其中，$R=\sum_{\tau=0}^{h-1} r_{\tau}^{\mathrm{step}}$ 表示采取适配动作 $a_{\tau}^{\mathrm{ref}}$ 后整个时间跨度的累计回报，$R_{\mathrm{base}}$ 表示基础原型策略下的累计回报，$R_{\mathrm{opt}}^{1}$ 表示通过 DP 识别出的最佳适配所对应的最大累计回报，$\beta_{1}$ 是控制对次优适配容忍度的超参数。该奖励结构鼓励相对基础策略实现可观盈利的调整，同时惩罚与最佳可能适配之间的差距。通过这种平衡，策略不会将唯一一次适配机会浪费在微不足道的改进上，而是尽量在最有价值的时刻进行干预。由于每个时间跨度内只允许一次适配，一旦适配器选择了非零动作，该强化学习回合便立即终止。最后，用于训练细化策略的目标函数为

$$
J' = \mathbb{E}_{\pi_{\omega}^{\mathrm{ref}}} \left[ \sum_{\tau=0}^{h-1} \left( \gamma^{\tau} r_{\tau}^{\mathrm{ref}} - \beta_2 L\!\left(\hat{a}_{\tau}^{\mathrm{ref}}, \pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})\right) \right) \right]
\tag{9}
$$

其中，$\hat{a}_{\tau}^{\mathrm{ref}}$ 表示最优适配动作（当 $\tau = \tau_{\mathrm{opt}}^{n}$ 时，$\hat{a}_{\tau}^{\mathrm{ref}} = a_{\mathrm{opt}}^{n}$，否则为 $0$），$L$ 是用于引导细化策略趋近最优行为的交叉熵损失。通过优化该目标，我们鼓励策略适配器对原型基础动作做出高影响力的调整，从而最终形成更具盈利性、更稳健的交易策略。

## Experiments / 实验

### Datasets / 数据集

**English**

To evaluate the effectiveness of our method, we use 10 minute data with orderbook depth M = 25 of BTC/ETH/DOT/BNB against USDT. The dataset spans from 2021-06-01 to 2023-05-31 for training, 2023-06-01 to 2023-12-31 for validation, and 2024-01-01 to 2024-09-01 for testing.

**中文**

为评估方法有效性，我们使用 BTC/ETH/DOT/BNB 对 USDT 的 10 分钟级别数据，订单簿深度设为 $M = 25$。数据集划分如下：训练集为 2021-06-01 至 2023-05-31，验证集为 2023-06-01 至 2023-12-31，测试集为 2024-01-01 至 2024-09-01。

### Evaluation Metrics / 评估指标

**English**

We evaluate our proposed method on 6 different financial metrics, including one profit criterion, two risk criteria, and three risk-adjusted profit criteria listed below.
- **Total Return (TR)** is the overall return rate of the entire trading period, which is defined as $ TR = \frac{V_T - V_1}{V_1} $, where $V_T$ is the final net value and $V_1$ is the initial net value.

- **Annual Volatility (AVOL)** is the variation of return over one year measured as $  \sigma[r] \times \sqrt{m},  $  where $r = [r_1, r_2, \ldots, r_T]$ is return, $\sigma[\cdot]$ is standard deviation, $m = 52560$ is the number of timesteps.

- **Maximum Drawdown (MDD)** measures the largest loss from any peak to show the worst case.

- **Annual Sharpe Ratio (ASR)** measures the amount of extra return a trader gets per unit of increased risk, calculated as  $  ASR = E[r] / \sigma[r] \times \sqrt{m}.  $

- **Annual Calmar Ratio (ACR)** measures the risk-adjusted return calculated as $  ACR = \frac{E[r]}{MDD} \times m.  $

- **Annual Sortino Ratio (ASoR)** measures risk with downside deviation (DD):  $  SoR = \frac{E[r]}{DD} \times \sqrt{m}.  $

**中文**

我们从 6 个不同金融指标评估所提方法，包括 1 个收益指标、2 个风险指标和 3 个风险调整后收益指标，定义如下。  
- **总收益率（TR）**：整个交易周期的总收益率，定义为 $ TR = \frac{V_T - V_1}{V_1} $，其中 $V_T$ 为最终净值，$V_1$ 为初始净值。  

- **年化波动率（AVOL）**：一年内收益波动程度，定义为 $  \sigma[r] \times \sqrt{m},  $，其中 $r = [r_1, r_2, \ldots, r_T]$ 为收益序列，$\sigma[\cdot]$ 为标准差，$m = 52560$ 为时间步数量。  

- **最大回撤（MDD）**：从任一峰值下跌到谷值的最大损失，用于衡量最坏情况。  

- **年化夏普比率（ASR）**：衡量单位风险所对应的超额收益，计算方式为 $  ASR = E[r] / \sigma[r] \times \sqrt{m}.  $  

- **年化卡玛比率（ACR）**：衡量风险调整后收益，计算方式为 $  ACR = \frac{E[r]}{MDD} \times m.  $  

- **年化索提诺比率（ASoR）**：使用下行偏差（DD）衡量风险，定义为 $  SoR = \frac{E[r]}{DD} \times \sqrt{m}.  $

### Baselines / 基线方法

**English**

To benchmark our method, we select 8 baselines including standard RL (DQN (Mnih et al. 2015), PPO (Schulman et al. 2017), CDQNRP (Zhu and Zhu 2022), CLSTM-PPO (Zou et al. 2024)), hierarchical RL (EarnHFT (Qin et al.2023), MacroHFT (Zong et al. 2024)), and rule-based strategies (IV (Chordia, Roll, and Subrahmanyam 2002),MACD (Krug, Dobaj, and Macher 2022)).

**中文**

为对比本文方法，我们选取了 8 个基线，包括标准强化学习方法（DQN（Mnih et al. 2015）、PPO（Schulman et al. 2017）、CDQNRP（Zhu and Zhu 2022）、CLSTM-PPO（Zou et al. 2024））、层次化强化学习方法（EarnHFT（Qin et al. 2023）、MacroHFT（Zong et al. 2024）），以及规则驱动策略（IV（Chordia, Roll, and Subrahmanyam 2002）、MACD（Krug, Dobaj, and Macher 2022））。

### Experiment Setup / 实验设置

**English**

Experiments are conducted on 4 RTX-4090 GPUs. All trades incur a commission of $\delta = 0.02\%$, and the max position $m$ varies by asset (BTC 8, ETH 100, DOT 2500, BNB 200). For archetype discovery, we sample 30k DP trajectories (horizon length $h = 72$) and train a 128-unit VQ encoder–decoder (archetype vector dimension=16, number of archetypes $K = 10$, $\beta_0 = 0.25$) for 100 epochs. Specifically, archetype dimension is set smaller than network dimension to create information bottleneck, forcing the VQ module to distill high-level trading strategies. For archetype selection, an RL selector with $\alpha = 1$ is optimized for 3M steps, and the checkpoint with the best validation performance is retained. For archetype refinement, the regret-aware adapter is trained for 1M steps with $\beta_2 = 1$ and asset-specific $\beta_1$ tuned over $\{0.3, 0.5, 0.7\}$ ($0.5$ for BTC/DOT, $0.7$ for ETH/BNB) via validation. We emphasize that all DP modules are applied solely for training and disabled during inference to avoid future information leakage.

**中文**

实验在 4 张 RTX-4090 GPU 上进行。所有交易均收取 $\delta = 0.02\%$ 的手续费，最大持仓 $m$ 随资产而变化（BTC 为 8，ETH 为 100，DOT 为 2500，BNB 为 200）。在原型发现阶段，我们采样 3 万条 DP 轨迹（时间跨度长度 $h = 72$），并训练一个 128 单元的 VQ 编码器—解码器（原型向量维度为 16，原型数量 $K = 10$，$\beta_0 = 0.25$），共训练 100 个 epoch。具体来说，我们将原型维度设置得小于网络维度，以形成信息瓶颈，迫使 VQ 模块提炼高层级交易策略。在原型选择阶段，一个设置 $\alpha = 1$ 的强化学习选择器被优化 300 万步，并保留验证集表现最佳的检查点。在原型细化阶段，遗憾感知型适配器训练 100 万步，$\beta_2 = 1$，并通过验证集在 $\{0.3, 0.5, 0.7\}$ 中调节资产特定的 $\beta_1$（BTC/DOT 取 $0.5$，ETH/BNB 取 $0.7$）。需要强调的是，所有 DP 模块仅在训练阶段使用，在推理阶段全部关闭，以避免未来信息泄漏。

### Results and Analysis / 结果与分析

**English**

Table 1 summarizes the performance of ArchetypeTrader alongside baseline methods across four cryptocurrency markets. Overall, ArchetypeTrader attains the highest profit and best risk-adjusted returns in all cryptocurrency markets except BNBUSDT. Even in the predominantly bullish BNB market, where ArchetypeTrader does not fully exploit the upward trend, our method still achieves competitive profit and risk control capacity. By contrast, RL approaches that rely on uniform policies (DQN, CDQNRP, PPO, CLSTMPPO) struggle to adapt to shifting market regimes, leading to inconsistent profit generation. Meanwhile, rule-based methods (IV, MACD) perform on specific datasets yet suffer catastrophic losses on others, indicating a lack of robustness. Although hierarchical RL methods (EarnHFT, MacroHFT) demonstrate relatively stable returns by segmenting markets into sub-policies, these sub-policies remain biased toward certain market dynamics by human-crafted indicators and fail to capture more nuanced, profitable opportunities. ArchetypeTrader avoids hand-crafted segmentation by discovering and refining archetypes directly from the effective use of demonstration data, resulting in a broader and more flexible set of trading behaviors that ultimately yield
better risk management and higher overall profitability.

**中文**

表 1 总结了 ArchetypeTrader 与各基线方法在四个加密货币市场上的表现。总体而言，除了 BNBUSDT 之外，ArchetypeTrader 在所有加密货币市场上都取得了最高收益和最佳风险调整后回报。即使在整体偏多的 BNB 市场中，尽管 ArchetypeTrader 未能完全吃满上涨趋势，它仍然表现出具有竞争力的收益与风险控制能力。相比之下，依赖统一策略的强化学习方法（DQN、CDQNRP、PPO、CLSTM-PPO）难以适应不断切换的市场状态，因此收益表现不稳定。与此同时，规则驱动方法（IV、MACD）虽然在个别数据集上表现良好，但在其他数据集上可能遭遇灾难性亏损，显示出较差的稳健性。尽管层次化强化学习方法（EarnHFT、MacroHFT）通过对市场进行划分并训练子策略，展示出相对稳定的收益，但这些子策略仍然受到人工构造指标的影响，偏向某些特定市场动态，难以捕捉更细微、更具盈利性的机会。ArchetypeTrader 则通过有效利用示范数据，直接发现并细化原型，避免了手工市场划分，由此形成更广泛、更灵活的交易行为集合，最终带来更好的风险管理与更高的整体收益。

### Interpretability Analysis of Archetypes / 原型可解释性分析

**English**

To illustrate that our data-driven archetypes capture meaningful expert behaviors beyond standard hand-crafted segmentation, we visualize several trading signals prescribed by the selected archetypes on BTCUSDT, as shown in Figure 2. For Archetype 9 (Figure 2(a, b)), examples 1 and 2 reveal a “short-and-hold” strategy suitable for bearish market horizons: the archetype short-sells near the horizon start, anticipating a downward trend. Thus, when the market is poised for further decline, the selection policy would probably pick this archetype. From examples 1 and 2 of Archetype 2 (Figure 2(c, d)), it is apparent that the archetype is geared toward shorting near local peaks and going long near local troughs, aiming to capture subsequent price reversals. By choosing this archetype, the resulting policy’s core behavior is identifying short-term extremes and trading against them, which is a typical counter-trend or mean-reversion style. Crucially, although Archetype 2 often exhibits these short-term extremes, it does not merely repeat identical positions. Instead, it extracts a high-level similarity of trading against the current price movement, leading to different specific trades but a consistent mean-reversion effect. These examples show how our learned archetypes retain intuitive and expert-like trading behaviors rather than following rigid trends.

**中文**

为了说明我们数据驱动学习到的原型捕捉了超越传统人工市场划分的、有意义的专家行为，我们在 BTCUSDT 数据上可视化了若干由选中原型给出的交易信号，如图 2 所示。对于 Archetype 9（图 2(a, b)），示例 1 和示例 2 展示了一种适用于看跌市场时间跨度的“做空并持有”策略：该原型在时间跨度起始附近建立空头，预期市场将继续下跌。因此，当市场有进一步下行趋势时，选择策略很可能会选中这个原型。再看 Archetype 2（图 2(c, d)）的示例 1 和示例 2，可以明显看出该原型倾向于在局部高点附近做空、在局部低点附近做多，以捕捉随后的价格反转。选中该原型后，所得策略的核心行为就是识别短期极值并逆势交易，这是一种典型的逆趋势或均值回归风格。关键在于，虽然 Archetype 2 往往出现在这类短期极值附近，但它并非机械地重复完全相同的持仓模式，而是抽取出一种更高层次的相似性——即逆当前价格运动方向进行交易，因此能够在具体交易上有所不同，却保持一致的均值回归效果。这些示例说明，我们学习到的原型保留了直观且类似专家的交易行为，而不是僵化地跟随趋势。

#### Ablation Study / 消融实验

**English**

To assess the effectiveness of our framework’s core components, we conduct various ablation experiments examining: vector quantization (VQ) for archetype embeddings, archetype refinement (Ref) via a policy adapter, and regret regularization (Reg) in the refinement phase. In Table 2, we compare three archetype embedding methods: continuous embeddings, clustered embeddings (k-means and k=10), and our vector-quantized discrete embeddings under the no-refinement setting for a fair comparison. The results show that discrete VQ embeddings consistently yield higher returns and better risk management metrics, highlighting the advantages of learnable discrete archetypes that the selector can easily interpret and switch among. Table 3 and Figure 3 present the performance for three refinement approaches: no refinement, refinement without regret, and refinement with regret (our original model). It can be observed that adding refinement via a policy adapter captures extra trading opportunities and corrects suboptimal archetype executions, leading to higher profit and better risk control. Moreover, incorporating a hindsight regret penalty ensures that refinements occur only at the most valuable moments while removing this term can even cause deficits compared with the archetype selector’s base policy, indicating that regret guidance is crucial for meaningful intra-horizon interventions.

**中文**

为了评估框架核心组件的有效性，我们进行了多组消融实验，分别考察：用于原型嵌入的向量量化（VQ）、通过策略适配器实现的原型细化（Ref），以及细化阶段中的遗憾正则化（Reg）。在表 2 中，我们在不进行细化的公平设定下，对比了三种原型嵌入方式：连续嵌入、聚类嵌入（k-means，$k=10$）以及本文采用的向量量化离散嵌入。结果表明，离散的 VQ 嵌入能够持续带来更高收益和更好的风险管理指标，说明可学习的离散原型更便于选择器理解并在其间切换。表 3 和图 3 展示了三种细化方式的表现：不进行细化、无遗憾项的细化，以及带遗憾项的细化（即我们的完整模型）。可以观察到，引入策略适配器进行细化能够捕捉额外交易机会，并纠正次优原型执行，从而获得更高收益和更好的风险控制。此外，加入基于后见信息的遗憾惩罚后，细化动作只会在最有价值的时刻发生；而移除该项时，模型甚至可能比原型选择器的基础策略表现更差，这表明遗憾引导对有意义的跨度内干预至关重要。

### Hyper-parameter Sensitivity / 超参数敏感性

**English**

We evaluate how some hyper-parameters shape performance on DOTUSDT dataset (Table 4). A small codebook of archetypes under-represents the diverse trading motifs found
in the demonstrations; a large one overloads the selector and adapter. Similarly, orizons that are too short trigger frequent, noise-driven shifts, whereas very long horizons postpone updates and blunt responsiveness. Intermediate values
for both K and h give the best trade-off between expressiveness and agility.

**中文**

我们在 DOTUSDT 数据集上评估了若干超参数对性能的影响（表 4）。过小的原型码本无法充分表示示范中丰富多样的交易模式；过大的码本又会给选择器和适配器带来过重负担。类似地，过短的时间跨度会导致频繁且易受噪声驱动的切换，而过长的时间跨度则会延迟更新、削弱响应能力。无论是 $K$ 还是 $h$，取中间值时都能在表达能力与敏捷性之间取得最佳平衡。

## Conclusion / 结论

**English**

In this paper, We present ArchetypeTrader, a novel RL framework that discovers, selects, and refines strategic archetypes for automated cryptocurrency trading. Firstly, ArchetypeTrader discovers representative and reusable strategic archetypes from DP–generated demonstrations via a vector-quantized encoder-decoder. A horizon-level RL agent then selects the best archetype for each chunk based on market context, while a step-level adapter refines archetype actions within each horizon through regret-aware updates. Comprehensive experiments on four major cryptocurrency pairs demonstrate that ArchetypeTrader consistently outper-forms multiple state-of-the-art trading algorithms in profitmaking while maintaining outstanding risk management ability in cryptocurrency trading.

**中文**

本文提出了 **ArchetypeTrader**，一个用于自动化加密货币交易的全新强化学习框架，能够发现、选择并细化策略原型。首先，ArchetypeTrader 通过向量量化编码器—解码器，从 DP 生成的示范中发现具有代表性、可复用的策略原型。随后，一个时间跨度级别的强化学习智能体会根据市场上下文为每个片段选择最佳原型，而一个逐步级别的适配器则通过遗憾感知更新在跨度内部细化原型动作。对四个主要加密货币交易对开展的综合实验表明，ArchetypeTrader 在获取收益的同时，也能维持出色的风险管理能力，并持续优于多种最先进的交易算法。

## Acknowledgments / 致谢

**English**

This research is supported by the Joint NTU-WeBank Research Centre on Fintech, Nanyang Technological University, Singapore.

**中文**

本研究得到南洋理工大学—微众银行金融科技联合研究中心（Joint NTU-WeBank Research Centre on Fintech, Nanyang Technological University, Singapore）的支持。

## References / 参考文献

**English**

Briola, A.; Turiel, J.; Marcaccioli, R.; Cauderan, A.; and Aste, T. 2021. Deep reinforcement learning for active high frequency trading. arXiv preprint arXiv:2101.07107.
Chordia, T.; Roll, R.; and Subrahmanyam, A. 2002. Order imbalance, liquidity, and market returns. Journal of Financial Economics, 65(1): 111–130.
Deng, Y.; Bao, F.; Kong, Y.; Ren, Z.; and Dai, Q. 2016. Deep direct reinforcement learning for financial signal representation and trading. IEEE Transactions on Neural Networks and Learning Systems, 28(3): 653–664.
Hung, N. H. 2016. Various moving average convergence divergence trading strategies: A comparison. Investment Management and Financial Innovations, (13, Iss. 2): 363–369.
Jia, W.; Chen, W.; Xiong, L.; and Hongyong, S. 2019. Quantitative trading on stock market based on deep reinforcement learning. In 2019 International Joint Conference on Neural Networks (IJCNN), 1–8.
Kakushadze, Z. 2016. 101 formulaic alphas. Wilmott,2016(84): 72–81.Krug, T.; Dobaj, J.; and Macher, G. 2022. Enforcing network safety-margins in industrial process control using MACD indicators. In European Conference on Software Process Improvement, 401–413. Springer.
Li, Y.; Zheng, W.; and Zheng, Z. 2019. Deep robust reinforcement learning for practical algorithmic trading. IEEE Access, 7: 108014–108022.
Liu, X.-Y.; Yang, H.; Chen, Q.; Zhang, R.; Yang, L.; Xiao, B.; and Wang, C. D. 2020a. FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. arXiv preprint arXiv:2011.09607.
Liu, Y.; Liu, Q.; Zhao, H.; Pan, Z.; and Liu, C. 2020b. Adaptive quantitative trading: An imitative deep reinforcement learning approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 2128–2135.
Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A. A.; Veness, J.; Bellemare, M. G.; Graves, A.; Riedmiller, M.; Fidjeland, A. K.; Ostrovski, G.; et al. 2015. Human-level control through deep reinforcement learning. nature, 518(7540):
529–533.
Murphy, J. J. 1999. Technical Analysis of the Futures Markets: A Comprehensive Guide to Trading Methods and Applications, New York Institute of Finance. Prentice-Hall. 
Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4195–4205.
Pertsch, K.; Lee, Y.; and Lim, J. 2021. Accelerating reinforcement learning with learned skill priors. In Conference on robot learning, 188–204. PMLR.
Qin, M.; Sun, S.; Zhang, W.; Xia, H.; Wang, X.; and An, B. 2023. Earnhft: Efficient hierarchical reinforcement learning for high frequency trading. arXiv preprint
arXiv:2309.12891.
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30.
Zhang, Z.; Zohren, S.; and Stephen, R. 2020. Deep reinforcement learning for trading. The Journal of Financial Data Science.
Zhu, T.; and Zhu, W. 2022. Quantitative trading through random perturbation Q-network with nonlinear transaction costs. Stats, 5(2): 546–560.
Zong, C.; Wang, C.; Qin, M.; Feng, L.; Wang, X.; and An, B. 2024. MacroHFT: Memory augmented context-aware reinforcement learning on high frequency trading. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4712–4721.
Zou, J.; Lou, J.; Wang, B.; and Liu, S. 2024. A novel deep reinforcement learning based automated stock trading system using cascaded lstm networks. Expert Systems with Applications, 242: 122801.

**中文**

Briola, A.; Turiel, J.; Marcaccioli, R.; Cauderan, A.; and Aste, T. 2021. Deep reinforcement learning for active high frequency trading. arXiv preprint arXiv:2101.07107.  
Chordia, T.; Roll, R.; and Subrahmanyam, A. 2002. Order imbalance, liquidity, and market returns. Journal of Financial Economics, 65(1): 111–130.  
Deng, Y.; Bao, F.; Kong, Y.; Ren, Z.; and Dai, Q. 2016. Deep direct reinforcement learning for financial signal representation and trading. IEEE Transactions on Neural Networks and Learning Systems, 28(3): 653–664.  
Hung, N. H. 2016. Various moving average convergence divergence trading strategies: A comparison. Investment Management and Financial Innovations, (13, Iss. 2): 363–369.  
Jia, W.; Chen, W.; Xiong, L.; and Hongyong, S. 2019. Quantitative trading on stock market based on deep reinforcement learning. In 2019 International Joint Conference on Neural Networks (IJCNN), 1–8.  
Kakushadze, Z. 2016. 101 formulaic alphas. Wilmott, 2016(84): 72–81.  
Krug, T.; Dobaj, J.; and Macher, G. 2022. Enforcing network safety-margins in industrial process control using MACD indicators. In European Conference on Software Process Improvement, 401–413. Springer.  
Li, Y.; Zheng, W.; and Zheng, Z. 2019. Deep robust reinforcement learning for practical algorithmic trading. IEEE Access, 7: 108014–108022.  
Liu, X.-Y.; Yang, H.; Chen, Q.; Zhang, R.; Yang, L.; Xiao, B.; and Wang, C. D. 2020a. FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. arXiv preprint arXiv:2011.09607.  
Liu, Y.; Liu, Q.; Zhao, H.; Pan, Z.; and Liu, C. 2020b. Adaptive quantitative trading: An imitative deep reinforcement learning approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 2128–2135.  
Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A. A.; Veness, J.; Bellemare, M. G.; Graves, A.; Riedmiller, M.; Fidjeland, A. K.; Ostrovski, G.; et al. 2015. Human-level control through deep reinforcement learning. Nature, 518(7540): 529–533.  
Murphy, J. J. 1999. Technical Analysis of the Futures Markets: A Comprehensive Guide to Trading Methods and Applications, New York Institute of Finance. Prentice-Hall.  
Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4195–4205.  
Pertsch, K.; Lee, Y.; and Lim, J. 2021. Accelerating reinforcement learning with learned skill priors. In Conference on Robot Learning, 188–204. PMLR.  
Qin, M.; Sun, S.; Zhang, W.; Xia, H.; Wang, X.; and An, B. 2023. EarnHFT: Efficient hierarchical reinforcement learning for high frequency trading. arXiv preprint arXiv:2309.12891.  
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.  
Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30.  
Zhang, Z.; Zohren, S.; and Stephen, R. 2020. Deep reinforcement learning for trading. The Journal of Financial Data Science.  
Zhu, T.; and Zhu, W. 2022. Quantitative trading through random perturbation Q-network with nonlinear transaction costs. Stats, 5(2): 546–560.  
Zong, C.; Wang, C.; Qin, M.; Feng, L.; Wang, X.; and An, B. 2024. MacroHFT: Memory augmented context-aware reinforcement learning on high frequency trading. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4712–4721.  
Zou, J.; Lou, J.; Wang, B.; and Liu, S. 2024. A novel deep reinforcement learning based automated stock trading system using cascaded lstm networks. Expert Systems with Applications, 242: 122801.
