# Phase II Double DQN Archetype Selection 技术设计

## 1. 目标

本文档重新设计 Phase II 训练框架：保留论文中的 Archetype Selection 问题定义，只把优化算法从 PPO 改为 Double DQN。除 Double DQN 必需机制外，不引入论文未提到的工程增强。

论文 Phase II 的定义来自 `docs/paper/AAAI26_ArchetypeTrader_core.md`：

- 对固定 horizon `H=[t,t+h-1]`，selector state `s^sel` 是 horizon 第一根 bar 的状态 `s_t`。
- `selected_code k_t in {0,...,K-1}` 是选择一个 Phase I 学到的 archetype code。
- 选中 archetype 后，将其 code 输入冻结的 Phase I decoder，生成未来 `h` 步 base actions。
- selector reward 是整个 horizon 的 step reward 之和：`r_t^sel = sum_{tau=t}^{t+h-1} r_tau^step`。
- demo label `demo_code_label_t` 是该 horizon 的 demonstration chunk 经 Phase I VQ encoder 分配的 ground-truth archetype label。
- 训练目标同时最大化 horizon reward，并用系数 `alpha` 鼓励 selector 接近 demonstration archetype choices。

本文档中的 Double DQN 是算法替换，不改变以上语义。

## 2. 非目标

第一版明确不做：

- 不做 PPO、GAE、actor-critic、entropy bonus、PPO clip。
- 不做并行探索、多 env、process worker、thread worker。
- 不做 dead-code code mask。
- 不做历史窗口 pooling、账户净值、近期收益等额外状态扩展。
- 不做 class-balanced demo loss、label smoothing、temperature scaling。
- 不做 prioritized replay、dueling DQN、noisy network、distributional DQN。
- 不做 Phase III refinement。
- 不在线调用 DP。

## 3. 目录

新代码独立放置，不复用现有 `src/rl/` PPO 训练器。

```text
src/phase2_dqn/
  config.py
  dataset.py
  env.py
  q_network.py
  replay_buffer.py
  losses.py
  trainer.py
  evaluator.py
  checkpoint.py
  report.py

scripts/train_phase2_dqn.py
scripts/backtest_phase2_dqn.py
```

职责边界：

- `src/phase2_dqn/trainer.py` 放训练主流程，包括加载 Phase I 产物、构建 dataset/env/network/replay、demo pretrain、Double DQN epoch loop、evaluate 调用和 checkpoint 保存。
- `scripts/train_phase2_dqn.py` 只负责 CLI 参数接收、配置覆盖、日志初始化、随机种子初始化、产物目录初始化，然后调用 `Phase2DQNTrainer.run()`。
- `scripts/train_phase2_dqn.py` 不实现训练循环、不直接采样 replay、不直接更新网络、不直接保存 checkpoint。
- `scripts/backtest_phase2_dqn.py` 只负责加载参数和 checkpoint，然后调用 `Phase2DQNEvaluator` 做独立 backtest。

## 4. 数据与上游产物

Phase II DQN 消费 Phase I 已冻结产物：

```text
artifacts/{PAIR}/{PHASE1_BATCH_ID}/phase1/
  decoder.pt
  codebook.pt
  input_schema.json
  sampled_horizons_train.feather
  non_overlap_horizons_val.feather
  sampled_horizon_labels_train.feather
  non_overlap_horizon_labels_val.feather
  phase1_config.yaml
  phase1_report.json
```

约束：

- Phase II 不重新训练 Phase I encoder、decoder、codebook。
- Phase II 不重新生成 demonstration label。
- train 的 `code_label` 固定来自 `sampled_horizon_labels_train.feather`；val/evaluation 的 `code_label` 固定来自 `non_overlap_horizon_labels_val.feather`。
- test label 不进入训练和 checkpoint selection。
- Phase II selector obs 特征列等于 Phase I `input_schema.json.feature_columns`。
- 持仓信息不进入 selector 决策输入。

## 5. Horizon-Level MDP

### 5.1 State

严格按论文：

```text
s^sel_t = s_t
```

实现上即 horizon 第一根 bar 的 Phase I market feature vector：

```python
obs = dataset.get_selector_state(horizon_idx)
```

不拼接上一 horizon 仓位、不拼接历史统计、不拼接账户净值、不拼接近期收益等额外状态。

持仓只属于交易环境内部状态，用于成交、换仓成本和 reward 结算；它不作为 selector 的 observation。

### 5.2 Selected Code

```text
selected_code_t = k_t in {0, 1, ..., K-1}
```

`K` 等于 Phase I codebook 中 archetype 数量。

### 5.3 Transition

执行 `selected_code = k`：

1. 从 Phase I codebook 取 archetype embedding `e_k`。
2. 冻结 Phase I decoder 逐步生成 base trading actions。
3. 用统一交易环境计算 step rewards。
4. 返回 horizon reward：

```text
r^sel_t = sum_{tau=t}^{t+h-1} r_tau^step
```

base trading action 生成必须是 streaming / step-by-step：

```text
for tau in [t, ..., t+h-1]:
  observe current state s_tau
  base_action_tau = frozen_decoder.decode_step(s_tau, e_k, decoder_hidden_state)
  execute/store base_action_tau
```

禁止用整段 `s_{t:t+h-1}` 一次性 batch decode 作为训练或评估主路径。批量 decode 容易让 decoder 或实现细节访问 horizon 内未来状态，造成未来信息泄漏；即使底层网络理论上是单向的，Phase II 环境也必须通过逐步接口约束可见信息边界。

### 5.4 Done

一个 episode 是 train split 的时间顺序 non-overlap horizon 序列。走到 split 末尾时 `done=True`，然后 reset 到 train split 开头。

### 5.5 HorizonDQNEnv 具体流程

`HorizonDQNEnv` 是 Phase II Double DQN 的环境层，接口保持单 env、单 cursor、按时间顺序遍历 horizon。

核心职责：

- `reset()` 返回当前 horizon 第一根 bar 的 selector obs。
- `step(selected_code)` 执行一个完整 horizon。
- 内部调用冻结 Phase I decoder 的 `decode_step()` 逐步生成 base trading actions。
- 内部调用交易结算逻辑计算 step rewards、手续费、滑点和 horizon reward。
- 返回 DQN transition 所需的 `next_obs, reward, done, info`。

环境内部状态：

```python
class HorizonDQNEnv:
    dataset
    frozen_decoder
    trading_env
    horizon_indices
    cursor
    initial_position = 0
```

按论文原始语义，`initial_position` 不进入 selector obs。第一版默认每个 horizon 的交易 replay 从 flat 仓位开始：

```text
initial_position = 0
```

这样 selector 的可见状态就是完整决策状态：

```text
obs = market_features_at_horizon_start
```

不设计跨 horizon 持仓延续。否则上一 horizon 终止仓位会成为影响 reward 的隐藏状态，但论文又没有把它放进 `s^sel`。

`reset()`：

```python
def reset():
    cursor = 0
    return selector_obs(cursor)
```

`selector_obs(cursor)`：

```python
idx = horizon_indices[cursor]
obs = dataset.get_selector_state(idx)
return obs
```

`step(selected_code)`：

```python
def step(selected_code):
    idx = horizon_indices[cursor]
    horizon_states = dataset.get_horizon_states(idx)
    horizon_inputs = dataset.get_horizon_inputs(idx)
    code_label, is_labeled = dataset.get_code_label(idx)

    frozen_decoder.reset(code_id=selected_code)

    base_actions = []
    for tau in range(horizon_len):
        state_tau = horizon_states[tau]
        base_action_tau = frozen_decoder.decode_step(state_tau)
        base_actions.append(base_action_tau)

    trading_env.reset(horizon_inputs, initial_position=0)
    step_rewards, step_infos = trading_env.replay(base_actions)

    reward = sum(step_rewards)
    cursor += 1
    done = cursor >= len(horizon_indices)
    next_obs = zero_selector_obs() if done else selector_obs(cursor)

    info = HorizonDQNStepInfo(
        selected_code=selected_code,
        code_label=code_label,
        is_labeled=is_labeled,
        reward=reward,
        base_actions=base_actions,
        step_rewards=step_rewards,
        final_position=step_infos[-1].filled_position,
        cost_paid=sum(i.fee + i.slippage for i in step_infos),
    )

    return next_obs, reward, done, info
```

`HorizonDQNStepInfo`：

```python
@dataclass
class HorizonDQNStepInfo:
    selected_code: int
    code_label: int | None
    is_labeled: bool
    reward: float
    base_actions: list[int]
    step_rewards: list[float]
    final_position: int
    cost_paid: float
```

重要边界：

- `selected_code` 是 Phase II 选择的 archetype code。
- `base_actions` 是 Phase I decoder 逐步生成的交易目标仓位。
- decoder 的每个 `decode_step(state_tau)` 只接收当前 `state_tau` 和已选 archetype code 的内部上下文。
- 交易环境在 replay 内部维护当前持仓，用于计算换仓成本和 step reward。
- 当前持仓不回灌给 selector，也不回灌给 decoder。
- 不做 gap mode、mid-horizon emergency flatten、risk guardrail、truncated rollout。

## 6. Q Network

`QNetwork` 输入 `s^sel`，输出每个 archetype 的 Q 值：

```text
Q_theta(s^sel) -> [Q(s,0), ..., Q(s,K-1)]
```

默认结构：

```text
Linear(obs_dim, 256)
ReLU
Linear(256, 128)
ReLU
Linear(128, K)
```

配置：

```yaml
q_network:
  hidden_dims: [256, 128]
  activation: relu
```

## 7. Double DQN

### 7.1 Replay Buffer

存储 transition：

```python
Transition(
    obs,
    selected_code,
    reward,
    next_obs,
    done,
    code_label,
    is_labeled,
)
```

Replay buffer 使用普通 uniform sampling。

### 7.2 TD Target

Double DQN 使用 online network 选择 next selected code，用 target network 估值：

```text
next_code = argmax_k Q_online(next_obs, k)
target = reward + gamma * (1 - done) * Q_target(next_obs, next_code)
td_loss = Huber(Q_online(obs, selected_code), target)
```

### 7.3 Demo Loss

论文目标中有 KL 项：

```text
alpha * KL(demo_code_label_t || pi_phi^sel(. | s_t^sel))
```

DQN 没有显式 policy。这里用 Q 值经 softmax 得到选择分布：

```text
p_theta(a | s) = softmax(Q_theta(s))[a]
```

对有 label 的样本：

```text
demo_loss = CE(Q_online(obs_labeled), code_label)
```

总损失：

```text
loss = td_loss + alpha * demo_loss
```

没有 label 的样本只计算 TD loss。

### 7.4 Target Network Update

Double DQN 必须维护 target network：

```text
Q_target <- Q_online
```

按固定间隔 hard update。

## 8. Exploration

使用标准 DQN 风格的 epsilon-greedy behavior policy。这里 `epsilon` 表示随机探索概率，`1 - epsilon` 表示利用模型当前 Q 值选择 code 的概率。

- `1 - epsilon` 的比例根据当前 `obs` 通过 `Q_online` 选择 code。
- `epsilon` 的比例执行随机探索。

例如 `epsilon=0.6` 时，40% 训练 horizon 使用模型 `argmax Q(obs)` 选 archetype，60% 训练 horizon 随机选 archetype。

```text
if np.random.uniform() < (1 - epsilon):
  selected_code = argmax_k Q_online(obs, k)
else:
  selected_code = random selected_code in [0, K)
```

含义：

- 训练采样阶段的利用分支必须依赖 `obs` 和当前 Q network。
- demo label 不直接替代模型决策，只通过 `demo_loss` 约束 Q 分布向 demonstration archetype choices 靠近。
- random 分支保证 DQN 能观察当前贪心 code 以外 archetype 的真实 horizon reward。
- 无 label 样本仍然可以通过 Q 贪心或随机探索生成 transition，只是不计算 demo loss。
- Validation 和 backtest 不使用 demo label，也不使用随机探索，始终 `argmax Q`。

`epsilon` 按 epoch 线性衰减：

```text
progress = min(current_epoch / epsilon_decay_epochs, 1.0)
epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
```

默认建议：

```text
起始阶段: 60% random / 40% Q greedy
衰减结束后: 20% random / 80% Q greedy
```

这样早期收集足够多非贪心 code 的 reward，对 Q 值做探索；后期逐步增加模型利用比例。demo 对模型的引导来自 `alpha * demo_loss`，而不是在 behavior policy 中直接强制选择 demo label。

配置：

```yaml
exploration:
  epsilon_start: 0.60
  epsilon_end: 0.20
  epsilon_decay_epochs: 20
```

Validation 和 backtest 使用：

```text
argmax_a Q_online(obs, a)
```

## 9. Demo Pretrain

可选使用 Phase I demo label 对 Q network 做预训练：

```text
loss = CE(Q_online(obs), code_label)
```

该步骤对应论文中让 selector 接近 demonstration archetype choices 的目标，但只作为 DQN 正式训练前的初始化。

配置：

```yaml
demo_pretrain:
  enabled: true
  epochs: 20
  lr: 0.0003
  batch_size: 512
```

记录：

- train label accuracy
- val label accuracy
- train demo CE
- val demo CE

## 10. 训练流程

训练组织参考 MacroHFT 这类高频交易 DQN 训练流程：按 epoch 顺序遍历训练时间段，使用 replay buffer 做 off-policy 更新，并定期用 target network 稳定 TD target。但算法语义仍然是本文档定义的 ArchetypeTrader Phase II horizon-level archetype selection；不引入 MacroHFT 的市场划分、memory module 或 hyper-agent。

训练主流程实现在 `src/phase2_dqn/trainer.py`：

```text
load Phase I artifacts
load train/val horizons and labels
build HorizonDQNEnv(train)
build Q_online and Q_target
optional demo pretrain
copy Q_online -> Q_target

global_step = 0
for epoch in range(total_epochs):
  epsilon = linear_decay_by_epoch(epoch)
  obs = env.reset()

  for horizon_step in train_horizons_in_chronological_order:
    selected_code = select_code_epsilon_greedy(obs, Q_online, epsilon)
    next_obs, reward, done, info = env.step(selected_code)
    replay.add(obs, selected_code, reward, next_obs, done, info.code_label, info.is_labeled)
    obs = env.reset() if done else next_obs
    global_step += 1

    if replay.size >= replay_warmup_steps:
      batch = replay.sample(batch_size)
      loss = double_dqn_loss(batch)
      optimizer.step()

    if global_step % target_update_interval == 0:
      Q_target <- Q_online

  evaluate validation at epoch end
  save checkpoint if validation metric improves
```

`scripts/train_phase2_dqn.py` 的伪代码：

```python
def main():
    args = parse_args()
    config = Phase2DQNConfig.from_yaml(args.config)
    config = config.override_from_cli(args)
    run_context = init_run_context(config)
    set_random_seed(config.seed)

    trainer = Phase2DQNTrainer(config=config, run_context=run_context)
    trainer.run()
```

脚本层不得包含 DQN 训练细节。所有会影响算法行为的逻辑必须进入 `src/phase2_dqn/`，方便单元测试和复用。

## 11. Evaluate

Evaluate 用于训练过程中的 validation、训练结束后的 best checkpoint 验证，以及独立 backtest。Evaluate 不参与探索，不读取 demo label 做决策。

### 11.1 Code 选择

Evaluate 阶段对每个 horizon 使用确定性贪心 code：

```text
selected_code = argmax_k Q_online(obs, k)
```

约束：

- 不使用 epsilon。
- 不使用 demo label 选择 code。
- 不随机采样 selected code。
- 不更新 replay buffer。
- 不更新 Q network。
- 不调用 DP。

### 11.2 Validation Evaluate

训练过程中按固定间隔对 val split 做 evaluate：

```text
for horizon in val_horizons in chronological order:
  obs = selector state at horizon start
  selected_code = argmax Q(obs)
  next_obs, reward, done, info = val_env.step(selected_code)
  collect per-horizon record
```

Validation evaluate 输出：

- `val_net_return`
- `val_sharpe_ratio`
- `val_max_drawdown`
- `val_code_counts`
- `val_active_archetype_ratio`
- `val_demo_label_accuracy`
- `val_per_horizon_records`

`val_demo_label_accuracy` 只作为诊断。它可以读取 val label 计算 selector 与 demo 的一致率，但不能参与 code 选择。

### 11.3 Checkpoint Selection

论文描述为保留 validation performance 最好的 checkpoint。默认选择指标：

```yaml
selection_metric: val_net_return
```

当当前 checkpoint 的 `val_net_return` 优于历史最佳时：

```text
save best_selector.pt
update checkpoint_manifest.json
```

同时每次 evaluate 都保存 `last_selector.pt`，便于中断后恢复。

### 11.4 Backtest Evaluate

训练完成后，独立脚本 `scripts/backtest_phase2_dqn.py` 加载：

```text
best_selector.pt
decoder.pt
codebook.pt
input_schema.json
test horizons
```

并在 test split 上做确定性 walk-forward：

```text
selected_code = argmax Q(obs)
```

test label 默认不加载。若为了 posthoc 诊断读取 test label，必须保证：

- 不影响 checkpoint selection。
- 不影响 code 选择。
- 不影响任何训练超参。
- report 中写明 `test_label_used_for_posthoc_only=true`。

### 11.5 Evaluate Records

每个 horizon 输出一条记录：

```text
sample_id
horizon_start
horizon_end
selected_code
code_label
is_labeled
reward_raw
reward_scaled
base_actions
step_rewards
```

train/val/test records 分别写出，test records 中 `code_label` 默认为空。

## 12. Validation Metrics

Validation 按时间顺序 non-overlap horizon deterministic replay。

输出：

- net return
- sharpe ratio
- max drawdown
- selected code counts
- active archetype ratio
- demo label accuracy
- per-horizon records

Checkpoint 选择遵循论文：“retain checkpoint with best validation performance”。默认主指标：

```yaml
selection_metric: val_net_return
```

也可以用已有 composite score，但文档默认不引入复杂 selection guardrails。

## 13. 配置

```yaml
run:
  pair: FU
  phase1_batch_id: batch_03
  phase2_batch_id: phase2_dqn_001
  artifact_root: artifacts
  seed: 42
  device: cuda

data:
  horizon: 72
  train_horizons: sampled_horizons_train.feather
  val_horizons: non_overlap_horizons_val.feather
  train_labels: sampled_horizon_labels_train.feather
  val_labels: non_overlap_horizon_labels_val.feather
  input_schema: input_schema.json
  selector_include_position: false
  horizon_initial_position: 0
  use_test_label: false

phase2_dqn:
  total_epochs: 50
  gamma: 0.99
  lr: 0.0003
  batch_size: 256
  replay_capacity: 100000
  replay_warmup_steps: 2048
  target_update_interval: 1000
  max_grad_norm: 10.0
  alpha: 1.0

q_network:
  hidden_dims: [256, 128]
  activation: relu

exploration:
  epsilon_start: 0.60
  epsilon_end: 0.20
  epsilon_decay_epochs: 20

demo_pretrain:
  enabled: true
  epochs: 20
  lr: 0.0003
  batch_size: 512

evaluation:
  eval_interval_epochs: 1
  selection_metric: val_net_return
  deterministic: true
  save_per_horizon_records: true

checkpoint:
  save_best: true
  save_last: true
  best_filename: best_selector.pt
  last_filename: last_selector.pt
```

配置语义：

- `run`：一次 Phase II DQN 训练运行的身份、产物根目录、随机种子和设备。
- `data`：Phase I 产物和 Phase II horizon/label 文件。`selector_include_position=false` 固定为论文原始语义；`horizon_initial_position=0` 表示每个 horizon 从 flat replay。
- `phase2_dqn`：Double DQN 主训练参数。`alpha` 是 demo CE 项权重，对应论文 KL/demo consistency 权重。
- `q_network`：selector Q network 结构。
- `exploration`：训练采样的 epsilon-greedy 参数，`epsilon` 按 epoch 线性衰减。
- `demo_pretrain`：正式 DQN 前的 demo label 预训练。
- `evaluation`：validation evaluate 频率和 checkpoint 选择指标。`deterministic=true` 表示 evaluate 永远使用 `argmax Q`。
- `checkpoint`：selector checkpoint 文件名。

## 14. 产物

```text
artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2_dqn/
  phase2_dqn_config.yaml
  best_selector.pt
  last_selector.pt
  checkpoint_manifest.json
  train_stats.feather
  val_stats.feather
  per_horizon_records_val.feather
  phase2_dqn_report.json
```

`phase2_dqn_report.json` 至少包含：

```json
{
  "algorithm": "double_dqn",
  "phase2_semantics": "paper_horizon_level_archetype_selection",
  "parallel_exploration": false,
  "phase1_batch_id": "",
  "phase2_batch_id": "",
  "config_hash": "",
  "demo_label_semantics": "vq_encoder_label_of_full_demonstration_chunk",
  "demo_pretrain": {},
  "val_metrics": {},
  "test_used_for_selection": false
}
```

## 15. 测试计划

单元测试：

```text
tests/unit/phase2_dqn/test_q_network.py
tests/unit/phase2_dqn/test_replay_buffer.py
tests/unit/phase2_dqn/test_double_dqn_loss.py
tests/unit/phase2_dqn/test_demo_pretrain.py
tests/unit/phase2_dqn/test_horizon_dqn_env.py
```

集成测试：

```text
tests/integration/test_phase2_dqn_smoke.py
tests/integration/test_phase2_dqn_no_test_label_leakage.py
tests/integration/test_phase2_dqn_backtest.py
```

必须覆盖：

- Q network 输出 shape `[batch, K]`。
- Double DQN target 使用 online argmax 和 target value。
- `done=True` 时 target 不 bootstrap。
- demo CE 只作用于 labeled 样本。
- HorizonDQNEnv step 调用冻结 decoder 和 TradingEnv。
- test label 不进入训练和 checkpoint selection。

## 16. 实施顺序

1. `config.py` 和 `scripts/train_phase2_dqn.py` CLI skeleton。
2. `dataset.py` 和 `env.py`。
3. `q_network.py`、`replay_buffer.py`、`losses.py`。
4. `demo_pretrain`。
5. `trainer.py` 中实现完整训练主流程。
6. `evaluator.py`、`checkpoint.py`、`report.py`。
7. smoke test。
8. FU 数据 pretrain-only 观察。
9. 完整 Double DQN 训练。
