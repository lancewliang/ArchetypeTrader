# Phase I Causal Online Validation 设计

## 背景

当前 Phase I checkpoint selection 的 `full_validation` 使用 validation split 上的
DP demonstration:

```text
states + teacher actions + teacher rewards
  -> encoder / quantizer -> code_id
  -> decoder(states, code_id) -> student actions
  -> TradingEnv replay -> validation metrics
```

这能验证 VQ encoder/codebook/decoder 是否能把未参与训练的 DP
demonstration 压缩成稳定 archetype，并重构出有收益的行为。但它不是线上推理语义:
validation encoder 看到了整段 teacher actions 和 rewards，真实交易时这些信息不可用。

因此 `promote_to_best` 只能解释为 teacher-conditioned reconstruction best，不能单独解释为
codebook 已具备线上泛化盈利能力。

## 目标

在保持当前 teacher-conditioned validation 的基础上，新增 Phase I
causal online validation，并把它接入 best checkpoint guardrail:

```text
validation horizon time order
  -> state-prefix encoder chooses code_id without teacher actions/rewards
  -> decoder(states, code_id) emits actions
  -> TradingEnv replay with inherited initial_position
  -> online net return / Sharpe / MDD / capture guardrail
```

目标是让 `promote_to_best` 至少同时满足:

1. teacher-conditioned DP archetype reconstruction 合格。
2. teacher-free causal code selection + decoder replay 在 validation 上没有明显失效。

## 非目标

- 不在 Phase I 内引入完整 Phase II selector 训练。
- 不声称该验证等价于最终策略线上泛化；最终仍需 Phase II/III walk-forward 与 test/backtest。
- 不使用 validation/test DP planner 在线重算 teacher。

## 算法

### 1. 样本集合

`full_validation=True` 时使用完整 validation records；fast probe 不允许 promote
依赖该线上式指标，除非配置显式关闭 `require_for_best`。

records 必须保持数据处理产物中的稳定顺序。若后续 manifest 提供 non-overlap val
标签来源，应优先用于该验证；当前最小实现先使用 `val_records[:sample_size]`。

### 2. Causal code selection

当前 encoder 结构需要 `(states, actions, rewards)` 三路输入。线上验证不得喂 teacher
action/reward，因此使用保守的 state-prefix surrogate:

```text
prefix_states = horizon.states[:online_validation.code_prefix_steps]
neutral_actions = flat action, shape = [prefix_steps]
neutral_rewards = 0, shape = [prefix_steps]
encoder(prefix_states, neutral_actions, neutral_rewards)
  -> quantizer -> online_code_id
```

默认 `code_prefix_steps=1`，只使用 horizon 起点可见 state。若输入特征本身包含过去滚动因子，
则该 state 可代表线上可见上下文。增大 prefix 会放宽验证，但必须保证 prefix 内状态在下单前可见。

### 3. Decoder replay

拿到每个 horizon 的 `online_code_id` 后:

```text
z_q = codebook[online_code_id]
logits = decoder(full_horizon_states, z_q)
actions = argmax(logits)
TradingEnv.reset(horizon, initial_position=prev_terminal_position)
TradingEnv.replay(actions)
prev_terminal_position = last StepInfo.filled_position
```

decoder 仍是单向 LSTM；虽然工程上一次 forward 得到整段 logits，但第 `t` 步 logits
只依赖 `states[:t]` 与固定 `code_id`。env replay 使用逐步 `step()` 语义，且相邻
horizon 继承仓位，第一步换仓成本由同一 `CostModel` 扣除。

### 4. 指标

新增指标前缀为 `online_`:

- `online_validation_measured`
- `online_code_prefix_steps`
- `online_code_usage_ratio`
- `online_val_student_net_return`
- `online_val_return_capture_ratio`
- `online_val_regret_to_dp`
- `online_val_cost_paid`
- `online_val_sharpe_ratio`
- `online_val_max_drawdown`
- `online_val_max_drawdown_abs`
- `online_horizon_boundary_turnover_cost`
- `online_horizon_boundary_position_consistency`

capture 仍以同一批 validation horizon 的 DP teacher replay 总收益为上限参考，但解释为
online surrogate 相对 DP teacher 的收益捕获率，不等价于最终 Phase II selector capture。

### 5. Selection guardrail

新增配置组:

```yaml
selection_policy:
  online_validation:
    enabled: true
    require_for_best: true
    code_prefix_steps: 1
    min_return_capture_ratio: 0.0
    min_sharpe_ratio: 0.0
    max_drawdown: 0.2
```

当 `enabled && require_for_best` 时，若 online validation 未测量，或任一指标不达标，
`Phase1SelectionPolicy` 返回 `reject`，reason 前缀为 `online_validation_guardrail`。

## 报告语义

`phase1_report.json` 和 epoch metrics 必须同时保留 teacher-conditioned 指标与 online
指标。最终解释:

- `val_*`: demonstration encoder 条件下的 archetype reconstruction/replay。
- `online_*`: teacher-free state-prefix code selection 下的 causal replay。
- `promote_to_best`: 两者都通过当前 selection policy。

若 local smoke 使用 relaxed guardrails，可关闭 `require_for_best`，但报告仍应标记
`local_smoke_relaxed_guardrails`，不得 sign off。

## 风险与后续

- state-prefix surrogate 没有经过单独监督训练，可能过严或噪声较大；这是可接受的保守 guardrail。
- 更完整方案是在 Phase I 增加 state-only causal encoder head，或把 Phase II selector 的
walk-forward 结果反向作为 Phase I sign-off 条件。
- 若 validation records 是 opportunity-sampled overlapping horizons，online 指标仍可能偏高；
后续应优先切换到 non-overlap/full-time validation records。
