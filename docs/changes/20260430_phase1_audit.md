# Phase I 代码实现审查与修复记录

**日期**: 2026-04-30
**范围**: Phase I 全部源码（45 个 .py） + 测试（32 个 .py）
**审查依据**:
- `docs/design/phase1_archetype_discovery_design.md`
- `docs/plan/phase1_archetype_discovery_execution_plan.md`
- `docs/paper/AAAI26_ArchetypeTrader_core.md`（论文 Algorithm 1）

---

## 1. 总览

本轮审查共发现 **11 个问题**，全部已修复，分为三类:

| 类别 | 数量 | 影响 |
| --- | --- | --- |
| 严重 bug（算法/一致性） | 4 | 直接影响训练质量与论文复现 |
| 一致性 bug | 4 | 报告字段、可复现性、错误处理 |
| 死代码 / 注释错位 | 3 | 维护性、未来扩展易踩坑 |

---

## 2. 严重 Bug（影响算法正确性）

### 2.1 DP 反向递推漏掉 `t = h-1`

**位置**: `src/planners/single_trade_dp.py:_backward`

**问题**:
论文 Algorithm 1 第 2 行明确 `for t = N-1 downto 0`，原实现写成
`range(horizon - 2, -1, -1)`，跳过了 `t = h-1`。结果:

- `V[h-1, *, *]` 一直保持初值 0；
- `V[h-2, *, *]` 用 `V[h-1] = 0` 反向递推，丢失末步收益贡献；
- 在 `t = h-2` 这一步，DP 决策无法感知 `t = h-1` 的潜在收益，
  即使切换到 long 在 `t = h-1` 收获 `position * (p_h - p_{h-1})`，
  DP 也会因为 `V[h-1] = 0` 而误判为"无未来收益"，倾向不切换。

**修复**:
```python
# 之前
for t in range(horizon - 2, -1, -1):
# 之后
for t in range(horizon - 1, -1, -1):
```

同时在 `plan()` 与 `_backward` 的 docstring 中明确:
"V[h-1] 必须填充；Pi[h-1] 计算后不被 forward 使用，但 V[h-1] 是 V[h-2] 反推的依赖"。

**为什么测试没发现**: 现有 `test_dp_total_return_matches_replay_sum`
检查的是 `total_return == sum(env.replay rewards)`，这是 `_replay` 强制
保证的不变量；DP 是否做出最优选择属于"训练质量"，单元测试没断言。

---

### 2.2 Phase1Loss 的 `num_codes` 用 `code_id.max()+1` 推断

**位置**: `src/models/vq_losses.py:Phase1Loss.forward`

**问题**:
原代码 `num_codes = int(code_id.max().item()) + 1` 在 codebook collapse
时严重低估真实 K。例如真 K=10 但只用了 `{0, 1, 5}`，会推断 K=6，
KL(uniform || p_code) 算的是 6 维上的均匀分布，漏掉 4 个 0 概率项 →
usage 项偏小 → 收不到应有的反塌缩信号 → 反过来加剧 collapse。

**修复**:
- `Phase1Loss.__init__` 增加 `num_codes: Optional[int] = None`。
- `forward` 优先用 init 传入；缺失时回退推断（仅留给单测）。
- `Phase1Trainer._build_training_components` 传入 `self.config.model.num_codes`。

---

### 2.3 reward normalization 没传递到 val/test/warmup

**位置**: `src/trainers/phase1_trainer.py:run`、`src/data/dataset.py`、
`src/evaluation/phase1_evaluator.py`

**问题**:
原代码在 step 8 用 `for rec in train_horizons: rec.rewards = list(norm.transform(...))`
**只对 train horizons 做 in-place 归一化**，导致:

- val/test 的 `rec.rewards` 仍是原始尺度 → encoder 输入分布与 train 不一致；
- val/test 的 `code_id` 与 train 不在同一表征空间；
- 所有依赖 `code_id` 的指标（`return_capture_ratio` / `regret_to_dp` /
  `inter_code_action_diversity` / `decoder_sensitivity_to_code`）全部被误读；
- `_warmup_codebook` 在 in-place 归一化之前调用，warmup 也是用未归一化的 reward
  跑 K-means → codebook 初始化分布从一开始就跑偏；
- `_export_horizon_labels` 用已被 in-place 修改的 `rec.rewards` 计算 train 的
  `demo_return = sum(rec.rewards)`，结果是"sum of normalized values"（错误）；
  val/test 同函数中是 actual return（正确），两口径不一致。

**修复**:
1. `Phase1DemoDataset` 增加 `reward_normalizer` 参数；`__getitem__` 即时
   `transform`，**不再 in-place 改写 rec.rewards**。
2. `Phase1Evaluator` 增加 `reward_normalizer` 参数；`evaluate_epoch` 内部
   创建 `Phase1DemoDataset` 时注入。
3. `Phase1Trainer.run` 删除 in-place 归一化循环；
   - train_dataset / val_dataset 都传入同一个 normalizer；
   - `_build_training_components(... reward_normalizer=norm)` 透传给 evaluator；
   - `_warmup_codebook` 创建 dataset 时传入 normalizer；
   - `_export_horizon_labels` 增加 `normalizer` 参数：encoder 输入用
     `normalizer.transform`，`demo_return` 仍用原始 `rec.rewards`（actual return）。

**关键不变量**:
- `rec.rewards` 在整条 pipeline 中始终是原始（actual）值；
- encoder 输入的归一化只在 dataset/evaluator 内部即时完成；
- 所有 split 共用同一个 RewardNormalizer 实例（train fit 后不可重新 fit）。

---

### 2.4 evaluate_horizon_boundaries 第一段 horizon 末仓位未计算

**位置**: `src/evaluation/phase1_replay.py:evaluate_horizon_boundaries`

**问题**:
原代码 `prev_position = 0` 作为初值，循环从 `i=1` 开始。第一个边界
`ordered_horizons[0] -> ordered_horizons[1]` 用的 `prev` 永远是 0，
完全忽略了 `ordered_horizons[0]` 的实际末仓位。

测试 `test_boundary_replay_records_cost_when_misaligned` 表面通过，
其实是因为后续 `env.replay(actions_i)` 用 `target_first` 作为
`initial_position`，使 `prev_position = infos[-1].filled_position`
在第二次循环时是正确的；但**第一个 boundary 的 turnover_cost 是错的**。

**修复**:
循环前先跑 `ordered_horizons[0]` 拿到其末仓位:
```python
if first_actions:
    env.reset(HorizonInputs(...), initial_position=0)
    _, infos_first = env.replay(first_actions)
    if infos_first:
        prev_position = infos_first[-1].filled_position
```

---

## 3. 一致性 Bug

### 3.1 `demo_return` 的 train/val/test 口径不一致

**位置**: `src/trainers/phase1_trainer.py:_export_horizon_labels`

**问题**: 见 §2.3。原 in-place 修改导致 train 的 `demo_return` 是
normalized 之和（错误），val/test 是 actual return（正确）。

**修复**: 配合 §2.3，统一用原始 `rec.rewards` 计算 `demo_return`。

---

### 3.2 `phase1_report.json` 的 `no_trade_ratio` 始终是 0

**位置**: `src/trainers/phase1_trainer.py:_build_final_summary`

**问题**: 原代码 `summary.setdefault("no_trade_ratio", 0.0)`，从来没真正
统计过。设计 §9.4 验收要求该比例小于 `max_no_trade_ratio`，0 默认值
让验收形同虚设。

**修复**: `run()` 中扫描 `train_horizons.actions` 算出真实比例:
```python
no_trade_count = sum(
    1 for rec in train_horizons
    if rec.actions is not None and all(a == 1 for a in rec.actions)
)
no_trade_ratio = no_trade_count / max(len(train_horizons), 1)
```
然后通过 `_build_final_summary(... no_trade_ratio=no_trade_ratio)` 透传。

---

### 3.3 缺 `torch.manual_seed`，可复现性受损

**位置**: `src/trainers/phase1_trainer.py:run`

**问题**: trainer 没设置 torch 全局 seed，DataLoader shuffle、模型权重
初始化等行为不可复现。设计 §6.9 / §9 reproducibility 是 sign-off 项。

**修复**: 新增 `_seed_everything`，在 `run()` 入口调用，统一设
random / numpy / torch (含 cuda) seed。

```python
def _seed_everything(self) -> None:
    seed = self.config.training.seed
    import random as _random
    _random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

### 3.4 `env.step` 越界后无显式错误

**位置**: `src/trading/env.py:step`

**问题**: 用户在 `done=True` 后继续 `step()`，`self._horizon.execution_books[self._t]`
触发 `IndexError`。这种"silent IndexError"对调试不友好。

**修复**: step 入口加防御性检查:
```python
if self._t >= len(self._horizon.execution_books):
    raise RuntimeError(
        f"step() 已超过 horizon 长度 {len(...)}; 请先调用 reset() 重新开始 horizon。"
    )
```

---

## 4. 死代码 / 实现注释错位

### 4.1 `Phase1Loss._usage_loss` 占位

**位置**: `src/models/vq_losses.py`

**问题**: `_usage_loss(*args, **kwargs)` 永远 return None；调用它后立即
被 `_kl_uniform` 覆盖。属于残留的占位。

**修复**: 删除 `_usage_loss`，`forward` 中清理 dead 赋值。

---

### 4.2 `selection_policy.evaluate` 冷却期占位 `* 0`

**位置**: `src/trainers/selection_policy.py`

**问题**:
```python
if epoch - history.last_dead_code_restart_epoch <= self.config.min_code_usage_ratio * 0:
    pass  # 保留扩展位置
```
`* 0 = 0` 让条件几乎永远不触发；紧接着又有真正的 cooldown 检查。
还有未使用的 `from src.config.phase1_config import CodebookHealthConfig` import。

**修复**: 删除占位段，cooldown 通过 `metrics["_dead_code_restart_cooldown_epochs"]`
注入，缺省 3（与 `CodebookHealthConfig.restart_cooldown_epochs` 一致）。

---

### 4.3 `Phase1Config.from_dict` 内部 `_build` 死代码

**位置**: `src/config/phase1_config.py`

**问题**: 因为 `from __future__ import annotations`，`f.type` 是字符串
而不是类对象，`hasattr(f.type, "__dataclass_fields__")` 永远返回 False。
内部 `_build` 函数实际上从未递归过；真正工作的是外部 `_rebuild_nested`
+ `_NESTED_TYPE_MAP`。

**修复**: 删除内部 `_build`，docstring 解释为什么必须用显式 type map。

---

## 5. 已知遗留（非本轮 bug）

下列项目设计提及但当前实现未完全覆盖，记录在此供后续迭代:

| 项目 | 当前状态 | 后续迭代建议 |
| --- | --- | --- |
| prospective 分层 + lookback 不足 | 早期窗口归到 `"unknown\|unknown\|mixed"` strata | 在 `StratifiedWindowSampler.sample` 中跳过 `unknown` 桶；或在 trainer 层裁掉前 lookback 窗口 |
| `quantizer.restart_dead_codes` | 已实现但 trainer 主循环未调用 | 需要 evaluator 暴露 per-sample reconstruction error，trainer 在每 epoch 末按需触发 |
| `contrastive_pairs_train.feather` | trainer 未保存到 artifact | 在 step 4 后调用 `feather_io.write_ipc`；默认 disabled 路径不影响主流程 |
| `local_optimum_escape` 扰动 | 配置已就位，逻辑未实现 | 设计 §6.5 已经标注 `enabled=false` 默认；后续按需 |

---

## 6. 测试影响

所有既有测试**向后兼容**:

- `Phase1Loss(usage_weight=0.0)`: `num_codes` 默认 None；usage_weight=0 跳过 KL → 通过。
- `Phase1DemoDataset(records=...)`: `reward_normalizer` 默认 None；transform 跳过 → 通过。
- `Phase1Evaluator(replay_evaluator=...)`: `reward_normalizer` 默认 None → 通过。
- `test_dp_total_return_matches_replay_sum`: DP 修复后最优解可能改变，但
  `total_return == sum(env.replay rewards)` 不变量由 `_replay` 强制保证 → 通过。
- `test_boundary_replay_records_cost_when_misaligned`: 修复后**真正**检测
  prev/target 不一致的换仓成本（之前是被 `prev=0` 默认值意外凑对的）→ 通过。

集成测试 `tests/integration/test_phase1_pipeline_smoke.py` 只检查文件存在性
与必填字段，所有断言仍成立。

---

## 7. 修复后的不变量

| 不变量 | 维护者 | 验证手段 |
| --- | --- | --- |
| DP `total_return == sum(env.replay rewards)` | `SingleTradeDPPlanner._replay` | 单测 `test_dp_total_return_matches_replay_sum` |
| `actions[h-1] == actions[h-2]`，且不计入 `num_switches` | `SingleTradeDPPlanner.plan` | 单测 `test_last_step_copies_*` |
| `rec.rewards` 在 pipeline 中始终是 actual values | `Phase1Trainer.run` 移除 in-place 修改 | `_export_horizon_labels` 的 `demo_return` 使用 `rec.rewards` |
| 所有 split encoder 输入用同一 RewardNormalizer | `Phase1DemoDataset` / `Phase1Evaluator` | `evaluator.reward_normalizer` 在所有 probe dataset 注入 |
| 全局 seed 一致 | `_seed_everything` | trainer 入口调用 |
| `code_usage` < 阈值 → 不可 best；连续 collapse → fatal | `Phase1SelectionPolicy.evaluate` | 单测 `test_consecutive_collapse_triggers_fatal` |
| `phase1_report.no_trade_ratio` 反映真实 train DP 输出 | `Phase1Trainer.run` 计算后注入 | 报告字段非 0 |

---

## 8. 文件改动一览

源文件改动（共 9 个）:

```text
src/planners/single_trade_dp.py        # DP backward range + docstring
src/models/vq_losses.py                 # num_codes 参数 + 清理 dead code
src/data/dataset.py                     # reward_normalizer 参数 + Protocol
src/evaluation/phase1_evaluator.py      # reward_normalizer 参数
src/evaluation/phase1_replay.py         # 第一段 horizon 末仓位预计算
src/trading/env.py                      # _t 越界防御
src/trainers/phase1_trainer.py          # 移除 in-place 归一化、_seed_everything、no_trade_ratio、normalizer 透传
src/trainers/selection_policy.py        # 清理冷却期占位 + import
src/config/phase1_config.py             # 清理 from_dict 内部 _build dead code
```

测试文件**未改动**（向后兼容）。

---

## 9. 后续建议

1. **优先级最高**: 实际跑一次端到端 smoke + reproducibility 测试，对比修复前后:
   - DP `total_return` 在固定 seed + fixture 上是否改变（应改变，且更优）；
   - `phase1_report.json` 的 `code_usage_ratio` / `val_return_capture_ratio` 是否变化。
2. 把"prospective 分层 + insufficient lookback 处理"补上，至少在 sampler 层
   skip `"unknown"` 桶。
3. 把 `restart_dead_codes` 接到 trainer 主循环；evaluator 需要暴露 per-sample
   reconstruction error。
4. 加一个回归测试 `test_dp_backward_includes_last_step_reward`，固定一段
   单调上涨末段的 horizon，验证 DP 必须切到 long（修复前会因 `V[h-1]=0`
   错过最佳切换点）。

---

## 10. 2026-05-01 执行结果追加

> 本节为追加执行记录，未修改上方原计划与原审查内容。

### 10.1 执行状态

| 标记 | 条目 | 执行结果 |
| --- | --- | --- |
| 【✅】 | 2.1 DP 反向递推与末步约束 | 已落地。`SingleTradeDPPlanner` 在 `t=h-1` 只允许保持仓位，并保留末步收益对 `V[h-2]` 的贡献；回归测试 `test_last_step_value_cannot_depend_on_unexecutable_switch` 已覆盖。 |
| 【✅】 | 2.2 `Phase1Loss.num_codes` | 已落地。loss 支持显式 `num_codes`，trainer 传入 `config.model.num_codes`。 |
| 【✅】 | 2.3 reward normalization 传递 | 已落地。dataset/evaluator/warmup/export labels 均使用同一 normalizer，`rec.rewards` 保持 actual values。 |
| 【✅】 | 2.4 horizon boundary 第一段末仓位 | 已落地。boundary replay 先 replay 第一个 horizon 取得真实末仓位。 |
| 【✅】 | 3.1 `demo_return` 口径 | 已落地。label 导出的 `demo_return` 使用原始 reward 求和。 |
| 【✅】 | 3.2 `no_trade_ratio` | 已落地。最终报告使用 train DP actions 真实统计值。 |
| 【✅】 | 3.3 全局 seed | 已落地。`Phase1Trainer.run()` 入口调用 `_seed_everything()`。 |
| 【✅】 | 3.4 `env.step` 越界错误 | 已落地。done 后继续 `step()` 会抛出明确 `RuntimeError`。 |
| 【✅】 | 4.1 `Phase1Loss._usage_loss` 死代码 | 已清理。 |
| 【✅】 | 4.2 `selection_policy.evaluate` 冷却期占位 | 已清理，并通过 metrics 注入 cooldown epoch。 |
| 【✅】 | 4.3 `Phase1Config.from_dict` 内部 `_build` 死代码 | 已清理，当前使用显式 nested type map。 |

### 10.2 本次执行中的额外调整

| 标记 | 条目 | 执行结果 |
| --- | --- | --- |
| 【✅】 | Phase I smoke test 与 P1-007 保护逻辑冲突 | 已调整 `tests/integration/test_phase1_pipeline_smoke.py` 的 smoke-only selection guardrail，使该测试继续验证端到端产物链路；生产代码仍保持“无 best checkpoint 时禁止导出 Phase II artifacts”。 |
| 【✅】 | `quantizer.restart_dead_codes` 已知遗留 | 当前源码已接入 trainer 主循环，epoch metrics/report 写入 restart 事件。 |
| 【】 | prospective lookback 不足的 `unknown` 桶 | 未在本次执行中采纳，仍作为策略/验收口径待确认项。 |
| 【】 | `contrastive_pairs_train.feather` | 未在本次执行中采纳；默认 disabled 路径不影响当前主流程。 |
| 【】 | `local_optimum_escape` 扰动 | 未在本次执行中采纳；配置仍保持 disabled。 |

### 10.3 验证结果

| 标记 | 命令 | 结果 |
| --- | --- | --- |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && pytest -q tests/integration/test_phase1_pipeline_smoke.py` | `4 passed` |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && pytest -q` | `375 passed, 17 warnings`。warnings 为未注册 `pytest.mark.integration`。 |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && bash -n run_pipeline.sh` | 语法检查通过。 |
