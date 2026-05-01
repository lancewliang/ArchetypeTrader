# Phase I 全量代码质量审查记录 (v2 复核版)

**日期**: 2026-05-01
**复核依据**: 论文、设计文档、全部 Phase I 源码、v1 审查报告
**审查方法**: 逐条交叉验证 v1 报告中每个问题与源码、设计文档、论文算法的一致性

---

## 第一部分: v1 审查报告逐条复核结论

**总体结论: v1 审查报告的 17 个问题全部属实，无错误项。** 以下逐条给出复核证据与源码定位。

---

### P1-001: DP 末步估值与实际 action 约束不一致

| 项目 | 内容 |
|------|------|
| **v1 结论** | Critical, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [single_trade_dp.py:L213](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L213): `for t in range(horizon - 1, -1, -1)` — `_backward()` 包含 `t = h-1`
- [single_trade_dp.py:L219](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L219): 在 `t=h-1, c=0` 时允许 `switch=1` (c+1=1 <= 1)，即 DP 估值允许最后一步切换
- [single_trade_dp.py:L246](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L246): `_forward()` 只遍历 `t in range(horizon - 1)` → `t=0..h-2`
- [single_trade_dp.py:L104-L105](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L104-L105): `actions[-1] = actions[-2]` — 末步强制复制，DP 在 `t=h-1` 计划的切换永远不执行

**算法级验证 (论文 Algorithm 1)**:

论文 Algorithm 1 第 13 行: `\hat{a}_{N-1} \leftarrow \hat{a}_{N-2}`。论文的正确语义是: **末步不参与 DP 最优化，仅复制倒数第二步**。当前实现的 `_backward()` 在 `t=h-1` 仍有 switch 自由度，这违反了论文约束。

以 `h=2`, prices=`[100, 100, 110]`, 零成本为例:
- `V[1][flat][0]` = 10 (允许在最后一步切 long)
- `V[0][flat][0]` = max(flat: 0+V[1][flat][0]=10, long: 0+V[1][long][1]=10) = 10
- Pi[0][flat][0] 可能选择 flat (把切换推迟到 t=1)
- forward: actions[0]=flat → plan(): actions[1]=actions[0]=flat
- **结果: [flat, flat] → 收益=0，而最优应为 10**

**建议**: 将 `_backward()` 的末步 (`t=h-1`) 强制只允许 `target_a == prev_a`，或把末步 reward 合并到 `t=h-2` 的同仓位递推。

---

### P1-002: reject_transition 统计没有统计 DP 候选转移拒绝

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [single_trade_dp.py:L172-L180](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L172-L180): `_precompute_transitions()` 中盘口深度不足 → `valid=False`，但**没有计数器**记录这些拒绝
- [single_trade_dp.py:L271-L272](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/single_trade_dp.py#L271-L272): `_replay()` 只收集**最终被选 action 路径**上的 reject_events
- [demo_generator.py:L107](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/planners/demo_generator.py#L107): `rejected = len(result.reject_events)` — 仅统计 replay 路径拒绝

**测试证据**:

`test_fail_when_dataset_reject_rate_exceeds` (line 99-106) 预期极小深度触发 `RejectTransitionExceeded`，实际未抛出。因为 DP 避开了所有不可成交换仓 → 全 flat → replay 无 reject → `dataset_reject_rate=0`。

**建议**: DPResult 增加 `precompute_rejected_count` 和 `precompute_rejected_by_pair` 字段。

---

### P1-003: prospective 对照 sign-off 未真正执行

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L819-L821](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L819-L821):
```python
def _hindsight_warning_triggered(self, summary: dict) -> bool:
    # 第一版没接前瞻对比数据，这里默认不告警；真正比较交给 Phase II 的对照流水线。
    return False
```
- 没有任何代码读取 `diagnostic_pair_batch_id` 对应的 `phase1_report.json`
- 没有比较 `val_return_capture_ratio`、`val_sharpe_ratio` 等指标差异
- 没有写入 `sampling_leakage_diagnostics.json` 中的 `metric_deltas`
- 没有在超阈值时标记 `hindsight_bias_warning="exceeded"` 并阻止 sign-off

**设计对照**: 设计 §3.4 明确要求:
> "任一指标差异超过阈值时，主实验 `phase1_report.json` 必须写入 `hindsight_bias_warning='exceeded'`，且主实验的 `best_vq_model.pt` 不可被声明为 sign-off 版本。"

---

### P1-004: 采样全局 min gap 没被 sampler 保证

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [stratified_sampler.py:L282-L303](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/stratified_sampler.py#L282-L303): `_pick_with_gap()` 的 `chosen_starts` 是**局部变量**，只检查当前 strata 内的已选窗口
- [stratified_sampler.py:L208-L220](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/stratified_sampler.py#L208-L220): 每个 strata 独立调用 `_pick_with_gap()`，跨 strata 无 gap 检查
- [stratified_sampler.py:L224-L256](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/stratified_sampler.py#L224-L256): 补采样时 `spare_pool` 排除了已选 indices，但**没有传入已选 window_start 做 gap 检查**
- [stratified_sampler.py:L258](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/stratified_sampler.py#L258): `sorted(set(sampled_indices))[:num_samples]` — 排序截断可能进一步破坏 gap 约束

**测试证据**: `test_min_gap_between_samples_enforced` 当前失败，出现 gap=5 < min_gap=10。

---

### P1-005: `next_row_execution` label 行号导出 off-by-one

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L726-L727](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L726-L727):
```python
last_execution_row = rec.start_index + len(rec.execution_books) - 1
last_markout_row = rec.start_index + len(rec.execution_books)
```

`len(rec.execution_books)` = h。所以:
- `last_execution_row = start + h - 1`
- `last_markout_row = start + h`

对于 `paper_formula`: 最后一步 `t=h-1`:
- execution_row = `start + h - 1` ✅
- markout_row = `start + h` ✅

对于 `next_row_execution`: 最后一步 `t=h-1`:
- [reward_alignment.py:L86-L89](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trading/reward_alignment.py#L86-L89): execution_row = `start + h`, markout_row = `start + h + 1`
- 但代码计算: execution_row = `start + h - 1`, markout_row = `start + h`
- **off-by-one** ❌

---

### P1-006: `_train_loop` 更新的 history 没返回给 `run()`

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L682](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L682):
```python
history = policy.update_history(history, metrics_for_select, verdict)
```
这是 `_train_loop()` 内部的局部变量重新赋值

- [phase1_trainer.py:L585-L598](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L585-L598): `_train_loop()` 签名中 `history` 是参数，但函数**没有 return history**

- [phase1_trainer.py:L309](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L309):
```python
best_epoch=history.best_epoch or 0
```
使用外层的 `history = SelectionHistory()` (line 239)，其 `best_epoch` 始终为 `None`

**影响**: `phase1_report.json` 的 `best_epoch` 固定为 0。

---

### P1-007: 无 best checkpoint 时仍导出 last/current 产物

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L269](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L269): `best_state = ckpt.load(ckpt.best_path) if ckpt.best_path.exists() else None`
- [phase1_trainer.py:L280-L282](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L280-L282):
```python
encoder_path, decoder_path, codebook_path = self._export_phase2_artifacts(
    best_state if best_state is not None else self._snapshot_state(model)
)
```
当 best 不存在时，用当前 model 的快照替代导出 Phase II 产物
- [phase1_trainer.py:L341](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L341): 返回 `best_vq_model=ckpt.best_path` — 指向不存在的文件

**设计对照**: 设计 §9.7 要求 `encoder.pt / decoder.pt / codebook.pt` 必须从 `best_vq_model.pt` 导出。当前在无 best 时产出 last/current 模型，违反设计约束。

---

### P1-008: `phase1_composite_score` 没进入 final report

| 项目 | 内容 |
|------|------|
| **v1 结论** | High, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [selection_policy.py:L77](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/selection_policy.py#L77): `score, debug = self.compute_composite_score(metrics)` — 计算了 composite score
- [selection_policy.py:L80](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/selection_policy.py#L80): `composite_score=score` — 存入 verdict，但**不写入 `metrics_for_select`**
- [phase1_trainer.py:L679](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L679): `checkpoint.save_last(state, ep_metrics.metrics, epoch)` — 保存的是 `ep_metrics.metrics`，不含 composite score
- [phase1_trainer.py:L811](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L811): `summary.setdefault("phase1_composite_score", 0.0)` — `_latest_metrics()` 读 epoch_metrics JSON 不含该 key → 默认 0.0

---

### P1-009: 采样健康报告被丢弃

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L454-L458](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L454-L458):
```python
checker.check(
    sampled=sampled,
    split_boundaries={"train_end_row": frame.height - 1},
    strata_labels=[s.strata_label for s in sampled],
)
```
- `SamplingHealthChecker.check()` 返回 `SamplingHealthReport` (line 70-151)，但**返回值未被捕获**
- [phase1_trainer.py:L791-L817](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L791-L817): `_build_final_summary()` 不含任何 sampling health 字段注入

---

### P1-010: 分层统计 off-by-one 与 prospective unknown 桶

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**horizon_return off-by-one**:

- 设计 §3.4: `(close[t+h] - close[t]) / close[t]`
- [window_indexer.py:L183-L185](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/window_indexer.py#L183-L185): `_compute_window_stats(close, start, h)` 
- [window_indexer.py:L50](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/window_indexer.py#L50): `horizon_return = (closes[-1] - closes[0]) / closes[0]` 其中 `closes[-1] = close[start+h-1]`
- 代码使用 `start+h-1`，设计要求 `start+h` → **off-by-one**

**past_return off-by-one**:

- 设计: `(close[t] - close[t-L]) / close[t-L]`
- [window_indexer.py:L77](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/window_indexer.py#L77): `closes = [float(close[start - lookback + i]) for i in range(lookback)]` → `[close[start-L], ..., close[start-1]]`
- `past_ret = (closes[-1] - closes[0]) / closes[0]` = `(close[start-1] - close[start-L]) / close[start-L]`
- 代码使用 `close[start-1]`，设计要求 `close[start]` → **off-by-one**

**unknown 桶未过滤**:

- [window_indexer.py:L75-L76](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/window_indexer.py#L75-L76): lookback 不足时 `past_*` 返回 NaN → strata = `"unknown|unknown|mixed"`
- [stratified_sampler.py:L138-L144](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/stratified_sampler.py#L138-L144): 桶分组不排除 `unknown` strata，prospective 诊断会被早期窗口污染

---

### P1-011: dead-code restart 默认开启但 trainer 未调用

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [vector_quantizer.py:L228-L267](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L228-L267): `restart_dead_codes()` 已完整实现
- [selection_policy.py:L84-L97](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/selection_policy.py#L84-L97): cooldown 逻辑已预留，等 `_dead_code_restart_triggered` 字段
- [phase1_trainer.py:L637-L682](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L637-L682): train loop 中**没有任何调用** `restart_dead_codes()`
- 配置默认 `dead_code_restart=True` ([phase1_config.py:L201](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/config/phase1_config.py#L201))

---

### P1-012: 多项设计要求评估指标未进入 evaluator/report

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_replay.py:L131](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase1_replay.py#L131): `evaluate_horizon_boundaries()` 已实现但**从未被 `Phase1Evaluator` 或 trainer 调用**
- [phase1_report.py:L107-L129](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase1_report.py#L107-L129): `write_diagnostics()` 已实现但**从未被 trainer 调用**

**缺失指标清单**:
- `horizon_boundary_turnover_cost` / `horizon_boundary_position_consistency`
- `confusion_matrix`
- `action_precision_recall_per_class`
- `per_code_switch_point_distribution`
- `dp_teacher_return_distribution`
- `epoch_code_stability` (虽然 selection_policy 引用了该 key，但 evaluator 未写入)

---

### P1-013: weighted reconstruction 指标 key 不一致

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_evaluator.py:L176](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase1_evaluator.py#L176): evaluator 写 `out.metrics["val_weighted_reconstruction_accuracy"]`
- [phase1_report.py:L17](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase1_report.py#L17): `REQUIRED_REPORT_KEYS` 要求 `"weighted_reconstruction_accuracy"` (无 `val_` 前缀)
- [phase1_trainer.py:L796](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L796): `summary.setdefault("weighted_reconstruction_accuracy", 0.0)` — key 不匹配 → 永远 fallback 到 0.0

**数据流**: evaluator → epoch_metrics JSON 含 `val_weighted_reconstruction_accuracy` → `_latest_metrics()` 读入 → `summary = dict(metrics)` 含该 key → `setdefault("weighted_reconstruction_accuracy", ...)` 找不到 → 设为 0.0

---

### P1-014: `torch` 缺失时 fallback 不完整

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [vector_quantizer.py:L16-L21](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L16-L21): `torch = None` fallback
- [vector_quantizer.py:L87](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L87): `@torch.no_grad()` — 类定义时执行，此时 `torch=None` → `AttributeError`
- 同样问题在 [vector_quantizer.py:L190](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L190), [L228](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L228), [L271](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vector_quantizer.py#L271)
- [vq_archetype.py:L172](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vq_archetype.py#L172): `@torch.no_grad()` 同样问题

---

### P1-015: schema 单测 fixture 自身先 ZeroDivision

| 项目 | 内容 |
|------|------|
| **v1 结论** | Medium, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [test_schema.py:L13](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/tests/unit/data/test_schema.py#L13):
```python
cols["return_1m"] = [0.0] + [(close[i] - close[i - 1]) / close[i - 1] for i in range(1, len(close))]
```
- [test_schema.py:L35-L38](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/tests/unit/data/test_schema.py#L35-L38): `test_close_non_positive_raises()` 传 `close=(100.0, 0.0, 100.0)`
- i=2: `(100.0 - 0.0) / 0.0` → **ZeroDivisionError** 在 `_make_frame()` 内，validator 从未被调用

---

### P1-016: `_schema_hash` 不是 schema hash

| 项目 | 内容 |
|------|------|
| **v1 结论** | Low, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [phase1_trainer.py:L181](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L181):
```python
schema_hash = config_hash  # 这里以 config_hash 为审计标记；避免又算一次。
```
- 注释明确承认这是 config_hash，不是真正的 schema hash

---

### P1-017: `run_pipeline.sh` Phase I 调用缺必填文件参数

| 项目 | 内容 |
|------|------|
| **v1 结论** | Low, 正确 |
| **复核** | ✅ 确认 |

**源码证据**:

- [train_phase1.py:L43-L45](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/scripts/train_phase1.py#L43-L45): `--train-file`、`--val-file`、`--test-file` 均为 `required=True`
- [run_pipeline.sh:L37](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/run_pipeline.sh#L37):
```bash
python scripts/train_phase1.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" "${EXTRA_ARGS[@]}"
```
默认不传文件路径，需用户通过 EXTRA_ARGS 补充

---

## 第二部分: v1 "其他观察" 项复核

v1 报告中的 "其他观察" 4 项经复核全部属实。

| 观察 | 复核 | 源码 |
|------|------|------|
| `NoTradeControlConfig` 基本未被 trainer 使用 | ✅ 确认 | [phase1_trainer.py](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py) 全文搜索 `no_trade_control` — 0 匹配 |
| `Phase1DemoStore.save_demos()` 不保存 `execution_books` | ✅ 确认 | [demo_store.py:L61](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/demo_store.py#L61): 注释明确 "execution_books 不写入" |
| `device`/`mixed_precision`/`early_stopping_patience` 未使用 | ✅ 确认 | [phase1_trainer.py](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py) 无引用 |
| `composite_score_sensitivity` 只重算不重新选择 | ✅ 确认 | [phase1_metrics.py](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase1_metrics.py) — 只对一组 metrics 重算 |

---

## 第三部分: 新增发现 (v1 未覆盖)

### 3.1 Decoder 因果性约束已满足

**结论**: v1 审查未提及，但这是设计 §6.7 的关键约束。

- [vq_archetype.py:L93-L97](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/vq_archetype.py#L93-L97): `ArchetypeDecoder` 使用 `bidirectional=False` 单向 LSTM ✅
- forward 逻辑 (lines 101-118) 每步只依赖当前及之前的 states + 固定 z_q，不访问未来 ✅

### 3.2 `HorizonRecord` 缺 `last_execution_row` / `last_markout_row` 字段

**位置**: [horizon_builder.py:L17-L32](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/horizon_builder.py#L17-L32)

`HorizonRecord` 有 `start_index`/`end_index` 但没有 `last_execution_row`/`last_markout_row`。`_export_horizon_labels()` (line 726-727) 不得不重新猜测这些行号，导致 P1-005 的 bug。设计 §3.3 中 SampledHorizon 有这两个字段但 HorizonRecord 丢失了。

### 3.3 `_build_final_summary` 读取 `_latest_metrics` 而非 best epoch metrics

**位置**: [phase1_trainer.py:L779-L789](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L779-L789)

`_latest_metrics()` 永远取 manifest 的 `entries[-1]` (最新 epoch)，而非 best epoch。结合 P1-006 → best_epoch=0，最终 report 的 metrics 来自 last epoch 而非 best epoch，与 `best_checkpoint_path` 不一致。

### 3.4 `_export_horizon_labels` 中 `num_switches` 统计包含末步复制

**位置**: [phase1_trainer.py:L716-L719](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase1_trainer.py#L716-L719)

```python
num_switches = sum(
    1 for i in range(1, len(rec.actions or []))
    if rec.actions[i] != rec.actions[i - 1]
)
```
这里统计了全部相邻帧切换，包含 `actions[h-1] != actions[h-2]` 的情况。但 `plan()` 强制 `actions[-1] = actions[-2]`，正常情况下末步不会切换。然而如果 horizon 构建过程有其他修改路径，统计方法不够健壮。

---

## 问题汇总 (含新增)

| ID | 严重度 | 分类 | 摘要 | 状态 |
|----|--------|------|------|------|
| P1-001 | Critical | DP 算法 | DP 末步估值切换与 forward 复制矛盾 | ✅ v1正确 |
| P1-002 | High | DP 质量门禁 | reject_transition 只看最终 replay 路径 | ✅ v1正确 |
| P1-003 | High | sign-off | prospective 对照检查未实质执行 | ✅ v1正确 |
| P1-004 | High | 采样 | min_gap 跨 strata 不保证 | ✅ v1正确 |
| P1-005 | High | 产物 | next_row_execution label 行号 offset | ✅ v1正确 |
| P1-006 | High | checkpoint | _train_loop history 没有返回 | ✅ v1正确 |
| P1-007 | High | checkpoint | 无 best 时仍导出 last 模型 | ✅ v1正确 |
| P1-008 | High | report | composite_score 不进入 report | ✅ v1正确 |
| P1-009 | Medium | 采样 | 健康报告被丢弃 | ✅ v1正确 |
| P1-010 | Medium | 分层 | horizon/past return off-by-one | ✅ v1正确 |
| P1-011 | Medium | VQ 健康 | dead-code restart 从未被调用 | ✅ v1正确 |
| P1-012 | Medium | 评估 | 多项设计指标未进入 evaluator | ✅ v1正确 |
| P1-013 | Medium | report | weighted_reconstruction_accuracy key 不一致 | ✅ v1正确 |
| P1-014 | Medium | 测试/依赖 | torch 缺失时模块 fallback 不完整 | ✅ v1正确 |
| P1-015 | Medium | 测试 | schema 测试 fixture ZeroDivision | ✅ v1正确 |
| P1-016 | Low | cache/审计 | _schema_hash 实际是 config_hash | ✅ v1正确 |
| P1-017 | Low | 集成脚本 | run_pipeline.sh 缺必填参数 | ✅ v1正确 |
| **P1-018** | **Medium** | **产物** | **HorizonRecord 缺 last_execution/markout 字段** | 🆕 新增 |
| **P1-019** | **Medium** | **report** | **final report 读 latest 而非 best epoch metrics** | 🆕 新增 |
| **P1-020** | **Low** | **产物** | **_export_horizon_labels 中 num_switches 统计不够健壮** | 🆕 新增 |

---

## 复核结论

1. **v1 审查报告质量**: 全部 17 个问题 + 4 个"其他观察"项均经源代码交叉验证确认属实，**无错误项**。
2. **v1 遗漏**: 新增 3 个问题 (P1-018 ~ P1-020)，严重度均为 Medium/Low。
3. **设计符合度**: 代码整体忠实于设计文档的模块划分和数据流，主要问题集中在:
   - DP 算法末步约束 (P1-001)
   - 采样和健康检查的结果未被消费 (P1-002, P1-009, P1-011)
   - 最终报告的指标完整性和准确性 (P1-003, P1-006, P1-008, P1-013)

---

## 2026-05-01 执行结果与采纳状态

> 本节为追加执行记录，未修改上方原 review 内容。

### v2 问题采纳状态

| 标记 | ID/条目 | 采纳状态 |
| --- | --- | --- |
| 【✅】 | P1-001 | 已采纳。末步 DP 只允许保持当前仓位，forward 复制约束与估值一致。 |
| 【✅】 | P1-002 | 已采纳。DP candidate reject 统计进入 demo generator 质量门禁。 |
| 【✅】 | P1-003 | 已采纳。hindsight/prospective paired report 对照已接入 final report 与 leakage diagnostics。 |
| 【✅】 | P1-004 | 已采纳。sampler 全局维护 min gap。 |
| 【✅】 | P1-005 | 已采纳。label 行号从 indexer/builder 真实边界字段传递。 |
| 【✅】 | P1-006 | 已采纳。selection history 从 `_train_loop()` 返回给 `run()`。 |
| 【✅】 | P1-007 | 已采纳。没有 best checkpoint 时 trainer fatal，禁止导出 last/current artifacts。 |
| 【✅】 | P1-008 | 已采纳。`phase1_composite_score` 进入 epoch metrics 和 final report。 |
| 【✅】 | P1-009 | 已采纳。sampling health report 已进入 final summary。 |
| 【✅】 | P1-010 | 已采纳。horizon/past return off-by-one 已修复；prospective `unknown` 桶过滤未作为本轮 bug 采纳。 |
| 【✅】 | P1-011 | 已采纳。dead-code restart 已接入 trainer 主循环。 |
| 【】 | P1-012 | 部分采纳，未标记为完成。boundary、epoch stability、confusion/per-class、per-code switch、teacher distribution 已进入 evaluator/report；latent snapshot/failure case 主流程触发仍未完整接线。 |
| 【✅】 | P1-013 | 已采纳。weighted reconstruction key 已统一。 |
| 【✅】 | P1-014 | 已采纳。torch fallback 的 no-op decorator 已覆盖模块级 no-grad 使用。 |
| 【✅】 | P1-015 | 已采纳。schema 测试 fixture 已修复。 |
| 【✅】 | P1-016 | 已采纳。`_schema_hash` 使用 schema hash。 |
| 【✅】 | P1-017 | 已采纳。`run_pipeline.sh` 已补默认 Phase I 数据文件参数。 |
| 【✅】 | P1-018 | 已采纳。`HorizonRecord` 已包含 `last_execution_row` / `last_markout_row`。 |
| 【✅】 | P1-019 | 已采纳。final report 读取 best epoch metrics。 |
| 【✅】 | P1-020 | 已采纳。`num_switches` 统计排除末步复制。 |

### v2 其他观察采纳状态

| 标记 | 条目 | 采纳状态 |
| --- | --- | --- |
| 【】 | `NoTradeControlConfig` 基本未被 trainer 使用 | 未采纳，仍需单独定义补采样/过滤/warning 口径。 |
| 【✅】 | `Phase1DemoStore.save_demos()` 不保存 `execution_books` | 已采纳。当前 demo store 保存 JSON 序列化的 execution books，并在读取时恢复。 |
| 【】 | `device` / `mixed_precision` / `early_stopping_patience` 未使用 | 未采纳，仍作为后续配置接线增强项。 |
| 【✅】 | `composite_score_sensitivity` 只重算不重新选择 | 已采纳。当前 sensitivity 会跨全部 epoch metrics 重新选择 best epoch。 |

### 本次补充调整

| 标记 | 文件 | 内容 |
| --- | --- | --- |
| 【✅】 | `tests/integration/test_phase1_pipeline_smoke.py` | smoke test 显式放宽 risk/behavior guardrail，确保该测试验证产物链路；P1-007 的生产保护逻辑保持不变。 |

### 验证结果

| 标记 | 命令 | 结果 |
| --- | --- | --- |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && pytest -q` | `375 passed, 17 warnings`。warnings 为未注册 `pytest.mark.integration`。 |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && bash -n run_pipeline.sh` | 语法检查通过。 |
