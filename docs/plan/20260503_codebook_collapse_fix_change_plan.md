# Phase I Codebook 坍塌修复变更计划

**日期**: 2026-05-03
**来源设计**: `docs/changes/20260503_codebook_collapse_root_cause_and_fix_design.md`
**计划目标**: 用最小结构风险建立“Phase A 预训练 + soft assignment + usage-profit alignment”的防坍塌闭环，并把高风险结构性改动排除出本轮。

---

## 1. 执行情况看板

状态含义：

- `DONE`: 文档或决策已完成。
- `PLANNED`: 纳入本轮代码实现计划，尚未执行。
- `DEFERRED`: 不进本轮，需独立设计或实验后再评审。
- `REJECTED`: 本轮明确不采纳。

| 项目 | 采纳结论 | 执行状态 | 风险 | 验收/测试要求 |
|------|----------|----------|------|---------------|
| 采纳范围判断 | 纳入 | DONE | 低 | 本文 §2 和原设计文档 §11 已记录 |
| C1 Phase A 预训练 | 采纳 | PLANNED | 中 | `test_forward_pretrain_*`, `test_phase_a_loss_only_rec`, trainer phase transition 单测 |
| C7 Soft Assignment 计算 | 采纳 | PLANNED | 低 | soft assignment shape、row-sum、temperature clamp 单测 |
| C2 Usage-Profit Alignment Loss | 采纳 | PLANNED | 中 | alignment loss 行为单测、dataset actual return 单测、paper_strict 关闭单测 |
| C3 Codebook Separation Loss | 本轮不采纳 | DEFERRED | 中-高 | 暂不改代码；若后续采纳需先解决 EMA 模式下梯度无效问题 |
| C4 Return Bucket Auxiliary Loss | 不采纳 | REJECTED | 高 | 暂不改代码；原因见 §4.2 |
| C5 方向感知初始化 | 不采纳 | REJECTED | 高 | 暂不改代码；原因见 §4.3 |
| C6 Dead Code Restart 增强 | 不采纳 | REJECTED | 高 | 暂不改代码；原因见 §4.4 |
| 单元测试代码变更 | 纳入 | PLANNED | 低 | 新增/更新 tests，覆盖配置、loss、model、dataset、trainer |
| 测试执行 | 纳入 | PLANNED | 低 | 执行 §6 命令并记录结果 |

---

## 2. 本轮采纳原则

本轮只采纳满足以下条件的变更：

1. 不改变 encoder / decoder / quantizer 的持久化结构，不引入新的 checkpoint 兼容风险。
2. 不改变 Phase II/III 读取 `encoder.pt` / `decoder.pt` / `codebook.pt` 的契约。
3. 能用局部单元测试验证核心行为，不依赖长时间训练才能判断正确性。
4. `paper_strict_reproduction=True` 时必须自动关闭新增机制。
5. 默认值保守，优先减少 collapse 风险，而不是一次性引入多个会相互干扰的训练动态。

据此，本轮采纳 C1、C7、C2；暂不采纳 C3、C4、C5、C6。

---

## 3. 纳入范围

### 3.1 C1 Phase A 预训练

采纳原因：

- 这是原设计识别出的根因修复点：先让 encoder/decoder 学会重构，再初始化 codebook。
- 不新增模型参数，不改变导出 artifact 契约，属于训练调度变更而非模型结构变更。
- 可以通过 `pretrain_epochs` 和 `paper_strict_reproduction` 控制开关。

计划变更：

1. `TrainingConfig` 新增：
   - `pretrain_epochs: int = 10`
   - `pretrain_lr: Optional[float] = None`
2. `apply_paper_strict_overrides` 在 strict 模式下强制 `pretrain_epochs=0`。
3. `VQArchetypeModel` 新增 `forward_pretrain(states, actions, rewards)`：
   - 跳过 `quantizer.quantize`。
   - 使用 `z_e` 直接作为 decoder condition。
   - `z_q == z_e`，`code_id is None`。
4. `Phase1Loss` 新增 `forward_pretrain`：
   - 只计算 reconstruction CE。
   - `codebook` 和 `commitment` 返回同 device 的 0 scalar。
5. Trainer 调整 warmup 时机：
   - `pretrain_epochs == 0`: 保持训练前 warmup。
   - `pretrain_epochs > 0`: Phase A 完成后再 warmup codebook，避免用随机 encoder 输出初始化。
6. `training.epochs` 仍表示总 epoch 数；要求 `0 <= pretrain_epochs < epochs`。

边界：

- 不新增 Phase A checkpoint 类型。
- 不改变 best checkpoint 选择逻辑。
- 不把 Phase A epoch 纳入 validation/checkpoint 选择，除非后续明确需要。

### 3.2 C7 Soft Assignment 计算

采纳原因：

- 是 C2 的低风险前置能力。
- 纯函数实现，容易单测。
- 不影响现有硬分配 quantize 路径。

计划变更：

1. 在 `src/models/vq_losses.py` 新增 `compute_soft_code_assignments(z_e, codebook, temperature)`。
2. 使用 squared L2 distance + softmax。
3. `temperature` clamp 到 `>= 1e-6`，避免除零。
4. `codebook` 传入时使用当前 `model.quantizer.codebook`；是否 detach 由 loss 侧决定。

### 3.3 C2 Usage-Profit Alignment Loss

采纳原因：

- 针对 KL-uniform “只管均匀、不管质量”的核心缺口。
- 不新增模型结构，只新增 loss 项和 batch 字段。
- 对 code 使用率和收益之间的负相关有直接约束。

计划变更：

1. `CodebookHealthConfig` 新增保守默认值：
   - `usage_profit_alignment_weight: float = 0.05`
   - `usage_profit_alignment_target_corr: float = 0.2`
   - `usage_profit_alignment_temperature: float = 2.0`
2. `Phase1Loss` 新增：
   - `alignment: Optional[torch.Tensor]` 输出字段。
   - `_usage_profit_alignment(...)` 私有方法。
   - `forward` 参数：`trajectory_returns`, `codebook`, `soft_assignment_temperature`。
3. `Phase1DemoDataset.__getitem__` 新增 `trajectory_return`：
   - 必须使用原始 `rec.rewards` 求和。
   - 不使用 normalized rewards，避免把 normalizer 后的数值当收益目标。
4. `collate_phase1` 新增 `trajectory_returns: Tensor[B]`。
5. Trainer 在 Phase B loss 中传入：
   - `trajectory_returns=batch["trajectory_returns"]`
   - `codebook=model.quantizer.codebook`
   - `soft_assignment_temperature=config.model.codebook.health.usage_profit_alignment_temperature`
6. `paper_strict_reproduction=True` 时 alignment weight 强制为 0。

边界：

- alignment 只在 Phase B 生效，Phase A 仍只做 reconstruction。
- 不替换现有 KL-uniform；本轮先并存，便于 ablation。
- 不改变 selection guardrail。

---

## 4. 不采纳与暂缓项

### 4.1 C3 Codebook Separation Loss: 本轮不采纳

原因：

- 当前默认 `update_method="ema"`，`codebook.requires_grad=False`，普通 separation loss 对 codebook 没有直接优化效果。
- 原设计提出的 in-place separation regularization 会改变 EMA 更新语义，属于训练动态重写，风险高。
- 如果只作为 loss 记录，容易给人“已修复软坍塌”的错觉。

后续条件：

- 先单独设计 EMA 模式下的分离机制，明确数学形式、更新顺序、cooldown 和指标影响。
- 至少用 unit test 验证相似 code 会被推开，并用小型集成实验验证不破坏 reconstruction。

### 4.2 C4 Return Bucket Auxiliary Loss: 不采纳

原因：

- 需要新增 `return_bucket_head`，改变模型 state_dict 和 checkpoint 兼容面。
- 会影响 Phase II/III artifact 切分逻辑，虽然辅助头不导出，也需要明确加载策略。
- 收益分桶边界、soft/hard 混合权重、重尾收益分布处理都需要额外实验，不适合与 C1/C2 同批上线。

后续条件：

- 只有当 C1+C2 后仍出现软坍塌，且 behavior metrics 明确指向“code 身份不足”时，再单独评审。

### 4.3 C5 方向感知初始化: 不采纳

原因：

- 需要在初始化阶段按 action 方向分配 code 容量，属于 codebook 语义和容量分配策略改写。
- long/short/flat 的固定配额对不同品种、不同 `max_position`、不同 no-trade 比例可能不稳。
- 与 Phase A 后的普通 k-means 效果存在重叠，应先验证 Phase A 是否已足够解决随机初始化问题。

后续条件：

- 只有当 C1 后仍出现方向性坍塌，并且 action 分布诊断证明普通 k-means 未覆盖方向时，再设计方向初始化。

### 4.4 C6 Dead Code Restart 增强: 不采纳

原因：

- 每 epoch restart、低使用率重置、高收益样本优先、EMA count 初始化策略等同时改变，会大幅改变训练轨迹。
- 该项容易掩盖 C1/C2 是否真正修复了根因。
- 当前 selection policy 已有 restart cooldown 和 fatal collapse 逻辑，同批重写 restart 会增加排障难度。

后续条件：

- C1+C2 完成后，若仍有可复现 dead code 现象，再只挑一个 restart 子项做小步实验。

---

## 5. 实施步骤

### Step 1: 配置与文档字段

涉及文件：

- `src/config/phase1_config.py`
- `tests/unit/config/test_phase1_config_docs.py`
- `tests/unit/trainers/test_phase1_trainer.py`

动作：

1. 增加 `TrainingConfig.pretrain_epochs/pretrain_lr`。
2. 增加 alignment 相关 `CodebookHealthConfig` 字段。
3. 更新 `PHASE1_CONFIG_FIELD_DOCS`。
4. 更新 `apply_paper_strict_overrides`。

验收：

- 配置 field docs 覆盖测试通过。
- strict 模式关闭 pretrain/alignment 的单测通过。

### Step 2: Dataset batch 增加 actual trajectory return

涉及文件：

- `src/data/dataset.py`
- `tests/unit/data/test_dataset.py`

动作：

1. `__getitem__` 返回 `trajectory_return=sum(original_rewards)`。
2. `collate_phase1` 返回 `trajectory_returns` tensor。
3. 单测确认 encoder 输入 rewards 仍可 normalized，但 `trajectory_returns` 保持原始收益。

### Step 3: Model 和 Loss 增量

涉及文件：

- `src/models/vq_archetype.py`
- `src/models/vq_losses.py`
- `tests/unit/models/test_vq_archetype.py`
- `tests/unit/models/test_vq_losses.py`

动作：

1. 新增 `forward_pretrain`。
2. 新增 `Phase1Loss.forward_pretrain`。
3. 新增 `compute_soft_code_assignments`。
4. 新增 usage-profit alignment loss。
5. `LossOutputs` 增加 `alignment` 字段。

验收：

- pretrain forward 不触发 quantizer。
- pretrain loss 只等于 reconstruction。
- soft assignment 每行和为 1。
- 高收益 code 使用率更高时 alignment loss 更小。

### Step 4: Trainer 接入 Phase A 和 C2

涉及文件：

- `src/trainers/phase1_trainer.py`
- `tests/unit/trainers/test_phase1_trainer.py`
- `tests/integration/test_phase1_collapse_handling.py`

动作：

1. Trainer 根据 `pretrain_epochs` 控制 Phase A/Phase B。
2. `pretrain_epochs > 0` 时，Phase A 后执行 codebook warmup。
3. Phase B loss 传入 `trajectory_returns` 和 `codebook`。
4. batch diagnostic 日志增加 `loss_alignment`。
5. 避免 Phase A 执行 EMA codebook update。

验收：

- pretrain epoch 中 `code_id is None` 不进入 VQ loss。
- Phase A 后 warmup 被调用一次。
- Phase B 正常执行 quantizer 和 EMA update。

---

## 6. 测试执行计划

本轮代码实现完成后，至少执行：

```bash
pytest tests/unit/models/test_vq_archetype.py \
  tests/unit/models/test_vq_losses.py \
  tests/unit/data/test_dataset.py \
  tests/unit/trainers/test_phase1_trainer.py \
  tests/unit/config/test_phase1_config_docs.py
```

随后执行轻量集成/回归：

```bash
pytest tests/integration/test_phase1_collapse_handling.py \
  tests/integration/test_phase1_pipeline_smoke.py
```

若本地 GPU 或 fixture 条件不足导致 smoke test 无法完整执行，必须在执行记录中注明阻塞原因，并至少保留所有 unit test 结果。

---

## 7. 完成标准

1. 所有纳入字段都能写入 `phase1_config.yaml`，且 `training_config_hash` 会随字段变化。
2. `paper_strict_reproduction=True` 时：
   - `pretrain_epochs == 0`
   - `usage_profit_alignment_weight == 0.0`
   - 现有 usage regularization / dead code restart strict 行为不回归。
3. Phase A 不走 quantizer、不更新 EMA codebook。
4. Phase A 后 warmup 使用已训练 encoder 的 `z_e`。
5. Phase B 的 total loss 包含 reconstruction、VQ、commitment、KL-uniform 和 alignment。
6. 新增/更新的单元测试和指定回归测试通过，测试命令和结果写入后续执行记录。

---

## 8. 执行结果记录（2026-05-03）

### 8.1 执行情况看板

| 项目 | 完成标记 | 执行结果 |
|------|----------|----------|
| C1 Phase A 预训练 | 【✅】 | 已实现 `TrainingConfig.pretrain_epochs/pretrain_lr`、`VQArchetypeModel.forward_pretrain`、`Phase1Loss.forward_pretrain`，trainer 在 Phase A 后执行 codebook warmup。 |
| C7 Soft Assignment 计算 | 【✅】 | 已实现 `compute_soft_code_assignments`，覆盖 shape、row-sum 和温度下限行为。 |
| C2 Usage-Profit Alignment Loss | 【✅】 | 已接入 `usage_profit_alignment_*` 配置、loss 输出 `alignment`、dataset 原始 `trajectory_returns`、trainer Phase B loss 参数。 |
| `paper_strict_reproduction` 兼容 | 【✅】 | strict 模式强制 `pretrain_epochs=0`、`usage_profit_alignment_weight=0.0`，保留原有 usage regularization/dead restart 关闭逻辑。 |
| CPU 测试稳定性 | 【✅】 | trainer/evaluator 在 CPU 下使用 `DataLoader(num_workers=0)`，CUDA 下保留 2 workers，避免沙箱 multiprocessing socket 权限问题。 |
| 单元测试代码变更 | 【✅】 | 已更新 model/loss/dataset/trainer/config 相关单元测试。 |
| 测试执行 | 【✅】 | conda 环境 `ArchetypeTrade` 下 unit 与 integration 均通过。 |
| C3-C6 不采纳项 | 【✅】 | 未实施结构性高风险改动，保持执行计划中的不采纳/暂缓决策。 |

### 8.2 实际代码变更

涉及文件：

- `src/config/phase1_config.py`: 新增 Phase A 与 usage-profit alignment 配置，更新 strict override 与配置字段文档。
- `scripts/train_phase1.py`: 新增 `--pretrain-epochs` / `--pretrain-lr` CLI 参数。
- `src/data/dataset.py`: batch 增加原始 `trajectory_return(s)`，不受 reward normalizer 影响。
- `src/models/vq_archetype.py`: 新增 `forward_pretrain`，Phase A 跳过 VQ。
- `src/models/vq_losses.py`: 新增 Phase A loss、soft assignment、usage-profit alignment loss 与 `alignment` 输出。
- `src/trainers/phase1_trainer.py`: 接入 Phase A/Phase B 切换、延后 warmup、Phase B alignment loss、CPU DataLoader worker 策略。
- `src/evaluation/phase1_evaluator.py`: CPU validation DataLoader 改为 0 worker，避免本地/沙箱权限问题。
- `tests/unit/data/test_dataset.py`: 覆盖 `trajectory_returns` 原始收益语义。
- `tests/unit/models/test_vq_archetype.py`: 覆盖 `forward_pretrain` 不走 quantizer。
- `tests/unit/models/test_vq_losses.py`: 覆盖 Phase A loss、soft assignment、alignment loss。
- `tests/unit/trainers/test_phase1_trainer.py`: 覆盖 strict 关闭、pretrain epoch 收敛、loss 配置注入。

### 8.3 测试执行记录

执行环境：

```bash
conda activate ArchetypeTrade
```

语法检查：

```bash
python -m py_compile src/config/phase1_config.py src/data/dataset.py src/models/vq_archetype.py src/models/vq_losses.py src/trainers/phase1_trainer.py src/evaluation/phase1_evaluator.py scripts/train_phase1.py
```

结果：通过。

单元测试：

```bash
pytest tests/unit/models/test_vq_archetype.py \
  tests/unit/models/test_vq_losses.py \
  tests/unit/data/test_dataset.py \
  tests/unit/trainers/test_phase1_trainer.py \
  tests/unit/config/test_phase1_config_docs.py
```

结果：`34 passed in 2.68s`。

轻量集成/回归：

```bash
pytest tests/integration/test_phase1_collapse_handling.py \
  tests/integration/test_phase1_pipeline_smoke.py
```

结果：`6 passed in 3.69s`。

### 8.4 执行备注

- 初次跑 integration smoke 时 CPU DataLoader worker 触发 `PermissionError: Operation not permitted`，原因是沙箱环境不允许 multiprocessing resource sharer 创建 socket。
- 已将 trainer/evaluator 的 CPU DataLoader worker 调整为 0；CUDA 路径仍保留 2 workers。
- 若 `pretrain_epochs >= epochs`，trainer 会自动收敛到 `epochs - 1`，保证至少保留一个 Phase B epoch。
- 未实施 C3/C4/C5/C6，符合本计划“高风险结构性改动不纳入本轮”的约束。
