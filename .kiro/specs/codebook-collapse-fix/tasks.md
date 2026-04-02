# 实现计划

- [x] 1. 编写 bug condition 探索测试
  - **Property 1: Bug Condition** — 随机初始化码本在 z_e 分布下产生 Dead Codes
  - **重要**: 此 property-based 测试必须在实施修复之前编写
  - **关键**: 此测试在未修复代码上必须 FAIL — 失败确认 bug 存在
  - **不要**在测试失败时尝试修复测试或代码
  - **说明**: 此测试编码了期望行为 — 修复后测试通过即验证修复正确
  - **目标**: 展示随机初始化的码本在量化 z_e 时产生 dead codes 的反例
  - **Scoped PBT 方法**: 生成集中在某个子区域的 z_e 样本（模拟 Phase A 训练后的 z_e 分布），用随机初始化的 `VQCodebook` 量化，断言 `used_code_count == K`（所有码本条目都被使用）
  - 具体步骤：
    1. 使用 Hypothesis 生成 z_e 样本：以随机均值 `mu` 和较小标准差 `sigma` 生成 N≥100 个 z_e 向量（模拟 Phase A 后 z_e 集中在某个子区域）
    2. 创建随机初始化的 `VQCodebook(num_codes=10, code_dim=16)`（不调用 `init_from_data`）
    3. 对所有 z_e 样本执行 `codebook.quantize(z_e)`
    4. 统计 `used_code_count = len(unique(indices))`
    5. 断言 `used_code_count == 10`（期望行为：所有码本条目都被使用）
  - 在未修复代码上运行 — 预期 FAIL（随机初始化码本远离 z_e 分布，部分条目成为 dead codes，`used_code_count < 10`）
  - 记录反例：例如 "z_e 集中在 mu=[2.0, ...], sigma=0.1 时，随机初始化码本 used_code_count=3/10"
  - 测试文件：`tests/test_codebook_collapse_fix.py`
  - _Requirements: 1.1, 1.2, 2.2, 2.3_

- [x] 2. 编写 preservation property 测试（在实施修复之前）
  - **Property 2: Preservation** — Quantize 方法、损失公式和流水线行为不变
  - **重要**: 遵循 observation-first 方法论
  - **说明**: 在未修复代码上观察行为，编写 property-based 测试捕获观察到的行为模式
  - 具体测试属性：
    1. **quantize() 接口不变**: 对任意 z_e (batch, 16)，`quantize()` 返回 `(z_q_st, indices, commitment_loss)`，形状分别为 `(batch, 16)`, `(batch,)`, `scalar`；无论码本是否经过 k-means 初始化
    2. **最近邻正确性不变**: 对任意 z_e，选中索引 k 满足 `||z_e - e_k||² ≤ ||z_e - e_j||² ∀j`
    3. **Straight-through 梯度不变**: `z_q_st` 对 `z_e` 的梯度为全 1（`z_q_st = z_e + (z_q - z_e).detach()`）
    4. **VQ 损失公式不变**: `commitment_loss = mean(||sg[z_e] - z_q||²)`，总损失 `L = L_rec + commitment_loss + β₀ × mean(||z_e - sg[z_q]||²)` 分解正确
    5. **编码器输出形状不变**: 对任意 `(batch, seq_len, 45)` 输入，z_e 形状为 `(batch, 16)`
    6. **解码器输出形状不变**: 对任意 `(batch, seq_len, 45)` 状态和 `(batch, 16)` z_q，logits 形状为 `(batch, seq_len, 3)`
    7. **Phase A 跳过 VQ**: Phase A 阶段 `is_phase_a=True` 时不执行量化
  - 在未修复代码上运行 — 预期 PASS（确认基线行为）
  - 测试文件：`tests/test_codebook_collapse_fix.py`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. 验证 k-means 初始化修复

  - [x] 3.1 验证 `init_from_data()` 单元测试
    - 编写 `init_from_data()` 的单元测试覆盖以下场景：
      - 基本功能：K=10, N=1000 的 z_e 样本，验证码本权重被更新且形状正确
      - 边界情况 N < K：N=5, K=10 时跳过初始化，码本权重不变
      - 空簇处理：构造会产生空簇的数据（如 9 个点在同一位置 + 1 个离群点），验证所有 K 个中心都有效（非 NaN/Inf）
      - k-means++ 分散性：验证初始中心之间的最小距离 > 0
      - 码本覆盖 z_e 分布：初始化后对 z_e 执行量化，验证 `used_code_count == K`
    - _Bug_Condition: isBugCondition(input) — codebook_weights 是随机初始化且 avg_dist_random >> avg_dist_kmeans_
    - _Expected_Behavior: init_from_data 后码本向量在 z_e 分布内，所有 K 个条目都被使用_
    - _Preservation: quantize() 方法接口和行为不变_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.2 验证 bug condition 探索测试现在通过
    - **Property 1: Expected Behavior** — K-Means 初始化消除 Dead Codes
    - **重要**: 重新运行任务 1 中的同一测试 — 不要编写新测试
    - 修改测试使其在量化前调用 `codebook.init_from_data(z_e_samples)` 初始化码本
    - 任务 1 的测试编码了期望行为（`used_code_count == K`）
    - 当此测试通过时，确认期望行为已满足
    - 运行 bug condition 探索测试
    - **预期结果**: 测试 PASS（确认 k-means 初始化修复了 dead codes 问题）
    - _Requirements: 2.2, 2.3_

  - [x] 3.3 验证 preservation 测试仍然通过
    - **Property 2: Preservation** — Quantize 方法、损失公式和流水线行为不变
    - **重要**: 重新运行任务 2 中的同一测试 — 不要编写新测试
    - 运行 preservation property 测试
    - **预期结果**: 测试 PASS（确认无回归）
    - 确认所有测试在修复后仍然通过

- [x] 4. Checkpoint — 确保所有测试通过
  - 运行完整测试套件 `pytest tests/test_codebook_collapse_fix.py -v`
  - 确保所有 property-based 测试和单元测试通过
  - 如有问题，询问用户
