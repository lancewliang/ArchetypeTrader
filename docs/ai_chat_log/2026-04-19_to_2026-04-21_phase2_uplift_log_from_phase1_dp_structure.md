# Phase II 收益提升记录（AL：batch_09 → batch_17，后续 batch_24 补充）

日期范围: 2026-04-19 至 2026-04-21  
交易对: AL  
关注点: 第二阶段（Phase II）收益为什么跌、怎么通过 Phase I 的 DP 结构与 profit-aware 训练把收益拉回来，并记录关键改动与测试结果。  
相关过程文档: `docs/2026-04-19_phase1_dp_structure_optimization_log.md`（更偏 Phase I 训练细节与完整时间线）

---

## 1. 结论摘要

- `batch_11/12` 的 Phase II 收益显著回撤，根因不是 PPO 训练器突然坏掉，而是 **Phase I 学到的 archetype 语义与收益结构脱钩**（`best_fixed`、`realizable_proxy` 变低，且 `return_usage_correlation` 变成强负相关）。
- 从 `batch_16 → batch_17` 的关键突破是把 Phase I 从“重建 DP 轨迹”推进到“让 code semantics 对齐收益结构”，组合策略是 `return bucket objective`（学出赚钱 archetype）+ `light usage-profit alignment`（轻推回正相关）+ `strict profit gate`（坏 checkpoint 禁止进入 Phase II）。
- `batch_17-short` 已超过 `batch_09-short` 的 Phase II 表现（val TR/ASR 明显更好、test MDD 更低）。
- 在 Phase I 已稳定可用后（维持 profit-aware Phase I），`batch_24-short` 通过 Phase II 的 **缩短训练 + alpha 退火 + imitation mask**，进一步显著缩小了与 DP 的 gap（尤其 test 集）。

---

## 2. 关键指标对比（收益视角）

说明:
- `avg_return(BEST/FINAL)` 是 Phase II 训练期间在验证集上的 horizon 平均 return（训练脚本内部指标）。
- `phase2_eval` 是完整回测口径（val/test）的 `TR/ASR/MDD`，并与同 split 的 `DP` 基准对比得到 `gap(TR)`。

| 批次 | Phase II avg_return (BEST → FINAL) | phase2_eval val TR / ASR / MDD | val gap(TR) | phase2_eval test TR / ASR / MDD | test gap(TR) |
|---|---:|---:|---:|---:|---:|
| batch_09-short | 96.0984 → 86.3404 | 1.1472 / 9.2787 / 0.0201 | -0.6798 | 2.3042 / 9.8067 / 0.0190 | -1.3351 |
| batch_11-short | -1.1703 → -10.7263 | 0.3908 / 3.5425 / 0.0159 | -1.4362 | 1.0567 / 4.8421 / 0.0520 | -2.5826 |
| batch_12-short | 22.9869 → 22.9869 | 0.5413 / 4.8049 / 0.0144 | -1.2857 | 1.1011 / 4.6680 / 0.0709 | -2.5382 |
| batch_17-short | 116.6090 → 107.9747 | 1.3378 / 11.0851 / 0.0189 | -0.4892 | 2.3079 / 11.0322 / 0.0114 | -1.3314 |
| batch_24-short | 146.2490 → 145.0575 | 1.5368 / 12.5244 / 0.0178 | -0.2902 | 2.9726 / 12.2121 / 0.0164 | -0.6666 |

关键解读:
- `batch_11/12` 的回撤不仅是 TR 下降，test MDD 也显著变差，属于“策略质量”整体退化。
- `batch_17` 相对 `batch_09` 的 test TR 接近，但 `ASR` 更高、`MDD` 更低，说明不是靠更大风险换收益。
- `batch_24` 的提升更明显，val/test 的 `gap(TR)` 都显著缩小，说明 Phase II 的泛化更强了。

---

## 3. Phase I 指标为何决定 Phase II

Phase II 的 selector 是在 “冻结 Phase I decoder 的动作空间” 内做决策，因此 Phase I 如果学坏了，会直接把 Phase II 的上限压低。

在 `batch_11/12` 的 Phase I 选择报告里，典型症状是:
- `best_fixed_archetype_return_mean` 很低: 表示“最赚钱的单一 archetype”本身就不赚钱。
- `phase2_realizable_proxy_return_mean` 很低: 表示“可实现的 proxy 收益”也很弱。
- `return_usage_correlation` 强负相关: 表示越赚钱的 archetype 越不被用，或使用率集中在低收益 archetype 上。

对应地，`batch_17` 的 Phase I 选择指标恢复为高值且 `corr>0`，Phase II 才有机会稳定学出可交易的选择策略。

---

## 4. 到 batch_17 为止的“主提升路径”（Phase I 驱动）

这一段是本轮最关键的提升路径，基本不依赖调 Phase II 的 PPO 超参。

### 4.1 DP 轨迹与采样结构（为 Phase I 提供可控训练分布）

- DP 轨迹是对时间序列做滑窗，`horizon=72`，对每个 start 运行 DP 规划生成 `(states, actions, rewards)`。
- `src/phase1/dp_planner.py` 增加了起点采样模式与 cache 元数据，采样模式包含 `uniform`、`stratified`、`hybrid_stratified_importance`。
- 实验结论是: importance 会把训练分布推偏，最稳的是 `pure stratified`。

建议复现实验时显式指定:
- `--phase1-start-sampling-mode stratified`
- `--phase1-stratified-ratio 1.0`
- `--phase1-importance-ratio 0.0`

### 4.2 strict profit gate（防止坏 checkpoint 进入 Phase II）

在 `scripts/train_phase1.py` 的 checkpoint 选择里加入 gate，核心约束:
- `realizable_proxy / oracle >= 0.40`
- `best_fixed / oracle >= 0.50`
- `return_usage_correlation >= 0`

效果:
- 训练阶段不再“凭重建指标/health 指标选 checkpoint”，而是确保 Phase I 本身具备进入 Phase II 的收益资格。
- `batch_13/14/15/16` 的失败能被明确定位到 gate 的哪一项，而不是进入 Phase II 后再发现收益塌掉。

### 4.3 profit-aware codebook init/reset（提高高收益 archetype 成型概率）

在 `src/phase1/codebook.py`:
- 初始化允许从“高收益子集”优先写入部分 code（按方向分组）。
- dead code reset 时优先从高收益样本抽 latent。

这一步的作用更像“成功率提升器”，不是最核心突破点，但在 reduced-data 条件下能提高稳定性。

### 4.4 return bucket objective + light alignment（真正决定 batch_17）

核心变化在 `scripts/train_phase1.py`:
- 把“连续收益回归”（弱约束）升级为“收益分桶分类”（强语义约束）。
- 训练日志中新增 `bucket_acc`、`usage_corr` 等可观测指标。
- 在 return bucket 已经把语义拉正后，只加很轻的 usage-profit alignment（`phase1_usage_profit_alignment_weight=0.01`, `phase1_usage_profit_alignment_target_corr=0.02`）。

直观理解:
- return bucket 负责让 archetype “语义上可区分收益谱系”
- light alignment 负责让 assignment “轻推回正相关”，但不主导训练

---

## 5. batch_24 的补充提升（Phase II 驱动，在 Phase I 已稳定后）

当 Phase I 已经稳定产出可交易的 archetype 后，Phase II 的收益仍会受 imitation/KL 约束强度和训练时长影响。

`batch_24` 这组改动的核心是:
- 缩短 Phase II 训练 (`phase2_total_steps: 5_000_000 → 1_000_000`)
- 降低 imitation/KL 初值并做线性退火 (`selection_alpha=0.5`, `alpha_schedule=linear`, `final_ratio=0.0`)
- 对 imitation 施加 raw-return mask (`phase2_imitation_min_raw_return`)

这些改动对应提交:
- `13266e4` 早停/缩短训练 + alpha anneal + raw-return mask
- `be5c968` Phase II 代码模块化重构（`src/phase2/*`）
- `b49caa0` Phase II 进一步重构与拆分（rollout/eval 逻辑更清晰）

---

## 6. 可复现与验证清单（建议你跑实验时按这个顺序看）

### 6.1 先验证 Phase I 是否“值得进 Phase II”

检查文件:
- `result/AL/<batch>/phase1_archetype_discovery/phase1_checkpoint_selection_report.json`

重点字段:
- `selection_best_fixed_archetype_return_mean`
- `selection_phase2_realizable_proxy_return_mean`
- `selection_return_usage_correlation`
- `profit_gate` 是否命中（log 会打印命中数量）

### 6.2 再看 Phase II 训练是否稳定

检查文件:
- `result/AL/<batch>/phase2_archetype_selection/AL_phase2_validation_report.json`

检查日志:
- `Phase II 结束验证（BEST/FINAL）: avg_return=...`
- 健康度 `weak_edge/bad_negative_return` 等

### 6.3 最后用 phase2_eval + DP gap 做回归验收

检查文件:
- `result/AL/<batch>/phase2_eval_val/AL_results.json`
- `result/AL/<batch>/phase2_eval_test/AL_results.json`

检查日志:
- `DP 评估结果 [AL/val|test]`
- `gap(TR)`

---

## 7. 追溯用日志与提交（最重要的几个）

关键日志:
- `logs/AL/batch_09-short/AL_pipeline_20260419_144814.log`（基线）
- `logs/AL/batch_11-short/AL_pipeline_20260419_162002.log`（回撤）
- `logs/AL/batch_12-short/AL_pipeline_20260419_171126.log`（回撤）
- `logs/AL/batch_17-short/AL_pipeline_20260419_215837.log`（首次超过基线）
- `logs/AL/batch_24-short/AL_pipeline_20260421_010148.log`（Phase II 补充提升）

关键提交（建议按时间顺序查看）:
- `883b97b` Phase I best model eval + 修复滑窗 start_index 回放验证
- `3316b5c` 固定随机（提高复现实验稳定性）
- `cdfed08` dp 采样分层 + Phase I profit-aware 目标（return bucket + gate + init/reset 等）
- `13266e4` 缩短 Phase II 训练 + alpha 退火 + raw-return imitation mask
- `be5c968` Phase II 模块化重构（`src/phase2/*`）
- `b49caa0` Phase II 进一步重构（rollout/eval 拆分）
