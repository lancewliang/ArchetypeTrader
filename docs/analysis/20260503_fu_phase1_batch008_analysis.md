# FU Phase I Batch 008 Analysis

日期: 2026-05-03

## 结论

`batch_008_prospective` 优先进入 Phase II。它使用 `prospective_past` 分层，只依赖窗口起点之前的信息，更接近线上可观测条件，并且没有 sign-off 阻塞。

`batch_008` 可保留为 hindsight 对照分析产物，但不可作为 sign-off 版本。它训练完成并产出 `best_vq_model.pt`，但 hindsight/prospective 对照诊断触发 `hindsight_vs_prospective_delta_exceeded`。

## 实验批次

| 批次 | 分层模式 | 用途 | 状态 |
| --- | --- | --- | --- |
| `batch_008_prospective` | `prospective_past` | Phase I 主候选与 Phase II 输入 | 通过 |
| `batch_008` | `hindsight_horizon` | hindsight 对照实验 | 训练完成, sign-off 阻塞 |

两个批次均使用:

- `pair=FU`
- `horizon=72`
- `num_demos=18000`
- `train/val/test horizons = 18000/64/64`
- `seed=42`
- `max_position=10`
- `prospective_lookback_minutes=720`
- `train_file=data/FU/train.feather`
- `val_file=data/FU/val.feather`
- `test_file=data/FU/test.feather`

## 数据预处理过程

### prospective 对照

目标: 生成只使用 past strata 的 Phase I 数据处理产物。

关键产物:

- `artifacts/FU/batch_008_prospective/phase1/data_process_manifest.json`
- `schema_hash=87165faf9d9f4f9f`
- `data_process_hash=8aee54cd87a7cadf`
- `dp_teacher_hash=333e545fb65fc20a`

### hindsight 对照

目标: 使用 horizon 内统计做分层，并通过 `--diagnostic-pair-batch-id batch_008_prospective` 绑定 prospective 对照。

最终成功命令需要显式放宽重叠阈值，因为 `num_demos=18000` 在 `horizon=72` 下理论最小 overlap 高于默认 `0.5`。

```bash
python scripts/process_phase1_data.py \
  --pair FU \
  --data-batch-id batch_008 \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --horizon 72 \
  --num-demos 18000 \
  --sampling-strategy stratified_uniform \
  --stratification-mode hindsight_horizon \
  --diagnostic-pair-batch-id batch_008_prospective \
  --prospective-lookback-minutes 720 \
  --sampling-max-overlap-ratio 0.65 \
  --max-position 10 \
  --seed 42
```

实际日志记录 `window_overlap_ratio=0.662616`，因此最终应以本次成功产物为准，并记录采样密度偏高这一风险。

关键产物:

- `artifacts/FU/batch_008/phase1/data_process_manifest.json`
- `schema_hash=87165faf9d9f4f9f`
- `data_process_hash=ac0d7a167fa714b8`
- `dp_teacher_hash=fd0001a14fb1c579`

## Phase I 训练过程

### prospective 训练

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id batch_008_prospective \
  --data-process-manifest artifacts/FU/batch_008_prospective/phase1/data_process_manifest.json \
  --horizon 72 \
  --epochs 61 \
  --pretrain-epochs 10 \
  --batch-size 4086 \
  --num-archetypes 10 \
  --max-position 10 \
  --seed 42 \
  --device cuda \
  --stratification-mode prospective_past \
  --prospective-lookback-minutes 720
```

结果:

- 日志: `logs/FU/batch_008_prospective/ts-20260503-183822-2161507.log`
- best epoch: `54`
- report: `artifacts/FU/batch_008_prospective/phase1/phase1_report.json`
- leakage warning: `not_applicable`
- sign-off blocked reason: 空

### hindsight 训练

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id batch_008 \
  --data-process-manifest artifacts/FU/batch_008/phase1/data_process_manifest.json \
  --horizon 72 \
  --epochs 61 \
  --pretrain-epochs 10 \
  --batch-size 4086 \
  --num-archetypes 10 \
  --max-position 10 \
  --seed 42 \
  --device cuda \
  --stratification-mode hindsight_horizon \
  --diagnostic-pair-batch-id batch_008_prospective \
  --prospective-lookback-minutes 720
```

结果:

- 日志: `logs/FU/batch_008/ts-20260503-191556-2195857.log`
- best epoch: `39`
- report: `artifacts/FU/batch_008/phase1/phase1_report.json`
- leakage warning: `exceeded`
- sign-off blocked reason: `hindsight_vs_prospective_delta_exceeded`

训练本身已完成，`best_vq_model.pt`、encoder、decoder、codebook 和 report 均已写出。最后的 fatal 是 sign-off 阻塞，不是训练中断。

## 指标对比

| 指标 | `batch_008_prospective` | `batch_008` | 判断 |
| --- | ---: | ---: | --- |
| `phase1_composite_score` | 5.491991 | 4.411499 | prospective 更高 |
| `val_return_capture_ratio` | 0.657790 | 0.636075 | prospective 更高 |
| `val_sharpe_ratio` | 47.005587 | 36.436668 | prospective 明显更高 |
| `val_max_drawdown` | 0.006874 | 0.014621 | prospective 更低 |
| `val_max_drawdown_abs` | 213.13 | 561.9275 | prospective 更低 |
| `code_usage_ratio` | 1.0 | 0.9 | prospective 更好 |
| `no_trade_ratio` | 0.005444 | 0.001778 | hindsight 更低 |
| `val_weighted_reconstruction_accuracy` | 0.958568 | 0.959076 | hindsight 略高 |

综合看，prospective 的收益捕获、Sharpe、回撤、code usage 和 sign-off 状态都更强。hindsight 只有 no-trade ratio 与 reconstruction accuracy 略占优，但不足以覆盖后视诊断阻塞。

## Leakage 诊断

`batch_008` 读取:

```text
artifacts/FU/batch_008_prospective/phase1/phase1_report.json
```

核心差异:

| 指标 | hindsight | prospective | abs delta | 阈值 | 是否超阈 |
| --- | ---: | ---: | ---: | ---: | --- |
| `val_return_capture_ratio` | 0.636075 | 0.657790 | 0.021716 | 0.2 | 否 |
| `val_sharpe_ratio` | 36.436668 | 47.005587 | 10.568920 | 0.5 | 是 |
| `val_max_drawdown` | 0.014621 | 0.006874 | 0.007747 | 0.1 | 否 |
| `code_usage_ratio` | 0.9 | 1.0 | 0.1 | 0.1 | 否 |

阻塞原因:

```text
hindsight_vs_prospective_delta_exceeded
```

触发指标:

```text
val_sharpe_ratio
```

## Phase II 建议

Phase II 应使用:

```text
--phase1-batch-id batch_008_prospective
```

不要使用 `batch_008` 作为默认 Phase II 输入。若强行使用 `batch_008`，`scripts/train_phase2.py` 需要加 `--allow-phase1-hindsight-warning`，但这只适合诊断或消融，不适合 sign-off。

推荐 Phase II 训练命令:

```bash
python scripts/train_phase2.py \
  --pair FU \
  --phase1-batch-id batch_008_prospective \
  --phase2-batch-id batch_008_prospective_phase2 \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --total-timesteps 3000000 \
  --num-envs 8 \
  --rollout-length 256 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --max-position 10 \
  --seed 42 \
  --device cuda
```

快速 smoke 可先把 `--total-timesteps` 改成 `100000`。

推荐 Phase II 回测命令:

```bash
python scripts/backtest_phase2.py \
  --pair FU \
  --phase1-batch-id batch_008_prospective \
  --phase2-batch-id batch_008_prospective_phase2 \
  --test-file data/FU/test.feather \
  --checkpoint artifacts/FU/batch_008_prospective_phase2/phase2/best_selector.pt \
  --max-position 10 \
  --seed 42
```

可选 KL/demo 消融:

```bash
python scripts/train_phase2.py \
  --pair FU \
  --phase1-batch-id batch_008_prospective \
  --phase2-batch-id batch_008_prospective_phase2_ablation \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --total-timesteps 3000000 \
  --num-envs 8 \
  --rollout-length 256 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --max-position 10 \
  --seed 42 \
  --device cuda \
  --run-kl-demo-ablation \
  --kl-demo-ablation-values 0.0 0.1 0.5 1.0
```

