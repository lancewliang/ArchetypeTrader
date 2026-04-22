我按 logs/AL/batch_01-short/AL_pipeline_20260422_151607.log、代码和 README 看完了。先说结论：现在最值得优先优化的不是继续堆 Phase I 架构，而是先修正 Phase I/Phase II 的窗口对齐问题，然后把 Phase II 从“带错位 imitation 的 PPO selector”往更直接的 horizon 级选择器优化。

关键发现

Phase I 重建已经很强，但原型健康度还不够干净。
best checkpoint 是 epoch 240，token accuracy 0.9998、exact match 0.9906，但仍有软告警：k=3 使用率 2.35%、k=8 几乎不用，且 14 对 codebook cosine > 0.8。也就是说，瓶颈不是“能不能复原 DP 动作”，而是 code 是否足够分散、可交易、可被 Phase II 稳定选择。

Phase II 有收益，但健康度还是 weak_edge。
best val horizon return 154.13，但平均成本 192.58，所以日志给了 weak_edge。完整评估里 val TR 1.6093 vs DP 1.8270，gap -0.2177；test TR 3.0404 vs DP 3.6392，gap -0.5989。这已经不是坏结果，但还有明显空间。

最重要：Phase II 的 imitation label 现在很可能错位。
Phase II 代码用 horizon_indices 直接索引 demo_states[horizon_indices] 取标签，见 rollout.py (line 325)。但 Phase I 的 DP 轨迹是随机滑窗采样，见 dp_planner.py (line 362)。我抽查了这次缓存：前 6098 个 Phase II horizon 里，sampled_start_indices[i] == i*72 的数量是 0/6098。这意味着 imitation 正则大概率在把 selector 拉向“别的窗口”的 VQ label。README 其实也已经提示过这个风险：README_zh.md (line 357)。
我建议的优先级

P0：先修对齐，不急着调参。
Phase I 新生成数据集时，TrajectoryDataset 没把 sampled_start_indices 挂回去，见 train_phase1.py (line 574)。而 env validation 明确需要真实滑窗起点，见 env_validation.py (line 194)。所以当前 Phase I checkpoint 的 env proxy 也可能受错位影响。方案是：生成后立刻保留 sampled_start_indices，或者统一从刚保存的 .npz 重新 from_npz() 加载。

P0：Phase II 标签协议二选一。
要么单独生成一份与 Phase II 非重叠 horizon 对齐的 DP label cache，起点就是 h_idx * horizon；要么让 Phase II rollout 也基于 sampled_start_indices 的任意滑窗起点执行。前者改动更小、更清晰。

P1：Phase II 可以考虑从 PPO 转成“全 K 反事实选择”训练。
现在 K=10，decoder 冻结，对每个 horizon 同时评估 10 个 archetype 的真实 return 是可承受的。这样可以直接训练 selector 选“同一窗口上真实 return 最好”的 archetype，减少对错位/噪声 imitation label 的依赖，比 PPO 的采样回报更稳。

P1：增强 selector 输入上下文。
当前 selector 只看 horizon 第一个 bar 的状态，selection_agent.py (line 19)。判断未来 72 步选哪个 archetype，只看单点状态太薄。可以加过去 N bar 的序列编码，或至少加趋势、波动、振幅、上一 horizon 收益等摘要。

P1：Phase I 做小修，不大改。
修低使用 code：现在 dead-code reset 只重置 count == 0，见 codebook.py (line 352)，但这次 k=8 只有 2/50000，基本也是死码。可以改成低使用阈值重置。另一个小实验是把当前 hybrid 0.95/0.05 改回文档里更稳定的 pure stratified，并和 middle 特征组做对照；这次跑的是 short，README 里记录 middle 通常更强。

P2：暂缓大结构实验。
K=16/20、多尺度 horizon、强 alignment 这些文档里已经有负结果或风险。decoder 加 reward、时间翻转增强可以留作后续，但我会先等窗口对齐和 Phase II 训练目标修正后再判断是否需要。