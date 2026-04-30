"""评估与诊断包。

顶层文件只做调度与门面: ``phase1_evaluator``、``phase1_replay``、``phase1_metrics``、``phase1_report``。

子包:
- ``metrics/``     : 按域拆分的纯函数指标（action / risk / archetype / behavior / stability）。
- ``diagnostics/`` : latent 可视化与 failure case 错题本。

边界:
- ``phase1_evaluator`` 不实现手续费/滑点/成交逻辑（交给 ``src.trading``）。
- ``phase1_evaluator`` 不直接写 report 文件（交给 ``phase1_report``）。
- ``phase1_metrics`` 是稳定 API 门面，对外只暴露重新导出的函数；内部子包可重构。
"""
