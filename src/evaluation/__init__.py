"""Shared and Phase II evaluation package.

Phase I evaluation/reporting lives in ``src.phase1.evaluation``.

子包:
- ``metrics/``     : 按域拆分的纯函数指标（action / risk / archetype / behavior / stability）。
- ``diagnostics/`` : shared/Phase II diagnostics.

边界:
- evaluator 不实现手续费/滑点/成交逻辑（交给 ``src.trading``）。
- metrics 子包保持纯函数或轻状态对象，便于离线重算与单元测试。
"""
