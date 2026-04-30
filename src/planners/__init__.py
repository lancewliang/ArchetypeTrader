"""DP planner 包。

边界:
- DP 只能在 Phase I 离线生成 demonstration 与 horizon label 时使用。
- Phase II/III 推理与回测严禁动态调用 DP（避免未来信息泄漏）。
- DP 转移 reward 必须经由共享的 ``TradingEnv`` / ``LobDepthCostModel``，
  禁止在 planner 内自实现一套手续费/滑点/盘口逻辑。
"""
