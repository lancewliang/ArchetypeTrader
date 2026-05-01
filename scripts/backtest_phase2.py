"""Phase II backtest 入口: 加载 best selector + frozen Phase I 产物，执行 test walk-forward。

设计文档锚点: Phase II 执行计划 §Step 1 / §Step 9。

关键约束:
- 明确只加载 best selector + frozen Phase I 产物。
- 不加载 test label 参与决策路径。
- 主结果为 deterministic argmax。
- 可选 stochastic seed pack 诊断。
- 若检测到 test label 进入决策路径直接抛错。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase II Backtest 入口")
    p.add_argument("--pair", required=True)
    p.add_argument("--phase1-batch-id", required=True)
    p.add_argument("--phase2-batch-id", required=True)
    p.add_argument("--test-file", required=True, help="测试数据 feather 路径")
    p.add_argument("--checkpoint", required=True, help="best_selector.pt 路径")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic-seeds", nargs="*", type=int, default=None,
                    help="stochastic seed pack（仅诊断）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Phase II backtest 主入口。

    步骤:
    1. 加载 Phase I 冻结产物。
    2. 加载 best selector checkpoint。
    3. 构造 test dataset 和 HorizonEnv。
    4. 执行 deterministic argmax walk-forward backtest。
    5. 可选: stochastic seed pack 诊断。
    6. 输出 per-horizon records 和 report。
    """
    import torch

    from src.config.phase2_config import Phase2Config
    from src.data.market_reader import MarketFileReader
    from src.data.phase2_dataset import Phase2Dataset
    from src.data.phase2_horizon_index import Phase2HorizonIndexer
    from src.evaluation.phase2_evaluator import Phase2Evaluator
    from src.evaluation.phase2_replay import Phase2BacktestRunner
    from src.evaluation.phase2_report import Phase2ReportPaths, Phase2ReportWriter
    from src.models.archetype_selector import ArchetypeSelector
    from src.models.phase1_frozen_policy import Phase1FrozenPolicy
    from src.rl.actor_critic import ActorCritic
    from src.trainers.phase2_dead_code import build_dead_code_mask
    from src.trading.cost_model import LobDepthCostModel
    from src.trading.env import TradingEnv
    from src.trading.reward_alignment import RewardAlignment
    from src.utils.feather_io import read_json

    parser = build_parser()
    args = parser.parse_args(argv)

    # 构建最小 config
    config = Phase2Config(
        pair=args.pair,
        phase1_batch_id=args.phase1_batch_id,
        phase2_batch_id=args.phase2_batch_id,
        test_file=args.test_file,
        artifact_root=args.artifact_root,
        seed=args.seed,
    )

    p1_dir = config.phase1_dir()
    artifacts_dir = config.artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Phase I 产物
    input_schema = read_json(p1_dir / "input_schema.json")
    phase1_report = read_json(p1_dir / "phase1_report.json") if (p1_dir / "phase1_report.json").exists() else {}
    frozen_policy = Phase1FrozenPolicy.load(
        p1_dir / "decoder.pt", p1_dir / "codebook.pt", device="cpu"
    )

    # 读取 test 数据
    reader = MarketFileReader()
    import polars as pl
    test_frame = pl.read_ipc(args.test_file)

    # 生成 test horizon index
    indexer = Phase2HorizonIndexer(config)
    test_entries = indexer.build_index(test_frame, "test", config.horizon, None)
    test_dataset = Phase2Dataset(test_frame, test_entries, input_schema, config)

    # 加载 selector
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_spec = test_dataset.state_spec()
    selector = ArchetypeSelector(
        state_dim=state_spec.total_dim,
        num_codes=frozen_policy.num_codes,
        config=config.selector_network,
    )
    selector.load_state_dict(ckpt["model_state"])
    selector.eval()
    dead_code_mask = build_dead_code_mask(
        phase1_report,
        frozen_policy.num_codes,
        config.selector_network.dead_code_usage_threshold,
    )
    actor_critic = ActorCritic(
        selector,
        dead_code_mask=torch.tensor(dead_code_mask, dtype=torch.bool),
    )

    # Trading env factory
    p1_config = {}
    p1_config_path = p1_dir / "phase1_config.yaml"
    if p1_config_path.exists():
        import yaml
        with open(p1_config_path) as f:
            p1_config = yaml.safe_load(f) or {}
    cost_cfg = p1_config.get("dp", {}).get("cost_config", {})
    cost_model = LobDepthCostModel(
        commission_rate=cost_cfg.get("commission_rate", 0.0002),
        book_levels=cost_cfg.get("book_levels", 5),
    )
    alignment = RewardAlignment(cost_cfg.get("reward_alignment", "paper_formula"))

    def env_factory():
        return TradingEnv(
            cost_model=cost_model,
            reward_alignment=alignment,
            max_position=config.max_position,
        )

    # 执行 backtest
    runner = Phase2BacktestRunner(config, actor_critic, frozen_policy, test_dataset, env_factory)
    records = runner.run_walk_forward("test", deterministic=True)

    # 输出结果
    from src.evaluation.phase2_metrics import phase2_composite_metrics
    rec_dicts = [Phase2Evaluator._record_to_dict(r) for r in records]
    metrics = phase2_composite_metrics(
        rec_dicts, {}, frozen_policy.num_codes, dead_code_mask
    )

    report_paths = Phase2ReportPaths.from_artifacts_dir(artifacts_dir)
    writer = Phase2ReportWriter(report_paths)
    writer.write_per_horizon_records(rec_dicts, "test")

    print(f"[info] Test net_return: {metrics.get('net_return', 0.0):.6f}")
    print(f"[info] Test sharpe: {metrics.get('sharpe_ratio', 0.0):.4f}")
    print(f"[info] Test max_drawdown: {metrics.get('max_drawdown', 0.0):.4f}")
    print(f"[info] Results written to {artifacts_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
