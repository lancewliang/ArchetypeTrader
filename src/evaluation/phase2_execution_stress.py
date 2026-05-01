"""Phase II execution-stress evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config.phase2_config import ExecutionStressConfig
from src.evaluation.phase2_evaluator import Phase2Evaluator
from src.evaluation.phase2_metrics import phase2_composite_metrics


@dataclass
class ExecutionStressScenario:
    """Single execution-stress scenario."""
    commission_multiplier: float
    slippage_multiplier: float
    execution_lag_offset: int


@dataclass
class ExecutionStressResult:
    """Execution-stress summary."""
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    selector_latency: Dict[str, float] = field(default_factory=dict)


class Phase2ExecutionStressRunner:
    """Run report-only execution stress scenarios.

    The runner accepts a callable so tests and production code can inject either
    the normal walk-forward runner or a cost-adjusted runner.
    """

    def __init__(
        self,
        config: ExecutionStressConfig,
        run_records: Callable[[ExecutionStressScenario], List[Any]],
        num_codes: int,
        dead_code_mask: Optional[List[bool]] = None,
    ) -> None:
        self.config = config
        self.run_records = run_records
        self.num_codes = num_codes
        self.dead_code_mask = dead_code_mask or [False] * num_codes

    def scenarios(self) -> List[ExecutionStressScenario]:
        return [
            ExecutionStressScenario(c, s, lag)
            for c in self.config.commission_multipliers
            for s in self.config.slippage_multipliers
            for lag in self.config.execution_lag_offsets
        ]

    def run(self) -> ExecutionStressResult:
        result = ExecutionStressResult(
            selector_latency={"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        )
        for scenario in self.scenarios():
            records = self.run_records(scenario)
            rec_dicts = [
                Phase2Evaluator._record_to_dict(r) if not isinstance(r, dict) else r
                for r in records
            ]
            metrics = phase2_composite_metrics(
                rec_dicts,
                {},
                self.num_codes,
                self.dead_code_mask,
            )
            result.scenarios.append({
                "commission_multiplier": scenario.commission_multiplier,
                "slippage_multiplier": scenario.slippage_multiplier,
                "execution_lag_offset": scenario.execution_lag_offset,
                "net_return": metrics.get("net_return", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "turnover": metrics.get("turnover", 0.0),
            })
        return result

