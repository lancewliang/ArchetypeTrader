"""Phase II 失败案例报告: worst return / largest regret / largest cost / unstable switching。

设计文档锚点: Phase II 执行计划 §Step 7。

职责:
- 筛选 worst return / largest regret / largest cost / unstable switching / risk trigger 案例。
- 输出结构化 JSON 和可选 HTML 报告。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Phase2FailureCase:
    """单个失败案例。"""
    sample_id: str
    category: str
    chosen_code: int
    reward: float
    regret: float = 0.0
    cost_paid: float = 0.0
    switch_count: int = 0
    risk_triggered: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class Phase2FailureCaseReportBuilder:
    """Phase II 失败案例报告构建器。"""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def select_cases(
        self,
        horizon_records: List[Dict[str, Any]],
    ) -> List[Phase2FailureCase]:
        """从 replay 记录中筛选失败案例。

        筛选规则:
        - worst_return: reward 最低的 top_k。
        - largest_cost: cost_paid 最大的 top_k。
        - unstable_switching: 切换次数最多的 top_k（基于相邻 horizon 的 code 变化）。
        - risk_trigger: 触发风控的案例。
        """
        cases: List[Phase2FailureCase] = []

        if not horizon_records:
            return cases

        # worst_return
        sorted_by_reward = sorted(horizon_records, key=lambda r: r.get("reward_raw", 0.0))
        for r in sorted_by_reward[: self.top_k]:
            cases.append(Phase2FailureCase(
                sample_id=r.get("sample_id", ""),
                category="worst_return",
                chosen_code=r.get("chosen_code", 0),
                reward=r.get("reward_raw", 0.0),
                cost_paid=r.get("cost_paid", 0.0),
            ))

        # largest_cost
        sorted_by_cost = sorted(
            horizon_records, key=lambda r: r.get("cost_paid", 0.0), reverse=True
        )
        for r in sorted_by_cost[: self.top_k]:
            cases.append(Phase2FailureCase(
                sample_id=r.get("sample_id", ""),
                category="largest_cost",
                chosen_code=r.get("chosen_code", 0),
                reward=r.get("reward_raw", 0.0),
                cost_paid=r.get("cost_paid", 0.0),
            ))

        # unstable_switching: 计算相邻 horizon 的 code 变化
        if len(horizon_records) > 1:
            switch_records = []
            for i in range(1, len(horizon_records)):
                prev_code = horizon_records[i - 1].get("chosen_code", 0)
                curr_code = horizon_records[i].get("chosen_code", 0)
                if prev_code != curr_code:
                    switch_records.append(horizon_records[i])
            # 取切换最频繁的区域
            for r in switch_records[: self.top_k]:
                cases.append(Phase2FailureCase(
                    sample_id=r.get("sample_id", ""),
                    category="unstable_switching",
                    chosen_code=r.get("chosen_code", 0),
                    reward=r.get("reward_raw", 0.0),
                    switch_count=1,
                ))

        # risk_trigger
        for r in horizon_records:
            if r.get("risk_triggered", False):
                cases.append(Phase2FailureCase(
                    sample_id=r.get("sample_id", ""),
                    category="risk_trigger",
                    chosen_code=r.get("chosen_code", 0),
                    reward=r.get("reward_raw", 0.0),
                    risk_triggered=True,
                ))

        return cases

    def write_jsonl(
        self,
        cases: List[Phase2FailureCase],
        output_path: Path,
    ) -> Path:
        """写 JSONL 格式的失败案例。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(asdict(case), ensure_ascii=False, sort_keys=True))
                f.write("\n")
        return output_path

    def write_html(
        self,
        cases: List[Phase2FailureCase],
        output_path: Path,
    ) -> Path:
        """写 HTML 格式的失败案例报告。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_parts = ["<html><head><title>Phase II Failure Cases</title></head><body>"]
        html_parts.append("<h1>Phase II Failure Case Report</h1>")
        html_parts.append(f"<p>Total cases: {len(cases)}</p>")

        for case in cases:
            html_parts.append(f"<div style='border:1px solid #ccc;margin:10px;padding:10px'>")
            html_parts.append(f"<h3>[{case.category}] {case.sample_id}</h3>")
            html_parts.append(f"<p>Code: {case.chosen_code}, Reward: {case.reward:.6f}, "
                              f"Cost: {case.cost_paid:.6f}</p>")
            if case.risk_triggered:
                html_parts.append("<p style='color:red'>Risk Triggered</p>")
            html_parts.append("</div>")

        html_parts.append("</body></html>")
        output_path.write_text("\n".join(html_parts), encoding="utf-8")
        return output_path
