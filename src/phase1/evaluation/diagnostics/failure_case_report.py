"""Failure case 错题本.

设计文档锚点: §4.13。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Sequence

from src.utils.feather_io import atomic_write_json, write_jsonl


CaseSet = Literal[
    "worst_student_return",
    "largest_regret_to_dp",
    "largest_cost_paid",
    "switch_mismatch",
    "false_trade_on_no_trade",
]


@dataclass
class FailureCase:
    sample_id: str
    case_set: str
    rank: int
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    price_series: list = field(default_factory=list)
    dp_actions: list = field(default_factory=list)
    student_actions: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    step_rewards: list = field(default_factory=list)
    cumulative_returns: list = field(default_factory=list)
    drawdowns: list = field(default_factory=list)


def _sort_records(records: Sequence[dict], key: str, reverse: bool, top_k: int) -> List[dict]:
    """按 key 排序取 top-k；缺 key 的记录视为 0 处理。"""
    return sorted(records, key=lambda r: r.get(key, 0.0), reverse=reverse)[:top_k]


class FailureCaseReportBuilder:
    """生成 failure case 错题本。

    用途
    ----
    在 evaluator 完成 validation replay 后，自动筛选最值得人工复盘的 horizon，
    生成静态 HTML（可在浏览器内打开）+ JSONL（机器可读）。
    HTML 仅诊断用，不参与 checkpoint 选择；其 case ID 必须能回溯到
    ``window_index_val.feather`` 与 replay records。
    """

    def __init__(
        self,
        top_k: int,
        max_cases_per_report: int,
        action_labels: Dict[int, str],
    ) -> None:
        self.top_k = top_k
        self.max_cases_per_report = max_cases_per_report
        self.action_labels = action_labels

    # ---------- 选择 ----------

    def select_cases(self, horizon_replay_records: List[dict]) -> Dict[str, List[FailureCase]]:
        """按各 case set 排序选 top-K。

        筛选规则（设计 §4.13）
        ---------------------
        - ``worst_student_return`` : ``student_net_return`` 升序。
        - ``largest_regret_to_dp`` : ``regret_to_dp`` 降序。
        - ``largest_cost_paid`` : ``cost_paid`` 降序。
        - ``switch_mismatch`` : ``switch_timing_error`` 降序。
        - ``false_trade_on_no_trade`` : 仅在 ``teacher_is_no_trade`` 中按
                                          ``student_turnover`` 降序。

        Returns
        -------
        ``{case_set: [FailureCase, ...]}`` ；空 records 时各 list 为空。
        """
        worst_ret = _sort_records(horizon_replay_records, "student_net_return", reverse=False, top_k=self.top_k)
        largest_regret = _sort_records(horizon_replay_records, "regret_to_dp", reverse=True, top_k=self.top_k)
        largest_cost = _sort_records(horizon_replay_records, "cost_paid", reverse=True, top_k=self.top_k)
        # switch mismatch / false_trade_on_no_trade: 需要 record 提供布尔字段 / 距离字段
        switch_mismatch = _sort_records(
            horizon_replay_records, "switch_timing_error", reverse=True, top_k=self.top_k
        )
        false_trade = _sort_records(
            [r for r in horizon_replay_records if r.get("teacher_is_no_trade")],
            "student_turnover",
            reverse=True,
            top_k=self.top_k,
        )

        def _build(case_set: str, rec: dict, rank: int) -> FailureCase:
            case = FailureCase(
                sample_id=rec.get("sample_id", "unknown"),
                case_set=case_set,
                rank=rank,
                metrics={
                    "student_net_return": rec.get("student_net_return", 0.0),
                    "teacher_net_return": rec.get("teacher_net_return", 0.0),
                    "regret_to_dp": rec.get("regret_to_dp", 0.0),
                    "cost_paid": rec.get("cost_paid", 0.0),
                    "switch_timing_error": rec.get("switch_timing_error", 0.0),
                },
                price_series=rec.get("price_series", []),
                dp_actions=rec.get("teacher_actions", []),
                student_actions=rec.get("student_actions", []),
                positions=rec.get("positions", []),
                step_rewards=rec.get("step_rewards", []),
                cumulative_returns=rec.get("cumulative_returns", []),
                drawdowns=rec.get("drawdowns", []),
            )
            case.failure_modes = self.classify_failure_modes(case)
            return case

        return {
            "worst_student_return": [_build("worst_student_return", r, i) for i, r in enumerate(worst_ret)],
            "largest_regret_to_dp": [_build("largest_regret_to_dp", r, i) for i, r in enumerate(largest_regret)],
            "largest_cost_paid": [_build("largest_cost_paid", r, i) for i, r in enumerate(largest_cost)],
            "switch_mismatch": [_build("switch_mismatch", r, i) for i, r in enumerate(switch_mismatch)],
            "false_trade_on_no_trade": [_build("false_trade_on_no_trade", r, i) for i, r in enumerate(false_trade)],
        }

    def classify_failure_modes(self, case: FailureCase) -> List[str]:
        """根据 case payload 推导 ``late_entry / wrong_direction / over_trading / ...`` 标签。

        当前规则
        --------
        - ``missed_trade``: regret > 0 且 student 收益为负 → 该交易但没交易。
        - ``cost_dominated``: cost 已经超过 student 净收益绝对值。
        - ``late_entry``: switch_timing_error > 5 bar。
        - ``wrong_direction``: DP 与 student 在切换点出现 long↔short 反向。

        返回去重后保持顺序的 list。
        """
        modes: List[str] = []
        m = case.metrics
        if m.get("regret_to_dp", 0.0) > 0 and m.get("student_net_return", 0.0) < 0:
            modes.append("missed_trade")
        if m.get("cost_paid", 0.0) > abs(m.get("student_net_return", 0.0)):
            modes.append("cost_dominated")
        if m.get("switch_timing_error", 0.0) > 5:
            modes.append("late_entry")
        # 方向: DP long 与 student short 的混合
        if case.dp_actions and case.student_actions:
            for a, b in zip(case.dp_actions, case.student_actions):
                if a == 0 and b == 2:
                    modes.append("wrong_direction")
                    break
                if a == 2 and b == 0:
                    modes.append("wrong_direction")
                    break
        return list(dict.fromkeys(modes))  # 去重保持顺序

    # ---------- 输出 ----------

    def write_html(self, cases: Dict[str, List[FailureCase]], path: Path) -> Path:
        """生成静态 HTML（无外网依赖；图表交给后续 matplotlib 扩展或 SVG 内嵌）。

        当前版本只输出文字与动作序列，避免引入 matplotlib 硬依赖。
        测试 ``test_html_no_external_dependencies`` 验证生成的文件不含 ``http`` 引用。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body_parts: List[str] = ["<html><head><meta charset='utf-8'><title>Phase I failure cases</title></head><body>"]
        for case_set, items in cases.items():
            body_parts.append(f"<h2>{case_set}</h2>")
            for case in items:
                body_parts.append(f"<h3>#{case.rank} {case.sample_id}</h3>")
                body_parts.append("<ul>")
                for k, v in case.metrics.items():
                    body_parts.append(f"<li>{k}: {v:.6f}</li>")
                body_parts.append("</ul>")
                if case.failure_modes:
                    body_parts.append("<p>Failure modes: " + ", ".join(case.failure_modes) + "</p>")
                body_parts.append(
                    "<pre>DP    : "
                    + " ".join(self.action_labels.get(int(a), str(a)) for a in case.dp_actions)
                    + "</pre>"
                )
                body_parts.append(
                    "<pre>Stud. : "
                    + " ".join(self.action_labels.get(int(a), str(a)) for a in case.student_actions)
                    + "</pre>"
                )
        body_parts.append("</body></html>")
        target.write_text("\n".join(body_parts), encoding="utf-8")
        return target

    def write_jsonl(self, cases: Dict[str, List[FailureCase]], path: Path) -> Path:
        records = []
        for case_set, items in cases.items():
            for case in items:
                d = asdict(case)
                records.append(d)
        return write_jsonl(records, path)
