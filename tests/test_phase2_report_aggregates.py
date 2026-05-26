import math

import numpy as np

from src.phase2.evaluators.phase2_validation_layers import (
    build_phase2_code_diagnostics,
    build_selector_pair_profitability_matrix,
)
from src.phase2.evaluators.phase2_evaluator import Phase2Evaluator
from src.phase2.phase2_config import Phase2RewardConfig
from src.phase2.phase2_selection_dataset import Phase2SelectionDataset
from src.phase2.report.phase2_selector_report_schema import (
    Phase2ReportDocument,
    Phase2ReportMeta,
    Phase2ReportPairProfitabilityCell,
    Phase2ReportPairProfitabilityMatrix,
    Phase2ReportPairProfitabilityRow,
    template_safe,
)
from src.phase2.metrics import (
    Phase2ReportCodeDiagnosticPayloadRow,
    Phase2ReportPairProfitabilityPayloadRow,
    Phase2ValidationMetrics,
    Phase2ValidationPayloads,
    Phase2ValidationResult,
)
from src.phase2.report.phase2_selector_report_context import (
    Phase2SelectorReportContextBuilder,
)
from src.phase2.report.phase2_selector_report import Phase2SelectorReport


def test_build_selector_pair_profitability_matrix_from_explicit_labels() -> None:
    rows = build_selector_pair_profitability_matrix(
        selected_code_ids=np.asarray([0, 1, 1, 0]),
        selector_returns=np.asarray([1.0, 2.0, -1.0, 3.0]),
        kl_returns=np.asarray([0.0, 1.0, 0.0, 1.0]),
        random_returns=np.asarray([0.0, 0.0, 0.0, 0.0]),
        morphologies=np.asarray(["uptrend", "uptrend", "downtrend", "downtrend"]),
        selector_motifs=np.asarray(["long", "long", "short", "long"]),
    )

    by_pair = {(row.morphology, row.motif): row for row in rows}
    up_long = by_pair[("uptrend", "long")]

    assert up_long.support == 2
    assert math.isclose(up_long.selector_mean_return, 1.5)
    assert math.isclose(up_long.kl_mean_return, 0.5)
    assert math.isclose(up_long.mean_advantage_vs_kl, 1.0)
    assert math.isclose(up_long.win_rate, 1.0)
    assert up_long.dominant_selected_code == 0


def test_build_phase2_code_diagnostics_adds_dominant_pair_and_risk_fields() -> None:
    rows = build_phase2_code_diagnostics(
        selected_code_ids=np.asarray([0, 1, 1, 0]),
        assigned_code_labels=np.asarray([0, 0, 1, 1]),
        selector_returns=np.asarray([1.0, 2.0, -1.0, 3.0]),
        kl_returns=np.asarray([0.0, 1.0, 0.0, 1.0]),
        q_margins=np.asarray([0.2, 0.05, 0.3, 0.4]),
        num_archetypes=2,
        morphologies=np.asarray(["uptrend", "uptrend", "downtrend", "downtrend"]),
        selector_motifs=np.asarray(["long", "long", "short", "long"]),
    )

    by_code = {row.code_id: row for row in rows}
    code_0 = by_code[0]

    assert code_0.selector_support == 2
    assert math.isclose(code_0.selector_usage_ratio, 0.5)
    assert code_0.dominant_morphology == "uptrend"
    assert code_0.dominant_motif == "long"
    assert code_0.dominant_pair == "uptrend / long"
    assert math.isclose(code_0.mean_q_margin, 0.3)
    assert code_0.status == "pass"


def test_phase2_pair_profitability_view_model_is_template_safe() -> None:
    cell = Phase2ReportPairProfitabilityCell(
        morphology="uptrend",
        motif="long",
        support="12",
        selector_mean_return="+1.2",
        kl_mean_return="+0.7",
        random_mean_return="+0.1",
        mean_advantage_vs_kl="+0.5",
        mean_advantage_vs_random="+1.1",
        win_rate="66.67%",
        fee_drag_ratio="10.00%",
        dominant_selected_code="2",
        dominant_selected_code_ratio="50.00%",
        display_value="+1.2 / adv +0.5 / n=12 c2",
    )
    matrix = Phase2ReportPairProfitabilityMatrix(
        motifs=("long",),
        rows=(Phase2ReportPairProfitabilityRow("uptrend", (cell,)),),
        cells=(cell,),
    )

    payload = template_safe(matrix)

    assert payload["rows"][0]["cells"][0]["display_value"] == (
        "+1.2 / adv +0.5 / n=12 c2"
    )


def _sample_validation_payloads() -> Phase2ValidationPayloads:
    return Phase2ValidationPayloads(
        selector_pair_profitability_matrix=(
            Phase2ReportPairProfitabilityPayloadRow(
                morphology="uptrend",
                motif="long",
                support=12,
                selector_mean_return=1.2,
                kl_mean_return=0.7,
                random_mean_return=0.1,
                mean_advantage_vs_kl=0.5,
                mean_advantage_vs_random=1.1,
                win_rate=0.667,
                fee_drag_ratio=0.1,
                dominant_selected_code=2,
                dominant_selected_code_ratio=0.5,
            ),
        ),
        code_diagnostics=(
            Phase2ReportCodeDiagnosticPayloadRow(
                code_id=2,
                status="pass",
                selector_support=12,
                selector_usage_ratio=0.25,
                kl_support=8,
                kl_usage_ratio=0.15,
                usage_delta=0.1,
                selector_mean_return=1.2,
                kl_mean_return=0.7,
                uplift_vs_kl=0.5,
                selector_win_rate=0.667,
                selector_fee_drag_ratio=0.1,
                selector_turnover=0.2,
                dominant_morphology="uptrend",
                dominant_morphology_ratio=0.75,
                dominant_motif="long",
                dominant_motif_ratio=0.8,
                dominant_pair="uptrend / long",
                dominant_pair_ratio=0.7,
                mean_q_margin=0.3,
                low_confidence_ratio=0.05,
                profitable_deviation_count=3,
                unprofitable_deviation_count=1,
                unprofitable_deviation_rate=0.083,
                risk_reason="ok",
            ),
        ),
    )


def test_phase2_context_builds_heatmap_and_code_diagnostics_from_payload() -> None:
    validation = Phase2ValidationResult(
        metrics=Phase2ValidationMetrics(
            mean_return=1.0,
            median_return=1.0,
            sharpe_like=1.0,
            win_rate=0.5,
            mean_turnover=0.2,
        ),
        payloads=_sample_validation_payloads(),
    )
    document = Phase2ReportDocument(
        report=Phase2ReportMeta.generated(),
        validation=validation,
    )

    context = Phase2SelectorReportContextBuilder().build(document)

    assert context.pair_profitability_matrix.rows[0].morphology == "uptrend"
    assert (
        context.pair_profitability_matrix.rows[0].cells[0].display_value
        == "1.2 / adv 0.5 / n=12 c2"
    )
    assert context.code_diagnostic_rows[0].code_id == "2"
    assert context.code_diagnostic_rows[0].selector_usage_ratio == "25.00%"
    assert context.code_diagnostic_rows[0].dominant_pair == "uptrend / long"


def test_phase2_selector_report_template_renders_heatmap_and_diagnostics() -> None:
    validation = Phase2ValidationResult(
        metrics=Phase2ValidationMetrics(
            mean_return=1.0,
            median_return=1.0,
            sharpe_like=1.0,
            win_rate=0.5,
            mean_turnover=0.2,
        ),
        payloads=_sample_validation_payloads(),
    )
    document = Phase2ReportDocument(
        report=Phase2ReportMeta.generated(),
        validation=validation,
    )

    html = Phase2SelectorReport().render_html(document)

    assert "Dominant Pair 热力图" in html
    assert "Code 级诊断表" in html
    assert "uptrend / long" in html
    assert "累计收益序列数据" not in html


def test_phase2_evaluator_report_fields_include_pair_profitability_matrix() -> None:
    evaluator = Phase2Evaluator(
        reward_config=Phase2RewardConfig(),
        device="cpu",
    )
    sample_count = 4
    horizon = 4
    states = np.zeros((sample_count, horizon, 1), dtype=np.float32)
    relative_states = np.zeros_like(states)
    trend_states = np.zeros_like(states)
    prices = np.asarray(
        [
            [[1.0], [1.1], [1.2], [1.3]],
            [[1.0], [0.9], [0.8], [0.7]],
            [[1.0], [1.0], [1.1], [1.0]],
            [[1.0], [1.2], [1.1], [1.3]],
        ],
        dtype=np.float32,
    )
    dataset = Phase2SelectionDataset(
        visible_states=tuple(np.zeros((sample_count, 1), dtype=np.float32) for _ in range(6)),
        horizon_dataset=(states, relative_states, trend_states, prices, None),
        demonstration_horizon_label_dataset=(
            np.arange(sample_count, dtype=np.int64),
            np.asarray([0, 1, 0, 1], dtype=np.int64),
        ),
    )

    payloads = evaluator._build_report_fields(
        selector_returns=np.asarray([1.0, 2.0, -1.0, 3.0]),
        assigned_label_returns=np.asarray([0.0, 1.0, 0.0, 1.0]),
        random_returns=np.zeros(sample_count),
        oracle_returns=np.ones(sample_count),
        hold_returns=np.zeros(sample_count),
        selector_fees=np.asarray([0.1, 0.1, 0.1, 0.1]),
        selector_turnover=np.asarray([0.2, 0.3, 0.4, 0.5]),
        selector_actions=np.asarray(
            [
                [2, 2, 2, 2],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 1, 2, 1],
            ],
            dtype=np.int64,
        ),
        q_margins=np.asarray([0.2, 0.1, 0.3, 0.4]),
        selected_code_ids=np.asarray([0, 1, 1, 0]),
        assigned_code_labels=np.asarray([0, 0, 1, 1]),
        per_code_diagnostics=(),
        dataset=dataset,
        num_archetypes=2,
    )

    assert payloads.selector_pair_profitability_matrix
    assert payloads.code_diagnostics
