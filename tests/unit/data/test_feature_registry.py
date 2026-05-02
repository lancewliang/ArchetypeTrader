"""``feature_registry`` 单元测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.feature_registry import (
    FIXED_FEATURES,
    default_factor_list_path,
    load_feature_selection,
)


def test_feature_registry_loads_pair_factor_list():
    spec = load_feature_selection(pair="AL", profile="short")
    assert spec.pair == "AL"
    assert spec.profile == "short"
    assert spec.factor_list_path.endswith("src/factors/AL/short.txt")
    assert spec.fixed_features == FIXED_FEATURES
    assert "close" not in spec.feature_columns
    assert spec.feature_columns[:4] == [
        "ask1_price",
        "ask1_size",
        "bid1_price",
        "bid1_size",
    ]
    assert "ask_gap_3_4" in spec.configured_factors


def test_feature_registry_strips_quotes_comments_and_blank_lines(tmp_path: Path):
    factor_file = tmp_path / "factors.txt"
    factor_file.write_text(
        """
        # comment
        "factor_a"

        'factor_b'
        factor_c
        """,
        encoding="utf-8",
    )
    spec = load_feature_selection("T", "short", str(factor_file))
    assert spec.configured_factors == ["factor_a", "factor_b", "factor_c"]


def test_feature_registry_rejects_close_in_factor_file(tmp_path: Path):
    factor_file = tmp_path / "factors.txt"
    factor_file.write_text("factor_a\nclose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="close"):
        load_feature_selection("T", "short", str(factor_file))


def test_feature_registry_deduplicates_fixed_feature(tmp_path: Path):
    factor_file = tmp_path / "factors.txt"
    factor_file.write_text("ask1_price\nfactor_a\n", encoding="utf-8")
    spec = load_feature_selection("T", "short", str(factor_file))
    assert spec.feature_columns.count("ask1_price") == 1
    assert spec.deduplicated_features == ["ask1_price"]
    assert spec.feature_columns[-1] == "factor_a"


def test_feature_registry_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        load_feature_selection("T", "short", str(missing))


def test_default_factor_list_path_points_inside_repo():
    assert default_factor_list_path("AL", "short").as_posix().endswith(
        "src/factors/AL/short.txt"
    )
