"""``InputSchemaValidator`` 单元测试."""
from __future__ import annotations

import pytest

from src.preprocess_data.schema import InputSchemaValidator


def _make_frame(close=(100.0, 100.5, 101.0), extra: dict | None = None):
    import polars as pl

    cols = {"timestamp": list(range(len(close))), "close": list(close)}
    returns = [0.0]
    for i in range(1, len(close)):
        prev = close[i - 1]
        returns.append(0.0 if prev <= 0 else (close[i] - prev) / prev)
    cols["return_1m"] = returns
    cols["mid_price"] = list(close)
    if extra:
        cols.update(extra)
    return pl.DataFrame(cols)


def test_valid_frame_passes():
    schema = InputSchemaValidator().validate(_make_frame())
    assert schema.price_column == "close"
    assert "close" not in schema.feature_columns
    assert "return_1m" in schema.feature_columns


def test_missing_close_raises():
    import polars as pl

    frame = pl.DataFrame({"timestamp": [0, 1], "mid_price": [100.0, 101.0]})
    with pytest.raises(ValueError):
        InputSchemaValidator().validate(frame)


def test_close_non_positive_raises():
    frame = _make_frame(close=(100.0, 0.0, 100.0))
    with pytest.raises(ValueError):
        InputSchemaValidator().validate(frame)


def test_nan_in_features_raises():
    frame = _make_frame(extra={"factor_a": [0.1, float("nan"), 0.3]})
    with pytest.raises(ValueError):
        InputSchemaValidator().validate(frame)


def test_meta_columns_excluded():
    frame = _make_frame(extra={"symbol": ["BTC", "BTC", "BTC"]})
    schema = InputSchemaValidator().validate(frame)
    assert "symbol" not in schema.feature_columns
    assert "symbol" in schema.excluded_columns


def test_close_excluded_from_features():
    schema = InputSchemaValidator().validate(_make_frame())
    assert schema.price_column not in schema.feature_columns


def test_orderbook_columns_recognized():
    frame = _make_frame(
        extra={
            "ask1_price": [100.5, 101.0, 101.5],
            "ask1_size": [10.0, 10.0, 10.0],
            "bid1_price": [99.5, 100.0, 100.5],
            "bid1_size": [10.0, 10.0, 10.0],
        }
    )
    schema = InputSchemaValidator().validate(frame)
    assert "ask1_price" in schema.orderbook_columns
    # 盘口字段也是 feature 的一部分（设计原则: 提供给 encoder 输入）
    assert "ask1_price" in schema.feature_columns


def test_assert_close_not_in_features_helper():
    schema = InputSchemaValidator().validate(_make_frame())
    schema.assert_close_not_in_features()


def test_schema_uses_explicit_feature_columns_only():
    frame = _make_frame(extra={"factor_a": [1.0, 2.0, 3.0], "unused_numeric_feature": [9.0, 9.0, 9.0]})
    schema = InputSchemaValidator(feature_columns=["factor_a"]).validate(frame)
    assert schema.feature_columns == ["factor_a"]
    assert "unused_numeric_feature" not in schema.feature_columns
    assert "return_1m" not in schema.feature_columns


def test_schema_requires_all_configured_features():
    with pytest.raises(ValueError, match="missing_feature"):
        InputSchemaValidator(feature_columns=["missing_feature"]).validate(_make_frame())


def test_schema_rejects_non_numeric_configured_feature():
    frame = _make_frame(extra={"factor_a": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="不是数值类型"):
        InputSchemaValidator(feature_columns=["factor_a"]).validate(frame)


def test_schema_validate_against_schema_rejects_missing_val_feature():
    train = _make_frame(extra={"factor_a": [1.0, 2.0, 3.0]})
    val = _make_frame()
    validator = InputSchemaValidator(feature_columns=["factor_a"])
    schema = validator.validate(train)
    with pytest.raises(ValueError, match="factor_a"):
        validator.validate_against_schema(val, schema)


def test_schema_feature_source_written_to_json(tmp_path):
    feature_source = {
        "mode": "fixed_plus_factor_list",
        "pair": "AL",
        "profile": "short",
    }
    schema = InputSchemaValidator(
        feature_columns=["return_1m"],
        feature_source=feature_source,
    ).validate(_make_frame())
    path = InputSchemaValidator().write_schema_json(schema, tmp_path / "schema.json")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["feature_source"] == feature_source
