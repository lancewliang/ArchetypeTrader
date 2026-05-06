from types import SimpleNamespace

import numpy as np
import pytest

from src.phase1.data.state_normalizer import StateNormalizer


def test_state_normalizer_scales_large_magnitude_features():
    records = [
        SimpleNamespace(
            states=[
                [100.0, 1.0e10, 500_000.0],
                [101.0, 2.0e10, 510_000.0],
            ]
        ),
        SimpleNamespace(
            states=[
                [99.0, 3.0e10, 490_000.0],
                [102.0, 4.0e10, 520_000.0],
            ]
        ),
    ]
    normalizer = StateNormalizer.fit_records(
        records,
        feature_columns=["ask1_price", "turnover", "open_interest"],
    )

    diag = normalizer.transform_records(records)

    assert diag["max_abs_before"] > 1.0e10
    assert diag["max_abs_after"] <= normalizer.stats.clip_value
    assert normalizer.stats.transform_kinds == [
        "identity",
        "signed_log1p",
        "signed_log1p",
    ]
    assert np.asarray(records[0].states).shape == (2, 3)
    assert np.isfinite(np.asarray(records[0].states)).all()


def test_state_normalizer_roundtrip_dict():
    matrix = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    normalizer = StateNormalizer.fit_matrix(
        matrix,
        feature_columns=["price", "volume"],
    )

    restored = StateNormalizer.from_dict(normalizer.to_dict())

    np.testing.assert_allclose(
        normalizer.transform_array(matrix),
        restored.transform_array(matrix),
    )


def test_state_normalizer_rejects_feature_dim_mismatch():
    normalizer = StateNormalizer.fit_matrix(
        np.array([[1.0, 2.0], [2.0, 3.0]]),
        feature_columns=["a", "b"],
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        normalizer.transform_array(np.array([[1.0]]))
