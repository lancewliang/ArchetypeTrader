import numpy as np
import polars as pl

from src.data.state_normalizer import StateNormalizer
from src.phase1.evaluators.phase1_validation_layers.layer4_label_predictability import (
    build_probe_features,
)


def test_build_probe_features_uses_previous_segment_and_current_t0() -> None:
    states = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )

    features = build_probe_features(states)

    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0],
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        ]
    )
    assert features.shape == (2, 8)
    np.testing.assert_allclose(features, expected)


def test_state_normalizer_fits_train_and_reuses_same_stats_for_validation() -> None:
    train = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 12.0, 14.0, 16.0],
        }
    )
    validation_values = np.asarray(
        [
            [2.0, 12.0],
            [4.0, 16.0],
        ],
        dtype=np.float32,
    )

    normalizer = StateNormalizer.fit(train, ["feature_a", "feature_b"])
    train_normalized = normalizer.transform(train.to_numpy())
    validation_normalized = normalizer.transform(validation_values)

    np.testing.assert_allclose(train_normalized.mean(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(train_normalized.std(axis=0), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        validation_normalized,
        (validation_values - normalizer.mean) / normalizer.std,
    )
