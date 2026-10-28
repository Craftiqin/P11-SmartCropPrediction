import numpy as np
import pytest

from src.features import FEATURE_NAMES, sensor_reading_to_features


def test_live_feature_order(reading):
    values = sensor_reading_to_features(reading)
    assert FEATURE_NAMES == (
        "temperature",
        "humidity",
        "soil_moisture",
        "light",
        "pressure",
    )
    np.testing.assert_array_equal(values, [[28.4, 64.2, 52.0, 423.5, 1008.4]])


@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_missing_or_off_sensor_blocks_prediction(reading, feature):
    reading[feature] = None
    with pytest.raises(ValueError, match=feature):
        sensor_reading_to_features(reading)


def test_selected_subset_preserves_order_and_ignores_unselected_pressure(reading):
    features = ("temperature", "humidity", "soil_moisture", "light")
    reading["pressure"] = None
    result = sensor_reading_to_features(reading, features)
    assert result.tolist() == [[28.4, 64.2, 52.0, 423.5]]
