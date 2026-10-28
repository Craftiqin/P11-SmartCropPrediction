import math
from typing import Any

import numpy as np


FEATURE_NAMES = ("temperature", "humidity", "soil_moisture", "light", "pressure")


def sensor_reading_to_features(
    reading: dict[str, Any], feature_names: tuple[str, ...] = FEATURE_NAMES
) -> np.ndarray:
    """Convert one API-validated reading to exact training feature order."""
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or any(name not in FEATURE_NAMES for name in feature_names)
    ):
        raise ValueError("prediction feature order is invalid")
    values: list[float] = []
    for name in feature_names:
        value = reading.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} is unavailable or non-numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} is non-finite")
        values.append(number)
    return np.asarray([values], dtype=float)
