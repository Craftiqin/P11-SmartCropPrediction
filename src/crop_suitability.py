from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dashboard import is_stale_reading


SENSOR_WEIGHTS = {
    "temperature": 0.35,
    "humidity": 0.25,
    "soil_moisture": 0.25,
    "light": 0.15,
}
PROFILE_COLUMNS = {
    "temperature": ("temperature_min_c", "temperature_max_c"),
    "humidity": ("humidity_min_pct", "humidity_max_pct"),
    "soil_moisture": ("soil_moisture_min_pct", "soil_moisture_max_pct"),
    "light": ("light_min_lux", "light_max_lux"),
}


@dataclass(frozen=True)
class CropSuitabilityResult:
    crop: str
    score: float
    sensor_score: float | None
    context_quality: str
    reason: str


def load_crop_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        profiles = list(csv.DictReader(handle))
    required = {
        "crop",
        "category",
        "scientific_name",
        "ecocrop_rainfall_min_mm",
        "ecocrop_rainfall_max_mm",
        *(column for pair in PROFILE_COLUMNS.values() for column in pair),
    }
    if not profiles or not required.issubset(profiles[0]):
        raise ValueError("crop suitability profiles are incomplete")
    crops = [profile["crop"].strip() for profile in profiles]
    if len(profiles) != 52 or len(set(crops)) != 52 or any(not crop for crop in crops):
        raise ValueError("crop suitability profiles must contain 52 unique crops")
    for profile in profiles:
        for low_column, high_column in PROFILE_COLUMNS.values():
            low = float(profile[low_column])
            high = float(profile[high_column])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"invalid suitability range for {profile['crop']}")
        rainfall_low = float(profile["ecocrop_rainfall_min_mm"])
        rainfall_high = float(profile["ecocrop_rainfall_max_mm"])
        if (
            not math.isfinite(rainfall_low)
            or not math.isfinite(rainfall_high)
            or rainfall_low >= rainfall_high
        ):
            raise ValueError(f"invalid rainfall range for {profile['crop']}")
    return profiles


def range_compatibility(value: float, low: float, high: float) -> float:
    """Full match inside range; linear decay to zero over half-range outside it."""
    if not all(math.isfinite(number) for number in (value, low, high)) or low >= high:
        raise ValueError("invalid compatibility range")
    if low <= value <= high:
        return 1.0
    shoulder = (high - low) / 2
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / shoulder)


def _usable_number(reading: dict[str, Any], field: str) -> float | None:
    value = reading.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if field == "temperature" and not -20 <= number <= 60:
        return None
    if field in {"humidity", "soil_moisture"} and not 0 <= number <= 100:
        return None
    if field == "soil_moisture" and number == 0:
        return None
    if field == "light" and not 0 <= number <= 200_000:
        return None
    return number


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float | None:
    available_weight = sum(weights[name] for name in scores)
    if available_weight == 0:
        return None
    return sum(scores[name] * weights[name] for name in scores) / available_weight


def score_crop(
    reading: dict[str, Any],
    profile: dict[str, str],
) -> CropSuitabilityResult:
    sensor_components: dict[str, float] = {}
    for field in SENSOR_WEIGHTS:
        value = _usable_number(reading, field)
        if value is None:
            continue
        low_column, high_column = PROFILE_COLUMNS[field]
        sensor_components[field] = range_compatibility(
            value, float(profile[low_column]), float(profile[high_column])
        )
    sensor_score = _weighted_average(sensor_components, SENSOR_WEIGHTS)
    if sensor_score is None:
        raise ValueError("no usable sensor inputs")

    sensor_coverage = len(sensor_components) / len(SENSOR_WEIGHTS)
    if sensor_coverage == 1:
        context_quality = "Full context"
    elif sensor_coverage >= 0.75:
        context_quality = "Partial context"
    else:
        context_quality = "Limited context"

    labels = {
        "temperature": "temperature match",
        "humidity": "humidity match",
        "soil_moisture": "soil-moisture match",
        "light": "light match",
    }
    factors = [
        labels[field]
        for field, value in sorted(
            sensor_components.items(), key=lambda item: (-item[1], item[0])
        )
        if value >= 0.75
    ]
    reason = ", ".join(factors[:3]) or "closest available environmental match"
    return CropSuitabilityResult(
        crop=profile["crop"].strip(),
        score=round(max(0.0, min(1.0, sensor_score)), 6),
        sensor_score=round(sensor_score, 6) if sensor_score is not None else None,
        context_quality=context_quality,
        reason=reason,
    )


def rank_crop_suitability(
    reading: dict[str, Any],
    profiles: list[dict[str, str]],
    *,
    limit: int = 3,
) -> list[CropSuitabilityResult]:
    ranked = sorted(
        (score_crop(reading, profile) for profile in profiles),
        key=lambda result: (-result.score, result.crop),
    )
    return ranked[: min(max(limit, 0), 3)]


def input_quality_warnings(
    reading: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if reading.get("soil_moisture") == 0:
        warnings.append(
            "Soil moisture is currently 0%. Verify sensor placement and calibration "
            "before relying on this result."
        )
    for field, label in (
        ("temperature", "Temperature"),
        ("humidity", "Humidity"),
        ("soil_moisture", "Soil moisture"),
        ("light", "Light"),
    ):
        if _usable_number(reading, field) is None and not (
            field == "soil_moisture" and reading.get(field) == 0
        ):
            warnings.append(f"{label} reading is missing or outside supported bounds.")
    pressure = reading.get("pressure")
    if isinstance(pressure, (int, float)) and not isinstance(pressure, bool):
        pressure_number = float(pressure)
        if not math.isfinite(pressure_number):
            warnings.append("Pressure reading is unavailable.")
        elif not 930 <= pressure_number <= 1050:
            warnings.append(
                "Pressure reading is unusual; verify sensor calibration."
            )
    else:
        warnings.append("Pressure reading is unavailable.")
    if is_stale_reading(reading.get("timestamp")) is not False:
        warnings.append("Sensor reading is stale or its timestamp cannot be verified.")
    return warnings
