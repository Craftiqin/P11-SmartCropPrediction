from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.crop_suitability import (
    input_quality_warnings,
    load_crop_profiles,
    range_compatibility,
    rank_crop_suitability,
    score_crop,
)
ML_ROOT = Path(__file__).resolve().parents[1]


def profile(crop: str, *, temperature=(20, 30), humidity=(50, 80)):
    return {
        "crop": crop,
        "temperature_min_c": str(temperature[0]),
        "temperature_max_c": str(temperature[1]),
        "humidity_min_pct": str(humidity[0]),
        "humidity_max_pct": str(humidity[1]),
        "soil_moisture_min_pct": "40",
        "soil_moisture_max_pct": "70",
        "light_min_lux": "10000",
        "light_max_lux": "50000",
    }


def test_existing_crop_profiles_load_for_all_supported_crops():
    profiles = load_crop_profiles(
        ML_ROOT / "data" / "synthetic" / "crop_profiles.csv"
    )
    assert len(profiles) == 52
    assert len({item["crop"] for item in profiles}) == 52


def test_range_compatibility_is_normalized_and_deterministic():
    assert range_compatibility(25, 20, 30) == 1
    assert range_compatibility(17.5, 20, 30) == 0.5
    assert range_compatibility(15, 20, 30) == 0
    with pytest.raises(ValueError):
        range_compatibility(20, 30, 20)


def test_sensor_score_uses_documented_weights(reading):
    result = score_crop(reading, profile("matched"))
    assert result.sensor_score is not None
    assert result.score == pytest.approx(result.sensor_score)
    assert 0 <= result.score <= 1
    assert result.context_quality == "Full context"


def test_top_three_sorting_and_final_crop_selection(reading):
    profiles = [
        profile("cold", temperature=(5, 10), humidity=(10, 20)),
        profile("best", temperature=(25, 30), humidity=(60, 70)),
        profile("warm", temperature=(28, 35), humidity=(50, 65)),
        profile("hot", temperature=(40, 50), humidity=(10, 30)),
    ]
    ranked = rank_crop_suitability(reading, profiles, limit=52)
    assert len(ranked) == 3
    assert ranked == sorted(ranked, key=lambda item: (-item.score, item.crop))
    assert ranked[0].crop == "best"


def test_missing_sensor_values_are_not_fabricated(reading):
    reading["soil_moisture"] = None
    reading["light"] = None
    result = score_crop(reading, profile("crop"))
    assert result.score == pytest.approx(result.sensor_score)
    assert result.context_quality == "Limited context"
    assert not hasattr(result, "probability")


def test_zero_soil_and_stale_warnings(reading):
    reading["soil_moisture"] = 0
    reading["timestamp"] = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    warnings = input_quality_warnings(reading)
    assert any("currently 0%" in warning for warning in warnings)
    assert any("stale" in warning for warning in warnings)
