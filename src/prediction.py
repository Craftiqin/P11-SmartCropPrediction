from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib

from src.features import FEATURE_NAMES, sensor_reading_to_features


SIMILAR_PROBABILITY_TOLERANCE = 0.01
HIGH_CONFIDENCE_THRESHOLD = 0.60
MODERATE_CONFIDENCE_THRESHOLD = 0.35


def load_selected_model(model_dir: Path):
    final_metadata = model_dir / "final_model_metadata.json"
    metadata_path = (
        final_metadata if final_metadata.exists() else model_dir / "training_summary.json"
    )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    feature_names = tuple(
        metadata.get("feature_order", metadata.get("feature_names", ()))
    )
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or any(name not in FEATURE_NAMES for name in feature_names)
    ):
        raise ValueError("model feature order is invalid")
    model_file = metadata.get("model_file", metadata.get("selected_model_file"))
    model = joblib.load(model_dir / model_file)
    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError("model input width does not match feature metadata")
    return model, metadata


def expected_features(model: Any) -> tuple[str, ...]:
    return tuple(getattr(model, "prediction_features_", FEATURE_NAMES))


def predict_reading(
    model: Any,
    reading: dict[str, Any],
    feature_names: tuple[str, ...] | None = None,
) -> tuple[str, float | None]:
    selected_features = expected_features(model) if feature_names is None else feature_names
    features = sensor_reading_to_features(reading, selected_features)
    crop = str(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
    return crop, confidence


def ranked_probabilities(
    model: Any,
    reading: dict[str, Any],
    limit: int = 3,
    feature_names: tuple[str, ...] | None = None,
    preferred_crop: str | None = None,
) -> list[dict[str, float | str]]:
    if not hasattr(model, "predict_proba"):
        return []
    selected_features = expected_features(model) if feature_names is None else feature_names
    features = sensor_reading_to_features(reading, selected_features)
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    mapped = [
        (str(crop), float(probability))
        for crop, probability in zip(classes, probabilities, strict=True)
    ]
    if (
        not mapped
        or len({crop for crop, _ in mapped}) != len(mapped)
        or any(
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0
            for _, probability in mapped
        )
        or not math.isclose(
            sum(probability for _, probability in mapped),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("model returned invalid class probabilities")
    if preferred_crop is not None:
        preferred_probabilities = [
            probability for crop, probability in mapped if crop == preferred_crop
        ]
        if not preferred_probabilities or not math.isclose(
            preferred_probabilities[0],
            max(probability for _, probability in mapped),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("predict output does not match class probabilities")
    ranked = sorted(
        ((crop, probability) for crop, probability in mapped if probability > 0.0),
        key=lambda item: (
            -item[1],
            item[0] != preferred_crop if preferred_crop is not None else False,
            item[0],
        ),
    )
    display_limit = min(max(limit, 0), 3)
    return [
        {"crop": crop, "probability": probability}
        for crop, probability in ranked[:display_limit]
    ]


def probabilities_are_similar(
    probabilities: list[dict[str, float | str]],
    tolerance: float = SIMILAR_PROBABILITY_TOLERANCE,
) -> bool:
    if len(probabilities) < 2:
        return False
    first = float(probabilities[0]["probability"])
    second = float(probabilities[1]["probability"])
    return abs(first - second) <= tolerance


def probability_confidence_status(
    probabilities: list[dict[str, float | str]],
) -> tuple[str, str]:
    """Apply documented conservative UI thresholds to ranked probabilities."""
    if not probabilities:
        return "Unavailable", "Model does not provide class probabilities."
    if probabilities_are_similar(probabilities):
        return (
            "Low confidence",
            "Top crop probabilities are equal or within 1 percentage point.",
        )
    highest = float(probabilities[0]["probability"])
    if highest >= HIGH_CONFIDENCE_THRESHOLD:
        return "High confidence", "Highest probability is at least 60%."
    if highest >= MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate confidence", "Highest probability is between 35% and 60%."
    return "Low confidence", "Highest probability is below 35%."
