"""Run finalized crop prediction locally without API or database access."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from src.crop_suitability import load_crop_profiles, rank_crop_suitability
from src.prediction import (
    load_selected_model,
    predict_reading,
    probability_confidence_status,
    ranked_probabilities,
)

ML_ROOT = Path(__file__).resolve().parent
MODEL_DIR = ML_ROOT / "models"
PROFILE_PATH = ML_ROOT / "data" / "synthetic" / "crop_profiles.csv"


def _finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a crop with the saved Smart Crop KNN model."
    )
    parser.add_argument("--temperature", required=True, type=_finite_number, help="degrees Celsius")
    parser.add_argument("--humidity", required=True, type=_finite_number, help="relative humidity, percent")
    parser.add_argument(
        "--soil-moisture", required=True, type=_finite_number,
        help="calibrated relative sensor percentage; not volumetric water content",
    )
    parser.add_argument("--light", required=True, type=_finite_number, help="illuminance, lux")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reading = {
        "temperature": args.temperature,
        "humidity": args.humidity,
        "soil_moisture": args.soil_moisture,
        "light": args.light,
    }
    try:
        model, metadata = load_selected_model(MODEL_DIR)
        features = tuple(metadata["feature_order"])
        crop, _ = predict_reading(model, reading, features)
        probabilities = ranked_probabilities(
            model, reading, limit=3, feature_names=features, preferred_crop=crop
        )
        profiles = load_crop_profiles(PROFILE_PATH)
        suitability = rank_crop_suitability(reading, profiles, limit=3)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Prediction unavailable: {exc}")
        return 1

    status, explanation = probability_confidence_status(probabilities)
    print(f"Selected model: {metadata['selected_model']} ({metadata['configuration']})")
    print(f"Feature order: {', '.join(features)}")
    print(f"Predicted crop: {crop}")
    print(f"KNN vote diagnostic: {status} — {explanation}")
    print("Top KNN vote fractions (not calibrated probabilities):")
    for item in probabilities:
        print(f"  {item['crop']}: {item['probability']:.1%}")
    print("Separate profile-based suitability (not model probability):")
    for item in suitability:
        print(f"  {item.crop}: {item.score:.1%} ({item.context_quality})")
    print("Note: model uses synthetic training data; output is not field-validated advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
