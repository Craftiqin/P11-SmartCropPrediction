import csv
import json
from pathlib import Path

import joblib
import pytest

from src.prediction import load_selected_model, predict_reading
from src.tuning import FEATURE_SETS


ML_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ML_ROOT / "models"


def test_training_only_feature_selection_results_and_metadata():
    with (MODEL_DIR / "final_feature_selection_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2 * 4
    assert {row["feature_set"] for row in rows} == set(FEATURE_SETS)
    selected = max(rows, key=lambda row: float(row["cv_f1_macro"]))

    metadata = json.loads((MODEL_DIR / "final_model_metadata.json").read_text())
    assert metadata["selected_model"] == selected["model"]
    assert metadata["configuration"] == selected["configuration"]
    assert metadata["selected_feature_set"] == selected["feature_set"]
    assert metadata["feature_order"] == json.loads(selected["feature_order"])
    assert metadata["cross_validation"]["selection_scope"] == "training split only"
    assert metadata["cross_validation"]["n_splits"] == 3
    assert metadata["pressure_used_for_prediction"] == (
        "pressure" in metadata["feature_order"]
    )


def test_final_pipeline_and_selected_alias_use_metadata_feature_order(reading):
    model, metadata = load_selected_model(MODEL_DIR)
    assert model.n_features_in_ == len(metadata["feature_order"])
    assert tuple(model.prediction_features_) == tuple(metadata["feature_order"])
    crop, confidence = predict_reading(model, reading)
    assert crop in metadata["class_labels"]
    assert confidence is not None

    alias = joblib.load(MODEL_DIR / "selected_model.joblib")
    assert alias.predict([[28.4, 64.2, 52.0, 423.5]])[0] == crop


def test_pressure_can_be_missing_but_required_feature_cannot(reading):
    model, metadata = load_selected_model(MODEL_DIR)
    assert "pressure" not in metadata["feature_order"]
    reading["pressure"] = None
    predict_reading(model, reading)
    reading["light"] = None
    with pytest.raises(ValueError, match="light"):
        predict_reading(model, reading)


def test_final_held_out_evaluation_contract():
    evaluation = json.loads((MODEL_DIR / "final_model_evaluation.json").read_text())
    assert evaluation["selection_basis"] == "training-only cross-validation macro F1"
    assert evaluation["held_out_test_usage"].startswith("Evaluated once after")
    assert set(evaluation["held_out_test_metrics"]) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    }
