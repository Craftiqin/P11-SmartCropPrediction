from pathlib import Path

import numpy as np
import pytest

from src.prediction import (
    load_selected_model,
    predict_reading,
    probability_confidence_status,
    probabilities_are_similar,
    ranked_probabilities,
)


ML_ROOT = Path(__file__).resolve().parents[1]


def test_selected_model_loads_and_predicts_with_confidence(reading):
    model, metadata = load_selected_model(ML_ROOT / "models")
    crop, confidence = predict_reading(model, reading)
    assert crop in metadata["class_labels"]
    assert confidence is not None
    assert 0 <= confidence <= 1


class PredictableProbabilityModel:
    classes_ = np.asarray(["apple", "cardamom", "cowpea", "sunflower"])

    def predict(self, _features):
        return np.asarray(["cardamom"])

    def predict_proba(self, _features):
        return np.asarray([[0.1, 0.3, 0.3, 0.3]])


class PredictOnlyModel:
    def predict(self, _features):
        return np.asarray(["cardamom"])


class MismatchedProbabilityModel(PredictableProbabilityModel):
    def predict(self, _features):
        return np.asarray(["apple"])


class InvalidProbabilityModel(PredictableProbabilityModel):
    def predict_proba(self, _features):
        return np.asarray([[0.1, 0.3, 0.3, 0.2]])


class SparseProbabilityModel(PredictableProbabilityModel):
    def predict(self, _features):
        return np.asarray(["apple"])

    def predict_proba(self, _features):
        return np.asarray([[0.6, 0.4, 0.0, 0.0]])


def test_prediction_and_probability_mapping_sorting_and_top_three(reading):
    model = PredictableProbabilityModel()
    crop, confidence = predict_reading(model, reading)
    ranked = ranked_probabilities(
        model, reading, limit=52, preferred_crop=crop
    )

    assert crop == "cardamom"
    assert confidence == 0.3
    assert ranked == [
        {"crop": "cardamom", "probability": 0.3},
        {"crop": "cowpea", "probability": 0.3},
        {"crop": "sunflower", "probability": 0.3},
    ]
    assert len(ranked) == 3


def test_probability_similarity_detection():
    assert probabilities_are_similar(
        [
            {"crop": "cardamom", "probability": 0.30},
            {"crop": "cowpea", "probability": 0.30},
        ]
    )
    assert probabilities_are_similar(
        [
            {"crop": "cardamom", "probability": 0.30},
            {"crop": "cowpea", "probability": 0.295},
        ]
    )
    assert not probabilities_are_similar(
        [
            {"crop": "cardamom", "probability": 0.30},
            {"crop": "cowpea", "probability": 0.20},
        ]
    )


def test_documented_confidence_statuses_and_close_probability_handling():
    assert probability_confidence_status([])[0] == "Unavailable"
    assert probability_confidence_status(
        [{"crop": "a", "probability": 0.7}]
    )[0] == "High confidence"
    assert probability_confidence_status(
        [{"crop": "a", "probability": 0.4}]
    )[0] == "Moderate confidence"
    assert probability_confidence_status(
        [{"crop": "a", "probability": 0.3}]
    )[0] == "Low confidence"
    assert probability_confidence_status(
        [
            {"crop": "a", "probability": 0.61},
            {"crop": "b", "probability": 0.605},
        ]
    )[0] == "Low confidence"


def test_prediction_without_predict_proba_has_no_probability_rows(reading):
    model = PredictOnlyModel()
    assert predict_reading(model, reading) == ("cardamom", None)
    assert ranked_probabilities(model, reading) == []


def test_prediction_probability_mismatch_is_not_hidden(reading):
    model = MismatchedProbabilityModel()
    crop, _ = predict_reading(model, reading)
    with pytest.raises(
        ValueError, match="predict output does not match class probabilities"
    ):
        ranked_probabilities(model, reading, preferred_crop=crop)


def test_invalid_probability_sum_is_rejected(reading):
    with pytest.raises(ValueError, match="invalid class probabilities"):
        ranked_probabilities(InvalidProbabilityModel(), reading)


def test_zero_probability_classes_are_not_presented_as_candidates(reading):
    ranked = ranked_probabilities(
        SparseProbabilityModel(), reading, preferred_crop="apple"
    )
    assert ranked == [
        {"crop": "apple", "probability": 0.6},
        {"crop": "cardamom", "probability": 0.4},
    ]
