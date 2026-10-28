import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.features import FEATURE_NAMES
from src.training import build_models, load_dataset, shared_split


ML_ROOT = Path(__file__).resolve().parents[1]
DATASET = ML_ROOT / "data" / "synthetic" / "synthetic_crop_sensor_data.csv"
MODEL_DIR = ML_ROOT / "models"


def test_dataset_load_and_shared_stratified_split():
    features, labels = load_dataset(DATASET)
    assert features.shape == (1000, 5)
    assert labels.shape == (1000,)
    assert np.isfinite(features).all()

    first = shared_split(features, labels)
    second = shared_split(features, labels)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(first, second, strict=True)
    )
    x_train, x_test, y_train, y_test = first
    assert x_train.shape == (800, 5)
    assert x_test.shape == (200, 5)
    assert set(y_train) == set(y_test) == set(labels)
    assert set(Counter(y_test).values()) <= {3, 4}


def test_required_pipeline_preprocessing():
    models = build_models()
    assert set(models) == {"Decision Tree", "KNN", "SVM"}
    assert "scaler" not in models["Decision Tree"].named_steps
    assert "scaler" in models["KNN"].named_steps
    assert "scaler" in models["SVM"].named_steps


def test_serialized_models_and_reports_exist_and_predict():
    features, _ = load_dataset(DATASET)
    filenames = (
        "decision_tree.joblib",
        "knn.joblib",
        "svm.joblib",
    )
    for filename in filenames:
        model = joblib.load(MODEL_DIR / filename)
        assert len(model.predict(features[:2])) == 2

    selected_metadata = json.loads(
        (MODEL_DIR / "final_model_metadata.json").read_text()
    )
    selected = joblib.load(MODEL_DIR / "selected_model.joblib")
    indices = [FEATURE_NAMES.index(name) for name in selected_metadata["feature_order"]]
    assert len(selected.predict(features[:2, indices])) == 2

    metadata = json.loads((MODEL_DIR / "training_summary.json").read_text())
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["training_samples"] == 800
    assert metadata["test_samples"] == 200
    assert metadata["classes"] == 52
    assert metadata["real_dataset_evaluation"].startswith("not performed")

    for slug in ("decision_tree", "knn", "svm"):
        matrix_path = MODEL_DIR / "evaluation" / f"confusion_matrix_{slug}.csv"
        with matrix_path.open() as handle:
            assert len(list(csv.reader(handle))) == 53
        with (MODEL_DIR / "evaluation" / f"per_class_{slug}.csv").open() as handle:
            assert len(list(csv.reader(handle))) == 53
