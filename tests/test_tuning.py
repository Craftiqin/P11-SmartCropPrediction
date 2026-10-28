import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURE_NAMES
from src.training import load_dataset, shared_split
from src.tuning import FEATURE_SETS, feature_indices, tuning_specs


ML_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ML_ROOT / "models"
EVALUATION_DIR = MODEL_DIR / "evaluation"
DATASET = ML_ROOT / "data" / "synthetic" / "synthetic_crop_sensor_data.csv"


def read_rows(filename: str):
    with (EVALUATION_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exact_hyperparameter_search_space_and_fold_policy():
    specs = tuning_specs()
    assert set(specs) == {"Decision Tree", "KNN", "SVM"}
    assert len(specs["Decision Tree"][1]["classifier__max_depth"]) == 5
    assert len(specs["Decision Tree"][1]["classifier__min_samples_leaf"]) == 3
    assert len(specs["KNN"][1]["classifier__n_neighbors"]) == 5
    assert len(specs["KNN"][1]["classifier__weights"]) == 2
    assert len(specs["KNN"][1]["classifier__p"]) == 2
    assert len(specs["SVM"][1]["classifier__estimator__C"]) == 3
    assert len(specs["SVM"][1]["classifier__estimator__kernel"]) == 2
    assert len(specs["SVM"][1]["classifier__estimator__gamma"]) == 2
    assert "scaler" not in specs["Decision Tree"][0].named_steps
    assert "scaler" in specs["KNN"][0].named_steps
    assert "scaler" in specs["SVM"][0].named_steps

    summary = json.loads((MODEL_DIR / "tuning_summary.json").read_text())
    assert summary["cross_validation"] == {
        "folds": 3,
        "type": "StratifiedKFold",
        "shuffle": True,
        "random_state": 42,
        "selection_metric": "macro F1",
        "scope": "training set only",
    }
    assert isinstance(StratifiedKFold(3), StratifiedKFold)


def test_feature_sets_reuse_shared_rows_and_preserve_order():
    assert FEATURE_SETS == {
        "All features": FEATURE_NAMES,
        "Without pressure": FEATURE_NAMES[:-1],
        "Core three features": FEATURE_NAMES[:3],
        "Temperature and humidity only": FEATURE_NAMES[:2],
    }
    features, labels = load_dataset(DATASET)
    x_train, x_test, y_train, y_test = shared_split(features, labels)
    for names in FEATURE_SETS.values():
        indices = feature_indices(names)
        assert np.array_equal(x_train[:, indices], x_train[:, : len(names)])
        assert np.array_equal(x_test[:, indices], x_test[:, : len(names)])
        assert len(y_train) == 800
        assert len(y_test) == 200


def test_search_and_ablation_result_files():
    hyperparameters = read_rows("hyperparameter_results.csv")
    ablation = read_rows("feature_ablation_results.csv")
    comparison = read_rows("tuned_model_comparison.csv")
    assert len(hyperparameters) == 15 + 20 + 12
    assert len(ablation) == 3 * 4
    assert len(comparison) == 3 * 2
    assert {row["baseline_or_tuned"] for row in comparison} == {"baseline", "tuned"}
    assert {row["feature_set"] for row in ablation} == set(FEATURE_SETS)
    for row in hyperparameters + ablation + comparison:
        for metric in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ):
            assert np.isfinite(float(row[metric]))


def test_training_only_selection_and_saved_tuned_pipelines():
    summary = json.loads((MODEL_DIR / "tuning_summary.json").read_text())
    cv_scores = {
        name: details["best_cv_macro_f1"]
        for name, details in summary["tuned_models"].items()
    }
    assert summary["cv_selected_tuned_model"] == max(cv_scores, key=cv_scores.get)
    assert "Ablation test scores are descriptive" in summary["selection_rule"]

    features, _ = load_dataset(DATASET)
    for slug in ("decision_tree", "knn", "svm"):
        pipeline = joblib.load(MODEL_DIR / f"tuned_{slug}.joblib")
        assert len(pipeline.predict(features[:2])) == 2
    svm = joblib.load(MODEL_DIR / "tuned_svm.joblib")
    assert svm.predict_proba(features[:1]).shape == (1, 52)

