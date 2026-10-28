from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_NAMES
from src.training import RANDOM_STATE, build_models, load_dataset, shared_split


FEATURE_SETS = OrderedDict(
    {
        "All features": tuple(FEATURE_NAMES),
        "Without pressure": tuple(FEATURE_NAMES[:-1]),
        "Core three features": tuple(FEATURE_NAMES[:3]),
        "Temperature and humidity only": tuple(FEATURE_NAMES[:2]),
    }
)

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(
        precision_score, average="macro", zero_division=0
    ),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
}

RESULT_FIELDS = (
    "model",
    "feature_set",
    "baseline_or_tuned",
    "hyperparameters",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
)


def tuning_specs() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "Decision Tree": (
            Pipeline(
                [("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]
            ),
            {
                "classifier__max_depth": [None, 5, 8, 12, 16],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "KNN": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        CalibratedClassifierCV(
                            SVC(random_state=RANDOM_STATE),
                            method="sigmoid",
                            cv=3,
                            ensemble=False,
                        ),
                    ),
                ]
            ),
            {
                "classifier__estimator__C": [0.1, 1, 10],
                "classifier__estimator__kernel": ["rbf", "linear"],
                "classifier__estimator__gamma": ["scale", "auto"],
            },
        ),
    }


def feature_indices(feature_names: tuple[str, ...]) -> list[int]:
    return [FEATURE_NAMES.index(name) for name in feature_names]


def evaluate(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def run_search(
    pipeline: Pipeline,
    parameter_grid: dict[str, list[Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring=SCORING,
        refit="f1_macro",
        cv=folds,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    return search


def clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("classifier__").removeprefix("estimator__"): value
        for key, value in sorted(parameters.items())
    }


def result_row(
    model: str,
    feature_set: str,
    status: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model,
        "feature_set": feature_set,
        "baseline_or_tuned": status,
        "hyperparameters": json.dumps(clean_parameters(parameters), sort_keys=True),
        **metrics,
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tune_and_ablate(dataset_path: Path, model_dir: Path) -> dict[str, Any]:
    features, labels = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = shared_split(features, labels)
    evaluation_dir = model_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    hyperparameter_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    tuned_models: dict[str, Pipeline] = {}
    tuned_details: dict[str, dict[str, Any]] = {}

    baseline_models = build_models()
    baseline_metrics: dict[str, dict[str, float]] = {}
    for name, model in baseline_models.items():
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_test, y_test)
        baseline_metrics[name] = metrics
        comparison_rows.append(result_row(name, "All features", "baseline", {}, metrics))

    for name, (pipeline, grid) in tuning_specs().items():
        search = run_search(pipeline, grid, x_train, y_train)
        cv_results = search.cv_results_
        for index, parameters in enumerate(cv_results["params"]):
            metrics = {
                metric: float(cv_results[f"mean_test_{metric}"][index])
                for metric in SCORING
            }
            hyperparameter_rows.append(
                result_row(
                    name,
                    "All features",
                    "tuned",
                    parameters,
                    metrics,
                )
            )

        best_model = search.best_estimator_
        tuned_models[name] = best_model
        joblib.dump(
            best_model,
            model_dir / f"tuned_{name.lower().replace(' ', '_')}.joblib",
        )
        test_metrics = evaluate(best_model, x_test, y_test)
        comparison_rows.append(
            result_row(name, "All features", "tuned", search.best_params_, test_metrics)
        )
        tuned_details[name] = {
            "best_parameters": clean_parameters(search.best_params_),
            "best_cv_macro_f1": float(search.best_score_),
            "held_out_metrics": test_metrics,
        }

        for feature_set_name, names in FEATURE_SETS.items():
            indices = feature_indices(names)
            ablation_model = clone(best_model)
            ablation_model.fit(x_train[:, indices], y_train)
            metrics = evaluate(ablation_model, x_test[:, indices], y_test)
            ablation_rows.append(
                result_row(
                    name,
                    feature_set_name,
                    "tuned",
                    search.best_params_,
                    metrics,
                )
            )

    write_results(evaluation_dir / "hyperparameter_results.csv", hyperparameter_rows)
    write_results(evaluation_dir / "feature_ablation_results.csv", ablation_rows)
    write_results(evaluation_dir / "tuned_model_comparison.csv", comparison_rows)

    cv_selected_name = max(
        tuned_details,
        key=lambda name: tuned_details[name]["best_cv_macro_f1"],
    )
    baseline_selected_name = max(
        baseline_metrics,
        key=lambda name: baseline_metrics[name]["f1_macro"],
    )
    tuned_test_f1 = tuned_details[cv_selected_name]["held_out_metrics"]["f1_macro"]
    baseline_test_f1 = baseline_metrics[baseline_selected_name]["f1_macro"]
    tuned_selected = tuned_test_f1 > baseline_test_f1

    training_summary_path = model_dir / "training_summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if not (model_dir / "final_model_metadata.json").exists():
        selected_name = cv_selected_name if tuned_selected else baseline_selected_name
        selected_model = (
            tuned_models[cv_selected_name]
            if tuned_selected
            else baseline_models[baseline_selected_name]
        )
        selected_parameters = (
            tuned_details[cv_selected_name]["best_parameters"]
            if tuned_selected
            else {}
        )
        selected_test_f1 = tuned_test_f1 if tuned_selected else baseline_test_f1
        joblib.dump(selected_model, model_dir / "selected_model.joblib")
        training_summary.update(
            {
                "selected_model": selected_name,
                "selected_model_status": "tuned" if tuned_selected else "baseline",
                "selected_model_feature_set": "All features",
                "selected_model_parameters": selected_parameters,
                "selected_model_test_macro_f1": selected_test_f1,
            }
        )
        training_summary_path.write_text(
            json.dumps(training_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    pressure_effect: dict[str, float] = {}
    for name in tuned_details:
        rows = [row for row in ablation_rows if row["model"] == name]
        all_f1 = next(row["f1_macro"] for row in rows if row["feature_set"] == "All features")
        no_pressure_f1 = next(
            row["f1_macro"] for row in rows if row["feature_set"] == "Without pressure"
        )
        pressure_effect[name] = float(all_f1 - no_pressure_f1)

    summary: dict[str, Any] = {
        "experiment_version": "step-7-tuning-ablation-v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        "records": 1000,
        "classes": 52,
        "samples_per_class": training_summary["full_class_distribution"],
        "available_features": list(FEATURE_NAMES),
        "feature_units": {
            "temperature": "degrees Celsius",
            "humidity": "percent relative humidity",
            "soil_moisture": "calibrated relative percent",
            "light": "lux",
            "pressure": "hPa",
        },
        "split": {
            "random_state": RANDOM_STATE,
            "training_samples": len(x_train),
            "test_samples": len(x_test),
            "stratified": True,
            "training_feature_fingerprint": split_fingerprint(x_train),
            "test_feature_fingerprint": split_fingerprint(x_test),
        },
        "cross_validation": {
            "folds": 3,
            "type": "StratifiedKFold",
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "selection_metric": "macro F1",
            "scope": "training set only",
        },
        "tuned_models": tuned_details,
        "cv_selected_tuned_model": cv_selected_name,
        "baseline_selected_model": baseline_selected_name,
        "baseline_selected_test_macro_f1": baseline_test_f1,
        "cv_selected_tuned_test_macro_f1": tuned_test_f1,
        "final_selected_status": "tuned" if tuned_selected else "baseline",
        "final_selected_model": cv_selected_name if tuned_selected else baseline_selected_name,
        "selection_rule": (
            "Tuned model chosen by training-only CV macro F1; held-out score used once "
            "as replacement gate against preserved baseline. Ablation test scores are "
            "descriptive and do not select the deployed feature set."
        ),
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "pressure_delta_macro_f1_all_minus_without": pressure_effect,
        "pressure_conclusion": (
            "Pressure was generated independently of crop. Held-out ablation deltas "
            "must be treated as sampling variation, not agronomic evidence."
        ),
        "data_limitations": {
            "environmental_range_overlap": (
                "Crop profiles overlap substantially across temperature, humidity, "
                "soil moisture, and light; pressure uses a shared crop-independent pool."
            ),
            "class_difficulty": (
                "Fifty-two classes share only five broad environmental measurements. "
                "Many crops therefore occupy similar feature regions."
            ),
            "more_samples": (
                "More samples from the same synthetic distributions would reduce sampling "
                "noise but would not add missing discriminative field variables or prove validity."
            ),
            "field_validity": (
                "Synthetic held-out performance measures recovery of generator assumptions, "
                "not real-world agricultural accuracy or field validation."
            ),
        },
    }
    (model_dir / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    ml_root = Path(__file__).resolve().parents[1]
    summary = tune_and_ablate(
        ml_root / "data" / "synthetic" / "synthetic_crop_sensor_data.csv",
        ml_root / "models",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
