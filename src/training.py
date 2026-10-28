from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_NAMES


DATASET_SHA256 = "0ec550bdbe0dc0165ab59b35401dd1bec81fc80bb89a34097ee2a74b14ea02c1"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_FILENAMES = {
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "SVM": "svm.joblib",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if file_sha256(path) != DATASET_SHA256:
        raise ValueError("synthetic dataset SHA-256 does not match finalized Step 6 input")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = (*FEATURE_NAMES, "crop")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"dataset columns must be {expected}")
        rows = list(reader)
    if len(rows) != 1000:
        raise ValueError("synthetic dataset must contain exactly 1000 rows")
    try:
        features = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("all training features must be numeric") from exc
    labels = np.asarray([row["crop"].strip() for row in rows], dtype=str)
    if not np.isfinite(features).all() or any(not label for label in labels):
        raise ValueError("dataset contains invalid feature or target values")
    if len(set(labels)) != 52:
        raise ValueError("synthetic dataset must contain exactly 52 crop classes")
    return features, labels


def shared_split(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline(
            [("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        SVC(kernel="rbf", random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def _write_confusion_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *map(int, row)])


def _write_per_class(path: Path, report: dict[str, Any], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("crop", "precision", "recall", "f1", "support"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in labels:
            values = report[label]
            writer.writerow(
                {
                    "crop": label,
                    "precision": f"{values['precision']:.6f}",
                    "recall": f"{values['recall']:.6f}",
                    "f1": f"{values['f1-score']:.6f}",
                    "support": int(values["support"]),
                }
            )


def _write_comparison(path: Path, results: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "accuracy", "precision_macro", "recall_macro", "f1_macro"),
            lineterminator="\n",
        )
        writer.writeheader()
        for name, result in results.items():
            writer.writerow(
                {
                    "model": name,
                    "accuracy": f"{result['accuracy']:.6f}",
                    "precision_macro": f"{result['precision_macro']:.6f}",
                    "recall_macro": f"{result['recall_macro']:.6f}",
                    "f1_macro": f"{result['f1_macro']:.6f}",
                }
            )


def train_and_evaluate(dataset_path: Path, model_dir: Path) -> dict[str, Any]:
    features, labels_array = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = shared_split(features, labels_array)
    class_labels = sorted(map(str, set(labels_array)))
    evaluation_dir = model_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    trained: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        report = classification_report(
            y_test,
            predictions,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=class_labels)
        if name == "Decision Tree":
            influence_values = model.named_steps["classifier"].feature_importances_
            influence_method = "native_feature_importance"
        else:
            permutation = permutation_importance(
                model,
                x_test,
                y_test,
                scoring="f1_macro",
                n_repeats=20,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            influence_values = permutation.importances_mean
            influence_method = "held_out_permutation_importance_f1_macro"

        slug = name.lower().replace(" ", "_")
        _write_confusion_matrix(
            evaluation_dir / f"confusion_matrix_{slug}.csv", class_labels, matrix
        )
        _write_per_class(evaluation_dir / f"per_class_{slug}.csv", report, class_labels)
        result = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "influence_method": influence_method,
            "feature_influence": {
                feature: float(value)
                for feature, value in zip(FEATURE_NAMES, influence_values, strict=True)
            },
            "confusion_matrix_file": f"evaluation/confusion_matrix_{slug}.csv",
            "per_class_file": f"evaluation/per_class_{slug}.csv",
        }
        results[name] = result
        trained[name] = model
        joblib.dump(model, model_dir / MODEL_FILENAMES[name])

    best_name = max(
        results,
        key=lambda name: (results[name]["f1_macro"], results[name]["accuracy"]),
    )
    joblib.dump(trained[best_name], model_dir / "selected_model.joblib")
    _write_comparison(evaluation_dir / "model_comparison.csv", results)

    summary: dict[str, Any] = {
        "dataset": dataset_path.name,
        "dataset_sha256": DATASET_SHA256,
        "synthetic_data": True,
        "feature_names": list(FEATURE_NAMES),
        "target": "crop",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "classes": len(class_labels),
        "class_labels": class_labels,
        "full_class_distribution": dict(sorted(Counter(map(str, labels_array)).items())),
        "training_class_distribution": dict(sorted(Counter(map(str, y_train)).items())),
        "test_class_distribution": dict(sorted(Counter(map(str, y_test)).items())),
        "metric_averaging": "macro; each crop class has equal metric weight",
        "models": results,
        "selected_model": best_name,
        "selected_model_file": "selected_model.joblib",
        "real_dataset_evaluation": (
            "not performed: real dataset lacks light and pressure"
        ),
    }
    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> int:
    ml_root = Path(__file__).resolve().parents[1]
    summary = train_and_evaluate(
        ml_root / "data" / "synthetic" / "synthetic_crop_sensor_data.csv",
        ml_root / "models",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
