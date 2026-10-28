from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features import FEATURE_NAMES
from src.training import RANDOM_STATE, build_models, load_dataset, shared_split
from src.tuning import FEATURE_SETS, SCORING, evaluate, feature_indices


SELECTION_FIELDS = (
    "model",
    "configuration",
    "feature_set",
    "feature_order",
    "hyperparameters",
    "cv_accuracy",
    "cv_precision_macro",
    "cv_recall_macro",
    "cv_f1_macro",
    "cv_f1_weighted",
    "cv_f1_macro_std",
)

BASELINE_PARAMETERS = {
    "Decision Tree": {"max_depth": None, "min_samples_leaf": 1},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "p": 2},
    "SVM": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
}


def load_candidates(model_dir: Path) -> list[dict[str, Any]]:
    tuning_summary = json.loads(
        (model_dir / "tuning_summary.json").read_text(encoding="utf-8")
    )
    candidates: list[dict[str, Any]] = []
    for name, pipeline in build_models().items():
        candidates.append(
            {
                "model": name,
                "configuration": "baseline",
                "pipeline": pipeline,
                "hyperparameters": BASELINE_PARAMETERS[name],
            }
        )
    for name in ("Decision Tree", "KNN", "SVM"):
        slug = name.lower().replace(" ", "_")
        candidates.append(
            {
                "model": name,
                "configuration": "tuned",
                "pipeline": joblib.load(model_dir / f"tuned_{slug}.joblib"),
                "hyperparameters": tuning_summary["tuned_models"][name][
                    "best_parameters"
                ],
            }
        )
    return candidates


def cross_validate_candidate(
    pipeline: Any,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        clone(pipeline),
        features,
        labels,
        cv=folds,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        metric: float(np.mean(scores[f"test_{metric}"])) for metric in SCORING
    } | {"f1_macro_std": float(np.std(scores["test_f1_macro"]))}


def select_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["cv_f1_macro"],
            -len(row["feature_order"]),
            row["configuration"] == "baseline",
            row["model"],
        ),
    )


def write_selection_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SELECTION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "feature_order": json.dumps(row["feature_order"]),
                    "hyperparameters": json.dumps(
                        row["hyperparameters"], sort_keys=True
                    ),
                }
            )


def finalize(dataset_path: Path, model_dir: Path) -> dict[str, Any]:
    features, labels = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = shared_split(features, labels)
    candidates = load_candidates(model_dir)

    rows: list[dict[str, Any]] = []
    candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["model"], candidate["configuration"])
        candidate_lookup[key] = candidate
        for feature_set, feature_order in FEATURE_SETS.items():
            indices = feature_indices(feature_order)
            metrics = cross_validate_candidate(
                candidate["pipeline"], x_train[:, indices], y_train
            )
            rows.append(
                {
                    "model": candidate["model"],
                    "configuration": candidate["configuration"],
                    "feature_set": feature_set,
                    "feature_order": list(feature_order),
                    "hyperparameters": candidate["hyperparameters"],
                    **{f"cv_{name}": value for name, value in metrics.items()},
                }
            )

    selected = select_configuration(rows)
    selected_features = tuple(selected["feature_order"])
    selected_indices = feature_indices(selected_features)
    selected_candidate = candidate_lookup[
        (selected["model"], selected["configuration"])
    ]
    final_pipeline = clone(selected_candidate["pipeline"])
    final_pipeline.fit(x_train[:, selected_indices], y_train)
    final_pipeline.prediction_features_ = selected_features
    held_out = evaluate(final_pipeline, x_test[:, selected_indices], y_test)

    joblib.dump(final_pipeline, model_dir / "final_model.joblib")
    joblib.dump(final_pipeline, model_dir / "selected_model.joblib")
    write_selection_results(model_dir / "final_feature_selection_results.csv", rows)

    metadata: dict[str, Any] = {
        "model_file": "final_model.joblib",
        "selected_model_file": "selected_model.joblib",
        "selected_model": selected["model"],
        "configuration": selected["configuration"],
        "selected_feature_set": selected["feature_set"],
        "feature_order": list(selected_features),
        "all_available_sensor_features": list(FEATURE_NAMES),
        "preprocessing_steps": list(final_pipeline.named_steps),
        "hyperparameters": selected["hyperparameters"],
        "training_dataset_path": dataset_path.name,
        "dataset_sha256": (
            "0ec550bdbe0dc0165ab59b35401dd1bec81fc80bb89a34097ee2a74b14ea02c1"
        ),
        "classes": 52,
        "class_labels": sorted(map(str, set(labels))),
        "training_records": len(x_train),
        "test_records": len(x_test),
        "random_state": RANDOM_STATE,
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "selection_metric": "macro F1",
            "selection_scope": "training split only",
            "macro_f1": selected["cv_f1_macro"],
        },
        "pressure_used_for_prediction": "pressure" in selected_features,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_data_limitation": (
            "Model was trained and evaluated on synthetic agricultural sensor data. "
            "Metrics do not represent field validation or real-world agricultural accuracy."
        ),
    }
    (model_dir / "final_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    evaluation = {
        "selected_model": selected["model"],
        "configuration": selected["configuration"],
        "selected_feature_set": selected["feature_set"],
        "feature_order": list(selected_features),
        "selection_basis": "training-only cross-validation macro F1",
        "cross_validation_metrics": {
            "accuracy": selected["cv_accuracy"],
            "precision_macro": selected["cv_precision_macro"],
            "recall_macro": selected["cv_recall_macro"],
            "f1_macro": selected["cv_f1_macro"],
            "f1_weighted": selected["cv_f1_weighted"],
            "f1_macro_std": selected["cv_f1_macro_std"],
        },
        "held_out_test_metrics": held_out,
        "held_out_test_usage": (
            "Evaluated once after model and feature-set selection was fixed."
        ),
        "pressure_used_for_prediction": "pressure" in selected_features,
        "synthetic_data": True,
        "field_validated": False,
    }
    (model_dir / "final_model_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "evaluation": evaluation, "selection_rows": rows}


def main() -> int:
    ml_root = Path(__file__).resolve().parents[1]
    result = finalize(
        ml_root / "data" / "synthetic" / "synthetic_crop_sensor_data.csv",
        ml_root / "models",
    )
    print(json.dumps({"metadata": result["metadata"], "evaluation": result["evaluation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
