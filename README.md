# Smart Crop Prediction: standalone ML project

This directory contains the project's existing training, evaluation, saved-model inference, sensor-profile suitability scoring, tests, and Streamlit dashboard. Local prediction needs no API, Worker, FastAPI service, database, or credentials.

## Model and input contract

The saved artifact `models/final_model.joblib` is the selected tuned KNN pipeline. It includes its fitted `StandardScaler`; inference must use metadata feature order:

1. `temperature`: degrees Celsius
2. `humidity`: relative humidity, percent
3. `soil_moisture`: calibrated relative sensor percent; not volumetric water content
4. `light`: illuminance in lux

The estimator uses nine neighbors, uniform vote weights, and Manhattan distance (`p=1`). The synthetic dataset also contains `pressure`, but it is excluded from final prediction. KNN vote fractions can tie; sklearn's predicted class is retained, and the CLI reports actual vote fractions plus the project's existing low/moderate/high diagnostic. These fractions are not calibrated probabilities. A separate profile compatibility score is also shown; it is not the KNN probability.

Training/evaluation dataset is `data/synthetic/synthetic_crop_sensor_data.csv`: 1,000 synthetic rows, 52 labels, fixed SHA-256 verified by loader. `data/synthetic/crop_profiles.csv` supplies separate heuristic suitability ranges. Synthetic metrics do not establish field accuracy or agronomic benefit. The real crop dataset remains separate and is not used to fit this selected model.

## Install

Verified runtime: Python 3.14.5. Use matching pinned scikit-learn and Joblib versions to load the checked-in model artifact:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Pinned runtime dependencies in `requirements.txt` preserve compatibility with the checked-in Joblib artifact. For artifact loading, keep scikit-learn and Joblib versions aligned with these pins.

## Run local prediction

Run from this directory. Sample uses the documented units above:

```bash
python predict.py --temperature 28 --humidity 65 --soil-moisture 40 --light 12000
```

Output separates the saved model's predicted crop and top KNN vote fractions from the independent sensor-profile suitability ranking. Soil-moisture value `0` is accepted by the classifier, but profile scoring excludes it as unreliable and labels context accordingly.

## Tests

```bash
python -m pytest -q
```

Tests cover fixed dataset integrity, feature order, saved pipeline loading, training-only selection, prediction behavior, the CLI, suitability scoring, API-client behavior, and dashboard helpers. The CLI test is fully local and makes no network requests.

## Reproduce training and selection

Commands run from this directory:

```bash
python -m src.training
python -m src.tuning
python -m src.finalize_model
```

`training` compares baseline Decision Tree, KNN and SVM. `tuning` performs the preserved tuning/ablation stage. `finalize_model` selects among existing baseline/tuned estimators and feature sets with three-fold stratified CV on the training partition, then saves the final pipeline, metadata and evaluation. This full workflow is compute-intensive; it is not needed to run inference because final model artifact is included.

The reported final held-out accuracy is `0.235` and macro F1 is `0.200301` on synthetic data. Earlier project development inspected the same holdout, so independence is limited; do not present these metrics as field validation.

## Optional dashboard and live API

The Streamlit dashboard is part of the existing ML application:

```bash
streamlit run app.py
```

Dashboard live-reading features use `API_BASE_URL` (default `https://api.craftiq.in`) and require network access. Local `predict.py` does not use this setting. Never put API tokens, database credentials, or Wi-Fi secrets in this repository.

## Included artifacts

- `predict.py`, `src/`: local inference, feature conversion, suitability scorer, training, preprocessing, tuning, and evaluation code.
- `models/final_model.joblib`, `models/final_model_metadata.json`, `models/final_model_evaluation.json`: authoritative finalized inference artifact and contract.
- Other `models/` files: existing baseline/tuned models and evaluation outputs used by reproducibility scripts and tests.
- `data/synthetic/`: fixed training data, crop profiles, range report, generation details.
- `tests/`: existing project tests plus a CLI smoke test.
- `requirements.txt`, `requirements-dev.txt`: pinned runtime and test dependencies.

Real-data exploratory preparation source is retained, but unrelated raw real-data CSVs are not required by this final synthetic training pipeline and are excluded from publishing. The generator and dataset README document the synthetic data provenance and limitations.
