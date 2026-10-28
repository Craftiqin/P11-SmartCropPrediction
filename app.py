from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.api_client import APIClientError, SmartCropAPIClient
from src.config import Settings
from src.crop_suitability import (
    input_quality_warnings,
    load_crop_profiles,
    rank_crop_suitability,
)
from src.dashboard import (
    format_timestamp,
    freshness_status,
    history_table_rows,
    is_stale_reading,
    latest_reading_rows,
    sensor_status_rows,
    sensor_value_rows,
)
from src.prediction import (
    load_selected_model,
    predict_reading,
)


ML_ROOT = Path(__file__).resolve().parent
MODEL_DIR = ML_ROOT / "models"
EVALUATION_DIR = MODEL_DIR / "evaluation"
CROP_PROFILE_PATH = ML_ROOT / "data" / "synthetic" / "crop_profiles.csv"
PAGES = (
    "Dashboard",
    "Live API Readings",
    "Live Prediction",
    "Model Comparison",
    "Feature Analysis",
    "System Information",
)


@st.cache_resource
def load_model():
    return load_selected_model(MODEL_DIR)


@st.cache_data
def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


@st.cache_data
def load_suitability_profiles() -> list[dict[str, str]]:
    return load_crop_profiles(CROP_PROFILE_PATH)


def fetch_latest(api_base_url: str) -> bool:
    try:
        with SmartCropAPIClient(api_base_url) as client:
            reading = client.get_latest_sensor_data()
    except APIClientError as exc:
        st.session_state.api_status = "Unavailable"
        st.session_state.api_error = str(exc)
        return False
    st.session_state.latest_reading = reading
    st.session_state.api_status = "Connected"
    st.session_state.api_error = None
    st.session_state.latest_fetch_time = datetime.now(timezone.utc).isoformat()
    return True


def fetch_history(api_base_url: str, limit: int) -> None:
    try:
        with SmartCropAPIClient(api_base_url) as client:
            readings = client.get_sensor_data(limit=limit)
    except APIClientError as exc:
        st.session_state.api_status = "Unavailable"
        st.session_state.api_error = str(exc)
        return
    st.session_state.api_readings = readings
    st.session_state.api_status = "Connected"
    st.session_state.api_error = None
    st.session_state.history_fetch_time = datetime.now(timezone.utc).isoformat()
    if readings:
        newest = sorted(
            readings,
            key=lambda item: item.get("timestamp") or "",
            reverse=True,
        )[0]
        st.session_state.latest_reading = newest


def merge_latest_into_history() -> None:
    latest = st.session_state.latest_reading
    if latest is None:
        return
    latest_id = latest.get("id")
    existing = st.session_state.get("api_readings", [])
    st.session_state.api_readings = [latest] + [
        reading for reading in existing if reading.get("id") != latest_id
    ]
    st.session_state.history_fetch_time = st.session_state.get("latest_fetch_time")


def show_disclaimer() -> None:
    st.warning(
        "Predictions are experimental decision-support estimates and should not be "
        "treated as guaranteed crop recommendations."
    )


def prediction_features(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(metadata.get("feature_order", metadata.get("feature_names", ())))


def readable_features(features: tuple[str, ...]) -> str:
    return ", ".join(name.replace("_", " ").title() for name in features)


def show_sensor_tiles(reading: dict[str, Any]) -> None:
    values = {row["Sensor value"]: row["Reading"] for row in sensor_value_rows(reading)}
    first = st.columns(3)
    first[0].metric("Temperature", values["Temperature"])
    first[1].metric("Humidity", values["Humidity"])
    first[2].metric("Soil Moisture", values["Soil moisture"])
    second = st.columns(2)
    second[0].metric("Light", values["Light"])
    second[1].metric("Pressure", values["Pressure"])


def show_reading_age(reading: dict[str, Any]) -> None:
    stale = is_stale_reading(reading.get("timestamp"))
    if stale is None:
        st.info("Timestamp unavailable; freshness cannot be verified.")
    elif stale:
        st.warning(
            "Warning: The latest API reading may be stale. Verify that the ESP32 is "
            "currently transmitting."
        )


def render_dashboard(
    reading: dict[str, Any] | None, metadata: dict[str, Any]
) -> None:
    st.title("Smart Crop Prediction")
    st.caption("IoT-based environmental monitoring and machine learning crop classification")
    show_disclaimer()

    first = st.columns(3)
    first[0].metric("API Status", st.session_state.api_status)
    first[1].metric("Device ID", reading.get("device_id", "Unavailable") if reading else "Unavailable")
    model_status = metadata.get(
        "configuration", metadata.get("selected_model_status", "baseline")
    )
    first[2].metric("Selected Model", f"{metadata['selected_model']} ({model_status})")
    second = st.columns(3)
    second[0].metric("Supported Crops", metadata.get("classes", 52))
    second[1].metric(
        "Last Reading Time",
        format_timestamp(reading.get("timestamp")) if reading else "Unknown timestamp",
    )
    second[2].metric(
        "Data Freshness",
        freshness_status(reading.get("timestamp")) if reading else "Unknown",
    )

    if reading:
        st.subheader("Latest sensor overview")
        show_sensor_tiles(reading)
        st.subheader("API-reported sensor status")
        st.dataframe(sensor_status_rows(reading), hide_index=True, width="stretch")
        st.subheader("Latest reading")
        st.write(f"Device: `{reading.get('device_id', 'Unknown')}`")
        st.write(f"Time: `{format_timestamp(reading.get('timestamp'))}`")
        st.write(f"Freshness: **{freshness_status(reading.get('timestamp'))}**")
        st.caption(
            'For complete API reading table, open “Live API Readings” from sidebar.'
        )
        show_reading_age(reading)
    else:
        st.info("No latest reading available. Use Live Prediction to retry the API.")


def numeric_average(readings: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(reading[field])
        for reading in readings
        if isinstance(reading.get(field), (int, float))
        and not isinstance(reading.get(field), bool)
        and math.isfinite(float(reading[field]))
    ]
    return sum(values) / len(values) if values else None


def render_live_api_readings(settings: Settings) -> None:
    st.title("Live API Readings")
    st.caption("View sensor readings received from the production IoT API.")
    st.code(
        f"Latest: {settings.api_base_url}/sensor-data/latest\n"
        f"Recent: {settings.api_base_url}/sensor-data?limit=N",
        language=None,
    )

    limit = st.selectbox("Records to request", (10, 25, 50, 100), index=1)
    if "api_readings" not in st.session_state:
        st.session_state.api_readings = []
        st.session_state.history_fetch_time = None
        with st.spinner("Fetching API readings..."):
            fetch_history(settings.api_base_url, limit)

    latest_button, recent_button = st.columns(2)
    if latest_button.button("Fetch Latest Reading", type="primary", width="stretch"):
        with st.spinner("Fetching latest API reading..."):
            if fetch_latest(settings.api_base_url):
                merge_latest_into_history()
    if recent_button.button("Refresh Recent Readings", width="stretch"):
        with st.spinner("Refreshing recent API readings..."):
            fetch_history(settings.api_base_url, limit)

    status, fetched = st.columns(2)
    status.metric("API Connection", st.session_state.api_status)
    fetched.metric(
        "Last Successful Fetch",
        format_timestamp(st.session_state.history_fetch_time)
        if st.session_state.history_fetch_time
        else "Not available",
    )
    if st.session_state.api_error:
        st.error(st.session_state.api_error)

    readings = st.session_state.api_readings
    if not readings:
        st.info("No API readings available.")
        return

    devices = sorted(
        {str(reading.get("device_id")) for reading in readings if reading.get("device_id")}
    )
    selected_device = st.selectbox("Device ID", ("All devices", *devices))
    filtered = (
        readings
        if selected_device == "All devices"
        else [reading for reading in readings if reading.get("device_id") == selected_device]
    )
    ordered = sorted(
        filtered,
        key=lambda item: item.get("timestamp") or "",
        reverse=True,
    )
    if not ordered:
        st.info("No readings match selected device.")
        return

    latest = ordered[0]
    summary = st.columns(2 if len(ordered) == 1 else 5)
    summary[0].metric("Records Loaded", len(ordered))
    summary[1].metric("Latest Reading", format_timestamp(latest.get("timestamp")))
    if len(ordered) > 1:
        for column, label, field, suffix, decimals in (
            (summary[2], "Average Temperature", "temperature", "°C", 1),
            (summary[3], "Average Humidity", "humidity", "% RH", 1),
            (summary[4], "Average Soil Moisture", "soil_moisture", "%", 1),
        ):
            average = numeric_average(ordered, field)
            column.metric(
                label,
                f"{average:.{decimals}f} {suffix}" if average is not None else "N/A",
            )

    st.subheader("Latest API Reading")
    st.dataframe(latest_reading_rows(latest), hide_index=True, width="stretch")
    st.write(f"Freshness: **{freshness_status(latest.get('timestamp'))}**")
    show_reading_age(latest)

    if len(ordered) > 1:
        st.subheader("Recent API Readings")
        st.dataframe(history_table_rows(ordered), hide_index=True, width="stretch")
        st.caption("Real records returned by existing `/sensor-data` API endpoint.")


def render_live_prediction(
    reading: dict[str, Any] | None,
    model: Any,
    metadata: dict[str, Any],
    settings: Settings,
) -> None:
    st.title("Live Prediction")
    features = prediction_features(metadata)
    st.info(f"Base ML prediction features: {readable_features(features)}")
    if "pressure" not in features:
        st.caption(
            "Pressure is displayed for monitoring but is not used by the final classifier."
        )

    st.code(f"{settings.api_base_url}/sensor-data/latest", language=None)
    st.write(f"API status: **{st.session_state.api_status}**")
    if st.session_state.api_error:
        st.error(st.session_state.api_error)

    fetch_col, predict_col = st.columns(2)
    if fetch_col.button("Fetch Latest Reading", type="secondary", width="stretch"):
        fetch_latest(settings.api_base_url)
        reading = st.session_state.latest_reading
    predict_pressed = predict_col.button(
        "Predict Crop", type="primary", disabled=reading is None, width="stretch"
    )

    if reading is None:
        st.info("No latest reading fetched yet.")
        return

    st.subheader("1. Input Context")
    context_columns = st.columns(3)
    context_values = (
        ("Device", str(reading.get("device_id") or "Unknown")),
        ("Last Reading", format_timestamp(reading.get("timestamp"))),
        ("Data Freshness", freshness_status(reading.get("timestamp"))),
    )
    for column, (label, value) in zip(context_columns, context_values, strict=True):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.write(f"**{value}**")
    st.subheader("2. Live Sensor Conditions")
    show_sensor_tiles(reading)
    with st.expander("Sensor connection status"):
        st.dataframe(sensor_status_rows(reading), hide_index=True, width="stretch")

    if predict_pressed:
        try:
            base_crop, _ = predict_reading(model, reading, features)
            suitability = rank_crop_suitability(
                reading,
                load_suitability_profiles(),
                limit=3,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Prediction unavailable: {exc}")
            return

        top = suitability[0]
        with st.container(border=True):
            st.subheader("Prediction Method")
            st.write("**Sensor-based crop suitability analysis**")
            method_columns = st.columns(2)
            method_columns[0].write(
                f"Base ML prediction: **{base_crop.replace('_', ' ').title()}**"
            )
            method_columns[1].write(
                f"Sensor-based top crop: **{top.crop.replace('_', ' ').title()}**"
            )
            st.caption(
                "Suitability compares current sensor conditions with existing crop "
                "profiles. It is separate from KNN probability."
            )

        st.subheader("3. Top Crop Suitability Results")
        with st.container(border=True):
            table_rows = []
            previous_score = None
            previous_rank = 0
            for position, result in enumerate(suitability, start=1):
                rank = (
                    previous_rank
                    if previous_score is not None
                    and math.isclose(
                        result.score,
                        previous_score,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    else position
                )
                table_rows.append(
                    {
                        "Rank": rank,
                        "Crop": result.crop.replace("_", " ").title(),
                        "Suitability Score": f"{result.score:.1%}",
                        "Matching Factors": result.reason.title(),
                    }
                )
                previous_score = result.score
                previous_rank = rank
            st.dataframe(
                table_rows,
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Scores are deterministic environmental compatibility estimates, "
                "not calibrated probabilities."
            )

        st.subheader("4. Final Crop Suggestion")
        with st.container(border=True):
            st.header(top.crop.replace("_", " ").upper())
            st.metric("Crop Suitability Score", f"{top.score:.1%}")
            st.write("Basis: **Current sensor compatibility + crop suitability profile**")
            st.write(f"Context quality: **{top.context_quality}**")

    st.subheader("5. Warnings")
    warnings = input_quality_warnings(reading)
    for warning in warnings:
        st.warning(warning)
    show_disclaimer()


def render_model_comparison(metadata: dict[str, Any]) -> None:
    st.title("Model Comparison")
    st.subheader("Held-out test performance")
    st.info(
        "These metrics are experimental and do not represent validated field performance."
    )
    st.success(
        f"Final selection: {metadata['selected_model']} ({metadata.get('configuration', 'baseline')}), "
        f"feature set: {metadata.get('selected_feature_set', 'All features')}."
    )
    comparison = load_csv(EVALUATION_DIR / "tuned_model_comparison.csv")
    if comparison is None:
        st.warning("Tuned model comparison is unavailable. Run `python -m src.tuning`.")
        return
    display = comparison.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "f1_macro": "Macro F1",
            "f1_weighted": "Weighted F1",
            "baseline_or_tuned": "Status",
        }
    )
    columns = [
        "Model",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Status",
    ]
    st.dataframe(display[columns], hide_index=True, width="stretch")
    chart = display.assign(Series=display["Model"] + " — " + display["Status"])
    st.bar_chart(chart, x="Series", y="Macro F1")


def render_feature_analysis(
    training_summary: dict[str, Any] | None,
    tuning_summary: dict[str, Any] | None,
    final_evaluation: dict[str, Any] | None,
) -> None:
    st.title("Feature Analysis")
    st.info(
        "Pressure is included because it is measured by the BMP280 sensor. Its "
        "usefulness for crop classification must be determined experimentally and "
        "should not be assumed."
    )
    if training_summary:
        rows = []
        for model_name, details in training_summary.get("models", {}).items():
            method = details.get("influence_method", "Unavailable")
            for feature, value in details.get("feature_influence", {}).items():
                rows.append(
                    {
                        "Model": model_name,
                        "Method": method,
                        "Feature": feature,
                        "Influence": value,
                    }
                )
        if rows:
            st.subheader("Baseline feature influence")
            st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.warning("Baseline feature influence file is unavailable.")

    selection = load_csv(MODEL_DIR / "final_feature_selection_results.csv")
    if selection is None:
        st.warning("Training-CV feature-set selection results are unavailable.")
    else:
        st.subheader("Training-only CV feature-set comparison")
        st.dataframe(selection, hide_index=True, width="stretch")
        st.bar_chart(
            selection,
            x="feature_set",
            y="cv_f1_macro",
            color="model",
        )

    ablation = load_csv(EVALUATION_DIR / "feature_ablation_results.csv")
    if ablation is None:
        st.warning("Feature ablation results are unavailable.")
    else:
        st.subheader("Previous held-out ablation — exploratory only")
        st.caption("These test-set results did not select the final configuration.")
        st.dataframe(ablation, hide_index=True, width="stretch")
        st.bar_chart(ablation, x="feature_set", y="f1_macro", color="model")

    if tuning_summary:
        deltas = tuning_summary.get("pressure_delta_macro_f1_all_minus_without", {})
        st.subheader("Pressure contribution")
        st.dataframe(
            [
                {"Model": model, "Macro F1 delta": value}
                for model, value in deltas.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(tuning_summary.get("pressure_conclusion", ""))

    if final_evaluation:
        st.subheader("Final held-out evaluation")
        metrics = final_evaluation.get("held_out_test_metrics", {})
        st.dataframe(
            [
                {"Metric": name.replace("_", " ").title(), "Value": value}
                for name, value in metrics.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Final test metrics were calculated after training-CV model and feature-set selection."
        )


def render_system_information(
    metadata: dict[str, Any], tuning_summary: dict[str, Any] | None
) -> None:
    st.title("System Information")
    hardware, software = st.columns(2)
    with hardware:
        st.subheader("Hardware")
        st.table(
            [
                "ESP32 DevKit V1",
                "DHT22 temperature/humidity sensor",
                "Capacitive soil moisture sensor",
                "BH1750 light sensor",
                "BMP280 pressure sensor",
                "SH1106 OLED display",
            ]
        )
    with software:
        st.subheader("Software")
        st.table(
            [
                "ESP32 firmware",
                "Cloudflare Worker API",
                "Neon PostgreSQL",
                "Python",
                "FastAPI-compatible data structures",
                "scikit-learn",
                "Streamlit",
            ]
        )

    st.subheader("Data and model")
    st.dataframe(
        [
            {"Field": "Model scope", "Value": "Experimental academic model"},
            {"Field": "Crop classes", "Value": str(metadata.get("classes", 52))},
            {
                "Field": "Training records",
                "Value": str(
                    metadata.get("training_records", metadata.get("training_samples", 800))
                ),
            },
            {
                "Field": "Prediction features",
                "Value": ", ".join(prediction_features(metadata)),
            },
            {"Field": "Selected model", "Value": metadata["selected_model"]},
            {
                "Field": "Experiment version",
                "Value": (tuning_summary or {}).get("experiment_version", "Baseline"),
            },
            {
                "Field": "Training time",
                "Value": (tuning_summary or {}).get("trained_at_utc", "Unavailable"),
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "This application is an academic prototype. The machine learning model was "
        "not validated against agricultural field observations. Predictions are for "
        "demonstration and research purposes only."
    )


st.set_page_config(
    page_title="Smart Crop Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    selected_model, selected_metadata = load_model()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Model unavailable: {exc}")
    st.stop()

settings = Settings()
if "api_status" not in st.session_state:
    st.session_state.api_status = "Not checked"
if "api_error" not in st.session_state:
    st.session_state.api_error = None
if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = None
    st.session_state.api_status = "Checking"
    st.session_state.api_error = None
    fetch_latest(settings.api_base_url)

training_summary = load_json(MODEL_DIR / "training_summary.json")
tuning_summary = load_json(MODEL_DIR / "tuning_summary.json")
final_evaluation = load_json(MODEL_DIR / "final_model_evaluation.json")

with st.sidebar:
    st.title("Smart Crop")
    st.subheader("Prediction")
    st.caption("IoT environmental monitoring and experimental crop classification")
    selected_page = st.radio("Navigation", PAGES, index=0)
    st.divider()
    st.subheader("System Status")
    st.write(f"API: **{st.session_state.api_status}**")
    st.write(
        f"Model: **{selected_metadata['selected_model']} "
        f"({selected_metadata.get('configuration', 'baseline')})**"
    )
    st.write("Scope: **Experimental decision support**")
    st.caption("Predictions are not field validated.")

reading = st.session_state.latest_reading
if selected_page == "Dashboard":
    render_dashboard(reading, selected_metadata)
elif selected_page == "Live API Readings":
    render_live_api_readings(settings)
elif selected_page == "Live Prediction":
    render_live_prediction(reading, selected_model, selected_metadata, settings)
elif selected_page == "Model Comparison":
    render_model_comparison(selected_metadata)
elif selected_page == "Feature Analysis":
    render_feature_analysis(training_summary, tuning_summary, final_evaluation)
else:
    render_system_information(selected_metadata, tuning_summary)
