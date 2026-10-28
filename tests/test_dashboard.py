from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.dashboard import (
    format_timestamp,
    freshness_status,
    history_table_rows,
    is_stale_reading,
    latest_reading_rows,
    sensor_status_rows,
    sensor_value_rows,
)
from src.api_client import APIConnectionError


ML_ROOT = Path(__file__).resolve().parents[1]


def test_timestamp_formatting_and_stale_detection():
    timestamp = "2026-09-08T12:42:31.106420Z"
    now = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
    assert format_timestamp(timestamp) == "08 Sep 2026, 06:12:31 PM IST"
    assert is_stale_reading(timestamp, now=now) is True
    assert is_stale_reading("2026-09-08T13:30:00Z", now=now) is False
    assert freshness_status("2026-09-08T13:58:00Z", now=now) == "Current"
    assert freshness_status("2026-09-08T13:30:00Z", now=now) == "Recently updated"
    assert freshness_status(timestamp, now=now) == "Stale"
    assert format_timestamp(None) == "Timestamp unavailable"
    assert freshness_status(None, now=now) == "Unknown"
    assert is_stale_reading("malformed", now=now) is None


def test_sensor_formatting_and_missing_status(reading):
    values = sensor_value_rows(reading)
    assert [row["Reading"] for row in values] == [
        "28.4 °C",
        "64.2 % RH",
        "52 %",
        "423.50 lux",
        "1008.40 hPa",
    ]
    reading["light"] = None
    reading["bh1750_status"] = "OFF"
    statuses = sensor_status_rows(reading)
    assert statuses[2] == {"Sensor": "BH1750", "Status": "Offline"}
    assert sensor_value_rows(reading)[3]["Reading"] == "N/A"
    reading.pop("bmp280_status")
    assert sensor_status_rows(reading)[3]["Status"] == "Unknown"


def test_latest_and_history_table_rows(reading):
    latest = latest_reading_rows(reading)
    assert {row["Field"]: row["Value"] for row in latest} == {
        "Device ID": "ESP32_CROP_01",
        "Record ID": "7",
        "Timestamp": "08 Sep 2026, 06:12:31 PM IST",
        "Temperature": "28.4 °C",
        "Humidity": "64.2 % RH",
        "Soil Moisture": "52 %",
        "Raw Soil Value": "2200",
        "Light": "423.50 lux",
        "Pressure": "1008.40 hPa",
        "DHT22 Status": "ON",
        "Soil Sensor Status": "ON",
        "BH1750 Status": "ON",
        "BMP280 Status": "ON",
    }
    older = reading | {"id": 6, "timestamp": "2026-09-08T11:00:00Z"}
    rows = history_table_rows([older, reading])
    assert [row["Record ID"] for row in rows] == ["7", "6"]
    assert rows[0]["Pressure"] == "1008.40 hPa"


class FakeAPIClient:
    latest_calls = 0
    history_calls = 0
    reading = {
        "id": 7,
        "timestamp": "2026-09-08T12:42:31.106420Z",
        "device_id": "ESP32_CROP_01",
        "uptime_ms": 12345,
        "temperature": 28.4,
        "humidity": 64.2,
        "soil_moisture": 52,
        "soil_raw": 2200,
        "light": 423.5,
        "pressure": 1008.4,
        "dht22_status": "ON",
        "soil_status": "ON",
        "bh1750_status": "ON",
        "bmp280_status": "ON",
    }

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def get_latest_sensor_data(self):
        type(self).latest_calls += 1
        return self.reading.copy()

    def get_sensor_data(self, limit=25):
        type(self).history_calls += 1
        return [
            self.reading.copy(),
            self.reading | {"id": 6, "timestamp": "2026-09-08T11:00:00Z"},
        ][:limit]


def test_navigation_pages_and_prediction_render_without_live_api():
    FakeAPIClient.latest_calls = 0
    FakeAPIClient.history_calls = 0
    with patch("src.api_client.SmartCropAPIClient", FakeAPIClient):
        app = AppTest.from_file(ML_ROOT / "app.py", default_timeout=20).run()
        assert not app.exception
        navigation = app.sidebar.radio[0]
        assert navigation.options == [
            "Dashboard",
            "Live API Readings",
            "Live Prediction",
            "Model Comparison",
            "Feature Analysis",
            "System Information",
        ]
        for page in navigation.options:
            navigation.set_value(page)
            app.run()
            assert not app.exception
            navigation = app.sidebar.radio[0]

        navigation.set_value("Live Prediction")
        app.run()
        next(button for button in app.button if button.label == "Predict Crop").click()
        app.run()
        assert not app.exception
        assert any("4. Final Crop Suggestion" == item.value for item in app.subheader)
        assert any("Prediction Method" == item.value for item in app.subheader)
        assert any(
            "3. Top Crop Suitability Results" == item.value for item in app.subheader
        )
        suitability_tables = [
            table.value
            for table in app.dataframe
            if list(table.value.columns)
            == ["Rank", "Crop", "Suitability Score", "Matching Factors"]
        ]
        assert len(suitability_tables) == 1
        assert len(suitability_tables[0]) == 3
        assert suitability_tables[0].iloc[0]["Rank"] == 1
        expected_final = suitability_tables[0].iloc[0]["Crop"].upper()
        assert any(item.value == expected_final for item in app.header)
        assert "Probability" not in suitability_tables[0].columns
        assert not any("Location" in warning.value for warning in app.warning)
        assert not app.get("vega_lite_chart")
        assert any(
            "Pressure is displayed for monitoring but is not used" in item.value
            for item in app.caption
        )
        assert FakeAPIClient.latest_calls == 1
        assert FakeAPIClient.history_calls == 1


def test_live_prediction_has_no_location_controls():
    with patch("src.api_client.SmartCropAPIClient", FakeAPIClient):
        app = AppTest.from_file(ML_ROOT / "app.py", default_timeout=20).run()
        app.sidebar.radio[0].set_value("Live Prediction")
        app.run()
        assert not app.selectbox
        assert not app.text_input
        assert not any(button.label == "Use Manual Location" for button in app.button)


def test_all_pages_avoid_repeated_dataset_wording():
    with patch("src.api_client.SmartCropAPIClient", FakeAPIClient):
        app = AppTest.from_file(ML_ROOT / "app.py", default_timeout=20).run()
        navigation = app.sidebar.radio[0]
        for page in navigation.options:
            navigation.set_value(page)
            app.run()
            visible_text = " ".join(
                str(element.value)
                for element_type in (
                    "caption",
                    "header",
                    "info",
                    "markdown",
                    "subheader",
                    "title",
                    "warning",
                )
                for element in app.get(element_type)
            ).lower()
            assert "synthetic" not in visible_text
            navigation = app.sidebar.radio[0]


class FailingAPIClient(FakeAPIClient):
    def get_latest_sensor_data(self):
        raise APIConnectionError("Cannot connect to production API")

    def get_sensor_data(self, limit=25):
        raise APIConnectionError("Cannot connect to production API")


def test_live_api_page_handles_api_failure():
    with patch("src.api_client.SmartCropAPIClient", FailingAPIClient):
        app = AppTest.from_file(ML_ROOT / "app.py", default_timeout=20).run()
        app.sidebar.radio[0].set_value("Live API Readings")
        app.run()
        assert not app.exception
        assert any("Cannot connect" in error.value for error in app.error)
