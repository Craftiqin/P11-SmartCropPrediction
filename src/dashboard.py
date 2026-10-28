from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


SENSOR_DISPLAY = (
    ("Temperature", "temperature", "°C", 1),
    ("Humidity", "humidity", "% RH", 1),
    ("Soil moisture", "soil_moisture", "%", None),
    ("Light", "light", "lux", 2),
    ("Pressure", "pressure", "hPa", 2),
)

SENSOR_STATUS = (
    ("DHT22", "dht22_status"),
    ("Soil moisture sensor", "soil_status"),
    ("BH1750", "bh1750_status"),
    ("BMP280", "bmp280_status"),
)


def parse_api_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: object) -> str:
    parsed = parse_api_timestamp(value)
    if parsed is None:
        return "Timestamp unavailable"
    return parsed.astimezone(IST).strftime("%d %b %Y, %I:%M:%S %p IST")


def is_stale_reading(
    value: object,
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(hours=1),
) -> bool | None:
    parsed = parse_api_timestamp(value)
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) - parsed > stale_after


def format_sensor_value(value: object, unit: str, decimals: int | None) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    if decimals is None:
        rendered = f"{number:.0f}" if number.is_integer() else f"{number:.1f}"
    else:
        rendered = f"{number:.{decimals}f}"
    return f"{rendered} {unit}"


def sensor_value_rows(reading: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "Sensor value": label,
            "Reading": format_sensor_value(reading.get(field), unit, decimals),
        }
        for label, field, unit, decimals in SENSOR_DISPLAY
    ]


def sensor_status_rows(reading: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, status_field in SENSOR_STATUS:
        explicit = reading.get(status_field)
        if explicit == "ON":
            status = "Connected"
        elif explicit == "OFF":
            status = "Offline"
        else:
            status = "Unknown"
        rows.append({"Sensor": label, "Status": status})
    return rows


def freshness_status(
    value: object,
    *,
    now: datetime | None = None,
) -> str:
    parsed = parse_api_timestamp(value)
    if parsed is None:
        return "Unknown"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age = current.astimezone(timezone.utc) - parsed
    if age <= timedelta(minutes=5):
        return "Current"
    if age <= timedelta(hours=1):
        return "Recently updated"
    return "Stale"


def format_integer(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    return str(int(number)) if math.isfinite(number) else "N/A"


def latest_reading_rows(reading: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"Field": "Device ID", "Value": str(reading.get("device_id") or "Unknown")},
        {"Field": "Record ID", "Value": format_integer(reading.get("id"))},
        {"Field": "Timestamp", "Value": format_timestamp(reading.get("timestamp"))},
        {
            "Field": "Temperature",
            "Value": format_sensor_value(reading.get("temperature"), "°C", 1),
        },
        {
            "Field": "Humidity",
            "Value": format_sensor_value(reading.get("humidity"), "% RH", 1),
        },
        {
            "Field": "Soil Moisture",
            "Value": format_sensor_value(reading.get("soil_moisture"), "%", None),
        },
        {"Field": "Raw Soil Value", "Value": format_integer(reading.get("soil_raw"))},
        {
            "Field": "Light",
            "Value": format_sensor_value(reading.get("light"), "lux", 2),
        },
        {
            "Field": "Pressure",
            "Value": format_sensor_value(reading.get("pressure"), "hPa", 2),
        },
        {"Field": "DHT22 Status", "Value": str(reading.get("dht22_status") or "Unknown")},
        {"Field": "Soil Sensor Status", "Value": str(reading.get("soil_status") or "Unknown")},
        {"Field": "BH1750 Status", "Value": str(reading.get("bh1750_status") or "Unknown")},
        {"Field": "BMP280 Status", "Value": str(reading.get("bmp280_status") or "Unknown")},
    ]


def history_table_rows(readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(reading: dict[str, Any]) -> datetime:
        return parse_api_timestamp(reading.get("timestamp")) or datetime.min.replace(
            tzinfo=timezone.utc
        )

    rows: list[dict[str, Any]] = []
    for reading in sorted(readings, key=sort_key, reverse=True):
        rows.append(
            {
                "Timestamp IST": format_timestamp(reading.get("timestamp")),
                "Device ID": str(reading.get("device_id") or "Unknown"),
                "Record ID": format_integer(reading.get("id")),
                "Temperature °C": format_sensor_value(
                    reading.get("temperature"), "°C", 1
                ),
                "Humidity": format_sensor_value(reading.get("humidity"), "% RH", 1),
                "Soil Moisture": format_sensor_value(
                    reading.get("soil_moisture"), "%", None
                ),
                "Light": format_sensor_value(reading.get("light"), "lux", 2),
                "Pressure": format_sensor_value(reading.get("pressure"), "hPa", 2),
                "DHT22": str(reading.get("dht22_status") or "Unknown"),
                "Soil": str(reading.get("soil_status") or "Unknown"),
                "BH1750": str(reading.get("bh1750_status") or "Unknown"),
                "BMP280": str(reading.get("bmp280_status") or "Unknown"),
            }
        )
    return rows
