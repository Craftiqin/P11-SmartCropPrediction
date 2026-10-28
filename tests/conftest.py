import pytest


@pytest.fixture
def reading():
    return {
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


@pytest.fixture
def openapi_document():
    nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    properties = {
        "id": {"type": "integer"},
        "timestamp": {"type": "string", "format": "date-time"},
        "device_id": {"type": "string"},
        "uptime_ms": {"type": "integer"},
        "temperature": nullable_number,
        "humidity": nullable_number,
        "soil_moisture": nullable_integer,
        "soil_raw": {"type": "integer"},
        "light": nullable_number,
        "pressure": nullable_number,
        "dht22_status": {"enum": ["ON", "OFF"]},
        "soil_status": {"enum": ["ON", "OFF"]},
        "bh1750_status": {"enum": ["ON", "OFF"]},
        "bmp280_status": {"enum": ["ON", "OFF"]},
    }
    return {
        "openapi": "3.1.0",
        "components": {
            "schemas": {
                "SensorReadingResponse": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                }
            }
        },
    }

