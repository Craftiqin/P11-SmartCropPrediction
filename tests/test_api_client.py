import json

import httpx
import pytest

from src.api_client import APIConnectionError, APIResponseError, SmartCropAPIClient


def transport_for(openapi_document, payload, expected_limit=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=openapi_document)
        if request.url.path == "/sensor-data/latest":
            return httpx.Response(200, json=payload)
        if request.url.path == "/sensor-data":
            if expected_limit is not None:
                assert request.url.params["limit"] == str(expected_limit)
            return httpx.Response(200, json=payload)
        raise AssertionError(f"Unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def test_latest_sensor_data(openapi_document, reading):
    client = SmartCropAPIClient(
        "https://api.craftiq.in", transport=transport_for(openapi_document, reading)
    )
    try:
        assert client.get_latest_sensor_data() == reading
    finally:
        client.close()


def test_recent_sensor_data(openapi_document, reading):
    client = SmartCropAPIClient(
        "https://api.craftiq.in/",
        transport=transport_for(openapi_document, [reading], expected_limit=10),
    )
    try:
        assert client.get_sensor_data(limit=10) == [reading]
    finally:
        client.close()


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_invalid_limit(limit):
    client = SmartCropAPIClient("https://api.craftiq.in", transport=httpx.MockTransport(lambda _: None))
    try:
        with pytest.raises(ValueError, match="1 to 100"):
            client.get_sensor_data(limit=limit)
    finally:
        client.close()


def test_connection_failure():
    def fail(request):
        raise httpx.ConnectError("offline", request=request)

    client = SmartCropAPIClient("https://api.craftiq.in", transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(APIConnectionError, match="Cannot connect"):
            client.get_latest_sensor_data()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"wrong": "shape"}, "failed SensorReadingResponse validation"),
        ("not-an-object", "failed SensorReadingResponse validation"),
    ],
)
def test_invalid_api_response(openapi_document, payload, message):
    client = SmartCropAPIClient(
        "https://api.craftiq.in", transport=transport_for(openapi_document, payload)
    )
    try:
        with pytest.raises(APIResponseError, match=message):
            client.get_latest_sensor_data()
    finally:
        client.close()


def test_invalid_json():
    def handler(request):
        return httpx.Response(200, content=b"not-json")

    client = SmartCropAPIClient("https://api.craftiq.in", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(APIResponseError, match="invalid JSON"):
            client.get_latest_sensor_data()
    finally:
        client.close()


def test_http_error_is_clean():
    client = SmartCropAPIClient(
        "https://api.craftiq.in",
        transport=httpx.MockTransport(lambda _: httpx.Response(503, json={"secret": "hidden"})),
    )
    try:
        with pytest.raises(APIResponseError, match="HTTP 503") as error:
            client.get_latest_sensor_data()
        assert "hidden" not in str(error.value)
    finally:
        client.close()

