from typing import Any

import httpx
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError


class APIClientError(RuntimeError):
    """Base error safe for CLI display."""


class APIConnectionError(APIClientError):
    """API could not be reached."""


class APIResponseError(APIClientError):
    """API returned an unsuccessful or invalid response."""


class SmartCropAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )
        self._openapi: dict[str, Any] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get_json(self, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.get(path, **kwargs)
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Cannot connect to {self.base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise APIResponseError(
                f"API returned HTTP {exc.response.status_code} for {path}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise APIResponseError(f"API returned invalid JSON for {path}") from exc

    def _response_schema(self, name: str) -> dict[str, Any]:
        if self._openapi is None:
            document = self._get_json("/openapi.json")
            if not isinstance(document, dict):
                raise APIResponseError("API OpenAPI document is not a JSON object")
            self._openapi = document

        components = self._openapi.get("components")
        schemas = components.get("schemas") if isinstance(components, dict) else None
        if not isinstance(schemas, dict) or name not in schemas:
            raise APIResponseError(f"API OpenAPI document has no {name} schema")
        return {"$ref": f"#/components/schemas/{name}", "components": components}

    def _validate(self, data: Any, schema_name: str) -> Any:
        try:
            Draft202012Validator(
                self._response_schema(schema_name), format_checker=FormatChecker()
            ).validate(data)
        except (SchemaError, ValidationError) as exc:
            raise APIResponseError(f"API response failed {schema_name} validation") from exc
        return data

    def get_latest_sensor_data(self) -> dict[str, Any]:
        data = self._get_json("/sensor-data/latest")
        self._validate(data, "SensorReadingResponse")
        return data

    def get_sensor_data(self, limit: int = 50) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        data = self._get_json("/sensor-data", params={"limit": limit})
        if not isinstance(data, list):
            raise APIResponseError("API recent sensor response is not a JSON array")
        for item in data:
            self._validate(item, "SensorReadingResponse")
        return data

