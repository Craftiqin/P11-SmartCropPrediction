from typing import Any

from src.api_client import SmartCropAPIClient


def load_latest(client: SmartCropAPIClient) -> dict[str, Any]:
    """Load one API-validated reading without database or backend imports."""
    return client.get_latest_sensor_data()


def load_recent(client: SmartCropAPIClient, limit: int = 50) -> list[dict[str, Any]]:
    """Load bounded API-validated history."""
    return client.get_sensor_data(limit=limit)

