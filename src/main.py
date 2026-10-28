from pathlib import Path

from src.api_client import APIClientError, SmartCropAPIClient
from src.config import Settings
from src.data_loader import load_latest
from src.prediction import load_selected_model, predict_reading


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def display_value(value: object, unit: str) -> str:
    return "OFF" if value is None else f"{value} {unit}"


def main() -> int:
    settings = Settings()
    print(f"Connecting to: {settings.api_base_url}")
    try:
        with SmartCropAPIClient(settings.api_base_url) as client:
            reading = load_latest(client)
    except APIClientError as exc:
        print(f"API error: {exc}")
        return 1

    print("\nLatest sensor data:")
    print(f"Device: {reading['device_id']}")
    print(f"Timestamp: {reading['timestamp']}")
    print(f"Temperature: {display_value(reading['temperature'], 'C')}")
    print(f"Humidity: {display_value(reading['humidity'], '%')}")
    print(f"Soil Moisture: {display_value(reading['soil_moisture'], '%')}")
    print(f"Light: {display_value(reading['light'], 'lux')}")
    print(f"Pressure: {display_value(reading['pressure'], 'hPa')}")
    try:
        model, metadata = load_selected_model(MODEL_DIR)
        feature_names = tuple(
            metadata.get("feature_order", metadata.get("feature_names", ()))
        )
        crop, confidence = predict_reading(model, reading, feature_names)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Prediction unavailable: {exc}")
        return 1

    print(f"\nSelected model: {metadata['selected_model']}")
    print(f"Predicted crop: {crop}")
    if confidence is not None:
        print(f"Model confidence: {confidence:.1%}")
    print("Warning: trained on SYNTHETIC data; not field-validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
