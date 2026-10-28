from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SOURCE_COLUMNS = ("Soil", "Soil_Moisture", "Humidity", "Temperature", "Crop Name")
OUTPUT_COLUMNS = ("temperature", "humidity", "soil_moisture", "crop")
LABEL_CORRECTIONS = {"Sugercane": "Sugarcane"}


@dataclass(frozen=True)
class PreparationSummary:
    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    class_distribution: dict[str, int]


def _finite_float(value: str, column: str, row_number: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return number


def _format_float(value: float) -> str:
    return format(value, ".15g")


def prepare_crop_dataset(source: Path, destination: Path) -> PreparationSummary:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(
                f"unexpected columns: expected {SOURCE_COLUMNS}, got {tuple(reader.fieldnames or ())}"
            )
        source_rows = list(reader)

    prepared: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    duplicates = 0

    for row_number, row in enumerate(source_rows, start=2):
        if any(row[column].strip() == "" for column in SOURCE_COLUMNS):
            raise ValueError(f"row {row_number}: missing value")

        soil = _finite_float(row["Soil"], "Soil", row_number)
        soil_moisture_value = _finite_float(
            row["Soil_Moisture"], "Soil_Moisture", row_number
        )
        humidity = _finite_float(row["Humidity"], "Humidity", row_number)
        temperature = _finite_float(row["Temperature"], "Temperature", row_number)

        if abs(soil * 100 - soil_moisture_value) > 1e-9:
            raise ValueError(
                f"row {row_number}: Soil is not Soil_Moisture divided by 100"
            )
        if not soil_moisture_value.is_integer():
            raise ValueError(f"row {row_number}: Soil_Moisture must be an integer percentage")
        if not 0 <= soil_moisture_value <= 100:
            raise ValueError(f"row {row_number}: Soil_Moisture outside 0..100")
        if not 0 <= humidity <= 100:
            raise ValueError(f"row {row_number}: Humidity outside 0..100")
        if not -40 <= temperature <= 80:
            raise ValueError(f"row {row_number}: Temperature outside -40..80")

        crop = row["Crop Name"].strip()
        if not crop:
            raise ValueError(f"row {row_number}: Crop Name must not be empty")
        crop = LABEL_CORRECTIONS.get(crop, crop)

        output_row = {
            "temperature": _format_float(temperature),
            "humidity": _format_float(humidity),
            "soil_moisture": str(int(soil_moisture_value)),
            "crop": crop,
        }
        key = tuple(output_row[column] for column in OUTPUT_COLUMNS)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        prepared.append(output_row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(prepared)

    distribution = dict(sorted(Counter(row["crop"] for row in prepared).items()))
    return PreparationSummary(
        input_rows=len(source_rows),
        output_rows=len(prepared),
        duplicate_rows_removed=duplicates,
        class_distribution=distribution,
    )


def main() -> int:
    ml_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare ESP32-compatible crop dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=ml_root / "data" / "final_crops_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ml_root / "data" / "processed" / "final_crops_sensor_subset.csv",
    )
    args = parser.parse_args()
    summary = prepare_crop_dataset(args.source, args.output)
    print(f"Prepared {summary.output_rows} rows from {summary.input_rows}")
    print(f"Removed {summary.duplicate_rows_removed} duplicate row(s)")
    print(f"Classes: {len(summary.class_distribution)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
