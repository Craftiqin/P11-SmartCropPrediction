import csv
import hashlib
from pathlib import Path

import pytest

from src.data.prepare_crop_dataset import OUTPUT_COLUMNS, prepare_crop_dataset


ML_ROOT = Path(__file__).resolve().parents[1]
SOURCE = ML_ROOT / "data" / "final_crops_data.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preparation_is_reproducible_and_preserves_source(tmp_path):
    source_hash = sha256(SOURCE)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    first_summary = prepare_crop_dataset(SOURCE, first)
    second_summary = prepare_crop_dataset(SOURCE, second)

    assert first.read_bytes() == second.read_bytes()
    assert sha256(SOURCE) == source_hash
    assert first_summary == second_summary
    assert first_summary.input_rows == 2892
    assert first_summary.output_rows == 2891
    assert first_summary.duplicate_rows_removed == 1
    assert len(first_summary.class_distribution) == 9


def test_processed_schema_contains_only_real_available_features(tmp_path):
    output = tmp_path / "processed.csv"
    prepare_crop_dataset(SOURCE, output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == OUTPUT_COLUMNS

    forbidden = {"light", "pressure", "ph", "N", "P", "K", "rainfall", "Soil"}
    assert forbidden.isdisjoint(rows[0])
    assert all(row["crop"] for row in rows)
    assert all(float(row["temperature"]) for row in rows)
    assert all(0 <= float(row["humidity"]) <= 100 for row in rows)
    assert all(0 <= int(row["soil_moisture"]) <= 100 for row in rows)
    assert "Sugarcane" in {row["crop"] for row in rows}
    assert "Sugercane" not in {row["crop"] for row in rows}


def test_unexpected_source_columns_are_rejected(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("temperature,crop\n25,Rice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected columns"):
        prepare_crop_dataset(source, tmp_path / "output.csv")
