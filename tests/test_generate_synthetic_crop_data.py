import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from src.data.generate_synthetic_crop_data import (
    OUTPUT_COLUMNS,
    SEED,
    TOTAL_SAMPLES,
    generate_rows,
    load_profiles,
    sample_allocation,
    validate_rows,
    write_rows,
)


ML_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ML_ROOT / "data" / "synthetic" / "crop_profiles.csv"
REAL_FILES = (
    ML_ROOT / "data" / "final_crops_data.csv",
    ML_ROOT / "data" / "processed" / "final_crops_sensor_subset.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_generation_is_balanced_valid_and_reproducible(tmp_path):
    profiles = load_profiles(PROFILES)
    first = generate_rows(profiles, seed=SEED)
    second = generate_rows(profiles, seed=SEED)
    validate_rows(first, profiles)

    assert first == second
    assert len(first) == TOTAL_SAMPLES == 1000
    assert len(profiles) == 52
    assert Counter(row["crop"] for row in first) == Counter(sample_allocation(profiles))
    assert Counter(Counter(row["crop"] for row in first).values()) == {19: 40, 20: 12}
    assert all(tuple(row) == OUTPUT_COLUMNS for row in first)
    assert len(first) == len({tuple(row.values()) for row in first})

    first_path, second_path = tmp_path / "first.csv", tmp_path / "second.csv"
    write_rows(first, first_path)
    write_rows(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_no_fabricated_or_unapproved_features():
    profiles = load_profiles(PROFILES)
    rows = generate_rows(profiles)
    assert set(rows[0]) == set(OUTPUT_COLUMNS)
    assert {"ph", "N", "P", "K", "rainfall"}.isdisjoint(rows[0])


def test_pressure_distribution_has_no_crop_label_signal():
    profiles = load_profiles(PROFILES)
    rows = generate_rows(profiles)
    pressure_by_crop = defaultdict(list)
    for row in rows:
        pressure_by_crop[row["crop"]].append(float(row["pressure"]))
    full = {
        tuple(sorted(values))
        for values in pressure_by_crop.values()
        if len(values) == 20
    }
    assert len(full) == 1
    full_values = set(next(iter(full)))
    assert all(set(values) <= full_values for values in pressure_by_crop.values())
    assert all(len(values) in {19, 20} for values in pressure_by_crop.values())


def test_default_allocation_is_deterministic_and_documented():
    profiles = load_profiles(PROFILES)
    allocation = sample_allocation(profiles)
    twenty = sorted(crop for crop, count in allocation.items() if count == 20)
    assert twenty == [
        "apple", "banana", "barley", "black_gram", "black_pepper", "cabbage",
        "cardamom", "carrot", "castor", "cauliflower", "chickpea", "chilli",
    ]


def test_real_datasets_remain_unchanged(tmp_path):
    before = {path: sha256(path) for path in REAL_FILES}
    profiles = load_profiles(PROFILES)
    output = tmp_path / "synthetic.csv"
    write_rows(generate_rows(profiles), output)
    assert {path: sha256(path) for path in REAL_FILES} == before


def test_generated_file_loads_with_numeric_features():
    path = ML_ROOT / "data" / "synthetic" / "synthetic_crop_sensor_data.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert all(all(float(row[column]) >= 0 for column in OUTPUT_COLUMNS[:-1]) for row in rows)
