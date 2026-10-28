from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260909
TOTAL_SAMPLES = 1000
EXTRA_SAMPLE_CROPS = 12
OUTPUT_COLUMNS = ("temperature", "humidity", "soil_moisture", "light", "pressure", "crop")
NUMERIC_COLUMNS = OUTPUT_COLUMNS[:-1]
PROFILE_RANGE_COLUMNS = {
    "temperature": ("temperature_min_c", "temperature_max_c"),
    "humidity": ("humidity_min_pct", "humidity_max_pct"),
    "soil_moisture": ("soil_moisture_min_pct", "soil_moisture_max_pct"),
    "light": ("light_min_lux", "light_max_lux"),
    "pressure": ("pressure_min_hpa", "pressure_max_hpa"),
}


def load_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"crop", *[name for pair in PROFILE_RANGE_COLUMNS.values() for name in pair]}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("crop profile file has missing required columns")
    crops = [row["crop"].strip() for row in rows]
    if any(not crop for crop in crops) or len(crops) != len(set(crops)):
        raise ValueError("crop profile labels must be non-empty and unique")
    for row in rows:
        for low_name, high_name in PROFILE_RANGE_COLUMNS.values():
            low, high = float(row[low_name]), float(row[high_name])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"invalid range for {row['crop']}: {low_name}/{high_name}")
    return rows


def _triangular(rng: random.Random, low: float, high: float) -> float:
    return rng.triangular(low, high, (low + high) / 2)


def sample_allocation(
    profiles: list[dict[str, str]], total_samples: int = TOTAL_SAMPLES
) -> dict[str, int]:
    crops = sorted(profile["crop"].strip() for profile in profiles)
    if total_samples < len(crops):
        raise ValueError("total_samples must represent every crop")
    base, extra = divmod(total_samples, len(crops))
    if total_samples == TOTAL_SAMPLES and extra != EXTRA_SAMPLE_CROPS:
        raise ValueError("default allocation must assign 12 extra samples")
    return {crop: base + (index < extra) for index, crop in enumerate(crops)}


def generate_rows(
    profiles: list[dict[str, str]],
    *,
    seed: int = SEED,
    total_samples: int = TOTAL_SAMPLES,
) -> list[dict[str, str]]:
    allocation = sample_allocation(profiles, total_samples)
    rng = random.Random(seed)
    pressure_rng = random.Random(seed + 1)
    pressure_low = float(profiles[0]["pressure_min_hpa"])
    pressure_high = float(profiles[0]["pressure_max_hpa"])
    if any(
        float(profile["pressure_min_hpa"]) != pressure_low
        or float(profile["pressure_max_hpa"]) != pressure_high
        for profile in profiles
    ):
        raise ValueError("pressure range must be identical for every crop")

    pressure_pool_size = max(allocation.values())
    base_pressures = [
        round(min(pressure_high, max(pressure_low, pressure_rng.gauss(1013.25, 10))), 2)
        for _ in range(pressure_pool_size)
    ]
    rows: list[dict[str, str]] = []
    for crop_index, profile in enumerate(profiles):
        crop = profile["crop"].strip()
        offset = crop_index % pressure_pool_size
        rotated_pressures = base_pressures[offset:] + base_pressures[:offset]
        pressure_values = rotated_pressures[: allocation[crop]]
        for sample_index in range(allocation[crop]):
            generated: dict[str, str] = {}
            for feature in ("temperature", "humidity", "soil_moisture", "light"):
                low_name, high_name = PROFILE_RANGE_COLUMNS[feature]
                low, high = float(profile[low_name]), float(profile[high_name])
                value = _triangular(rng, low, high)
                generated[feature] = str(int(round(value))) if feature == "soil_moisture" else f"{value:.2f}"
            generated["pressure"] = f"{pressure_values[sample_index]:.2f}"
            generated["crop"] = crop
            rows.append(generated)
    rng.shuffle(rows)
    return rows


def validate_rows(rows: list[dict[str, str]], profiles: list[dict[str, str]]) -> None:
    profile_by_crop = {profile["crop"]: profile for profile in profiles}
    if any(tuple(row) != OUTPUT_COLUMNS for row in rows):
        raise ValueError("synthetic output schema mismatch")
    if any(not row[column].strip() for row in rows for column in OUTPUT_COLUMNS):
        raise ValueError("synthetic output contains missing values")
    if len(rows) != len({tuple(row[column] for column in OUTPUT_COLUMNS) for row in rows}):
        raise ValueError("synthetic output contains duplicate rows")

    pressure_sets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        crop = row["crop"]
        if crop not in profile_by_crop:
            raise ValueError(f"unknown crop label: {crop}")
        profile = profile_by_crop[crop]
        for feature in NUMERIC_COLUMNS:
            value = float(row[feature])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {feature}")
            low_name, high_name = PROFILE_RANGE_COLUMNS[feature]
            if not float(profile[low_name]) <= value <= float(profile[high_name]):
                raise ValueError(f"{crop} {feature} outside profile range")
        pressure_sets[crop].append(float(row["pressure"]))

    maximum_count = max(map(len, pressure_sets.values()))
    reference = Counter(
        next(values for values in pressure_sets.values() if len(values) == maximum_count)
    )
    for values in pressure_sets.values():
        observed = Counter(values)
        if len(values) == maximum_count and observed != reference:
            raise ValueError("full pressure pools differ across crop labels")
        if len(values) == maximum_count - 1:
            if observed - reference or sum((reference - observed).values()) != 1:
                raise ValueError("reduced pressure pool is not a one-value subset")


def write_rows(rows: list[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_statistics(rows: list[dict[str, str]], destination: Path) -> None:
    by_crop: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_crop[row["crop"]].append(row)
    fields = ["crop", "samples"] + [f"{feature}_{stat}" for feature in NUMERIC_COLUMNS for stat in ("min", "max", "mean")]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for crop in sorted(by_crop):
            values = by_crop[crop]
            record: dict[str, str | int] = {"crop": crop, "samples": len(values)}
            for feature in NUMERIC_COLUMNS:
                numbers = [float(row[feature]) for row in values]
                record[f"{feature}_min"] = f"{min(numbers):.2f}"
                record[f"{feature}_max"] = f"{max(numbers):.2f}"
                record[f"{feature}_mean"] = f"{sum(numbers) / len(numbers):.2f}"
            writer.writerow(record)


def main() -> int:
    ml_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate documented synthetic crop sensor data")
    parser.add_argument("--profiles", type=Path, default=ml_root / "data" / "synthetic" / "crop_profiles.csv")
    parser.add_argument("--output", type=Path, default=ml_root / "data" / "synthetic" / "synthetic_crop_sensor_data.csv")
    parser.add_argument("--statistics", type=Path, default=ml_root / "data" / "synthetic" / "synthetic_crop_sensor_ranges.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    rows = generate_rows(profiles, seed=args.seed)
    validate_rows(rows, profiles)
    write_rows(rows, args.output)
    write_statistics(rows, args.statistics)
    counts = Counter(row["crop"] for row in rows)
    print(f"Generated {len(rows)} SYNTHETIC rows across {len(counts)} crops with seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
