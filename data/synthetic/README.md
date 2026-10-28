# Synthetic crop sensor dataset

## Status and purpose

`synthetic_crop_sensor_data.csv` is **SYNTHETIC**. It contains generated scenarios, not field observations. It exists because the available real dataset lacks `light` and `pressure` and covers only nine crop labels. Never present these rows as real measurements or use them as proof of agronomic accuracy.

Real data remains separate at `../processed/final_crops_sensor_subset.csv`. No real and synthetic rows are merged.

## Contents

- `crop_profiles.csv`: 52 curated crop profiles and source descriptors.
- `synthetic_crop_sensor_data.csv`: exactly 1,000 generated rows across all 52 crops.
- `synthetic_crop_sensor_ranges.csv`: observed per-crop min/max/mean validation report.

Generator: `../../src/data/generate_synthetic_crop_data.py`. Random seed: `20260909`.

Run from `ml/`:

```bash
python -m src.data.generate_synthetic_crop_data
```

## Features

| Feature | Unit | Meaning |
|---|---|---|
| `temperature` | °C | Ambient temperature sampled within FAO EcoCrop optimal temperature bounds. |
| `humidity` | % RH | Synthetic ambient relative-humidity scenario. Broad humid/intermediate/dry bands; not measured crop thresholds. |
| `soil_moisture` | calibrated relative % | Synthetic operational reading matching ESP32 output concept. Not volumetric water content. |
| `light` | lux | BH1750-compatible ambient illuminance. EcoCrop qualitative light classes mapped to conservative synthetic lux bands capped at 65,000 lux. |
| `pressure` | hPa | Realistic near-surface environmental measurement, 970–1040 hPa. Identical sampled distribution for every crop; not a crop requirement or primary suitability variable. |
| `crop` | label | Crop class. |

## Generation method

Temperature bounds and qualitative light classes come from FAO EcoCrop entries identified by scientific name. Triangular distributions emphasize each profile midpoint while remaining inside bounds. Humidity uses broad climate-scenario bands. Relative soil-moisture bands use low/intermediate/high water-demand scenarios informed by FAO crop-water guidance; they are design assumptions, not converted rainfall or VWC. Light-category mappings are synthetic: light shade 5,000–20,000 lux, cloudy skies 10,000–30,000 lux, clear skies 25,000–55,000 lux, very bright 40,000–65,000 lux.

Allocation is deterministic after sorting crop labels: 12 crops receive 20 rows and 40 crops receive 19 rows. The 20-row crops are `apple`, `banana`, `barley`, `black_gram`, `black_pepper`, `cabbage`, `cardamom`, `carrot`, `castor`, `cauliflower`, `chickpea`, and `chilli`. This gives exactly 1,000 rows without truncation.

Pressure uses one shared seeded 20-value pool centered on standard sea-level pressure. Every 20-row crop receives the complete pool. Each 19-row crop receives a rotated subset missing one pool value; omitted values rotate by crop. Pressure range and generating distribution remain crop-independent. Rows are shuffled deterministically. Generation rejects missing values, unknown labels, non-finite values, profile-range violations, duplicates, or invalid pressure pools.

## Sources

- FAO, ECOCROP database overview: https://www.fao.org/geospatial/data-and-tools/data-portals/ecocrop/en
- FAO ECOCROP application: https://ecocrop.apps.fao.org/ecocrop/srv/en/home
- OpenCLIM mirror of FAO EcoCrop database used to extract profile values: https://github.com/OpenCLIM/ecocrop/blob/main/EcoCrop_DB.csv
- OpenCLIM EcoCrop variable definitions and suitability method: https://github.com/OpenCLIM/ecocrop
- FAO, *Crop Water Needs*, including crop-specific seasonal water-demand guidance: https://www.fao.org/4/s2022e/s2022e02.htm
- FAO, *Guidelines for Predicting Crop Water Requirements*, humidity definitions: https://www.fao.org/4/f2430e/f2430e.pdf
- Tamil Nadu Agricultural University, relative humidity and crop growth: https://www.agritech.tnau.ac.in/agriculture/agri_agrometeorology_relativehumidity.html
- ROHM BH1750FVI datasheet mirror, illuminance output/range: https://dfimg.dfrobot.com/enshop/image/data/SEN0097/BH1750FVI.pdf
- NOAA/National Weather Service, normal surface pressure range: https://www.weather.gov/media/zhu/ZHU_Training_Page/winds/pressure_winds/Temp_Density_Pressure.pdf
- NOAA, standard sea-level pressure 1013.25 hPa: https://www.ndbc.noaa.gov/education/pressure.shtml

## Limitations

- Synthetic distributions encode profile assumptions. A model can learn those assumptions and appear accurate without demonstrating field performance.
- FAO EcoCrop temperatures describe crop suitability, not synchronized ESP32 observations.
- Humidity scenarios are broad; crop stage, location, season, disease pressure, and time of day are omitted.
- Relative soil sensor percentages depend on local wet/dry calibration, soil texture, salinity, installation depth, and sensor-to-sensor variation.
- Lux measures human-weighted illuminance, not photosynthetically active radiation. Canopy position, shade, clouds, time, and sensor saturation affect readings.
- Pressure is deliberately non-discriminative. Altitude correction is absent.
- Pests, nutrients, pH, rainfall, soil type, cultivar, planting season, and management are absent.
- Dataset must be externally validated against real, locally collected observations before production crop recommendations.
