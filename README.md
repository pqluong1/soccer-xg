# Soccer Expected Goals (xG) Model

An end-to-end expected goals project built on StatsBomb open data. Phase 1 is a
baseline model that estimates the probability a shot becomes a goal using only
its geometry.

## Data

- StatsBomb open data: FIFA World Cup 2018 and 2022, UEFA Euro 2020 and 2024
- 230 matches, 5,606 shots after excluding penalties and shootout kicks
- Train/test split is **grouped by match**, so no match appears in both sets

## Features

- `distance`: straight-line distance from the shot to the center of the goal
- `angle`: the angle the goal posts make as seen from the shot location

## Model and results (Phase 1)

Logistic regression on distance and angle, evaluated on held-out matches:

| Model | Log Loss | ROC-AUC |
|---|---|---|
| Naive (average goal rate) | 0.2930 | 0.5000 |
| Logistic regression (distance + angle) | 0.2567 | 0.7595 |
| StatsBomb xG (reference) | 0.2227 | 0.8440 |

StatsBomb's xG also uses defender and goalkeeper positions, so it is a ceiling
to aim at, not a like-for-like comparison. The test set has only about 90 goals,
so small differences in these numbers are within noise.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/baseline.py
```

The first run downloads the data (a minute or two) and caches it in `data/`.

## Roadmap

- [x] Phase 1: geometry features and baseline model
- [ ] Phase 2: context features and probability calibration
- [ ] Phase 3: pitch maps and SHAP explanations
- [ ] Phase 4: FastAPI endpoint, Streamlit app, Docker