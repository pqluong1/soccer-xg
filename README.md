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

Logistic regression on distance and angle. Results are the mean ± standard
deviation over 20 random train/test splits, each grouped by match:

| Model | Log Loss | ROC-AUC |
|---|---|---|
| Naive (average goal rate) | 0.3005 ± 0.0183 | 0.500 ± 0.000 |
| Logistic regression (distance + angle) | 0.2702 ± 0.0166 | 0.731 ± 0.025 |
| StatsBomb xG (reference) | 0.2431 ± 0.0148 | 0.796 ± 0.018 |

The baseline beat the naive model on Log Loss in all 20 splits and closes
roughly half of the gap to StatsBomb's xG. StatsBomb's model also sees
defender and goalkeeper positions, so it is a ceiling to aim at, not a
like-for-like comparison. Each test set has only about 90 goals, so single-split
scores vary noticeably, which is why results are averaged across splits.

### What the metrics mean

- **ROC-AUC** measures how well the model ranks shots. Pick one shot that was a
  goal and one that wasn't: ROC-AUC is how often the model gave the goal the
  higher xG. 0.5 is a coin flip and 1.0 is perfect ranking.
- **Log Loss** measures how good the probabilities are. Each shot is penalized
  based on how much probability the model put on what actually happened, and
  confident mistakes are penalized the most. Lower is better.

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