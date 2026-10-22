import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import load_data, split_by_match

NUMERIC = ["distance", "angle"]
BASE = "Baseline (distance + angle)"

# model name -> (text columns to one-hot encode, 0/1 flag columns)
CONFIGS = {
    BASE: ([], []),
    "+ body part": (["shot_body_part"], []),
    "+ technique": (["shot_technique"], []),
    "+ play pattern": (["play_pattern"], []),
    "+ pressure": ([], ["under_pressure"]),
    "All four": (["shot_body_part", "shot_technique", "play_pattern"], ["under_pressure"]),
    "Three (no technique)": (["shot_body_part", "play_pattern"], ["under_pressure"]),
    "+ pass type": (["key_pass_type"], []),
    "All five": (["shot_body_part", "shot_technique", "play_pattern", "key_pass_type"], ["under_pressure"]),
}


def make_model(categorical, flags):
    parts = [("num", StandardScaler(), NUMERIC)]
    if categorical:
        encoder = OneHotEncoder(
            min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=False
        )
        parts.append(("cat", encoder, categorical))
    if flags:
        parts.append(("flag", "passthrough", flags))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=1000))


if __name__ == "__main__":
    shots = load_data()
    shots["under_pressure"] = shots["under_pressure"].astype(int)

    rows = []
    for seed in range(20):
        train, test = split_by_match(shots, seed=seed)
        y = test["is_goal"]
        for name, (categorical, flags) in CONFIGS.items():
            cols = NUMERIC + categorical + flags
            model = make_model(categorical, flags).fit(train[cols], train["is_goal"])
            p = model.predict_proba(test[cols])[:, 1]
            rows.append({"seed": seed, "model": name, "log_loss": log_loss(y, p)})

    ll = pd.DataFrame(rows).pivot(index="seed", columns="model", values="log_loss")
    gain = ll.rsub(ll[BASE], axis=0)  # baseline Log Loss minus this model's, per exam

    summary = pd.DataFrame(
        {
            "avg Log Loss": ll.mean(),
            "avg gain": gain.mean(),
            "exams improved": (gain > 0).sum(),
        }
    ).loc[list(CONFIGS)]
    print(summary.round(4))