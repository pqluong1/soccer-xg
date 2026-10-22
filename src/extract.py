from pathlib import Path
import warnings

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

DATA_DIR = Path("data")
SHOTS_CACHE = DATA_DIR / "shots_v2.parquet"

# (competition_id, season_id, label), the IDs you found earlier
TARGETS = [
    (43, 106, "World Cup 2022"),
    (43, 3, "World Cup 2018"),
    (55, 282, "Euro 2024"),
    (55, 43, "Euro 2020"),
]

SHOT_COLS = [
    "id", "period", "minute", "second", "team", "player", "play_pattern",
    "location", "under_pressure", "shot_outcome", "shot_type",
    "shot_body_part", "shot_technique", "shot_statsbomb_xg", "shot_key_pass_id",
]

PASS_COLS = ["id", "pass_type", "pass_cross", "pass_through_ball", "pass_cut_back"]


def classify_passes(passes: pd.DataFrame) -> pd.Series:
    """Give every pass in a match a simple category, indexed by pass ID."""
    p = passes.reindex(columns=PASS_COLS)
    category = np.select(
        [
            p["pass_type"].isin(["Corner", "Free Kick", "Throw-in"]),
            p["pass_through_ball"].eq(True),
            p["pass_cut_back"].eq(True),
            p["pass_cross"].eq(True),
        ],
        ["Set piece pass", "Through ball", "Cut-back", "Cross"],
        default="Other pass",
    )
    return pd.Series(category, index=p["id"])

def extract_shots(max_matches=None):
    """Download all shots for the target tournaments.

    max_matches limits matches per tournament (for quick tests).
    Full runs are cached to disk; test runs are not.
    """
    use_cache = max_matches is None
    if use_cache and SHOTS_CACHE.exists():
        print("Loading cached shots...")
        return pd.read_parquet(SHOTS_CACHE)

    frames = []
    for comp_id, season_id, label in TARGETS:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        match_ids = matches["match_id"].tolist()[:max_matches]

        for match_id in tqdm(match_ids, desc=label):
            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].reindex(columns=SHOT_COLS).copy()
            passes = events[events["type"] == "Pass"]
            lookup = classify_passes(passes)
            shots["key_pass_type"] = shots["shot_key_pass_id"].map(lookup).fillna("No pass")
            shots["match_id"] = match_id
            shots["competition"] = label
            frames.append(shots)

    df = pd.concat(frames, ignore_index=True)

    # Split the [x, y] list into two numeric columns
    df["x"] = df["location"].apply(lambda loc: loc[0])
    df["y"] = df["location"].apply(lambda loc: loc[1])
    df = df.drop(columns="location")

    # under_pressure only appears (as True) when it applies; missing means no pressure
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(bool)

    if use_cache:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_parquet(SHOTS_CACHE, index=False)
    return df


if __name__ == "__main__":
    df = extract_shots()
    print(df.shape)
    print(df.groupby("key_pass_type").size().sort_values(ascending=False))