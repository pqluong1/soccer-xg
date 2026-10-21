import pandas as pd

from extract import extract_shots


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to the shots we want to model and add the target column."""
    df = raw.copy()

    # Target: 1 if the shot was a goal, 0 for everything else (saved, blocked, off target...)
    df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)

    # Penalties are a different situation (fixed spot, no defenders), and shootout
    # kicks (period 5) are not normal chances, so both are excluded.
    df = df[(df["shot_type"] != "Penalty") & (df["period"] != 5)]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = extract_shots()
    raw["is_goal"] = (raw["shot_outcome"] == "Goal").astype(int)

    print("Raw shots:", len(raw))
    print("\nGoal rate by shot type:")
    print(raw.groupby("shot_type")["is_goal"].agg(["count", "mean"]).round(3))
    print("\nShots by period:")
    print(raw["period"].value_counts().sort_index())

    shots = prepare_dataset(raw)
    print("\nAfter cleaning:", len(shots), "shots")
    print("Overall goal rate:", round(shots["is_goal"].mean(), 3))