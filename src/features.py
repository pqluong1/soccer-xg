import numpy as np
import pandas as pd

from extract import extract_shots
from prepare import prepare_dataset

# Goal geometry (StatsBomb pitch is 120 x 80; goal center at x=120, y=40; posts 8 yards apart)
GOAL_X, GOAL_Y = 120, 40
GOAL_WIDTH = 8


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance (yards) and angle (radians) to goal for every shot."""
    df = df.copy()

    dx = GOAL_X - df["x"]             # how far the shot is from the goal line
    dy = (df["y"] - GOAL_Y).abs()     # how far off-center it is

    df["distance"] = np.sqrt(dx**2 + dy**2)
    df["angle"] = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    return df


if __name__ == "__main__":
    shots = add_geometry_features(prepare_dataset(extract_shots()))

    print(shots[["distance", "angle"]].describe().round(2))

    # Sanity check: goal rate should fall as distance grows
    dist_bins = pd.cut(shots["distance"], [0, 6, 12, 18, 25, np.inf])
    print("\nGoal rate by distance (yards):")
    print(shots.groupby(dist_bins, observed=True)["is_goal"].agg(["count", "mean"]).round(3))

    # ...and rise as the angle gets wider
    angle_bins = pd.cut(np.degrees(shots["angle"]), [0, 10, 20, 30, 45, 180])
    print("\nGoal rate by angle (degrees):")
    print(shots.groupby(angle_bins, observed=True)["is_goal"].agg(["count", "mean"]).round(3))