import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from extract import extract_shots
from prepare import prepare_dataset
from features import add_geometry_features

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score

FEATURES = ["distance", "angle"]
SEED = 42


def load_data() -> pd.DataFrame:
    """Download (or load cached) shots, clean them, and add geometry features."""
    return add_geometry_features(prepare_dataset(extract_shots()))


def split_by_match(df: pd.DataFrame, test_size=0.2, seed=SEED):
    """Split into train/test so that all shots from a match land on the same side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]

def train_model(train: pd.DataFrame):
    """Fit a logistic regression on the training shots."""
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(train[FEATURES], train["is_goal"])
    return model

if __name__ == "__main__":
    shots = load_data()
    train, test = split_by_match(shots)
    y_test = test["is_goal"]

    # 1. Lazy model: predicts the training goal rate for every shot
    lazy = DummyClassifier(strategy="prior").fit(train[FEATURES], train["is_goal"])
    p_lazy = lazy.predict_proba(test[FEATURES])[:, 1]

    # 2. Our logistic regression
    model = train_model(train)
    p_model = model.predict_proba(test[FEATURES])[:, 1]

    # 3. StatsBomb's own xG (reference only)
    p_sb = test["shot_statsbomb_xg"]

    results = pd.DataFrame(
        {
            "Log Loss": [log_loss(y_test, p) for p in (p_lazy, p_model, p_sb)],
            "ROC-AUC": [roc_auc_score(y_test, p) for p in (p_lazy, p_model, p_sb)],
        },
        index=["Lazy (average rate)", "Logistic (distance + angle)", "StatsBomb xG (reference)"],
    )
    print(results.round(4))