import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import FEATURES, load_data, split_by_match, train_model

NUMERIC = ["distance", "angle"]
CATEGORICAL = ["shot_body_part", "shot_technique", "play_pattern", "key_pass_type"]
FLAGS = ["under_pressure"]
ALL_FEATURES = NUMERIC + CATEGORICAL + FLAGS


def build_model():
    """Pipeline: prepare each column type, then fit logistic regression."""
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    min_frequency=50,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
            ("flag", "passthrough", FLAGS),
        ]
    )
    return make_pipeline(preprocess, LogisticRegression(max_iter=1000))


if __name__ == "__main__":
    shots = load_data()
    shots["under_pressure"] = shots["under_pressure"].astype(int)
    train, test = split_by_match(shots)
    y_test = test["is_goal"]

    # Phase 1 model: distance + angle only
    p_base = train_model(train).predict_proba(test[FEATURES])[:, 1]

    # Phase 2 model: adds body part, technique, play pattern, pressure
    context = build_model().fit(train[ALL_FEATURES], train["is_goal"])
    p_ctx = context.predict_proba(test[ALL_FEATURES])[:, 1]

    results = pd.DataFrame(
        {
            "Log Loss": [log_loss(y_test, p_base), log_loss(y_test, p_ctx)],
            "ROC-AUC": [roc_auc_score(y_test, p_base), roc_auc_score(y_test, p_ctx)],
        },
        index=["Phase 1 (distance + angle)", "Phase 2 (+ context features)"],
    )
    print(results.round(4))