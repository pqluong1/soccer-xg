import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score

from baseline import FEATURES, load_data, split_by_match, train_model

LAZY = "Lazy (average rate)"
LOGIT = "Logistic (distance + angle)"
SB = "StatsBomb xG (reference)"

if __name__ == "__main__":
    shots = load_data()
    rows = []

    for seed in range(20):
        train, test = split_by_match(shots, seed=seed)
        y = test["is_goal"]

        lazy = DummyClassifier(strategy="prior").fit(train[FEATURES], train["is_goal"])
        preds = {
            LAZY: lazy.predict_proba(test[FEATURES])[:, 1],
            LOGIT: train_model(train).predict_proba(test[FEATURES])[:, 1],
            SB: test["shot_statsbomb_xg"],
        }

        for name, p in preds.items():
            rows.append(
                {"seed": seed, "model": name,
                 "log_loss": log_loss(y, p), "roc_auc": roc_auc_score(y, p)}
            )

    results = pd.DataFrame(rows)

    summary = results.groupby("model")[["log_loss", "roc_auc"]].agg(["mean", "std"])
    print(summary.round(4))

    # Same-split comparison: how much does our model improve on the lazy one?
    ll = results.pivot(index="seed", columns="model", values="log_loss")
    gain = ll[LAZY] - ll[LOGIT]
    print("\nLog Loss improvement over lazy, per split:")
    print("  mean:", round(gain.mean(), 4), "| min:", round(gain.min(), 4))
    print("  splits where our model beat lazy:", int((gain > 0).sum()), "of 20")