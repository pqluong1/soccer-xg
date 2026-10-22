import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from baseline import FEATURES, load_data, split_by_match, train_model
from context_model import ALL_FEATURES, build_model

if __name__ == "__main__":
    shots = load_data()
    shots["under_pressure"] = shots["under_pressure"].astype(int)

    rows = []
    for seed in range(20):
        train, test = split_by_match(shots, seed=seed)
        y = test["is_goal"]

        base = train_model(train)
        p_base = base.predict_proba(test[FEATURES])[:, 1]

        ctx = build_model().fit(train[ALL_FEATURES], train["is_goal"])
        p_ctx = ctx.predict_proba(test[ALL_FEATURES])[:, 1]

        rows.append(
            {
                "ll_base": log_loss(y, p_base),
                "ll_ctx": log_loss(y, p_ctx),
                "auc_base": roc_auc_score(y, p_base),
                "auc_ctx": roc_auc_score(y, p_ctx),
            }
        )

    r = pd.DataFrame(rows)
    print(r.mean().round(4))
    print()
    print(r.std().round(4))
    print()
    print("Exams where Log Loss improved:", int((r["ll_ctx"] < r["ll_base"]).sum()), "of 20")
    print("Exams where ROC-AUC improved: ", int((r["auc_ctx"] > r["auc_base"]).sum()), "of 20")