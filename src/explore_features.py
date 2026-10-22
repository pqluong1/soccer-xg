from baseline import load_data

shots = load_data()

for col in ["key_pass_type"]:
    print(f"\n{col}")
    summary = shots.groupby(col)["is_goal"].agg(["count", "mean"])
    print(summary.sort_values("count", ascending=False).round(3))