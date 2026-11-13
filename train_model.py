"""Train and compare models that predict: will she understand this session?

Reads sessions.csv (made by make_dataset.py) and prints a results report.
It also saves the chosen model as ml_model.json, so the app can use it
without needing scikit-learn.

Run it with:  python3 train_model.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_dataset import FEATURES

DATA_FILE = "sessions.csv"
MODEL_FILE = "ml_model.json"
RESULTS_FILE = "ml_results.txt"

lines = []


def out(text=""):
    print(text)
    lines.append(text)


def scores(y, p):
    """Four numbers that describe how good the predicted chances are."""
    return {
        "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def print_table(title, results):
    out(title)
    out(f"{'Model':<34}{'Accuracy':>10}{'AUC':>8}{'Brier':>8}{'LogLoss':>9}")
    for name, s in results.items():
        out(f"{name:<34}{s['accuracy']:>10.3f}{s['auc']:>8.3f}{s['brier']:>8.3f}{s['logloss']:>9.3f}")
    out()


def main():
    data = pd.read_csv(DATA_FILE)
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    shifted = data[data["split"] == "shifted"]

    x_train, y_train = train[FEATURES], train["understood"]

    out("DATA")
    for name, part in (("train", train), ("test", test), ("shifted", shifted)):
        out(f"  {name:<8}{part['girl_id'].nunique():>5} girls {len(part):>7} sessions "
            f"{part['understood'].mean():>7.1%} understood")
    out()

    # ----- Baselines -----
    base_rate = y_train.mean()
    by_weakness = train.groupby("weakness")["understood"].mean()

    def constant(part):
        return np.full(len(part), base_rate)

    def weakness_only(part):
        return part["weakness"].map(by_weakness).fillna(base_rate).to_numpy()

    # ----- Models -----
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                    random_state=0, n_jobs=-1)

    # Cross-validation on the training girls. GroupKFold keeps every girl
    # entirely inside one fold, so no girl is seen in both training and testing.
    folds = GroupKFold(n_splits=5)
    out("CROSS-VALIDATION ON TRAINING GIRLS (5 folds, split by girl, AUC)")
    for name, mdl in (("Logistic regression", logistic), ("Random forest", forest)):
        cv = cross_val_score(mdl, x_train, y_train, groups=train["girl_id"],
                             cv=folds, scoring="roc_auc")
        out(f"  {name:<22}{cv.mean():.3f} +/- {cv.std():.3f}")
    out()

    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    for title, part in (("RESULTS ON NEW GIRLS, SAME KIND OF WORLD (test)", test),
                        ("RESULTS ON NEW GIRLS, DIFFERENT WORLD (shifted)", shifted)):
        y = part["understood"]
        results = {
            "Always the average rate": scores(y, constant(part)),
            "By weakness level only": scores(y, weakness_only(part)),
            "Logistic regression": scores(y, logistic.predict_proba(part[FEATURES])[:, 1]),
            "Random forest": scores(y, forest.predict_proba(part[FEATURES])[:, 1]),
        }
        print_table(title, results)

    # ----- Calibration: does "70 percent" really mean about 70 percent? -----
    p = logistic.predict_proba(test[FEATURES])[:, 1]
    bins = pd.cut(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    table = (pd.DataFrame({"bin": bins, "pred": p, "real": test["understood"].to_numpy()})
             .groupby("bin", observed=True).agg(sessions=("real", "size"),
                                                predicted=("pred", "mean"),
                                                real=("real", "mean")))
    out("CALIBRATION OF THE LOGISTIC MODEL (test girls)")
    out(f"{'Predicted range':<18}{'Sessions':>10}{'Predicted':>11}{'Real':>8}")
    for interval, row in table.iterrows():
        out(f"{str(interval):<18}{int(row['sessions']):>10}{row['predicted']:>11.2f}{row['real']:>8.2f}")
    out()

    # ----- What does the model use? -----
    scaler = logistic.named_steps["standardscaler"]
    lr = logistic.named_steps["logisticregression"]
    out("WHAT THE LOGISTIC MODEL LEARNED (features scaled to the same size;")
    out("positive = more likely to understand, negative = less likely)")
    for name, coef in sorted(zip(FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1])):
        out(f"  {name:<18}{coef:>+8.3f}")
    out()
    out("RANDOM FOREST FEATURE IMPORTANCE (share of the decisions, adds up to 1)")
    for name, imp in sorted(zip(FEATURES, forest.feature_importances_), key=lambda t: -t[1]):
        out(f"  {name:<18}{imp:>8.3f}")
    out()

    # ----- Save the logistic model as plain numbers -----
    saved = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "trained_on": "synthetic sessions from make_dataset.py",
    }
    with open(MODEL_FILE, "w") as f:
        json.dump(saved, f, indent=2)
    out(f"Saved the logistic model to {MODEL_FILE}")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()