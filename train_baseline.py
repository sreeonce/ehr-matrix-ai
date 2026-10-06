"""
Fri-Sat: train the baseline models and record AUROC / AUPRC.

Trains TWO baselines so you have a comparison point:
  1. Logistic regression (simple, interpretable, fast)
  2. XGBoost (usually stronger, handles missing/skewed data well)

Both are evaluated on val (tuning) and test (final, hospital B, look once).

Run: python train_baseline.py
Needs: pip install scikit-learn xgboost
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

TABLE = Path("modeling_table.pkl")
LABEL_COL = "label_24h"
ID_COLS = ["patient_id", "hospital_set", "ICULOS", "SepsisLabel", "label_24h", "split"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]

# Saturday ablation: run "python train_baseline.py --vitals-only" to drop
# the 5 labs and their _hours_since/_measured columns, and compare against
# Friday's full run.
VITALS_ONLY = "--vitals-only" in sys.argv


def load_splits():
    df = pd.read_pickle(TABLE)
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    if VITALS_ONLY:
        drop = set()
        for lab in LABS:
            drop.update({lab, f"{lab}_hours_since", f"{lab}_measured"})
        feature_cols = [c for c in feature_cols if c not in drop]
        print(f"[--vitals-only] Dropped {len(drop)} lab-related columns.")

    print(f"Using {len(feature_cols)} features.")

    splits = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        splits[name] = (part[feature_cols], part[LABEL_COL])
    return splits, feature_cols


def evaluate(name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict:
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    print(f"  {name:5s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
          f"(positive rate={y_true.mean():.3f})")
    return {"split": name, "AUROC": auroc, "AUPRC": auprc}


def run_logistic_regression(splits: dict) -> list[dict]:
    print("\n" + "=" * 60)
    print("BASELINE 1: Logistic Regression")
    print("=" * 60)

    X_train, y_train = splits["train"]
    # Logistic regression can't handle NaN natively (unlike XGBoost), so
    # median-impute any leftover gaps (e.g. static columns) before scaling.
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train)

    results = []
    for name in ("train", "val", "test"):
        X, y = splits[name]
        X_i = imputer.transform(X)
        X_s = scaler.transform(X_i)
        prob = model.predict_proba(X_s)[:, 1]
        results.append({**evaluate(name, y, prob), "model": "logreg"})
    return results


def run_xgboost(splits: dict) -> list[dict]:
    try:
        import xgboost as xgb
    except ImportError:
        print("\nxgboost not installed - skipping. Install with: pip install xgboost")
        return []

    print("\n" + "=" * 60)
    print("BASELINE 2: XGBoost")
    print("=" * 60)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # scale_pos_weight compensates for the ~5.7% positive rate
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    results = []
    for name in ("train", "val", "test"):
        X, y = splits[name]
        prob = model.predict_proba(X)[:, 1]
        results.append({**evaluate(name, y, prob), "model": "xgboost"})

    print("\nTop 10 feature importances:")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).head(10).to_string())

    return results


def main():
    splits, feature_cols = load_splits()

    all_results = []
    all_results += run_logistic_regression(splits)
    all_results += run_xgboost(splits)

    print("\n" + "=" * 60)
    print("SUMMARY (save this table for your case study)")
    print("=" * 60)
    summary = pd.DataFrame(all_results).pivot_table(
        index="model", columns="split", values=["AUROC", "AUPRC"]
    )
    print(summary.round(3).to_string())
    out_name = "baseline_results_vitals_only.csv" if VITALS_ONLY else "baseline_results.csv"
    summary.to_csv(out_name)
    print(f"\nSaved to {out_name}")


if __name__ == "__main__":
    main()
