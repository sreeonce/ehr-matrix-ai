"""
Thursday sanity checks on modeling_table.pkl (checks 1-3 from the plan).

Run: python sanity_checks.py
"""
from pathlib import Path

import pandas as pd

TABLE = Path("modeling_table.pkl")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]

df = pd.read_pickle(TABLE)

# ---------------------------------------------------------------
# Check 1: label_24h looks FORWARD, never backward
# ---------------------------------------------------------------
print("=" * 60)
print("CHECK 1: label_24h vs SepsisLabel, one septic patient")
print("=" * 60)

# pick a patient who is septic at some point but NOT from the first hour
# (so we can see the label switch on ahead of the real onset)
ever_septic = df.groupby("patient_id")["SepsisLabel"].max()
starts_healthy = df.groupby("patient_id")["SepsisLabel"].first() == 0
candidates = ever_septic[(ever_septic == 1) & starts_healthy].index

if len(candidates) == 0:
    # fallback: any patient who is ever septic, even if septic from hour 1
    candidates = ever_septic[ever_septic == 1].index
if len(candidates) == 0:
    raise SystemExit("No septic patients found in this table - check SepsisLabel.")

pid = candidates[0]
one = df.loc[df["patient_id"] == pid, ["ICULOS", "SepsisLabel", "label_24h"]]
one = one.sort_values("ICULOS")

first_true_onset = one.loc[one["SepsisLabel"] == 1, "ICULOS"].min()
first_label_on = one.loc[one["label_24h"] == 1, "ICULOS"].min()

print(f"Patient {pid}")
print(one.to_string(index=False))
print(f"\nSepsisLabel first turns 1 at hour {first_true_onset}")
print(f"label_24h first turns 1 at hour {first_label_on}")
gap = first_true_onset - first_label_on
print(f"label_24h switches on {gap} hour(s) BEFORE the real onset.")
if 0 < gap <= 24:
    print("PASS: label looks forward by <=24h, as intended.")
elif gap == 0:
    print("CHECK: label turned on at the same hour as onset - only ok if "
          "onset was already <24h from admission.")
else:
    print("INVESTIGATE: gap is outside 0-24h - check the rolling window logic.")

# ---------------------------------------------------------------
# Check 2: hours_since resets to 0 on a real reading, climbs between
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 2: hours_since behaviour for one vital, one patient")
print("=" * 60)

vital = "Temp"  # sparsest vital - easiest to see gaps
pid2 = df["patient_id"].iloc[0]
raw = pd.read_pickle("sepsis_all.pkl")
raw_one = raw.loc[raw["patient_id"] == pid2, ["ICULOS", vital]].sort_values("ICULOS")
filled_one = df.loc[df["patient_id"] == pid2,
                     ["ICULOS", vital, f"{vital}_hours_since"]].sort_values("ICULOS")

merged = raw_one.merge(filled_one, on="ICULOS", suffixes=("_raw", "_filled"))
print(f"Patient {pid2}, vital = {vital}")
print(merged.head(20).to_string(index=False))
print("\nExpect: hours_since = 0 exactly where <vital>_raw is not NaN, "
      "and increases by 1 each hour it stays NaN.")

# ---------------------------------------------------------------
# Check 3: train (set A) vs test (set B) distribution shift
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 3: feature means, train vs test")
print("=" * 60)

cols = VITALS + LABS
summary = (
    df.groupby("split")[cols]
    .mean()
    .T.rename(columns={"train": "train_mean", "test": "test_mean", "val": "val_mean"})
)
summary["pct_diff_train_vs_test"] = (
    (summary["test_mean"] - summary["train_mean"]) / summary["train_mean"] * 100
).round(1)
print(summary.round(2).to_string())
print("\nFlag any row where pct_diff is large (say >15-20%) - note it for "
      "the case study as a train/test hospital difference.")