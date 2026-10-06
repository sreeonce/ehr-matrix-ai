"""
Wed-Thu: turn sepsis_all.pkl into a modelling-ready table.

Steps:
  1. Load the raw patient-hour table (from build_cache.py).
  2. Keep only the chosen input columns (7 vitals, 3 static, 5 labs).
  3. Per patient: forward-fill vitals and labs; add "hours since last
     measured" for both, and a "was measured this hour" flag for labs.
  4. Define the label: will this patient be septic at any point in the
     NEXT 24 hours? (forward-looking, so it's an early-warning target,
     not "is septic right now").
  5. Split by PATIENT, not by row: train/val from hospital set A,
     held-out test = hospital set B (tests generalisation to a new
     hospital).

Run: python build_modeling_table.py
Output: modeling_table.pkl  (one row per patient-hour, ready for the
        baseline script)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CACHE = Path("sepsis_all.pkl")
OUT_PATH = Path("modeling_table.pkl")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
STATIC = ["Age", "Gender", "HospAdmTime"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]
LABEL_WINDOW_HOURS = 24


def add_ffill_and_gap(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    """For each column: forward-fill within patient, and add
    '<col>_hours_since' = hours since the last real (non-NaN) reading.
    Rows before the first reading get filled with the column's overall
    median and hours_since = a large sentinel (999)."""
    df = df.sort_values([group, "ICULOS"]).copy()
    for col in cols:
        had_value = df[col].notna()
        # hours since last real reading, per patient
        last_seen_hour = df["ICULOS"].where(had_value)
        last_seen_hour = last_seen_hour.groupby(df[group]).ffill()
        gap = df["ICULOS"] - last_seen_hour
        df[f"{col}_hours_since"] = gap.fillna(999).clip(upper=999)

        filled = df.groupby(group)[col].ffill()
        median = df[col].median()
        df[col] = filled.fillna(median)
    return df


def add_measured_flags(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    """Add a 0/1 '<col>_measured' flag BEFORE forward-filling overwrites
    the missingness. Call this first."""
    for col in cols:
        df[f"{col}_measured"] = df[col].notna().astype("int8")
    return df


def make_label(df: pd.DataFrame, group: str, window: int) -> pd.Series:
    """1 if SepsisLabel is 1 at any point in the next `window` hours
    (looking forward from this row), else 0. This is what the model
    predicts: sepsis risk in the near future, not right now."""
    def per_patient(s: pd.Series) -> pd.Series:
        # reverse, rolling max, reverse back = "max over next `window` rows"
        return s[::-1].rolling(window=window, min_periods=1).max()[::-1]

    return (
        df.groupby(group)["SepsisLabel"]
        .apply(per_patient)
        .reset_index(level=0, drop=True)
        .astype("int8")
    )


def split_by_patient(df: pd.DataFrame, group: str, seed: int = 0):
    """Train/val from hospital set A (80/20 by patient), test = all of
    set B. Stratify the A split so both halves have a similar fraction
    of ever-septic patients."""
    rng = np.random.default_rng(seed)

    a_patients = (
        df.loc[df["hospital_set"] == "A", [group]]
        .drop_duplicates()
        .merge(
            df.groupby(group)["SepsisLabel"].max().rename("ever_septic"),
            on=group,
        )
    )

    train_ids, val_ids = [], []
    for label_val in (0, 1):
        ids = a_patients.loc[a_patients["ever_septic"] == label_val, group].to_numpy()
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_ids.extend(ids[:cut])
        val_ids.extend(ids[cut:])

    split = pd.Series("test", index=df[group].unique())  # default: set B patients
    split.loc[train_ids] = "train"
    split.loc[val_ids] = "val"
    return df[group].map(split)


def fill_static(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    """Static columns (Age, Gender, HospAdmTime) are recorded once per
    patient, not every hour, so most rows are NaN. Forward-fill AND
    back-fill within each patient to spread the one real value across
    all their rows; if a patient has no value at all, fall back to the
    overall median."""
    df = df.copy()
    for col in cols:
        filled = df.groupby(group)[col].transform(lambda s: s.ffill().bfill())
        df[col] = filled.fillna(df[col].median())
    return df


def main():
    print(f"Loading {RAW_CACHE} ...")
    raw = pd.read_pickle(RAW_CACHE)

    keep = ["patient_id", "hospital_set", "ICULOS", "SepsisLabel"] + VITALS + STATIC + LABS
    df = raw[keep].copy()

    print("Filling static columns (Age, Gender, HospAdmTime) per patient ...")
    df = fill_static(df, STATIC, group="patient_id")

    print("Adding measured flags for labs ...")
    df = add_measured_flags(df, LABS, group="patient_id")

    print("Forward-filling vitals + adding hours-since ...")
    df = add_ffill_and_gap(df, VITALS, group="patient_id")

    print("Forward-filling labs + adding hours-since ...")
    df = add_ffill_and_gap(df, LABS, group="patient_id")

    print(f"Building the {LABEL_WINDOW_HOURS}h-ahead label ...")
    df["label_24h"] = make_label(df, group="patient_id", window=LABEL_WINDOW_HOURS)

    print("Splitting by patient (train/val = set A, test = set B) ...")
    df["split"] = split_by_patient(df, group="patient_id")

    print("\n=== Rows per split ===")
    print(df["split"].value_counts())
    print("\n=== Patients per split ===")
    print(df.groupby("split")["patient_id"].nunique())
    print("\n=== Label rate (fraction of hours where label_24h=1) per split ===")
    print(df.groupby("split")["label_24h"].mean().round(4))

    df.to_pickle(OUT_PATH)
    print(f"\nSaved {len(df)} rows, {df.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()