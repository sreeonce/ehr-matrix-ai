from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("physionet-sepsis-2019")
CACHE = Path("sepsis_all.pkl")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]


def load_all(data_dir: Path) -> pd.DataFrame:
    frames = []
    files = sorted(data_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"No .psv files found under {data_dir.resolve()}")
    for f in files:
        df = pd.read_csv(f, sep="|")
        df.insert(0, "patient_id", f.stem)
        df.insert(1, "hospital_set", f.parent.name[-1].upper())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if CACHE.exists():
    data = pd.read_pickle(CACHE)
    print(f"Loaded cached table from {CACHE}")
else:
    data = load_all(DATA_DIR)
    data.to_pickle(CACHE)
    print(f"Read files and cached to {CACHE}")

print("\n=== Size ===")
print("Patient-hours (rows):", len(data))
print(data.groupby("hospital_set")["patient_id"].nunique().rename("patients"))

stay_len = data.groupby("patient_id")["ICULOS"].max()
print("\n=== ICU stay length (hours) ===")
print(stay_len.describe())

per_patient = data.groupby(["patient_id", "hospital_set"])["SepsisLabel"].max()
per_patient = per_patient.reset_index()
print("\n=== Patients who ever have SepsisLabel = 1 ===")
print(per_patient.groupby("hospital_set")["SepsisLabel"].mean().round(3))
print("Fraction of all patient-hours labelled 1:",
      round(data["SepsisLabel"].mean(), 4))

print("\n=== Fraction missing per column (highest first) ===")
print(data.drop(columns=["patient_id", "hospital_set"])
      .isna().mean().sort_values(ascending=False).round(2).to_string())

sepsis_ids = per_patient.loc[per_patient["SepsisLabel"] == 1, "patient_id"]
pid = sepsis_ids.iloc[0]
one = data[data["patient_id"] == pid].set_index("ICULOS")
first_pos = one.index[one["SepsisLabel"] == 1][0]

cols = [c for c in VITALS if c in one.columns]
one[cols].plot(subplots=True, figsize=(9, 9), marker=".", title=f"Patient {pid}")
for ax in plt.gcf().axes:
    ax.axvline(first_pos, color="red", linestyle="--")
plt.suptitle(f"{pid}: red line = first hour labelled sepsis", y=0.99)
plt.tight_layout()
plt.show()