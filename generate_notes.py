"""
Mon (Week 2): generate synthetic clinical notes from each patient's own
vitals and labs.

Design rules (these matter for keeping the project honest):
  1. A note written at ICU hour T uses ONLY data at or before hour T.
     (Values are already forward-filled, so "value at T" = last reading
     at or before T. Trends compare T with T-6.)
  2. Notes NEVER use SepsisLabel / label_24h, and never contain the word
     "sepsis". They only describe what the numbers show.
  3. A vital or lab is only mentioned if it was measured recently
     (vitals: within 6h, labs: within 12h), like a real note that only
     comments on what was actually checked.
  4. Wording varies between notes (seeded, so results are reproducible)
     so the text model sees realistic variety, not one fixed sentence.

Because every note is a function of the same numbers the LSTM/XGBoost
already see, the text CANNOT add new information. It exists to prove the
text -> ClinicalBERT -> fusion pipeline works end to end. Say this
plainly in your case study.

Run:  python generate_notes.py
Input:  modeling_table.pkl
Output: notes.pkl  (one row per note: patient_id, ICULOS, split, note_text, ...)
"""
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path("modeling_table.pkl")
OUT = Path("notes.pkl")

NOTE_EVERY = 6        # a note every 6 ICU hours (like a shift-based check-in)
VITAL_RECENT = 6      # only mention a vital if measured within this many hours
LAB_RECENT = 12       # only mention a lab if measured within this many hours
TREND_WINDOW = 6      # compare now vs this many hours ago
SEED = 0

TREND_VITALS = ["HR", "MAP", "Resp", "O2Sat"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]


def pick(rng, options):
    return options[int(rng.integers(len(options)))]


# ---------------------------------------------------------------
# Building blocks: each returns (sentence_fragment, is_abnormal)
# ---------------------------------------------------------------
def heart_rate(r, rng):
    if r["HR_hours_since"] > VITAL_RECENT:
        return None
    hr = round(r["HR"])
    prev = r["HR_prev"]
    if hr >= 100:
        text, abn = pick(rng, [
            f"tachycardic with HR {hr:.0f}",
            f"heart rate elevated at {hr:.0f} bpm",
            f"persistent tachycardia, HR {hr:.0f}",
        ]), True
    elif hr < 60:
        text, abn = pick(rng, [
            f"bradycardic with HR {hr:.0f}",
            f"heart rate low at {hr:.0f} bpm",
        ]), True
    else:
        text, abn = pick(rng, [
            f"HR {hr:.0f}, regular",
            f"heart rate stable at {hr:.0f} bpm",
        ]), False
    if not np.isnan(prev):
        if hr - prev >= 15:
            text += f", trending up from {prev:.0f} six hours ago"
        elif prev - hr >= 15:
            text += f", down from {prev:.0f} six hours ago"
    return text, abn


def resp_rate(r, rng):
    if r["Resp_hours_since"] > VITAL_RECENT:
        return None
    rr = round(r["Resp"])
    if rr >= 22:
        text, abn = pick(rng, [
            f"tachypneic, RR {rr:.0f}",
            f"breathing fast at {rr:.0f}/min",
            f"increased work of breathing, RR {rr:.0f}",
        ]), True
    else:
        text, abn = pick(rng, [
            f"RR {rr:.0f}, comfortable",
            f"respiratory rate normal at {rr:.0f}",
        ]), False
    prev = r["Resp_prev"]
    if abn and not np.isnan(prev) and rr - prev >= 6:
        text += f" (up from {prev:.0f})"
    return text, abn


def oxygen(r, rng):
    if r["O2Sat_hours_since"] > VITAL_RECENT:
        return None
    s = round(r["O2Sat"])
    if s < 92:
        text, abn = pick(rng, [
            f"hypoxic, SpO2 {s:.0f}%",
            f"SpO2 low at {s:.0f}%",
            f"oxygen saturation dropping, {s:.0f}%",
        ]), True
    elif s < 95:
        text, abn = pick(rng, [
            f"SpO2 borderline at {s:.0f}%",
            f"saturating {s:.0f}%",
        ]), False
    else:
        text, abn = pick(rng, [
            f"SpO2 {s:.0f}%",
            f"saturating well at {s:.0f}%",
        ]), False
    return text, abn


def temperature(r, rng):
    if r["Temp_hours_since"] > VITAL_RECENT:
        return None
    t = r["Temp"]
    if t >= 38.0:
        return pick(rng, [
            f"febrile, T {t:.1f}C",
            f"temperature raised at {t:.1f}C",
        ]), True
    if t < 36.0:
        return pick(rng, [
            f"hypothermic, T {t:.1f}C",
            f"temperature low at {t:.1f}C",
        ]), True
    return pick(rng, [f"afebrile, T {t:.1f}C", f"T {t:.1f}C, no fever"]), False


def blood_pressure(r, rng):
    map_ok = r["MAP_hours_since"] <= VITAL_RECENT
    sbp_ok = r["SBP_hours_since"] <= VITAL_RECENT
    if map_ok and round(r["MAP"]) < 65:
        m = round(r["MAP"])
        text = pick(rng, [
            f"hypotensive, MAP {m:.0f}",
            f"blood pressure low, MAP {m:.0f}",
        ])
        prev = r["MAP_prev"]
        if not np.isnan(prev) and prev - m >= 10:
            text += f" (down from {prev:.0f})"
        return text, True
    if sbp_ok and round(r["SBP"]) < 100:
        return pick(rng, [
            f"soft blood pressure, SBP {r['SBP']:.0f}",
            f"SBP low at {r['SBP']:.0f}",
        ]), True
    if map_ok or sbp_ok:
        return pick(rng, ["BP stable", "blood pressure within range"]), False
    return None


def lab_findings(r, rng):
    """Returns (list of abnormal lab phrases, n_measured_recently)."""
    abnormal, measured = [], 0
    for lab in LABS:
        if r[f"{lab}_hours_since"] > LAB_RECENT:
            continue
        measured += 1
        v = r[lab]
        if lab == "Lactate" and v >= 2.0:
            abnormal.append(pick(rng, [f"lactate elevated at {v:.1f}", f"raised lactate {v:.1f}"]))
        elif lab == "WBC" and v > 12:
            abnormal.append(f"WBC raised at {v:.1f}")
        elif lab == "WBC" and v < 4:
            abnormal.append(f"WBC low at {v:.1f}")
        elif lab == "Creatinine" and v >= 1.5:
            abnormal.append(f"creatinine elevated at {v:.1f}")
        elif lab == "Platelets" and v < 150:
            abnormal.append(f"platelets low at {v:.0f}")
        elif lab == "Glucose" and v > 180:
            abnormal.append(f"glucose high at {v:.0f}")
        elif lab == "Glucose" and v < 70:
            abnormal.append(f"glucose low at {v:.0f}")
    return abnormal, measured


# ---------------------------------------------------------------
# Assemble one note
# ---------------------------------------------------------------
def make_note(r, rng):
    age = int(round(r["Age"]))
    sex = "male" if r["Gender"] >= 0.5 else "female"  # dataset: 0=female, 1=male
    if r["is_first"]:
        opening = f"Admission note, ICU hour {int(r['ICULOS'])}. {age}-year-old {sex}."
    else:
        opening = f"ICU hour {int(r['ICULOS'])}. {age}-year-old {sex}."

    n_abn = 0
    frags = []
    for fn in (heart_rate, blood_pressure, resp_rate, oxygen, temperature):
        out = fn(r, rng)
        if out is None:
            continue
        text, abn = out
        frags.append(text)
        n_abn += int(abn)

    if frags:
        vitals_sentence = "Vitals: " + "; ".join(frags) + "."
    else:
        vitals_sentence = "Vitals not recently recorded."

    abnormal_labs, n_measured = lab_findings(r, rng)
    n_abn += len(abnormal_labs)
    if abnormal_labs:
        lab_sentence = "Labs: " + "; ".join(abnormal_labs) + "."
    elif n_measured:
        lab_sentence = pick(rng, ["Recent labs unremarkable.", "Labs within acceptable limits."])
    else:
        lab_sentence = "No recent labs available."

    if n_abn == 0:
        impression = pick(rng, [
            "Patient appears stable. Continue routine monitoring.",
            "No acute concerns at this time. Continue current plan.",
        ])
    elif n_abn == 1:
        impression = pick(rng, [
            "Mild abnormality noted. Continue to monitor closely.",
            "One parameter out of range. Recheck in a few hours.",
        ])
    else:
        impression = pick(rng, [
            "Multiple abnormal findings. Concern for clinical deterioration; recommend urgent review.",
            "Several parameters out of range. Escalate and reassess promptly.",
        ])

    return f"{opening} {vitals_sentence} {lab_sentence} {impression}", n_abn


# ---------------------------------------------------------------
def main():
    print(f"Loading {TABLE} ...")
    df = pd.read_pickle(TABLE).sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)

    contiguous = (df.groupby("patient_id")["ICULOS"].diff().dropna() == 1).mean()
    print(f"Hours contiguous within patients: {contiguous:.4f} (should be ~1.0)")

    # values from TREND_WINDOW hours ago; blank them if that reading was stale
    g = df.groupby("patient_id")
    for v in TREND_VITALS:
        prev = g[v].shift(TREND_WINDOW)
        prev_hs = g[f"{v}_hours_since"].shift(TREND_WINDOW)
        df[f"{v}_prev"] = prev.where(prev_hs <= VITAL_RECENT)

    df["is_first"] = g["ICULOS"].transform("min") == df["ICULOS"]
    note_rows = df[(df["ICULOS"] % NOTE_EVERY == 0) | df["is_first"]].copy()
    print(f"Writing {len(note_rows):,} notes for {note_rows['patient_id'].nunique():,} patients ...")

    rng = np.random.default_rng(SEED)
    texts, n_abns = [], []
    for r in note_rows.to_dict("records"):
        text, n_abn = make_note(r, rng)
        texts.append(text)
        n_abns.append(n_abn)

    notes = note_rows[["patient_id", "hospital_set", "split", "ICULOS"]].copy()
    notes["note_text"] = texts
    notes["n_abnormal"] = n_abns
    notes = notes.reset_index(drop=True)

    # hygiene checks
    assert not notes["note_text"].str.contains("sepsis", case=False).any(), "note mentions sepsis!"
    notes.to_pickle(OUT)

    # ---------- summary ----------
    print("\n=== Notes per split ===")
    print(notes["split"].value_counts().to_string())
    words = notes["note_text"].str.split().str.len()
    print(f"\nWords per note: mean={words.mean():.0f}, max={words.max()} "
          f"(ClinicalBERT limit is 512 tokens - plenty of room)")

    print("\n=== Notes by abnormal-finding count ===")
    print(notes["n_abnormal"].clip(upper=4).value_counts().sort_index().to_string())

    # does the text carry the same signal as the numbers? (it should - it's derived from them)
    labelled = notes.merge(df[["patient_id", "ICULOS", "label_24h"]], on=["patient_id", "ICULOS"])
    print("\n=== Mean abnormal-finding count by label_24h (0 = no sepsis in next 24h) ===")
    print(labelled.groupby("label_24h")["n_abnormal"].mean().round(2).to_string())

    # show a real timeline for one septic patient, approaching onset
    sepsis = df.loc[df["SepsisLabel"] == 1, "patient_id"]
    if len(sepsis):
        pid = sepsis.iloc[0]
        onset = df.loc[(df["patient_id"] == pid) & (df["SepsisLabel"] == 1), "ICULOS"].min()
        window = notes[(notes["patient_id"] == pid) &
                       (notes["ICULOS"].between(onset - 24, onset + 6))]
        print(f"\n=== Example: patient {pid}, sepsis onset at hour {int(onset)} ===")
        for _, row in window.iterrows():
            print(f"\n[hour {int(row['ICULOS'])}, {row['n_abnormal']} abnormal]\n{row['note_text']}")

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()