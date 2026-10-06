"""
Thu (Week 2): turn the trained fusion model into static data the Next.js
dashboard can read with no backend at all.

Picks a small set of patients that tell a good story (stable patients who
deteriorate, plus one who stays stable throughout), runs the already-trained
fusion model over their timeline, and writes everything the dashboard needs
- risk per hour, vitals, labs, notes, a plain-language summary, and a few
generic AI suggestions - into dashboard/data/patients.json.

Run:  python export_dashboard_data.py
Needs: modeling_table.pkl, notes.pkl, note_embeddings.npy, model_fusion.pt
       (all produced by earlier scripts this week)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TABLE = Path("modeling_table.pkl")
NOTES = Path("notes.pkl")
EMB = Path("note_embeddings.npy")
MODEL_PATH = Path("model_fusion.pt")
OUT = Path("dashboard/data/patients.json")

LABEL_COL = "label_24h"
ID_COLS = ["patient_id", "hospital_set", "ICULOS", "SepsisLabel", "label_24h", "split"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

HIDDEN, TEXT_HIDDEN = 64, 64
MAX_LEN = 200
N_PATIENTS = 5
N_DETERIORATING = 4   # the rest (1) is the stable contrast case


# ---------------------------------------------------------------
# Model - same architecture as train_fusion.py's --mode fusion
# ---------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, n_num, text_dim):
        super().__init__()
        self.lstm = nn.LSTM(n_num, HIDDEN, batch_first=True)
        self.text = nn.Sequential(
            nn.LayerNorm(text_dim + 1),
            nn.Linear(text_dim + 1, TEXT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(nn.Linear(HIDDEN + TEXT_HIDDEN, 64), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, lengths, idx, hsn, emb_table):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        e = emb_table[idx].float()
        e = torch.cat([e, hsn.unsqueeze(-1) / 6.0], dim=-1)
        t = self.text(e)
        return self.head(torch.cat([h, t], dim=-1)).squeeze(-1)


def link_notes(df, notes):
    notes = notes.reset_index(drop=True)
    notes["note_idx"] = np.arange(len(notes))
    df = df.merge(notes[["patient_id", "ICULOS", "note_idx"]], on=["patient_id", "ICULOS"], how="left")
    df["note_hour"] = df["ICULOS"].where(df["note_idx"].notna())
    df["note_idx"] = df.groupby("patient_id")["note_idx"].ffill()
    df["note_hour"] = df.groupby("patient_id")["note_hour"].ffill()
    df["note_idx"] = df["note_idx"].astype("int64")
    df["hours_since_note"] = (df["ICULOS"] - df["note_hour"]).astype("float32")
    return df


def pick_deteriorating(df):
    """Septic patients with a clear stable-then-bad arc and a stay short
    enough to show on one screen."""
    stats = df.groupby("patient_id").agg(
        ever_septic=("SepsisLabel", "max"),
        n_hours=("ICULOS", "size"),
        onset=("ICULOS", lambda s: df.loc[s.index, "ICULOS"][df.loc[s.index, "SepsisLabel"] == 1].min()),
    )
    good_arc = stats[(stats["ever_septic"] == 1) &
                     (stats["n_hours"].between(40, 120)) &
                     (stats["onset"] / stats["n_hours"] > 0.4) &
                     (stats["onset"] / stats["n_hours"] < 0.9)]
    return list(good_arc.sample(min(N_DETERIORATING, len(good_arc)), random_state=0).index)


def pick_stable_candidates(df, pool_size=40):
    """Candidate pool of patients who never meet sepsis criteria. The final
    choice among these is made AFTER running the model (see pick_stable),
    since a patient can look calm in the raw vitals but still get flagged
    by the model - that's worth knowing before picking the "reassuring"
    demo case, not after."""
    stats = df.groupby("patient_id").agg(
        ever_septic=("SepsisLabel", "max"),
        n_hours=("ICULOS", "size"),
    )
    stable = stats[(stats["ever_septic"] == 0) & (stats["n_hours"].between(40, 120))]
    return list(stable.sample(min(pool_size, len(stable)), random_state=0).index)


DRIVER_LABELS = {
    "HR": "Heart rate", "O2Sat": "SpO2", "Temp": "Temperature",
    "SBP": "Systolic BP", "DBP": "Diastolic BP", "MAP": "MAP", "Resp": "Resp. rate",
    "Lactate": "Lactate", "WBC": "WBC", "Creatinine": "Creatinine",
    "Platelets": "Platelets", "Glucose": "Glucose",
}


def compute_drivers(g, mean, std, top_n=3, z_threshold=1.0):
    """For each hour, the top few vitals/labs that deviate most from the
    training-set typical value (a simple, honest proxy for "what stands
    out" - not a real SHAP/attribution value, so we don't claim it is)."""
    cols = list(DRIVER_LABELS.keys())
    vals = g[cols].to_numpy("float32")
    z = (vals - mean[cols].to_numpy("float32")) / std[cols].to_numpy("float32")

    out = []
    for row_z, row_v in zip(z, vals):
        order = np.argsort(-np.abs(row_z))
        picks = []
        for i in order[:top_n]:
            if abs(row_z[i]) < z_threshold:
                break
            key = cols[i]
            direction = "higher" if row_z[i] > 0 else "lower"
            typical = mean[key]
            picks.append(
                f"{DRIVER_LABELS[key]} {direction} than typical "
                f"({row_v[i]:.1f} vs ~{typical:.0f} usually)"
            )
        out.append(picks if picks else ["Nothing notably abnormal this hour."])
    return out


def nearest_recent(series_val, series_hours_since, limit):
    """Last real value if measured within `limit` hours, else None."""
    if series_hours_since <= limit:
        return round(float(series_val), 1)
    return None


def summarize(pid, hours, risk, onset_hour, vitals_now, labs_now):
    peak_idx = int(np.argmax(risk))
    peak_hour, peak_risk = hours[peak_idx], risk[peak_idx]
    crossed = next((h for h, r in zip(hours, risk) if r >= 0.5), None)

    lines = [f"Over this {hours[-1]}-hour stay, predicted risk peaked at "
             f"{peak_risk:.0%} around hour {peak_hour}."]
    if crossed is not None and onset_hour is not None:
        lead = onset_hour - crossed
        if lead > 0:
            lines.append(f"Model risk crossed 50% at hour {crossed}, "
                         f"{lead:.0f}h before sepsis criteria were met at hour {onset_hour:.0f}.")
        else:
            lines.append(f"Sepsis criteria were met at hour {onset_hour:.0f}; "
                         f"model risk crossed 50% at hour {crossed}.")
    elif onset_hour is not None:
        lines.append(f"Sepsis criteria were met at hour {onset_hour:.0f}.")
    else:
        lines.append("This patient did not meet sepsis criteria during the stay.")
    return " ".join(lines)


def ai_suggestions(vitals_now, labs_now, risk_now):
    """Generic, non-diagnostic, template-based - this is a demo panel, not
    real clinical guidance. Always framed as suggestions to consider."""
    out = []
    if risk_now >= 0.5:
        out.append("Risk score is elevated - consider closer monitoring and clinical review.")
    hr = vitals_now.get("HR")
    resp = vitals_now.get("Resp")
    if hr and hr >= 110:
        out.append("Heart rate is persistently high - consider reassessing volume status.")
    if resp and resp >= 24:
        out.append("Respiratory rate is elevated - consider a repeat oxygenation check.")
    lac = labs_now.get("Lactate")
    if lac and lac >= 2.0:
        out.append("Lactate is raised - consider a repeat lactate to check the trend.")
    if not out:
        out.append("No specific concerns flagged - continue routine monitoring.")
    return out[:3]


def main():
    print("Loading data and model ...")
    df = pd.read_pickle(TABLE).sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    notes = pd.read_pickle(NOTES)
    emb = np.load(EMB)
    df = link_notes(df, notes)

    feature_cols = [c for c in df.columns if c not in
                    ID_COLS + ["note_idx", "note_hour", "hours_since_note"]]
    train_df = df[df["split"] == "train"]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)
    df_norm = df.copy()
    df_norm[feature_cols] = ((df_norm[feature_cols] - mean) / std).fillna(0.0)

    model = FusionModel(len(feature_cols), emb.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    emb_table = torch.from_numpy(emb)

    def run_model(pid):
        g = df[df["patient_id"] == pid].iloc[:MAX_LEN]
        g_norm = df_norm[df_norm["patient_id"] == pid].iloc[:MAX_LEN]
        x = torch.tensor(g_norm[feature_cols].to_numpy("float32")).unsqueeze(0)
        idx = torch.tensor(g_norm["note_idx"].to_numpy("int64")).unsqueeze(0)
        hsn = torch.tensor(g_norm["hours_since_note"].to_numpy("float32")).unsqueeze(0)
        length = torch.tensor([len(g)])
        with torch.inference_mode():
            logits = model(x, length, idx, hsn, emb_table)
        return torch.sigmoid(logits)[0].numpy()

    chosen = pick_deteriorating(df)
    print(f"Deteriorating patients: {chosen}")

    # Pick the stable contrast case by actually checking the model's
    # output, not just the raw label - a patient with calm vitals can
    # still get flagged, and that's not the "reassuring" example we want.
    stable_pool = pick_stable_candidates(df)
    stable_scores = {pid: run_model(pid).max() for pid in stable_pool}
    best_stable = min(stable_scores, key=stable_scores.get)
    print(f"Stable candidates' peak risk: "
          f"{sorted(stable_scores.values())[:3]} (lowest) ... "
          f"{sorted(stable_scores.values())[-3:]} (highest)")
    print(f"Chosen stable patient: {best_stable} (peak risk {stable_scores[best_stable]:.0%})")
    if stable_scores[best_stable] > 0.3:
        print(f"NOTE: even the calmest stable candidate peaks at "
              f"{stable_scores[best_stable]:.0%} - the model may be miscalibrated "
              f"on this cohort, or this cohort simply has few truly low-risk patients. "
              f"Worth a line in the case study either way.")
    chosen.append(best_stable)
    print(f"Chosen patients: {chosen}")

    patients_out = []
    for pid in chosen:
        g = df[df["patient_id"] == pid].iloc[:MAX_LEN]
        risk = run_model(pid).tolist()
        hours = g["ICULOS"].astype(int).tolist()
        onset = g.loc[g["SepsisLabel"] == 1, "ICULOS"]
        onset_hour = float(onset.min()) if len(onset) else None

        vitals = {v: [round(float(val), 1) for val in g[v]] for v in VITALS}
        vitals_hs = {v: g[f"{v}_hours_since"].tolist() for v in VITALS}
        labs = {l: [round(float(val), 1) for val in g[l]] for l in LABS}
        labs_hs = {l: g[f"{l}_hours_since"].tolist() for l in LABS}

        last = g.iloc[-1]
        vitals_now = {v: nearest_recent(last[v], last[f"{v}_hours_since"], 6) for v in VITALS}
        labs_now = {l: nearest_recent(last[l], last[f"{l}_hours_since"], 12) for l in LABS}

        pid_notes = notes[(notes["patient_id"] == pid) & (notes["ICULOS"] <= hours[-1])]
        note_list = [
            {"hour": int(r.ICULOS), "text": r.note_text, "n_abnormal": int(r.n_abnormal)}
            for r in pid_notes.itertuples()
        ]

        drivers = compute_drivers(g, mean, std)

        patients_out.append({
            "id": pid,
            "hospital_set": last["hospital_set"],
            "age": int(round(last["Age"])),
            "gender": "male" if last["Gender"] >= 0.5 else "female",
            "hours": hours,
            "risk": [round(r, 3) for r in risk],
            "sepsis_onset_hour": onset_hour,
            "vitals": vitals,
            "vitals_hours_since": vitals_hs,
            "labs": labs,
            "labs_hours_since": labs_hs,
            "notes": note_list,
            "drivers": drivers,
            "summary": summarize(pid, hours, risk, onset_hour, vitals_now, labs_now),
            "ai_suggestions": ai_suggestions(vitals_now, labs_now, risk[-1]),
        })
        print(f"  {pid}: {len(hours)}h, peak risk {max(risk):.0%}, "
              f"onset={'none' if onset_hour is None else f'h{onset_hour:.0f}'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"patients": patients_out}, f, indent=2)
    print(f"\nSaved {len(patients_out)} patients to {OUT}")


if __name__ == "__main__":
    main()