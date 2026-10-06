"""
Tue (Week 2), part 2: the fusion model ("the blender").

Three modes, so Wednesday's ablation uses ONE identical code path:
  --mode numeric   LSTM on vitals/labs only            (no text)
  --mode text      ClinicalBERT note vectors only      (no numbers)
  --mode fusion    both, blended                       (the full model)
Add --vitals-only to drop the 5 labs from the numeric branch.

How fusion works, hour by hour, for every patient:
  numbers of hour t --> LSTM ----------------> h_t (64 numbers)  \
                                                                   +--> blend --> risk at hour t
  latest note written at or before hour t ---> z_t (64 numbers)  /
                        (768-number ClinicalBERT vector, squeezed by a small layer)

No leakage: the note used at hour t is the most recent one written at or
before hour t, and each note only describes data up to its own hour.

Run (after embed_notes.py has produced note_embeddings.npy):
  python train_fusion.py --mode fusion
  python train_fusion.py --mode numeric
  python train_fusion.py --mode text
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

TABLE = Path("modeling_table.pkl")
NOTES = Path("notes.pkl")
EMB = Path("note_embeddings.npy")

LABEL_COL = "label_24h"
ID_COLS = ["patient_id", "hospital_set", "ICULOS", "SepsisLabel", "label_24h", "split"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]

MAX_LEN = 200       # same cap as train_lstm.py
HIDDEN = 64
TEXT_HIDDEN = 64
BATCH_SIZE = 64
LR = 1e-3
SEED = 0

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["numeric", "text", "fusion"], default="fusion")
ap.add_argument("--vitals-only", action="store_true")
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()

torch.manual_seed(SEED)
np.random.seed(SEED)
USE_NUM = args.mode in ("numeric", "fusion")
USE_TEXT = args.mode in ("text", "fusion")


# ---------------------------------------------------------------
# Data: one sequence per patient, each hour linked to its latest note
# ---------------------------------------------------------------
def load_data():
    df = pd.read_pickle(TABLE).sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    notes = pd.read_pickle(NOTES)  # must be in the SAME order embed_notes.py used
    emb = np.load(EMB)
    assert len(emb) == len(notes), f"{len(emb)} embeddings vs {len(notes)} notes - re-run embed_notes.py"
    print(f"Loaded {len(df):,} patient-hours, {len(notes):,} notes, embeddings {emb.shape}")

    # link each hour to the most recent note written at or before it
    notes = notes.reset_index(drop=True)
    notes["note_idx"] = np.arange(len(notes))
    df = df.merge(notes[["patient_id", "ICULOS", "note_idx"]], on=["patient_id", "ICULOS"], how="left")
    df["note_hour"] = df["ICULOS"].where(df["note_idx"].notna())
    df["note_idx"] = df.groupby("patient_id")["note_idx"].ffill()
    df["note_hour"] = df.groupby("patient_id")["note_hour"].ffill()
    assert df["note_idx"].notna().all(), "some hours have no earlier note"
    df["note_idx"] = df["note_idx"].astype("int64")
    df["hours_since_note"] = (df["ICULOS"] - df["note_hour"]).astype("float32")
    # safety: a note is never from the future
    assert (df["hours_since_note"] >= 0).all(), "found a note from the future!"

    feature_cols = [c for c in df.columns if c not in ID_COLS + ["note_idx", "note_hour", "hours_since_note"]]
    if args.vitals_only:
        drop = set()
        for lab in LABS:
            drop.update({lab, f"{lab}_hours_since", f"{lab}_measured"})
        feature_cols = [c for c in feature_cols if c not in drop]

    # standardize numeric features using TRAIN statistics only
    train_df = df[df["split"] == "train"]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)
    df[feature_cols] = ((df[feature_cols] - mean) / std).fillna(0.0)
    return df, feature_cols, emb


class SeqDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.items = []
        for _, g in df.groupby("patient_id", sort=False):
            g = g.iloc[:MAX_LEN]
            self.items.append((
                g[feature_cols].to_numpy("float32").copy(),
                g[LABEL_COL].to_numpy("float32").copy(),
                g["note_idx"].to_numpy("int64").copy(),
                g["hours_since_note"].to_numpy("float32").copy(),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
    lengths = [len(b[0]) for b in batch]
    B, T, F = len(batch), max(lengths), batch[0][0].shape[1]
    x = torch.zeros(B, T, F)
    y = torch.zeros(B, T)
    mask = torch.zeros(B, T)
    idx = torch.zeros(B, T, dtype=torch.long)
    hsn = torch.zeros(B, T)
    for i, (xi, yi, ii, hi) in enumerate(batch):
        n = len(xi)
        x[i, :n] = torch.from_numpy(xi)
        y[i, :n] = torch.from_numpy(yi)
        idx[i, :n] = torch.from_numpy(ii)
        hsn[i, :n] = torch.from_numpy(hi)
        mask[i, :n] = 1.0
    return x, y, mask, idx, hsn, torch.tensor(lengths)


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, n_num, text_dim):
        super().__init__()
        head_in = 0
        if USE_NUM:
            self.lstm = nn.LSTM(n_num, HIDDEN, batch_first=True)
            head_in += HIDDEN
        if USE_TEXT:
            # LayerNorm tames the large-valued dimensions BERT vectors tend to have
            self.text = nn.Sequential(
                nn.LayerNorm(text_dim + 1),
                nn.Linear(text_dim + 1, TEXT_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            head_in += TEXT_HIDDEN
        # the "blender": a small network that turns the joined vector into one risk logit
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, lengths, idx, hsn, emb_table):
        parts = []
        if USE_NUM:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
            out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
            parts.append(h)
        if USE_TEXT:
            e = emb_table[idx].float()                              # [B, T, 768]
            e = torch.cat([e, hsn.unsqueeze(-1) / 6.0], dim=-1)     # + "hours since that note"
            parts.append(self.text(e))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)      # [B, T]


def run_epoch(model, loader, opt, pos_weight, emb_table, device, train):
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    tot, steps, probs_all, labels_all = 0.0, 0, [], []
    for x, y, mask, idx, hsn, lengths in loader:
        x, y, mask, idx, hsn = (t.to(device) for t in (x, y, mask, idx, hsn))
        with torch.set_grad_enabled(train):
            logits = model(x, lengths, idx, hsn, emb_table)
            w = torch.where(y == 1, pos_weight, torch.ones_like(y))
            loss = (loss_fn(logits, y) * w * mask).sum() / mask.sum()
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * mask.sum().item()
        steps += mask.sum().item()
        m = mask.cpu().numpy().astype(bool)
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy()[m])
        labels_all.append(y.cpu().numpy()[m])
    return tot / steps, np.concatenate(probs_all), np.concatenate(labels_all)


def score(labels, probs):
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.mode + ("_vitals" if args.vitals_only and args.mode != "text" else "")
    print(f"Mode: {args.mode}  |  vitals_only={args.vitals_only}  |  device: {device}")

    df, feature_cols, emb = load_data()
    print(f"Numeric features: {len(feature_cols) if USE_NUM else 'not used'}")
    emb_table = torch.from_numpy(emb).to(device)

    sets = {s: SeqDataset(df[df["split"] == s], feature_cols) for s in ("train", "val", "test")}
    for s, d in sets.items():
        print(f"  {s}: {len(d):,} patients")
    train_loader = DataLoader(sets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    eval_loaders = {s: DataLoader(d, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
                    for s, d in sets.items()}

    pos_rate = np.concatenate([it[1] for it in sets["train"].items]).mean()
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate, device=device)
    print(f"Train positive rate {pos_rate:.4f}, pos_weight {pos_weight.item():.1f}")

    model = FusionModel(len(feature_cols), emb.shape[1]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_auprc, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, opt, pos_weight, emb_table, device, True)
        va_loss, vp, vl = run_epoch(model, eval_loaders["val"], opt, pos_weight, emb_table, device, False)
        va_auroc, va_auprc = score(vl, vp)
        flag = ""
        if va_auprc > best_auprc:
            best_auprc, best_state, best_epoch = va_auprc, copy.deepcopy(model.state_dict()), ep
            flag = "  <- best so far"
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_AUROC={va_auroc:.3f}  val_AUPRC={va_auprc:.3f}  ({time.time() - t0:.0f}s){flag}")

    model.load_state_dict(best_state)
    print(f"\nUsing the best epoch by validation AUPRC (epoch {best_epoch}).")
    print("=" * 60)
    print(f"FINAL RESULTS - mode: {tag}")
    print("=" * 60)
    rows = []
    for s in ("train", "val", "test"):
        _, p, l = run_epoch(model, eval_loaders[s], opt, pos_weight, emb_table, device, False)
        auroc, auprc = score(l, p)
        print(f"  {s:5s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
        rows.append({"split": s, "AUROC": auroc, "AUPRC": auprc, "model": tag, "best_epoch": best_epoch})

    pd.DataFrame(rows).to_csv(f"fusion_results_{tag}.csv", index=False)
    torch.save(best_state, f"model_{tag}.pt")
    print(f"\nSaved fusion_results_{tag}.csv and model_{tag}.pt")


if __name__ == "__main__":
    main()