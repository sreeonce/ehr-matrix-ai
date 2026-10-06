"""
Sun: train an LSTM on the hourly vitals (and optionally labs) to predict
label_24h, and compare it against Friday/Saturday's XGBoost numbers.

Unlike XGBoost (which sees each hour independently), the LSTM reads each
patient's whole sequence of hours in order, so it can pick up on TRENDS
(e.g. "heart rate has been climbing for the last 6 hours") rather than
just the current snapshot.

Run:
  python train_lstm.py                 # vitals + static only (17 features)
  python train_lstm.py --with-labs     # + the 5 labs (32 features)

Needs: pip install torch
"""
import sys
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
LABEL_COL = "label_24h"
ID_COLS = ["patient_id", "hospital_set", "ICULOS", "SepsisLabel", "label_24h", "split"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Glucose"]

WITH_LABS = "--with-labs" in sys.argv
MAX_LEN = 200          # cap very long stays (median is 39h, this covers most)
HIDDEN_SIZE = 64
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------
# Data prep: one padded sequence per patient
# ---------------------------------------------------------------
def build_sequences(df: pd.DataFrame, feature_cols: list[str]):
    """Returns, per patient: (features [T, F], labels [T], length T)."""
    df = df.sort_values(["patient_id", "ICULOS"])
    seqs, labels, lengths, pids = [], [], [], []
    for pid, g in df.groupby("patient_id", sort=False):
        x = g[feature_cols].to_numpy(dtype="float32")[:MAX_LEN]
        y = g[LABEL_COL].to_numpy(dtype="float32")[:MAX_LEN]
        seqs.append(x)
        labels.append(y)
        lengths.append(len(x))
        pids.append(pid)
    return seqs, labels, lengths, pids


class SepsisSeqDataset(Dataset):
    def __init__(self, seqs, labels, lengths):
        self.seqs, self.labels, self.lengths = seqs, labels, lengths

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i], self.lengths[i]


def collate(batch):
    batch = sorted(batch, key=lambda b: b[2], reverse=True)  # pack_padded needs sorted lengths
    seqs, labels, lengths = zip(*batch)
    max_t = max(lengths)
    n_feat = seqs[0].shape[1]

    x = torch.zeros(len(batch), max_t, n_feat)
    y = torch.zeros(len(batch), max_t)
    mask = torch.zeros(len(batch), max_t)
    for i, (s, l, length) in enumerate(zip(seqs, labels, lengths)):
        x[i, :length] = torch.from_numpy(s)
        y[i, :length] = torch.from_numpy(l)
        mask[i, :length] = 1.0
    return x, y, mask, torch.tensor(lengths)


# ---------------------------------------------------------------
# Model: LSTM -> per-timestep risk logit
# ---------------------------------------------------------------
class RiskLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        return self.head(out).squeeze(-1)  # [B, T] logits


# ---------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------
def run_epoch(model, loader, optimizer, pos_weight, device, train: bool):
    model.train(mode=train)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    total_loss, total_steps = 0.0, 0
    all_probs, all_labels = [], []

    for x, y, mask, lengths in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x, lengths)
            weight = torch.where(y == 1, pos_weight, torch.tensor(1.0, device=device))
            loss_per_step = loss_fn(logits, y) * weight * mask
            loss = loss_per_step.sum() / mask.sum()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_steps += mask.sum().item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        m = mask.cpu().numpy().astype(bool)
        all_probs.append(probs[m])
        all_labels.append(y.cpu().numpy()[m])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / total_steps, probs, labels


def main():
    print(f"Loading {TABLE} ...")
    df = pd.read_pickle(TABLE)

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    if not WITH_LABS:
        drop = set()
        for lab in LABS:
            drop.update({lab, f"{lab}_hours_since", f"{lab}_measured"})
        feature_cols = [c for c in feature_cols if c not in drop]
    print(f"Using {len(feature_cols)} features. with_labs={WITH_LABS}")

    # standardize using TRAIN stats only
    train_df = df[df["split"] == "train"]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - mean) / std
    df[feature_cols] = df[feature_cols].fillna(0.0)  # safety net: 0 = "at the train mean"

    splits = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        seqs, labels, lengths, pids = build_sequences(part, feature_cols)
        splits[name] = SepsisSeqDataset(seqs, labels, lengths)
        print(f"  {name}: {len(seqs)} patients")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    eval_loaders = {
        name: DataLoader(splits[name], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        for name in ("train", "val", "test")
    }

    # pos_weight from train label rate (same idea as scale_pos_weight in XGBoost)
    train_pos_rate = np.concatenate([l for l in splits["train"].labels]).mean()
    pos_weight = torch.tensor((1 - train_pos_rate) / train_pos_rate, device=device)
    print(f"Train positive rate: {train_pos_rate:.4f}  pos_weight: {pos_weight.item():.2f}")

    model = RiskLSTM(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, pos_weight, device, train=True)
        val_loss, val_probs, val_labels = run_epoch(model, eval_loaders["val"], optimizer, pos_weight, device, train=False)
        val_auroc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_AUROC={val_auroc:.3f}  val_AUPRC={val_auprc:.3f}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 60)
    print("FINAL RESULTS (compare against baseline_results.csv)")
    print("=" * 60)
    rows = []
    for name in ("train", "val", "test"):
        _, probs, labels = run_epoch(model, eval_loaders[name], optimizer, pos_weight, device, train=False)
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        print(f"  {name:5s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
        rows.append({"split": name, "AUROC": auroc, "AUPRC": auprc, "model": "lstm"})

    out_name = "lstm_results_with_labs.csv" if WITH_LABS else "lstm_results_vitals.csv"
    pd.DataFrame(rows).to_csv(out_name, index=False)
    print(f"\nSaved to {out_name}")


if __name__ == "__main__":
    main()