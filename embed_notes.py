"""
Tue (Week 2), part 1: turn every synthetic note into a 768-number vector
using ClinicalBERT (a BERT model pre-trained on clinical text).

What "embedding" means here: ClinicalBERT reads a note and we average its
internal representation of each word piece, giving ONE list of 768 numbers
per note. Notes with similar meaning get similar numbers. The model is
FROZEN (we only read from it, never train it), which is why this step only
needs to run once.

The script is built to be safe to interrupt:
  * it saves progress in chunks of 10,000 notes (folder: embedding_chunks/)
  * if you stop it (Ctrl+C) and run it again, it resumes where it left off
  * at the end it joins the chunks into note_embeddings.npy

Run:
  python embed_notes.py --speed-test   # just measure speed, then exit
  python embed_notes.py                # embed everything (resumable)

Needs: pip install transformers torch
The first run downloads the model (~440 MB) from Hugging Face.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

NOTES = Path("notes.pkl")
CHUNK_DIR = Path("embedding_chunks")
OUT = Path("note_embeddings.npy")

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
CHUNK_SIZE = 10_000
MAX_LEN = 192  # notes are ~37 words on average; this leaves lots of headroom


def embed_texts(texts, tokenizer, model, device, batch_size):
    """Mean-pooled embeddings for a list of strings (order preserved)."""
    hidden = model.config.hidden_size
    out = np.zeros((len(texts), hidden), dtype=np.float32)
    # sort by length so each batch has little padding, then put results back in order
    order = np.argsort([len(t) for t in texts])
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
        out[idx] = pooled.float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-test", action="store_true", help="measure speed and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face name or local folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256 if device == "cuda" else 64
    print(f"Device: {device}  |  batch size: {batch_size}  |  model: {args.model}")

    notes = pd.read_pickle(NOTES)
    texts = notes["note_text"].tolist()
    n = len(texts)
    print(f"{n:,} notes to embed")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()

    # ---- token-length check: make sure MAX_LEN doesn't cut off the end of notes ----
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(2000, n), replace=False)
    lens = [len(tokenizer(texts[i])["input_ids"]) for i in sample_idx]
    print(f"Token length (2,000-note sample): median={int(np.median(lens))}, "
          f"95th pct={int(np.percentile(lens, 95))}, max={max(lens)}  (limit used: {MAX_LEN})")
    if max(lens) > MAX_LEN:
        print("WARNING: some notes exceed MAX_LEN and will be cut off at the end.")

    # ---- speed test on a random sample ----
    test_texts = [texts[i] for i in sample_idx[:256]]
    t0 = time.time()
    embed_texts(test_texts, tokenizer, model, device, batch_size)
    rate = len(test_texts) / (time.time() - t0)
    est_min = n / rate / 60
    print(f"\nSpeed: {rate:.0f} notes/sec  ->  estimated {est_min:.0f} minutes for all {n:,} notes")
    if args.speed_test:
        print("(--speed-test: stopping here.)")
        return

    # ---- chunked embedding, resumable ----
    CHUNK_DIR.mkdir(exist_ok=True)
    starts = list(range(0, n, CHUNK_SIZE))
    t_start = time.time()
    done_now = 0
    for cid, s in enumerate(starts):
        path = CHUNK_DIR / f"chunk_{cid:04d}.npy"
        if path.exists():
            continue
        emb = embed_texts(texts[s:s + CHUNK_SIZE], tokenizer, model, device, batch_size)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb.astype(np.float16))
        tmp.replace(path)  # only counts as done once fully written
        done_now += 1
        elapsed = time.time() - t_start
        left = len(starts) - cid - 1
        print(f"  chunk {cid + 1}/{len(starts)} saved  "
              f"({elapsed / 60:.1f} min elapsed, ~{elapsed / done_now * left / 60:.0f} min left)", flush=True)

    # ---- join chunks ----
    parts = [np.load(CHUNK_DIR / f"chunk_{cid:04d}.npy") for cid in range(len(starts))]
    full = np.concatenate(parts, axis=0)
    assert full.shape[0] == n, f"row mismatch: {full.shape[0]} vs {n}"
    np.save(OUT, full)
    print(f"\nSaved {OUT}  shape={full.shape}  dtype={full.dtype}  "
          f"({full.nbytes / 1e6:.0f} MB)")
    print("Row i of this file = note i in notes.pkl. Do NOT re-sort notes.pkl.")


if __name__ == "__main__":
    main()