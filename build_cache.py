import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATA_DIR = Path("physionet-sepsis-2019")
CACHE = Path("sepsis_all.pkl")


def read_one(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f, sep="|", dtype="float32")
    df.insert(0, "patient_id", f.stem)
    df.insert(1, "hospital_set", f.parent.name[-1].upper())
    return df


start = time.time()
files = sorted(DATA_DIR.rglob("*.psv"))
print(f"Found {len(files)} files in {time.time() - start:.1f}s", flush=True)
if not files:
    raise SystemExit(f"No .psv files under {DATA_DIR.resolve()}")

frames = []
with ThreadPoolExecutor(max_workers=8) as pool:
    for i, df in enumerate(pool.map(read_one, files), 1):
        frames.append(df)
        if i % 2000 == 0 or i == len(files):
            print(f"  read {i}/{len(files)}  ({time.time() - start:.0f}s elapsed)", flush=True)

print("Combining...", flush=True)
data = pd.concat(frames, ignore_index=True)
data.to_pickle(CACHE)
print(f"Done: {len(data)} rows saved to {CACHE} in {time.time() - start:.0f}s", flush=True)