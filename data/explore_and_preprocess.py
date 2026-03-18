"""
Phase 1: Data Exploration and Preprocessing
Loads my_fed_data.csv, explores its structure, applies Z-score normalization,
and splits into train/val/test sets (7:1:2) saved as pickle files.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── configuration ────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(ROOT_DIR, "my_fed_data.csv")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

TIME_COL   = "date"
TARGET_COL = "Wind"           # wind-power prediction target
FEAT_COLS  = ["Load", "Solar", "Wind"]   # all numeric feature columns

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
# ──────────────────────────────────────────────────────────────────────────


def explore(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"Date range     : {df[TIME_COL].min()} → {df[TIME_COL].max()}")
    print(f"Missing values :\n{df.isnull().sum()}")
    print(f"\nDescriptive stats for {TARGET_COL}:")
    print(df[TARGET_COL].describe())
    print("=" * 60)


def preprocess(df: pd.DataFrame):
    """
    Apply preprocessing aligned with paper §4.1.2:
    1. Parse timestamps, sort chronologically.
    2. Fill short gaps by forward-fill (≤2 h).
    3. Z-score normalization fitted ONLY on training portion.
    4. Build time features (month, day, weekday, hour).
    5. Split 7 : 1 : 2.
    Returns (train, val, test) each as dict with keys
      'data', 'data_stamp', 'scaler', 'dates'.
    """

    # ── 1. parse & sort ──────────────────────────────────────────────────
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # ── 2. fill short gaps ────────────────────────────────────────────────
    df[FEAT_COLS] = df[FEAT_COLS].ffill(limit=2)
    df = df.dropna(subset=FEAT_COLS).reset_index(drop=True)

    n = len(df)
    num_train = int(n * TRAIN_RATIO)
    num_val   = int(n * VAL_RATIO)
    num_test  = n - num_train - num_val

    print(f"\nTotal samples  : {n}")
    print(f"Train / Val / Test : {num_train} / {num_val} / {num_test}")

    # ── 3. Z-score — fit on training portion only ──────────────────────
    scaler = StandardScaler()
    feat_array = df[FEAT_COLS].values.astype(np.float32)
    scaler.fit(feat_array[:num_train])
    scaled = scaler.transform(feat_array)

    print(f"\nScaler mean  : {dict(zip(FEAT_COLS, scaler.mean_.round(4)))}")
    print(f"Scaler std   : {dict(zip(FEAT_COLS, scaler.scale_.round(4)))}")

    # ── 4. time features ─────────────────────────────────────────────────
    dates = df[TIME_COL]
    data_stamp = np.stack([
        dates.dt.month.values,
        dates.dt.day.values,
        dates.dt.dayofweek.values,
        dates.dt.hour.values,
    ], axis=1).astype(np.float32)

    # ── 5. split ─────────────────────────────────────────────────────────
    splits = {
        "train": (0,            num_train),
        "val":   (num_train,    num_train + num_val),
        "test":  (num_train + num_val, n),
    }

    results = {}
    for name, (s, e) in splits.items():
        results[name] = {
            "data":        scaled[s:e],
            "data_stamp":  data_stamp[s:e],
            "dates":       dates.iloc[s:e].values,
            "scaler":      scaler,
            "feat_cols":   FEAT_COLS,
            "target_col":  TARGET_COL,
            "target_idx":  FEAT_COLS.index(TARGET_COL),
        }

    return results


def save_splits(results: dict) -> None:
    for split_name, data in results.items():
        out_path = os.path.join(OUT_DIR, f"{split_name}.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump(data, fh)
        print(f"Saved {out_path}  (rows={len(data['data'])})")


def main():
    print(f"Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV)

    explore(df)
    results = preprocess(df)
    save_splits(results)

    print("\nPreprocessing complete.")
    print("Output files:")
    for name in ("train", "val", "test"):
        p = os.path.join(OUT_DIR, f"{name}.pkl")
        print(f"  {p}")


if __name__ == "__main__":
    main()
