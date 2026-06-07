"""
Prepare Fakeddit and LIAR datasets for fake-news / misinformation detection.
Run from project root: python -m src.prepare_datasets
"""

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Paths – adjust these if your folders differ from data_raw/fakeddit and data_raw/liar
# -----------------------------------------------------------------------------
# If your Fakeddit files are not in data_raw/fakeddit/, set this to the folder
# that contains the TSV/CSV file(s). Example if you have a nested zip layout:
#   FAKEDDIT_DIR = "data_raw/all_samples (also includes non multimodal)-20260214T164746Z-1-001/all_samples (also includes non multimodal)"
# You can also set this to a path that has only one main file (e.g. all_train.tsv).
# Folder containing Fakeddit TSV/CSV (all_train.tsv, all_validate.tsv, etc.)
FAKEDDIT_DIR = "data_raw/all_samples (also includes non multimodal)-20260214T164746Z-1-001/all_samples (also includes non multimodal)"

# Folder containing LIAR train.tsv, valid.tsv, test.tsv
LIAR_DIR = "data_raw/liar_dataset"

PROCESSED_DIR = "data_processed"

# Fakeddit: max rows to keep (optional cap; None = use all)
FAKEDDIT_MAX_ROWS = 160_000

# Stratified split ratios for Fakeddit (train / val / test)
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


def clean_text(series: pd.Series) -> pd.Series:
    """Lowercase, remove URLs, collapse whitespace."""
    s = series.astype(str).str.lower()
    s = s.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def prepare_fakeddit() -> None:
    """
    Load Fakeddit TSV/CSV from FAKEDDIT_DIR, select text and binary label columns,
    clean text, optionally cap at FAKEDDIT_MAX_ROWS, then stratified train/val/test split.
    """
    base = Path(FAKEDDIT_DIR)
    if not base.exists():
        print(f"[Fakeddit] Directory not found: {base.resolve()}")
        print("  Change FAKEDDIT_DIR at the top of this script if your files are elsewhere.")
        return

    # Find all TSV/CSV files; use the first or concatenate (adjust if you have one "main" file)
    tsv_files = list(base.glob("*.tsv")) + list(base.glob("*.csv"))
    if not tsv_files:
        print(f"[Fakeddit] No .tsv or .csv files in {base.resolve()}")
        return

    # Load: if single file use it, else concatenate (e.g. all_train + all_validate + all_test)
    dfs = []
    for f in sorted(tsv_files):
        sep = "\t" if f.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(f, sep=sep, low_memory=False)
        except Exception as e:
            print(f"[Fakeddit] Skipping {f.name}: {e}")
            continue
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"[Fakeddit] Loaded {len(raw)} rows from {len(dfs)} file(s).")
    print("  Column names:", list(raw.columns))

    # --- Choose text column: prefer clean_title, else title ---
    # If your dataset uses different names, set TEXT_COL in this block:
    if "clean_title" in raw.columns:
        text_col = "clean_title"
    elif "title" in raw.columns:
        text_col = "title"
    else:
        # Fallback: first column that looks like text (e.g. "title", "text", "content")
        cand = [c for c in raw.columns if c.lower() in ("title", "text", "content", "sentence")]
        text_col = cand[0] if cand else raw.columns[0]
        print(f"  No 'clean_title' or 'title'; using: {text_col}")

    # --- Choose binary label column ---
    # Prefer a 2-way label column (0/1 or real/fake). If none found, we print options and exit.
    label_candidates = [c for c in raw.columns if "label" in c.lower() or c in ("label", "binary", "2_way_label")]
    binary_label_col = None
    for c in ("2_way_label", "binary_label", "label", "binary"):
        if c in raw.columns:
            binary_label_col = c
            break
    if binary_label_col is None and label_candidates:
        binary_label_col = label_candidates[0]

    if binary_label_col is None:
        print("[Fakeddit] No obvious binary label column found.")
        all_cols = list(raw.columns)
        label_like = [c for c in all_cols if "label" in c.lower() or "class" in c.lower()]
        print("  Label-like columns:", label_like or "none")
        print("  All columns:", all_cols)
        print("  To fix: in prepare_fakeddit(), set binary_label_col = 'your_column' (e.g. '2_way_label').")
        print("  Common Fakeddit columns: 2_way_label (0/1), 3_way_label, 6_way_label.")
        return

    # Keep only text and label; rename to standard names
    df = raw[[text_col, binary_label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text", "label"])
    # Ensure numeric binary labels 0/1 if needed (e.g. 2_way_label might already be 0/1)
    if df["label"].dtype == object or df["label"].dtype.name == "category":
        # Map common string labels to 0/1
        lower = df["label"].astype(str).str.lower()
        if set(lower.unique()) <= {"0", "1", "real", "fake", "true", "false"}:
            df["label"] = lower.replace({"real": 0, "fake": 1, "true": 1, "false": 0}).astype(int)
        else:
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    # Keep only 0 and 1 if multiple values (e.g. 2_way_label should be 0/1)
    if set(df["label"].unique()) - {0, 1}:
        print(f"  Binary label column '{binary_label_col}' has values: {sorted(df['label'].unique())}")
        print("  Restricting to 0 and 1. If you need a different mapping, adjust the script.")
        df = df[df["label"].isin([0, 1])]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        print("[Fakeddit] No rows left after selecting text/label and dropping missing.")
        return

    # Optional cap
    if FAKEDDIT_MAX_ROWS is not None and len(df) > FAKEDDIT_MAX_ROWS:
        df = df.sample(n=FAKEDDIT_MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"[Fakeddit] Sampled down to {len(df)} rows.")

    # Clean text
    df["text"] = clean_text(df["text"])

    # Stratified split
    train, rest = train_test_split(df, test_size=(1 - TRAIN_RATIO), stratify=df["label"], random_state=42)
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(rest, test_size=(1 - val_ratio_adj), stratify=rest["label"], random_state=42)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(PROCESSED_DIR, f"fakeddit_{name}.csv")
        part.to_csv(out, index=False)
        print(f"  {out}: {len(part)} rows, label dist. 0/1: {part['label'].value_counts().sort_index().to_dict()}")

    print()


def prepare_liar() -> None:
    """
    Load LIAR train/valid/test TSVs, keep label (col 1) and statement (col 2),
    map 6-way labels to binary, clean text, save to data_processed/liar_*.csv.
    """
    base = Path(LIAR_DIR)
    if not base.exists():
        print(f"[LIAR] Directory not found: {base.resolve()}")
        print("  Change LIAR_DIR at the top of this script (e.g. to data_raw/liar_dataset).")
        return

    # LIAR TSV: no header. Col 0 = ID, col 1 = label, col 2 = statement (and more after)
    col_names = ["id", "label_orig", "text"] + [f"col_{i}" for i in range(3, 15)]

    splits = {}
    for part, fname in [("train", "train.tsv"), ("val", "valid.tsv"), ("test", "test.tsv")]:
        path = base / fname
        if not path.exists():
            print(f"[LIAR] File not found: {path}")
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=col_names, on_bad_lines="warn")
        df = df[["label_orig", "text"]].copy()
        df.columns = ["label_orig", "text"]
        df = df.dropna(subset=["text", "label_orig"])
        splits[part] = df

    if not splits:
        return

    # Map 6-way to binary: Unreliable (1): pants-fire, false, barely-true; Reliable (0): half-true, mostly-true, true
    unreliable = {"pants-fire", "false", "barely-true"}
    reliable = {"half-true", "mostly-true", "true"}

    def to_binary(lbl: str) -> int | float:
        lbl = str(lbl).strip().lower()
        if lbl in unreliable:
            return 1
        if lbl in reliable:
            return 0
        return float("nan")

    for part in splits:
        splits[part]["label"] = splits[part]["label_orig"].map(to_binary)
        splits[part] = splits[part].dropna(subset=["label"])
        splits[part]["label"] = splits[part]["label"].astype(int)
        splits[part]["text"] = clean_text(splits[part]["text"])
        splits[part] = splits[part][["text", "label"]]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for part, df in splits.items():
        out = os.path.join(PROCESSED_DIR, f"liar_{part}.csv")
        df.to_csv(out, index=False)
        print(f"  {out}: {len(df)} rows, label dist. 0/1: {df['label'].value_counts().sort_index().to_dict()}")

    print()


def main() -> None:
    print("Preparing Fakeddit...")
    prepare_fakeddit()
    print("Preparing LIAR...")
    prepare_liar()
    print("Done.")


if __name__ == "__main__":
    main()
