#!/usr/bin/env python3
"""
Assign Fakeddit-style binary labels (0=reliable, 1=unreliable) to live posts by
transferring labels from the nearest neighbors in fakeddit_train.csv.

This matches the same label convention as data_processed/fakeddit_test.csv without
using the models you evaluate (avoids circular evaluation).

Usage (from project root):
  python scripts/label_live_knn_fakeddit_train.py \\
    --live-csv data_processed/live_balanced_proxy_labeled.csv \\
    --output data_processed/live_1k_fakeddit_knn_labeled.csv

Optional:
  --train-max-rows 80000   # subsample train for speed/RAM
  --k 11
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAIN = os.path.join(PROJECT_ROOT, "data_processed", "fakeddit_train.csv")
DEFAULT_LIVE = os.path.join(PROJECT_ROOT, "data_processed", "live_balanced_proxy_labeled.csv")


def _normalize_text(s: str) -> str:
    t = (s or "").strip().lower()
    return t if t else " "


def main() -> None:
    parser = argparse.ArgumentParser(description="k-NN Fakeddit train labels for live CSV")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN)
    parser.add_argument("--live-csv", default=DEFAULT_LIVE)
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data_processed", "live_1k_fakeddit_knn_labeled.csv"),
    )
    parser.add_argument("--train-max-rows", type=int, default=0, help="0 = all train rows")
    parser.add_argument("--k", type=int, default=11, help="odd k recommended for majority vote")
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    if args.k < 1 or args.k % 2 == 0:
        print("Use an odd --k (e.g. 11) for clearer majority votes.")
        sys.exit(2)

    train_path = os.path.abspath(args.train_csv)
    live_path = os.path.abspath(args.live_csv)
    out_path = os.path.abspath(args.output)

    if not os.path.isfile(train_path):
        print(f"Missing train CSV: {train_path}")
        sys.exit(2)
    if not os.path.isfile(live_path):
        print(f"Missing live CSV: {live_path}")
        sys.exit(2)

    print("Loading Fakeddit train ...")
    train = pd.read_csv(train_path, usecols=["text", "label"])
    train["text"] = train["text"].astype(str).map(_normalize_text)
    train["label"] = train["label"].astype(int)
    if args.train_max_rows and args.train_max_rows > 0:
        train = train.head(args.train_max_rows).copy()
        print(f"  using first {len(train)} train rows")
    else:
        print(f"  train rows: {len(train)}")

    print("Loading live CSV ...")
    live = pd.read_csv(live_path)
    if "text" not in live.columns:
        print("live CSV must have a 'text' column")
        sys.exit(2)
    live["_text_norm"] = live["text"].astype(str).map(_normalize_text)

    print("Fitting TF-IDF on train texts ...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["text"])
    y_train = train["label"].values
    print(f"  X_train shape: {X_train.shape}")

    print(f"Fitting NearestNeighbors (cosine, k={args.k}) ...")
    nn = NearestNeighbors(n_neighbors=args.k, metric="cosine", algorithm="brute")
    nn.fit(X_train)

    print("Transforming live + querying neighbors ...")
    X_live = vectorizer.transform(live["_text_norm"])
    dist, idx = nn.kneighbors(X_live, return_distance=True)

    # Majority vote; tie-break: label of closest neighbor
    labels = np.empty(len(live), dtype=np.int64)
    for i in range(len(live)):
        neigh_labels = y_train[idx[i]]
        votes = np.bincount(neigh_labels, minlength=2)
        if votes[0] == votes[1]:
            labels[i] = int(neigh_labels[0])
        else:
            labels[i] = int(np.argmax(votes))

    out = live.drop(columns=["_text_norm"], errors="ignore").copy()
    if "label" in out.columns:
        out["proxy_label"] = out["label"]
    out["label"] = labels
    # Optional diagnostics
    out["knn_mean_dist"] = dist.mean(axis=1)

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    # Extra columns (source, knn_mean_dist, ...) are ignored by verify_*; text + label required
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")
    print(
        "Label distribution (Fakeddit k-NN):",
        out["label"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
