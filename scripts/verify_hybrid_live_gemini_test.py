#!/usr/bin/env python3
"""
Evaluate the Advanced Hybrid Fakeddit model (VeriPulsePredictor) on the live dataset
labeled with Gemini: data_processed/live_posts_text_label.csv

Columns required: text, label  (0=reliable, 1=unreliable — same as Fakeddit training)

Metrics match verify_advanced_hybrid_metrics.py. Treat Gemini labels as weak / approximate
ground truth when interpreting results (domain shift + labeling noise).

Run from project root:
  python scripts/verify_hybrid_live_gemini_test.py
  python scripts/verify_hybrid_live_gemini_test.py --limit 200
  python scripts/verify_hybrid_live_gemini_test.py --csv path/to/other.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.veripulse_predictor import get_predictor

DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data_processed", "live_posts_text_label.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV with text,label columns")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Missing CSV: {args.csv}")
        sys.exit(2)

    pred = get_predictor()
    st = pred.load()
    if not st.model_loaded:
        print("=" * 60)
        print("HYBRID CHECKPOINT: NOT FOUND OR FAILED TO LOAD")
        print("=" * 60)
        print(st.message)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        print("CSV must have columns: text, label")
        print("Found:", list(df.columns))
        sys.exit(2)

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    texts = df["text"].astype(str).fillna("").tolist()
    y_true = df["label"].astype(int).values

    y_pred: list[int] = []
    confidences: list[float] = []

    for i, t in enumerate(texts):
        out = pred.predict(t)
        if not out.get("model_loaded"):
            print("ERROR: predictor fell back to heuristic mid-run — aborting.")
            sys.exit(3)
        y_pred.append(int(out["prediction"]))
        confidences.append(float(out["confidence"]))
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(texts)} ...")

    y_pred_arr = np.array(y_pred, dtype=int)
    confidences_arr = np.array(confidences, dtype=float)

    acc = accuracy_score(y_true, y_pred_arr)
    prec = precision_score(y_true, y_pred_arr, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred_arr, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred_arr, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_arr)
    cm = confusion_matrix(y_true, y_pred_arr, labels=[0, 1])

    correct = (y_true == y_pred_arr).astype(float)
    mean_conf_when_correct = confidences_arr[correct == 1].mean()
    mean_conf_when_wrong = confidences_arr[correct == 0].mean()
    if np.std(confidences_arr) > 1e-12 and np.std(correct) > 1e-12:
        r_pb = float(np.corrcoef(correct, confidences_arr)[0, 1])
    else:
        r_pb = float("nan")

    csv_l = os.path.basename(args.csv).lower()
    if "distilbert" in csv_l:
        title = "ADVANCED HYBRID - LIVE TEST SET (DistilBERT teacher labels)"
        note = (
            "Note: y_true = DistilBERT predictions on the same texts; metrics = hybrid agreement "
            "with DistilBERT (not independent ground truth; both trained on Fakeddit-style data)."
        )
    else:
        title = "ADVANCED HYBRID - LIVE TEST SET (Gemini-labeled CSV)"
        note = "Note: y_true = Gemini weak labels; metrics = alignment with that proxy, not oracle accuracy."

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"CSV:       {args.csv}")
    print(f"Samples:   {len(texts)}")
    print(f"Model dir: {pred._model_dir}")  # type: ignore[attr-defined]
    print(f"Weight:    {pred._weight_file}")  # type: ignore[attr-defined]
    print(f"Variant:   {getattr(pred, '_checkpoint_variant', None)}")  # type: ignore[attr-defined]
    print()
    print(note)
    t1 = (y_true == 1).mean()
    if t1 < 0.05 or t1 > 0.95:
        print(f"Warning: extreme class imbalance in CSV (true label=1 rate = {t1:.4f}); MCC/PRF1 for class 1 are unstable.")
    print()
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (label=1): {prec:.4f}")
    print(f"Recall    (label=1): {rec:.4f}")
    print(f"F1        (label=1): {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print()
    print("Confusion matrix [rows=true 0,1 | cols=pred 0,1]:")
    print(cm)
    print()
    print("Confidence diagnostics:")
    print(f"  Mean confidence (correct):   {mean_conf_when_correct:.4f}")
    print(f"  Mean confidence (incorrect): {mean_conf_when_wrong:.4f}")
    print(f"  Pearson r (confidence vs correct): {r_pb:.4f}")
    print()
    print(f"  True label=1 rate:  {(y_true == 1).mean():.4f}")
    print(f"  Pred label=1 rate:  {(y_pred_arr == 1).mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
