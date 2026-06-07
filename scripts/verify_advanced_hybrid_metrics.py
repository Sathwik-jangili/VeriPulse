#!/usr/bin/env python3
"""
Verify Advanced Hybrid Fakeddit checkpoint on data_processed/fakeddit_test.csv.

Computes: accuracy, precision/recall/F1 (positive = label 1), MCC,
mean confidence when correct vs wrong, point-biserial correlation(confidence, correct).

Run from project root:
  python scripts/verify_advanced_hybrid_metrics.py
  python scripts/verify_advanced_hybrid_metrics.py --limit 500
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.veripulse_predictor import get_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max test rows (0 = all)")
    parser.add_argument(
        "--test-csv",
        default=os.path.join(PROJECT_ROOT, "data_processed", "fakeddit_test.csv"),
    )
    args = parser.parse_args()

    pred = get_predictor()
    st = pred.load()
    if not st.model_loaded:
        print("=" * 60)
        print("CHECKPOINT: NOT FOUND OR FAILED TO LOAD")
        print("=" * 60)
        print(st.message)
        print("\nDocumented Colab / training reference (see FAKEDDIT_60K_RESULTS.md):")
        print("  - Test accuracy (reported): ~84.84% on ~24k Fakeddit test samples")
        print("  - Mean confidence (reported): ~0.748 (training log)")
        print("  - Test class balance (reported): ~50/50 reliable vs unreliable")
        print("\nAfter you copy advanced_hybrid_model.bin (or fakeddit_model.bin) into")
        print("  models/advanced_hybrid_fakeddit_60k/  re-run this script for live metrics.")
        sys.exit(2)

    df = pd.read_csv(args.test_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    texts = df["text"].astype(str).fillna("").tolist()
    y_true = df["label"].astype(int).values

    y_pred = []
    confidences = []

    for i, t in enumerate(texts):
        out = pred.predict(t)
        y_pred.append(int(out["prediction"]))
        confidences.append(float(out["confidence"]))
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(texts)} ...")

    y_pred = pd.Series(y_pred).values
    confidences = pd.Series(confidences).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    correct = (y_true == y_pred).astype(float)
    mean_conf_when_correct = confidences[correct == 1].mean()
    mean_conf_when_wrong = confidences[correct == 0].mean()

    if np.std(confidences) > 1e-12 and np.std(correct) > 1e-12:
        r_pb = float(np.corrcoef(correct, confidences)[0, 1])
    else:
        r_pb = float("nan")

    print("=" * 60)
    print("ADVANCED HYBRID FAKEDDIT — EMPIRICAL METRICS (local run)")
    print("=" * 60)
    print(f"Samples: {len(texts)}")
    print(f"Model dir: {pred._model_dir}")  # type: ignore[attr-defined]
    print(f"Weight:    {pred._weight_file}")  # type: ignore[attr-defined]
    print(f"Variant:   {getattr(pred, '_checkpoint_variant', None)}")  # type: ignore[attr-defined]
    print(f"Fusion:    text_only={getattr(pred._model, 'fusion_text_only', False)}")  # type: ignore[attr-defined]
    print()
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (label=1): {prec:.4f}")
    print(f"Recall    (label=1): {rec:.4f}")
    print(f"F1        (label=1): {f1:.4f}")
    print(f"MCC (bias/balance):  {mcc:.4f}")
    print()
    print("Confusion matrix [rows=true 0,1 | cols=pred 0,1]:")
    print(cm)
    print()
    print("Confidence diagnostics:")
    print(f"  Mean confidence (correct):   {mean_conf_when_correct:.4f}")
    print(f"  Mean confidence (incorrect): {mean_conf_when_wrong:.4f}")
    print(f"  Pearson r (confidence vs correct): {r_pb:.4f}")
    print()
    pred_pos_rate = (y_pred == 1).mean()
    true_pos_rate = (y_true == 1).mean()
    print("Distribution:")
    print(f"  True label=1 rate:  {true_pos_rate:.4f}")
    print(f"  Pred label=1 rate:  {pred_pos_rate:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
