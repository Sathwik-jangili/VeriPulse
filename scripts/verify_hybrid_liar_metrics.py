#!/usr/bin/env python3
"""
Evaluate LIAR hybrid checkpoint on data_processed/liar_test.csv.

Supports Colab export (distilbert + linguistic_projection + fusion_bottleneck)
or repo HybridTransformerModel (text_encoder + feature_processor + classifier).

Run from project root:
  python scripts/verify_hybrid_liar_metrics.py
  python scripts/verify_hybrid_liar_metrics.py --limit 500

Set VERIPULSE_LIAR_MODEL_DIR to point at the folder containing the .bin + tokenizer.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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

from src.liar_hybrid_checkpoint import (
    find_liar_hybrid_checkpoint,
    linguistic_batch_for_liar,
    load_liar_hybrid,
)

MAX_LENGTH = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max test rows (0 = all)")
    parser.add_argument(
        "--test-csv",
        default=os.path.join(PROJECT_ROOT, "data_processed", "liar_test.csv"),
    )
    args = parser.parse_args()

    d, wp = find_liar_hybrid_checkpoint()
    if not wp:
        print("=" * 60)
        print("LIAR HYBRID: NO WEIGHT FILE FOUND")
        print("=" * 60)
        print("Expected one of:", "hybrid_liar_model.bin, hybrid_model.bin, model.bin, ...")
        print("Place a weight file in the Colab export folder, e.g.:")
        print(
            "  models/hybrid_liar_model-20260322T181556Z-3-001/hybrid_liar_model/"
            "hybrid_liar_model.bin"
        )
        print("(alongside tokenizer.json). Or set VERIPULSE_LIAR_MODEL_DIR.")
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, info = load_liar_hybrid(device=device)
    variant = info["variant"]
    model_dir = info["model_dir"]

    df = pd.read_csv(args.test_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    texts = df["text"].astype(str).fillna("").tolist()
    y_true = df["label"].astype(int).values

    y_pred = []
    confidences = []

    for i, t in enumerate(texts):
        inputs = tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        meta = linguistic_batch_for_liar([t], model_dir, variant).to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta_features=meta,
            )
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(torch.max(probs, dim=-1).values.item())

        y_pred.append(pred)
        confidences.append(conf)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(texts)} ...")

    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    correct = (y_true == y_pred).astype(float)
    mean_conf_correct = confidences[correct == 1].mean()
    mean_conf_wrong = confidences[correct == 0].mean()
    if np.std(confidences) > 1e-12 and np.std(correct) > 1e-12:
        r_pb = float(np.corrcoef(correct, confidences)[0, 1])
    else:
        r_pb = float("nan")

    print("=" * 60)
    print("LIAR HYBRID — EMPIRICAL METRICS (local run)")
    print("=" * 60)
    print(f"Samples:   {len(texts)}")
    print(f"Model dir: {model_dir}")
    print(f"Weight:    {info['weight_path']}")
    print(f"Variant:   {variant}")
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
    print(f"  Mean confidence (correct):   {mean_conf_correct:.4f}")
    print(f"  Mean confidence (incorrect): {mean_conf_wrong:.4f}")
    print(f"  Pearson r (confidence vs correct): {r_pb:.4f}")
    print()
    print(f"  True label=1 rate: {(y_true == 1).mean():.4f}")
    print(f"  Pred label=1 rate: {(y_pred == 1).mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
