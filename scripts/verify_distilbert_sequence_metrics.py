#!/usr/bin/env python3
"""
Evaluate a HuggingFace DistilBertForSequenceClassification folder on a CSV
with columns: text, label (0=reliable, 1=unreliable).

Presets (same style as hybrid verify scripts):
  --preset fakeddit_30k   -> models/distilbert_fakeddit_30k_full + fakeddit_test.csv
  --preset liar           -> models/distilbert_liar_trained + liar_test.csv

Or pass --model-dir and --test-csv explicitly.

Run from project root:
  python scripts/verify_distilbert_sequence_metrics.py --preset fakeddit_30k --limit 500
  python scripts/verify_distilbert_sequence_metrics.py --preset liar --limit 500
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MAX_LENGTH = 128

PRESETS = {
    "fakeddit_30k": (
        os.path.join(PROJECT_ROOT, "models", "distilbert_fakeddit_30k_full"),
        os.path.join(PROJECT_ROOT, "data_processed", "fakeddit_test.csv"),
    ),
    "liar": (
        os.path.join(PROJECT_ROOT, "models", "distilbert_liar_trained"),
        os.path.join(PROJECT_ROOT, "data_processed", "liar_test.csv"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use bundled model dir + test CSV",
    )
    parser.add_argument("--model-dir", default=None, help="Override or use without preset")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    args = parser.parse_args()

    if args.preset:
        model_dir, test_csv = PRESETS[args.preset]
    else:
        model_dir = args.model_dir
        test_csv = args.test_csv
    if not model_dir or not test_csv:
        print("Provide --preset fakeddit_30k|liar OR both --model-dir and --test-csv")
        sys.exit(2)

    model_dir = os.path.abspath(model_dir)
    test_csv = os.path.abspath(test_csv)

    if not os.path.isdir(model_dir):
        print(f"Model directory not found: {model_dir}")
        sys.exit(2)
    weights_ok = any(
        os.path.isfile(os.path.join(model_dir, name))
        for name in ("model.safetensors", "pytorch_model.bin", "model.bin")
    )
    if not weights_ok:
        print(f"No model.safetensors / pytorch_model.bin in: {model_dir}")
        sys.exit(2)
    if not os.path.isfile(test_csv):
        print(f"Test CSV not found: {test_csv}")
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    n_labels = int(getattr(model.config, "num_labels", 2) or 2)
    print(f"num_labels={n_labels} device={device}")

    df = pd.read_csv(test_csv)
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
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
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

    tl = test_csv.lower()
    if "liar" in tl:
        title = "LIAR"
    elif "fakeddit" in tl or "live_posts" in tl:
        title = "FAKEDDIT / LIVE CSV"
    else:
        title = "CUSTOM CSV"
    print("=" * 60)
    print(f"DISTILBERT SEQ CLASS - {title} - EMPIRICAL METRICS")
    print("=" * 60)
    print(f"Preset:    {args.preset or '(custom)'}")
    print(f"Samples:   {len(texts)}")
    print(f"Model dir: {model_dir}")
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
    print(f"  True label=1 rate:  {(y_true == 1).mean():.4f}")
    print(f"  Pred label=1 rate:  {(y_pred == 1).mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
