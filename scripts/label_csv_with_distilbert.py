#!/usr/bin/env python3
"""
Assign label = DistilBERT argmax prediction (0/1) for each row in a CSV.

Use case: build a teacher-labeled set, then evaluate another model (e.g. hybrid) on it.

Note: y_true is DistilBERT's own predictions — agreement with hybrid measures teacher–student
alignment, not independent accuracy.

Usage:
  python scripts/label_csv_with_distilbert.py \\
    --input data_processed/live_balanced_proxy_labeled.csv \\
    --output data_processed/live_1k_distilbert_labeled.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "distilbert_fakeddit_30k_full")
MAX_LENGTH = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data_processed", "live_balanced_proxy_labeled.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data_processed", "live_1k_distilbert_labeled.csv"),
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if "text" not in df.columns:
        print("CSV must have a 'text' column.", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).fillna("").tolist()
    preds: list[int] = []
    confs: list[float] = []

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
            probs = F.softmax(out.logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(torch.max(probs, dim=-1).values.item())
        preds.append(pred)
        confs.append(conf)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(texts)} ...")

    out = df.copy()
    if "label" in out.columns:
        out["prior_label"] = out["label"]
    out["label"] = preds
    out["distilbert_confidence"] = confs
    out["label_teacher"] = f"distilbert:{os.path.basename(model_dir)}"

    d = os.path.dirname(args.output) or "."
    os.makedirs(d, exist_ok=True)
    out.to_csv(args.output, index=False)
    s = pd.Series(preds)
    print(f"Wrote {len(out)} rows -> {args.output}")
    print("DistilBERT label distribution:", s.value_counts().sort_index().to_dict())
    print(f"Mean confidence: {np.mean(confs):.4f}")


if __name__ == "__main__":
    main()
