#!/usr/bin/env python3
"""
Smoke test: load DistilBERT sequence-classification presets and run one forward each.

Run from project root:
  python test_distilbert_exports.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PRESETS = {
    "Fakeddit 30k": os.path.join(PROJECT_ROOT, "models", "distilbert_fakeddit_30k_full"),
    "LIAR": os.path.join(PROJECT_ROOT, "models", "distilbert_liar_trained"),
}

MAX_LENGTH = 128


def try_one(name: str, model_dir: str, sample: str) -> bool:
    print(f"\n--- {name} ---")
    if not os.path.isdir(model_dir):
        print(f"  [SKIP] Missing folder: {model_dir}")
        return False
    if not any(
        os.path.isfile(os.path.join(model_dir, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    ):
        print(f"  [SKIP] No weights in {model_dir}")
        return False
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tok(
        sample,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        conf = float(probs[0, pred].item())
    lab = "Unreliable" if pred == 1 else "Reliable"
    print(f"  OK: {lab} (class {pred}) conf={conf:.4f}")
    print(f"  sample: {sample[:70]}...")
    return True


def main() -> bool:
    print("DistilBERT sequence-classification — smoke test")
    print("=" * 60)
    ok_any = False
    ok_any |= try_one(
        "Fakeddit 30k full",
        PRESETS["Fakeddit 30k"],
        "Breaking: miracle cure discovered overnight doctors shocked",
    )
    ok_any |= try_one(
        "LIAR",
        PRESETS["LIAR"],
        "The governor claimed unemployment fell by half last year.",
    )
    if ok_any:
        print("\n[OK] At least one DistilBERT preset loaded.")
    else:
        print("\n[FAIL] No presets could be loaded.")
    return ok_any


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
