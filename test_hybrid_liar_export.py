#!/usr/bin/env python3
"""
Smoke test for LIAR hybrid Colab export (same search order as verify_hybrid_liar_metrics).

Run from project root:
  python test_hybrid_liar_export.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.liar_hybrid_checkpoint import linguistic_batch_for_liar, load_liar_hybrid


def main() -> bool:
    print("LIAR hybrid export — smoke test")
    print("=" * 60)
    try:
        device = "cpu"
        tokenizer, model, info = load_liar_hybrid(device=device)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        print("Add hybrid_liar_model.bin (or hybrid_model.bin) next to tokenizer.json.")
        return False

    print(f"OK: {info['weight_path']}")
    print(f"variant={info['variant']}")

    samples = [
        "The senator proposed a bill to increase education funding statewide.",
        "Scientists confirm aliens built the pyramids using secret NASA technology!",
    ]
    for text in samples:
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        meta = linguistic_batch_for_liar([text], info["model_dir"], info["variant"])
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=meta,
            )
            probs = F.softmax(out["logits"], dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(probs[0, pred].item())
        label = "Unreliable" if pred == 1 else "Reliable"
        print(f"\n{text[:70]}...")
        print(f"  -> {label} (class {pred}), confidence={conf:.4f}")

    print("\n[OK] Forward pass complete.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
