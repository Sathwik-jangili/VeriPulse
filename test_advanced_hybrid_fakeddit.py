#!/usr/bin/env python3
"""
Quick smoke test for Advanced Hybrid Fakeddit (shared loader with three-step test).
Run from project root: python test_advanced_hybrid_fakeddit.py
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from advanced_hybrid_transformer import AdvancedHybridTransformerModel
from fakeddit_colab_hybrid_model import colab_linguistic_features
from fakeddit_hybrid_checkpoint import load_advanced_hybrid_fakeddit


def test_advanced_hybrid_fakeddit():
    print("Testing Advanced Hybrid Fakeddit (quick)")
    print("=" * 60)

    try:
        tokenizer, model, info = load_advanced_hybrid_fakeddit(device="cpu")
        print(f"Using: {info['weight_path']}")
        print(f"variant={info.get('variant')} fusion_text_only={info['fusion_text_only']}")

        text = "BREAKING: miracle cure discovered overnight — doctors shocked!"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        if info.get("variant") == "colab_distil_hybrid":
            meta = colab_linguistic_features([text], info["model_dir"])
        else:
            meta = AdvancedHybridTransformerModel.extract_linguistic_features([text])

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=meta,
            )
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        pred_label = "unreliable" if prediction.item() == 1 else "reliable"
        print(f"Sample text: {text[:70]}...")
        print(
            f"Predicted: {pred_label} (class {prediction.item()}), "
            f"confidence={confidence.item():.4f}"
        )
        print("[OK] Forward pass works.")
        return True

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = test_advanced_hybrid_fakeddit()
    sys.exit(0 if ok else 1)
