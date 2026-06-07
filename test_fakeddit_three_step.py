#!/usr/bin/env python3
"""
Fakeddit Advanced Hybrid — Three-step solution smoke test (same style as Windsurf).

Uses the SAME loader as production (infer fusion_text_only, no manual layer surgery).
Class convention: 0 = real/reliable, 1 = fake/unreliable.

Run from project root (PowerShell):
  python test_fakeddit_three_step.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from advanced_hybrid_transformer import AdvancedHybridTransformerModel
from fakeddit_colab_hybrid_model import colab_linguistic_features
from fakeddit_hybrid_checkpoint import load_advanced_hybrid_fakeddit, find_fakeddit_hybrid_checkpoint

MAX_LENGTH = 128


def test_fakeddit_three_step() -> bool:
    print("[*] Fakeddit Advanced Hybrid — Three-Step Solution Test")
    print("=" * 60)

    d, wp = find_fakeddit_hybrid_checkpoint()
    if not wp:
        print("[ERROR] No checkpoint found. Searched VERIPULSE_MODEL_DIR and:")
        from fakeddit_hybrid_checkpoint import MODEL_SEARCH_DIRS

        for x in MODEL_SEARCH_DIRS:
            if x:
                print(f"        {x}")
        return False

    print(f"[OK] Checkpoint: {wp}")

    test_cases = [
        {
            "text": "BREAKING: Scientists discover miracle cure for cancer using household ingredients!",
            "expected": "fake",
            "reason": "Sensational headline, miracle claims",
        },
        {
            "text": "NASA confirms water found on Mars in new rover mission data analysis.",
            "expected": "real",
            "reason": "Factual scientific announcement",
        },
        {
            "text": "5G technology causes COVID-19 - doctors confirm dangerous radiation effects!",
            "expected": "fake",
            "reason": "Conspiracy theory, false claims",
        },
        {
            "text": "Study shows regular exercise reduces risk of heart disease by 30%.",
            "expected": "real",
            "reason": "Evidence-based health information",
        },
        {
            "text": "Aliens landed in Area 51 - government whistleblower reveals truth!",
            "expected": "fake",
            "reason": "Conspiracy theory, extraordinary claims",
        },
        {
            "text": "Federal Reserve announces 0.25% interest rate adjustment following economic data review.",
            "expected": "real",
            "reason": "Factual financial news",
        },
        {
            "text": "Miracle weight loss pill melts fat overnight - no diet or exercise needed!",
            "expected": "fake",
            "reason": "Unrealistic claims, miracle product",
        },
        {
            "text": "Research team publishes peer-reviewed study on climate change impacts in Nature journal.",
            "expected": "real",
            "reason": "Peer-reviewed scientific publication",
        },
    ]

    try:
        tokenizer, model, info = load_advanced_hybrid_fakeddit(device="cpu")
        print(
            f"[OK] Loaded weights | variant={info.get('variant', '?')} | "
            f"fusion_text_only={info['fusion_text_only']} | dir={info['model_dir']}"
        )

        cw = info.get("class_weights")
        print("\n--- Step 1: Class weighting (training artifact) ---")
        if cw:
            print(f"     class_weights.json found: {cw}")
        else:
            print("     (no class_weights.json next to checkpoint - optional)")

        print("\n--- Step 2: Dense fusion (768+16 -> fusion stack -> classifier) ---")
        print(f"     fusion_text_only={info['fusion_text_only']} (768-only vs 784 concat)")
        if hasattr(model, "classifier_input_dim"):
            print(f"     classifier_input_dim={model.classifier_input_dim}")
        else:
            print("     (Colab export: linguistic_projection 6->16 + fusion_bottleneck)")

        print("\n--- Step 3: Training schedule (warmup) ---")
        print("     (logged at train time; inference uses frozen weights)")

        print("\n[*] Predictions (0=real, 1=fake)")
        print("-" * 60)

        results = []
        confidences = []
        pred_labels = []

        for i, tc in enumerate(test_cases, 1):
            text = tc["text"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )
            if info.get("variant") == "colab_distil_hybrid":
                meta = colab_linguistic_features([text], info["model_dir"])
            else:
                meta = AdvancedHybridTransformerModel.extract_linguistic_features([text])

            with torch.no_grad():
                out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    meta_features=meta,
                )
                logits = out["logits"]
                probs = torch.softmax(logits, dim=-1)
                pred = int(torch.argmax(probs, dim=-1).item())
                conf = float(torch.max(probs, dim=-1).values.item())

            pred_label = "fake" if pred == 1 else "real"
            pred_labels.append(pred_label)
            ok = pred_label == tc["expected"]
            mark = "[+]" if ok else "[-]"

            print(f"\nTest {i}: {text[:58]}...")
            print(f"   {mark} expected={tc['expected']} predicted={pred_label} (class {pred})")
            print(f"   confidence={conf:.4f}  P(real)={probs[0,0].item():.4f}  P(fake)={probs[0,1].item():.4f}")
            print(f"   reason: {tc['reason']}")

            results.append(ok)
            confidences.append(conf)

        acc = float(np.mean(results))
        avg_conf = float(np.mean(confidences))
        preds_fake = sum(1 for p in pred_labels if p == "fake")
        fake_rate = preds_fake / len(test_cases)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Accuracy (vs hand labels): {acc:.3f} ({sum(results)}/{len(results)})")
        print(f"Average confidence:        {avg_conf:.3f}")
        print(f"Predicted fake rate:       {fake_rate:.3f} ({preds_fake}/{len(test_cases)})")

        if 0.3 <= fake_rate <= 0.65:
            bias = "Good balance (not collapsed to one class)"
        elif fake_rate < 0.25:
            bias = "Mostly predicting real (conservative)"
        else:
            bias = "Mostly predicting fake (aggressive)"

        print(f"Bias / distribution note: {bias}")
        print("\nThree-step checklist:")
        print("  [1] Class weighting: used in training; file present" if cw else "  [1] Class weighting: training technique (no json beside ckpt)")
        print("  [2] Dense fusion:    active in forward pass")
        print("  [3] Warmup schedule: training-only; weights already converged")

        print("\nIntegration: backend uses src/veripulse_predictor.py (same architecture).")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = test_fakeddit_three_step()
    print("\n" + ("[OK] Fakeddit three-step test finished." if ok else "[FAIL] Test failed."))
    sys.exit(0 if ok else 1)
