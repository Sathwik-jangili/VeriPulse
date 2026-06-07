"""
Inference script for trained DistilBERT models and Hybrid model on Fakeddit/LIAR datasets.
Supports 2K test model, 30K full model, LIAR model, and Hybrid model.
Run from project root: python -m src.model_inference
"""

import os
import sys
import numpy as np
import pandas as pd
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install torch transformers")
    sys.exit(1)

# Import advanced hybrid model
try:
    from src.models.hybrid_transformer import HybridTransformerModel
    HYBRID_AVAILABLE = True
except ImportError as e:
    print("Hybrid model not available:", e)
    HYBRID_AVAILABLE = False

# Import advanced hybrid model
try:
    from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel
    ADVANCED_HYBRID_AVAILABLE = True
except ImportError as e:
    print("Advanced hybrid model not available:", e)
    ADVANCED_HYBRID_AVAILABLE = False

# Model paths
MODEL_2K_PATH = "models/distilbert_fakeddit_2k_test"
MODEL_30K_PATH = "models/distilbert_fakeddit_30k_full"
MODEL_LIAR_PATH = "models/distilbert_liar_trained"
MODEL_HYBRID_LIAR_PATH = "models/hybrid_liar_trained"
MODEL_HYBRID_FAKEDDIT_PATH = "models/hybrid_fakeddit_trained"
MODEL_ADVANCED_HYBRID_LIAR_PATH = "models/advanced_hybrid_liar_trained"
MODEL_ADVANCED_HYBRID_FAKEDDIT_PATH = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
MODEL_ADVANCED_HYBRID_FAKEDDIT_60K_PATH = os.path.join(
    "models",
    "advanced_hybrid_fakeddit_60k-20260322T162137Z-3-001",
    "advanced_hybrid_fakeddit_60k",
)
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

def load_model(model_path):
    """Load tokenizer and model from specified path."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def load_advanced_hybrid_model(model_path, model_name):
    """Load advanced hybrid model and tokenizer from specified path."""
    print(f"Loading advanced hybrid model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Using base tokenizer (distilbert-base-uncased)")
    
    from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel

    # Load trained weights (Colab exports may use different filenames)
    weight_names = (
        "advanced_hybrid_model.bin",
        "hybrid_fakeddit_model.bin",
        "fakeddit_model.bin",
        "model.bin",
        "pytorch_model.bin",
    )
    model_file = next(
        (os.path.join(model_path, n) for n in weight_names if os.path.exists(os.path.join(model_path, n))),
        None,
    )
    model = AdvancedHybridTransformerModel(mode="full")
    if model_file:
        try:
            try:
                state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(model_file, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            fusion_text_only = AdvancedHybridTransformerModel.infer_fusion_text_only_from_state_dict(
                state_dict
            )
            model = AdvancedHybridTransformerModel(mode="full", fusion_text_only=fusion_text_only)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(
                f"[OK] Advanced hybrid weights loaded from {os.path.basename(model_file)}"
                f" (fusion_text_only={fusion_text_only})"
            )
        except Exception as e:
            print(f"[WARN] Loading advanced weights: {e}")
            print(f"   Using base advanced model weights")
    else:
        print(f"[WARN] No weight file in {model_path}; using random init")

    model.eval()
    return tokenizer, model

def load_hybrid_model(model_path, model_name):
    """Load hybrid model and tokenizer from specified path."""
    print(f"Loading hybrid model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Using base tokenizer (distilbert-base-uncased)")
    
    # Load the hybrid model architecture
    model = HybridTransformerModel(model_name=MODEL_NAME, mode="full")
    
    # Load trained weights with proper filename
    if "liar" in model_name.lower():
        weight_file = os.path.join(model_path, "hybrid_model.bin")
    else:  # fakeddit
        weight_file = os.path.join(model_path, "hybrid_model_fakeddit.bin")
    
    if os.path.exists(weight_file):
        try:
            state_dict = torch.load(weight_file, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Hybrid weights loaded successfully")
        except Exception as e:
            print(f"[WARN] Loading weights: {e}")
            print(f"   Using base model weights")
    else:
        print(f"[WARN] Weight file not found: {weight_file}")
        print(f"   Using base model weights")
    
    return tokenizer, model

def predict_text(text, tokenizer, model, is_hybrid=False):
    """Make prediction on a single text input."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    if is_hybrid or is_hybrid == "advanced":
        if is_hybrid == "advanced":
            # Advanced hybrid model
            from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel
            meta_features = AdvancedHybridTransformerModel.extract_linguistic_features([text])
        else:
            # Regular hybrid model
            meta_features = HybridTransformerModel.extract_linguistic_features([text])
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=meta_features
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs, dim=-1)[0].item()
        
        return pred, confidence, probs[0].tolist(), outputs
    else:
        # Standard transformer model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        
        return pred, confidence, probs[0].tolist(), None

def test_models():
    """Test all models with sample inputs."""
    sample_texts = [
        "Breaking: Scientists discover cure for cancer!",
        "President signs new healthcare bill into law",
        "Celebrity spotted with alien - exclusive photos!",
        "Stock market reaches all-time high"
    ]
    
    # Standard models
    standard_models = [
        ("2K Fakeddit Test Model", MODEL_2K_PATH, False),
        ("30K Fakeddit Full Model", MODEL_30K_PATH, False),
        ("LIAR Trained Model", MODEL_LIAR_PATH, False)
    ]
    
    # Hybrid models (if available)
    if HYBRID_AVAILABLE:
        standard_models.append(("Hybrid LIAR Model", MODEL_HYBRID_LIAR_PATH, True))
        standard_models.append(("Hybrid Fakeddit Model", MODEL_HYBRID_FAKEDDIT_PATH, True))

    # Advanced hybrid model (if available)
    if os.path.exists(MODEL_ADVANCED_HYBRID_LIAR_PATH):
        standard_models.append(("Advanced Hybrid LIAR Model", MODEL_ADVANCED_HYBRID_LIAR_PATH, "advanced"))
    
    if os.path.exists(MODEL_ADVANCED_HYBRID_FAKEDDIT_PATH):
        standard_models.append(("Advanced Hybrid Fakeddit Model", MODEL_ADVANCED_HYBRID_FAKEDDIT_PATH, "advanced"))
    
    if os.path.exists(MODEL_ADVANCED_HYBRID_FAKEDDIT_60K_PATH):
        standard_models.append(("Advanced Hybrid Fakeddit 60K Model", MODEL_ADVANCED_HYBRID_FAKEDDIT_60K_PATH, "advanced"))
    
    for model_name, model_path, is_hybrid in standard_models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            continue
        
        # Load model (standard, hybrid, or advanced)
        if is_hybrid == "advanced":
            tokenizer, model = load_advanced_hybrid_model(model_path, model_name)
        elif is_hybrid:
            tokenizer, model = load_hybrid_model(model_path, model_name)
        else:
            tokenizer, model = load_model(model_path)
        
        for i, text in enumerate(sample_texts, 1):
            pred, confidence, probs, outputs = predict_text(text, tokenizer, model, is_hybrid)
            label = "Reliable" if pred == 0 else "Unreliable"
            
            print(f"\nText {i}: {text}")
            print(f"Prediction: {label} (Class {pred})")
            print(f"Confidence: {confidence:.4f}")
            print(f"Probabilities: [Reliable: {probs[0]:.4f}, Unreliable: {probs[1]:.4f}]")
            
            # Show linguistic features for hybrid model
            if is_hybrid and outputs:
                features = outputs.get('feature_embeddings')
                if features is not None:
                    print(f"Feature embeddings shape: {features.shape}")
                    print(f"Text embeddings shape: {outputs['text_embeddings'].shape}")

def test_advanced_hybrid_fakeddit_60k_only():
    """
    Run inference only on Advanced Hybrid Fakeddit 60k (your Colab export folder).
    Uses hybrid_fakeddit_model.bin / advanced_hybrid_model.bin etc. in that directory.
    """
    path = MODEL_ADVANCED_HYBRID_FAKEDDIT_60K_PATH
    name = "Advanced Hybrid Fakeddit 60K Model"

    sample_texts = [
        "Breaking: Scientists discover miracle cure for cancer overnight!",
        "President signs new healthcare bill into law after bipartisan vote.",
        "5G towers cause COVID — doctors confirm shocking cover-up!",
        "Federal Reserve raises interest rate 0.25% citing inflation data.",
    ]

    print("=" * 60)
    print(f"Inference test: {name}")
    print(f"Path: {os.path.abspath(path)}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"[ERROR] Folder not found: {path}")
        sys.exit(1)
    if not ADVANCED_HYBRID_AVAILABLE:
        print("[ERROR] AdvancedHybridTransformerModel not importable.")
        sys.exit(1)

    tokenizer, model = load_advanced_hybrid_model(path, name)

    for i, text in enumerate(sample_texts, 1):
        pred, confidence, probs, outputs = predict_text(text, tokenizer, model, "advanced")
        label = "Reliable" if pred == 0 else "Unreliable"
        print(f"\n--- Sample {i} ---")
        print(f"Text: {text}")
        print(f"Prediction: {label} (class {pred})")
        print(f"Confidence: {confidence:.4f}")
        print(f"P(Reliable)={probs[0]:.4f}  P(Unreliable)={probs[1]:.4f}")
        if outputs is not None:
            fe = outputs.get("feature_embeddings")
            if fe is not None:
                print(f"Feature emb. shape: {tuple(fe.shape)}")

    print("\n" + "=" * 60)
    print("Done. (0 = Reliable, 1 = Unreliable)")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VeriPulse model inference smoke tests")
    parser.add_argument(
        "--only-advanced-fakeddit-60k",
        action="store_true",
        help="Only test Advanced Hybrid Fakeddit 60k (hybrid_fakeddit_model.bin in 60k folder)",
    )
    args = parser.parse_args()

    if args.only_advanced_fakeddit_60k:
        test_advanced_hybrid_fakeddit_60k_only()
        return

    print("DistilBERT Model Inference Test")
    print("Available models:")
    print(f"1. 2K Fakeddit Test Model: {MODEL_2K_PATH}")
    print(f"2. 30K Fakeddit Full Model: {MODEL_30K_PATH}")
    print(f"3. LIAR Trained Model: {MODEL_LIAR_PATH}")
    if HYBRID_AVAILABLE:
        print(f"5. Hybrid LIAR Model: {MODEL_HYBRID_LIAR_PATH}")
    print(f"6. Hybrid Fakeddit Model: {MODEL_HYBRID_FAKEDDIT_PATH}")
    if os.path.exists(MODEL_ADVANCED_HYBRID_LIAR_PATH):
        print(f"7. Advanced Hybrid LIAR Model: {MODEL_ADVANCED_HYBRID_LIAR_PATH}")
    print(f"8. Advanced Hybrid Fakeddit 60K: {MODEL_ADVANCED_HYBRID_FAKEDDIT_60K_PATH}")
    print("\nTip: to test only your 60k hybrid checkpoint, run:")
    print("  python -m src.model_inference --only-advanced-fakeddit-60k")

    test_models()

if __name__ == "__main__":
    main()
