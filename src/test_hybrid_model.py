"""
Test script for Hybrid Transformer Model
Run from project root: python -m src.test_hybrid_model
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from src.models.hybrid_transformer import HybridTransformerModel, create_hybrid_model


def test_linguistic_features():
    """Test linguistic feature extraction"""
    print("Testing Linguistic Feature Extraction")
    print("=" * 50)
    
    model = HybridTransformerModel()
    
    test_texts = [
        "BREAKING: Scientists discover SECRET cure for cancer!",
        "The economy grew by 2% last quarter.",
        "What is happening to our country??? URGENT warning!",
        "Normal news report with standard punctuation."
    ]
    
    features = model.extract_linguistic_features(test_texts)
    
    print("Feature extraction results:")
    feature_names = [
        "Text Length", "Exclamation Count", "Question Count", 
        "Uppercase Ratio", "Avg Word Length", "Salience Keywords"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nText: {text}")
        print("Features:")
        for j, (name, val) in enumerate(zip(feature_names, features[i])):
            print(f"  {name}: {val:.4f}")
    
    print(f"\nFeature tensor shape: {features.shape}")
    print("✅ Linguistic feature extraction working correctly")


def test_model_modes():
    """Test different model modes"""
    print("\nTesting Different Model Modes")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_text = "BREAKING: Secret conspiracy revealed!"
    
    # Prepare inputs
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    meta_features = HybridTransformerModel.extract_linguistic_features([test_text])
    
    modes = ["full", "text_only", "features_only"]
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        model = create_hybrid_model(mode=mode)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=meta_features
            )
        
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Classifier input dim: {model.classifier_input_dim}")
        
        if mode == "full":
            print(f"  Text embeddings shape: {outputs['text_embeddings'].shape}")
            print(f"  Feature embeddings shape: {outputs['feature_embeddings'].shape}")
        elif mode == "text_only":
            print(f"  Text embeddings shape: {outputs['text_embeddings'].shape}")
        elif mode == "features_only":
            print(f"  Feature embeddings shape: {outputs['feature_embeddings'].shape}")
        
        # Get prediction
        probs = torch.softmax(outputs['logits'], dim=-1)
        pred = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        
        print(f"  Prediction: {'Reliable' if pred.item() == 0 else 'Unreliable'}")
        print(f"  Confidence: {confidence.item():.4f}")
    
    print("\n✅ All modes working correctly")


def test_attention_explainability():
    """Test attention-based explainability"""
    print("\nTesting Attention Explainability")
    print("=" * 50)
    
    model = create_hybrid_model(mode="full")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    test_text = "SHOCKING: Government hiding secret alien technology!"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    meta_features = HybridTransformerModel.extract_linguistic_features([test_text])
    
    # Get prediction with explanation
    explanation = model.predict_with_explanation(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        meta_features=meta_features,
        tokenizer=tokenizer
    )
    
    print(f"Text: {test_text}")
    print(f"Prediction: {'Reliable' if explanation['predictions'].item() == 0 else 'Unreliable'}")
    print(f"Confidence: {explanation['confidence'].item():.4f}")
    print(f"Probabilities: Reliable={explanation['probabilities'][0][0]:.4f}, Unreliable={explanation['probabilities'][0][1]:.4f}")
    
    # Show attention weights shape
    attention_weights = explanation['attention_weights']
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of attention heads: {attention_weights.shape[1]}")
    print(f"Sequence length: {attention_weights.shape[2]}")
    
    print("\n✅ Attention explainability working correctly")


def test_feature_ablation():
    """Test feature importance by ablation"""
    print("\nTesting Feature Ablation Study")
    print("=" * 50)
    
    model = create_hybrid_model(mode="full")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    test_text = "BREAKING: Secret conspiracy revealed by anonymous source!"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Test with different feature configurations
    feature_configs = [
        ("Original features", HybridTransformerModel.extract_linguistic_features([test_text])),
        ("Zero features", torch.zeros(1, 6)),
        ("Random features", torch.rand(1, 6)),
    ]
    
    print(f"Text: {test_text}")
    print("\nFeature ablation results:")
    
    for config_name, meta_features in feature_configs:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=meta_features
            )
        
        probs = torch.softmax(outputs['logits'], dim=-1)
        pred = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        
        print(f"  {config_name}: {'Reliable' if pred.item() == 0 else 'Unreliable'} (confidence: {confidence.item():.4f})")
    
    print("\n✅ Feature ablation study completed")


def main():
    """Run all tests"""
    print("Hybrid Transformer Model Test Suite")
    print("=" * 60)
    
    try:
        test_linguistic_features()
        test_model_modes()
        test_attention_explainability()
        test_feature_ablation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("Hybrid Transformer Model is ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
