"""
Diagnose and Fix Model Bias Issues
Run from project root: python -m src.diagnose_model_bias
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from src.models.hybrid_transformer import HybridTransformerModel
from collections import Counter
import os

def analyze_model_predictions(model_path, model_name, test_texts):
    """Deep analysis of model predictions"""
    print(f"\n{'='*60}")
    print(f"🔍 Deep Analysis: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HybridTransformerModel(model_name="distilbert-base-uncased", mode="full")
    
    # Load weights
    if "liar" in model_name.lower():
        weight_file = os.path.join(model_path, "hybrid_model.bin")
    else:
        weight_file = os.path.join(model_path, "hybrid_model_fakeddit.bin")
    
    if os.path.exists(weight_file):
        state_dict = torch.load(weight_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    predictions = []
    confidences = []
    probabilities = []
    logits_values = []
    
    print(f"\nAnalyzing {len(test_texts)} predictions...")
    
    for text in test_texts:
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # Extract features
        meta_features = HybridTransformerModel.extract_linguistic_features([text])
        
        # Predict
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
        
        predictions.append(pred)
        confidences.append(confidence)
        probabilities.append(probs[0].tolist())
        logits_values.append(logits[0].tolist())
    
    # Convert to numpy
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    probabilities = np.array(probabilities)
    logits_values = np.array(logits_values)
    
    # Analysis
    print(f"\n📊 Prediction Analysis:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Class 0 (Reliable): {np.sum(predictions == 0)} ({np.mean(predictions == 0):.1%})")
    print(f"   Class 1 (Unreliable): {np.sum(predictions == 1)} ({np.mean(predictions == 1):.1%})")
    
    print(f"\n🎯 Confidence Analysis:")
    print(f"   Mean confidence: {np.mean(confidences):.4f}")
    print(f"   Std confidence: {np.std(confidences):.4f}")
    print(f"   Min confidence: {np.min(confidences):.4f}")
    print(f"   Max confidence: {np.max(confidences):.4f}")
    
    print(f"\n📈 Probability Distribution:")
    print(f"   Class 0 mean prob: {np.mean(probabilities[:, 0]):.4f} ± {np.std(probabilities[:, 0]):.4f}")
    print(f"   Class 1 mean prob: {np.mean(probabilities[:, 1]):.4f} ± {np.std(probabilities[:, 1]):.4f}")
    
    print(f"\n🔢 Raw Logits Analysis:")
    print(f"   Class 0 mean logit: {np.mean(logits_values[:, 0]):.4f} ± {np.std(logits_values[:, 0]):.4f}")
    print(f"   Class 1 mean logit: {np.mean(logits_values[:, 1]):.4f} ± {np.std(logits_values[:, 1]):.4f}")
    
    # Bias detection
    print(f"\n🚨 Bias Detection:")
    
    # Class imbalance check
    class_ratio = np.sum(predictions == 1) / len(predictions)
    if class_ratio > 0.8:
        print(f"   ⚠️  SEVERE BIAS: Model predicts class 1 {class_ratio:.1%} of the time")
    elif class_ratio > 0.7:
        print(f"   ⚠️  MODERATE BIAS: Model predicts class 1 {class_ratio:.1%} of the time")
    elif class_ratio < 0.2:
        print(f"   ⚠️  SEVERE BIAS: Model predicts class 0 {1-class_ratio:.1%} of the time")
    elif class_ratio < 0.3:
        print(f"   ⚠️  MODERATE BIAS: Model predicts class 0 {1-class_ratio:.1%} of the time")
    else:
        print(f"   ✅ BALANCED: Class distribution looks reasonable")
    
    # Confidence check
    mean_conf = np.mean(confidences)
    if mean_conf < 0.6:
        print(f"   ⚠️  LOW CONFIDENCE: Mean confidence {mean_conf:.4f} is too low")
        print(f"      Model is essentially guessing")
    elif mean_conf < 0.7:
        print(f"   ⚠️  MODERATE CONFIDENCE: Mean confidence {mean_conf:.4f}")
    else:
        print(f"   ✅ GOOD CONFIDENCE: Mean confidence {mean_conf:.4f}")
    
    # Probability distribution check
    prob_std = np.std(probabilities[:, 1])
    if prob_std < 0.1:
        print(f"   ⚠️  LOW VARIANCE: Probability std {prob_std:.4f} - model not discriminating")
    else:
        print(f"   ✅ GOOD VARIANCE: Probability std {prob_std:.4f}")
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'probabilities': probabilities,
        'logits': logits_values,
        'bias_level': 'severe' if class_ratio > 0.8 or class_ratio < 0.2 else 'moderate' if class_ratio > 0.7 or class_ratio < 0.3 else 'balanced'
    }

def generate_balanced_test_set():
    """Generate a more diverse test set"""
    return [
        # Clearly reliable
        "Scientists at Harvard University published a peer-reviewed study in Nature journal confirming the effectiveness of the new vaccine.",
        "The Federal Reserve announced yesterday that interest rates will remain unchanged at 5.25%.",
        "According to the Bureau of Labor Statistics, unemployment rate decreased to 3.8% last month.",
        "NASA confirmed that the James Webb Space Telescope successfully captured images of distant galaxies.",
        
        # Clearly unreliable  
        "BREAKING: Secret government files prove aliens have been living among us for 50 years!",
        "SHOCKING: One weird trick discovered by mom that doctors hate! Lose 30 pounds in 3 days!",
        "URGENT: Bitcoin will reach $1 million by next week according to anonymous whistleblower!",
        "EXCLUSIVE: Celebrity caught in scandalous affair with time traveler from the future!",
        
        # Ambiguous/neutral
        "Local officials announced new infrastructure projects for the downtown area.",
        "Researchers are studying the effects of social media on mental health.",
        "The company reported quarterly earnings that exceeded analyst expectations.",
        "Weather forecast predicts rain for the upcoming weekend."
    ]

def suggest_solutions(bias_analysis_liar, bias_analysis_fakeddit):
    """Suggest solutions for model bias"""
    print(f"\n{'='*60}")
    print(f"💡 SOLUTIONS FOR MODEL BIAS")
    print(f"{'='*60}")
    
    print(f"\n🔧 IMMEDIATE FIXES:")
    print(f"1. **Threshold Adjustment**")
    print(f"   - Current: argmax(logits)")
    print(f"   - Suggested: Use 0.5 threshold with confidence margin")
    print(f"   - Implementation: pred = 1 if prob[1] > 0.5 + margin else 0")
    
    print(f"\n2. **Confidence Thresholding**")
    print(f"   - Reject predictions with confidence < 0.65")
    print(f"   - Mark as 'uncertain' instead of forcing prediction")
    
    print(f"\n3. **Calibration**")
    print(f"   - Use temperature scaling to calibrate probabilities")
    print(f"   - Apply Platt scaling on validation set")
    
    print(f"\n🎯 TRAINING FIXES:")
    print(f"1. **Class Balance**")
    print(f"   - Check original dataset class distribution")
    print(f"   - Use class weights in loss function")
    print(f"   - Apply oversampling/undersampling")
    
    print(f"\n2. **Loss Function**")
    print(f"   - Current: CrossEntropyLoss")
    print(f"   - Suggested: FocalLoss for better class separation")
    print(f"   - Add label smoothing")
    
    print(f"\n3. **Training Strategy**")
    print(f"   - Longer training (more epochs)")
    print(f"   - Lower learning rate")
    print(f"   - Better validation monitoring")
    
    print(f"\n📊 EVALUATION FIXES:")
    print(f"1. **Better Metrics**")
    print(f"   - Track AUC-ROC, not just accuracy")
    print(f"   - Monitor calibration curves")
    print(f"   - Use confusion matrix analysis")
    
    print(f"\n2. **Threshold Optimization**")
    print(f"   - Find optimal threshold on validation set")
    print(f"   - Use Youden's J statistic")
    print(f"   - Consider cost-sensitive thresholds")

def main():
    """Main diagnostic function"""
    print("🔍 MODEL BIAS DIAGNOSIS")
    print("=" * 60)
    
    # Model paths
    hybrid_liar_path = "models/hybrid_liar_trained"
    hybrid_fakeddit_path = "models/hybrid_fakeddit_trained"
    
    # Better test set
    test_texts = generate_balanced_test_set()
    
    print(f"Using {len(test_texts)} diverse test texts")
    
    # Analyze both models
    liar_analysis = analyze_model_predictions(hybrid_liar_path, "Hybrid LIAR Model", test_texts)
    fakeddit_analysis = analyze_model_predictions(hybrid_fakeddit_path, "Hybrid Fakeddit Model", test_texts)
    
    # Suggest solutions
    suggest_solutions(liar_analysis, fakeddit_analysis)
    
    print(f"\n{'='*60}")
    print(f"🎯 SUMMARY")
    print(f"{'='*60}")
    
    liar_bias = liar_analysis['bias_level']
    fakeddit_bias = fakeddit_analysis['bias_level']
    
    print(f"LIAR Model Bias: {liar_bias.upper()}")
    print(f"Fakeddit Model Bias: {fakeddit_bias.upper()}")
    
    if liar_bias in ['severe', 'moderate'] or fakeddit_bias in ['severe', 'moderate']:
        print(f"\n⚠️  ACTION REQUIRED: Models show bias - implement fixes")
    else:
        print(f"\n✅ Models appear balanced")

if __name__ == "__main__":
    main()
