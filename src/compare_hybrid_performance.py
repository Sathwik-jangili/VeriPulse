"""
Compare Hybrid Model Performance across Datasets
Run from project root: python -m src.compare_hybrid_performance
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from src.models.hybrid_transformer import HybridTransformerModel

# Model paths
HYBRID_LIAR_PATH = "models/hybrid_liar_trained"
HYBRID_FAKEDDIT_PATH = "models/hybrid_fakeddit_trained"
MODEL_NAME = "distilbert-base-uncased"

# Test texts for different domains
test_texts = {
    "Political News": [
        "President signs new healthcare bill into law today",
        "BREAKING: Secret conspiracy revealed in Washington!",
        "Senator promises to lower taxes for middle class families",
        "SHOCKING: Government hiding alien technology from public!"
    ],
    "Social Media": [
        "Scientists discover breakthrough cure for cancer in lab study",
        "You won't believe what this celebrity did last night!",
        "New study shows coffee may have health benefits",
        "URGENT: This one weird trick will change your life forever!"
    ],
    "Financial News": [
        "Stock market reaches all-time high amid economic growth",
        "EXCLUSIVE: Insider trading scheme exposed by whistleblowers",
        "Federal Reserve announces interest rate decision",
        "WARNING: Market crash predicted by anonymous analyst!"
    ]
}


def load_hybrid_model(model_path):
    """Load hybrid model"""
    print(f"Loading hybrid model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HybridTransformerModel(model_name=MODEL_NAME, mode="full")
    return tokenizer, model


def predict_with_analysis(texts, tokenizer, model, dataset_name):
    """Make predictions with detailed analysis"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} Model")
    print(f"{'='*60}")
    
    results = []
    
    for category, category_texts in texts.items():
        print(f"\n{category} Category:")
        print("-" * 40)
        
        category_results = []
        
        for text in category_texts:
            # Tokenize
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            # Extract linguistic features
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
            
            label = "Reliable" if pred == 0 else "Unreliable"
            
            # Get linguistic features for analysis
            feature_values = meta_features[0].numpy()
            feature_names = [
                "Text Length", "Exclamation", "Question", 
                "Uppercase", "Avg Word Len", "Salience"
            ]
            
            print(f"  Text: {text}")
            print(f"  Prediction: {label} (Confidence: {confidence:.3f})")
            print(f"  Features: {[f'{name}:{val:.2f}' for name, val in zip(feature_names, feature_values)]}")
            print()
            
            category_results.append({
                'text': text,
                'prediction': label,
                'confidence': confidence,
                'reliable_prob': probs[0][0].item(),
                'unreliable_prob': probs[0][1].item(),
                'features': dict(zip(feature_names, feature_values))
            })
        
        results.append({
            'category': category,
            'results': category_results
        })
    
    return results


def analyze_feature_importance(results, dataset_name):
    """Analyze feature importance across predictions"""
    print(f"\n{dataset_name} Feature Analysis:")
    print("=" * 40)
    
    all_features = []
    unreliable_predictions = []
    reliable_predictions = []
    
    for category_data in results:
        for result in category_data['results']:
            features = result['features']
            all_features.append(features)
            
            if result['prediction'] == 'Unreliable':
                unreliable_predictions.append(features)
            else:
                reliable_predictions.append(features)
    
    # Calculate average features
    if all_features:
        feature_names = list(all_features[0].keys())
        
        print("Overall Feature Averages:")
        for name in feature_names:
            avg_val = np.mean([f[name] for f in all_features])
            print(f"  {name}: {avg_val:.3f}")
        
        print("\nUnreliable Predictions Feature Averages:")
        for name in feature_names:
            if unreliable_predictions:
                avg_val = np.mean([f[name] for f in unreliable_predictions])
                print(f"  {name}: {avg_val:.3f}")
            else:
                print(f"  {name}: N/A")
        
        print("\nReliable Predictions Feature Averages:")
        for name in feature_names:
            if reliable_predictions:
                avg_val = np.mean([f[name] for f in reliable_predictions])
                print(f"  {name}: {avg_val:.3f}")
            else:
                print(f"  {name}: N/A")


def compare_models():
    """Compare hybrid models trained on different datasets"""
    print("Hybrid Model Performance Comparison")
    print("=" * 60)
    
    # Check which models are available
    liar_available = os.path.exists(HYBRID_LIAR_PATH)
    fakeddit_available = os.path.exists(HYBRID_FAKEDDIT_PATH)
    
    if not liar_available and not fakeddit_available:
        print("No hybrid models found! Please train models first:")
        print("  - python -m src.transformer_liar_hybrid")
        print("  - python -m src.transformer_fakeddit_hybrid")
        return
    
    # Load and test LIAR model
    if liar_available:
        print("\n" + "="*60)
        print("LIAR-TRAINED HYBRID MODEL")
        print("="*60)
        tokenizer_liar, model_liar = load_hybrid_model(HYBRID_LIAR_PATH)
        liar_results = predict_with_analysis(test_texts, tokenizer_liar, model_liar, "LIAR")
        analyze_feature_importance(liar_results, "LIAR")
    
    # Load and test Fakeddit model
    if fakeddit_available:
        print("\n" + "="*60)
        print("FAKEDDIT-TRAINED HYBRID MODEL")
        print("="*60)
        tokenizer_fakeddit, model_fakeddit = load_hybrid_model(HYBRID_FAKEDDIT_PATH)
        fakeddit_results = predict_with_analysis(test_texts, tokenizer_fakeddit, model_fakeddit, "Fakeddit")
        analyze_feature_importance(fakeddit_results, "Fakeddit")
    
    # Summary comparison
    if liar_available and fakeddit_available:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Compare overall prediction patterns
        liar_unreliable_rate = sum(1 for cat in liar_results for r in cat['results'] if r['prediction'] == 'Unreliable') / sum(len(cat['results']) for cat in liar_results)
        fakeddit_unreliable_rate = sum(1 for cat in fakeddit_results for r in cat['results'] if r['prediction'] == 'Unreliable') / sum(len(cat['results']) for cat in fakeddit_results)
        
        print(f"Overall Unreliable Prediction Rate:")
        print(f"  LIAR Model: {liar_unreliable_rate:.2%}")
        print(f"  Fakeddit Model: {fakeddit_unreliable_rate:.2%}")
        
        print(f"\nDataset Specialization:")
        print(f"  LIAR Model: Trained on political statements, fact-checking data")
        print(f"  Fakeddit Model: Trained on social media content, news articles")


def main():
    """Main comparison function"""
    try:
        compare_models()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
