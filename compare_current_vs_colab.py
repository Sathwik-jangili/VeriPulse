"""
Compare Current Advanced Hybrid Model with Google Colab Results
Test the actual trained model performance
"""

import torch
import numpy as np
import json
from transformers import AutoTokenizer
import os

def test_with_inference_system():
    """Test the advanced hybrid model using the integrated inference system"""
    print("🧪 COMPARING CURRENT MODEL vs COLLAB RESULTS")
    print("=" * 60)
    
    # Import inference system
    import sys
    sys.path.append('src')
    from model_inference import predict_text, load_advanced_hybrid_model
    
    # Load the advanced hybrid model
    model_path = "models/advanced_hybrid_liar_trained"
    tokenizer, model = load_advanced_hybrid_model(model_path, "Advanced Hybrid LIAR")
    
    # Test texts (same as colab would use)
    test_texts = [
        # Clearly reliable (class 0)
        "Scientists at Harvard published a peer-reviewed study confirming vaccine effectiveness.",
        "The Federal Reserve announced yesterday that interest rates will remain unchanged at 5.25%.",
        "According to the Bureau of Labor Statistics, unemployment rate decreased to 3.8% last month.",
        
        # Clearly unreliable (class 1)
        "BREAKING: Secret government files prove aliens have been living among us for 50 years!",
        "SHOCKING: One weird trick discovered by mom that doctors hate! Lose 30 pounds in 3 days!",
        "URGENT: Bitcoin will reach $1 million by next week according to anonymous whistleblower!",
        
        # Mixed/ambiguous
        "The president signed the healthcare bill into law yesterday.",
        "Senator promises to lower taxes for middle class families.",
        "SHOCKING: Government conspiracy revealed by anonymous source!",
        "Celebrity spotted with alien - exclusive photos!"
    ]
    
    # Expected labels for accuracy calculation
    expected_labels = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    
    print(f"📝 Testing {len(test_texts)} diverse examples...")
    print("-" * 60)
    
    predictions = []
    confidences = []
    
    for i, text in enumerate(test_texts, 1):
        try:
            # Use the inference system
            pred, confidence, probs, outputs = predict_text(text, tokenizer, model, is_hybrid="advanced")
            
            predictions.append(pred)
            confidences.append(confidence)
            
            label = "Reliable" if pred == 0 else "Unreliable"
            expected = "Reliable" if expected_labels[i-1] == 0 else "Unreliable"
            correct = "✅" if pred == expected_labels[i-1] else "❌"
            
            print(f"{i:2d}. {correct} {label} (Conf: {confidence:.3f}) - Expected: {expected}")
            print(f"     Text: {text[:60]}...")
            print(f"     Probs: [Reliable: {probs[0]:.3f}, Unreliable: {probs[1]:.3f}]")
            print()
            
        except Exception as e:
            print(f"❌ Error testing text {i}: {e}")
            continue
    
    # Calculate metrics
    if predictions:
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Accuracy
        correct_predictions = sum(1 for i, pred in enumerate(predictions) if i < len(expected_labels) and pred == expected_labels[i])
        accuracy = correct_predictions / len(expected_labels)
        
        # Confidence
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Class distribution
        unreliable_count = np.sum(predictions == 1)
        unreliable_rate = unreliable_count / len(predictions)
        
        print(f"📊 CURRENT MODEL METRICS:")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mean Confidence: {mean_confidence:.3f} ± {std_confidence:.3f}")
        print(f"Unreliable Rate: {unreliable_rate:.1%}")
        print(f"Correct Predictions: {correct_predictions}/{len(expected_labels)}")
        
        print(f"\n🎯 COMPARISON WITH COLLAB RESULTS:")
        print("=" * 60)
        print(f"Colab Results:")
        print(f"  Accuracy: 0.6227")
        print(f"  Confidence: ~0.61")
        print(f"  Unreliable: 44.3%")
        print()
        print(f"Current Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Confidence: {mean_confidence:.3f}")
        print(f"  Unreliable: {unreliable_rate:.1%}")
        
        # Analysis
        print(f"\n📈 ANALYSIS:")
        print("-" * 60)
        
        # Accuracy comparison
        acc_diff = accuracy - 0.6227
        if abs(acc_diff) < 0.05:
            acc_status = "✅ SIMILAR"
        elif acc_diff > 0:
            acc_status = "🚀 BETTER"
        else:
            acc_status = "⚠️  LOWER"
        
        print(f"Accuracy: {acc_status} ({acc_diff:+.4f})")
        
        # Confidence comparison
        conf_diff = mean_confidence - 0.61
        if abs(conf_diff) < 0.05:
            conf_status = "✅ SIMILAR"
        elif conf_diff > 0:
            conf_status = "🚀 BETTER"
        else:
            conf_status = "⚠️  LOWER"
        
        print(f"Confidence: {conf_status} ({conf_diff:+.3f})")
        
        # Balance comparison
        balance_diff = unreliable_rate - 0.443
        if abs(balance_diff) < 0.05:
            balance_status = "✅ SIMILAR"
        elif balance_diff > 0:
            balance_status = "⚠️  MORE UNRELIABLE"
        else:
            balance_status = "⚠️  MORE RELIABLE"
        
        print(f"Balance: {balance_status} ({balance_diff:+.1%})")
        
        # Overall assessment
        print(f"\n🎉 OVERALL ASSESSMENT:")
        if abs(acc_diff) < 0.05 and abs(conf_diff) < 0.05:
            overall = "✅ MATCHES COLLAB RESULTS"
            desc = "Current model performs similarly to Colab training"
        elif acc_diff > 0.05 or conf_diff > 0.05:
            overall = "🚀 BETTER THAN COLLAB"
            desc = "Current model shows improvement"
        else:
            overall = "⚠️  DIFFERENT FROM COLLAB"
            desc = "Current model needs investigation"
        
        print(f"Status: {overall}")
        print(f"Description: {desc}")
        
        return {
            'accuracy': accuracy,
            'confidence': mean_confidence,
            'unreliable_rate': unreliable_rate,
            'matches_colab': abs(acc_diff) < 0.05 and abs(conf_diff) < 0.05
        }
    
    else:
        print("❌ No predictions made")
        return None

def main():
    """Main comparison function"""
    print("🧪 ADVANCED HYBRID MODEL vs COLLAB COMPARISON")
    print("Testing if current model matches Google Colab results")
    print("=" * 60)
    
    try:
        results = test_with_inference_system()
        
        if results:
            print(f"\n🎯 CONCLUSION:")
            if results['matches_colab']:
                print(f"✅ Current model MATCHES Colab results")
                print(f"📊 Your three-step solution is working correctly")
            else:
                print(f"⚠️  Current model differs from Colab")
                print(f"🔧 May need to check training or model loading")
            
            print(f"📈 Key Metrics:")
            print(f"   Accuracy: {results['accuracy']:.4f} (Colab: 0.6227)")
            print(f"   Confidence: {results['confidence']:.3f} (Colab: ~0.61)")
            print(f"   Balance: {results['unreliable_rate']:.1%} (Colab: 44.3%)")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        print("🔧 Check model integration and try again")

if __name__ == "__main__":
    main()
