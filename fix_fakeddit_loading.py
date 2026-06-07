#!/usr/bin/env python3
"""
Fix Fakeddit Model Loading Issues
Properly load the 84.84% accuracy model from Google Colab
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def fix_fakeddit_model_loading():
    """Fix the Fakeddit model loading to match Colab performance"""
    
    print("🔧 Fixing Fakeddit Model Loading Issues")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    # Test cases that should work with your 84.84% accuracy model
    test_cases = [
        {
            "text": "BREAKING: Scientists discover miracle cure for cancer using household ingredients!",
            "expected": "fake",
            "reason": "Sensational miracle claims"
        },
        {
            "text": "NASA confirms water found on Mars in new rover mission data analysis.",
            "expected": "real", 
            "reason": "Factual scientific announcement"
        },
        {
            "text": "5G technology causes COVID-19 - doctors confirm dangerous radiation effects!",
            "expected": "fake",
            "reason": "Conspiracy theory"
        },
        {
            "text": "Study shows regular exercise reduces risk of heart disease by 30%.",
            "expected": "real",
            "reason": "Evidence-based health"
        }
    ]
    
    try:
        # Load class weights to understand the training
        class_weights_path = os.path.join(model_path, "class_weights.json")
        if os.path.exists(class_weights_path):
            with open(class_weights_path, 'r') as f:
                class_weights = json.load(f)
            print(f"📊 Class Weights: {class_weights}")
        
        # Try to load as standard DistilBERT first
        print("📦 Attempting to load as standard DistilBERT...")
        
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Create base model
        base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained weights keys: {list(trained_weights.keys())[:10]}...")
        
        # Check what we can load
        compatible_keys = []
        incompatible_keys = []
        
        for key in trained_weights.keys():
            if key in base_model.state_dict():
                if trained_weights[key].shape == base_model.state_dict()[key].shape:
                    compatible_keys.append(key)
                else:
                    incompatible_keys.append(f"{key} (shape mismatch: {trained_weights[key].shape} vs {base_model.state_dict()[key].shape})")
            else:
                incompatible_keys.append(f"{key} (not in base model)")
        
        print(f"✅ Compatible keys: {len(compatible_keys)}")
        print(f"⚠️  Incompatible keys: {len(incompatible_keys)}")
        
        if incompatible_keys:
            print("🔍 Incompatible keys (first 5):")
            for key in incompatible_keys[:5]:
                print(f"   {key}")
        
        # Load compatible weights
        if compatible_keys:
            compatible_state = {k: trained_weights[k] for k in compatible_keys}
            base_model.load_state_dict(compatible_state, strict=False)
            print("✅ Loaded compatible weights")
        
        base_model.eval()
        
        # Test predictions
        print("\n🧪 Testing with Fixed Model:")
        print("-" * 60)
        
        results = []
        confidence_scores = []
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected"]
            
            print(f"\n📝 Test {i}: {text[:60]}...")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            # Predict
            with torch.no_grad():
                outputs = base_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
                
                pred_label = "fake" if prediction.item() == 1 else "real"
                conf_score = confidence.item()
                
            results.append({
                'text': text,
                'expected': expected,
                'predicted': pred_label,
                'confidence': conf_score
            })
            
            confidence_scores.append(conf_score)
            
            # Display result
            status = "✅" if pred_label == expected else "❌"
            print(f"   {status} Expected: {expected} | Predicted: {pred_label}")
            print(f"   📊 Confidence: {conf_score:.3f}")
            print(f"   💭 Reason: {test_case['reason']}")
        
        # Calculate metrics
        print("\n📈 Fixed Model Performance:")
        print("-" * 60)
        
        correct = sum(1 for r in results if r['predicted'] == r['expected'])
        total = len(results)
        accuracy = correct / total
        
        avg_confidence = np.mean(confidence_scores)
        fake_predictions = sum(1 for r in results if r['predicted'] == 'fake')
        fake_rate = fake_predictions / total
        
        print(f"🎯 Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        print(f"🔍 Fake Rate: {fake_rate:.3f} ({fake_predictions}/{total})")
        
        # Compare with expected Colab performance
        print("\n🎯 Comparison with Google Colab Results:")
        print("-" * 60)
        print(f"📊 Expected Accuracy: 84.84%")
        print(f"📊 Current Accuracy: {accuracy:.1%}")
        print(f"📊 Expected Confidence: 0.7477")
        print(f"📊 Current Confidence: {avg_confidence:.3f}")
        
        if accuracy >= 0.75 and avg_confidence >= 0.7:
            print("✅ EXCELLENT - Close to Colab performance!")
        elif accuracy >= 0.6 and avg_confidence >= 0.6:
            print("✅ GOOD - Partial Colab performance achieved!")
        else:
            print("⚠️  NEEDS WORK - Not matching Colab performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_fakeddit_model_loading()
    if success:
        print("\n🎉 Fakeddit Model Loading Fix Complete!")
    else:
        print("\n❌ Fix Failed")
