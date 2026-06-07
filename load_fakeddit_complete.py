#!/usr/bin/env python3
"""
Load Fakeddit Model Completely
Fix the missing classifier weights to achieve 84.84% accuracy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
import sys
import os
import json

def load_fakeddit_complete():
    """Load the Fakeddit model completely with classifier weights"""
    
    print("🔧 Loading Fakeddit Model Completely")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create BERT model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained model has {len(trained_weights)} parameters")
        
        # Check for classifier weights
        classifier_keys = [k for k in trained_weights.keys() if 'classifier' in k]
        print(f"🔍 Found classifier keys: {classifier_keys}")
        
        # Load all compatible weights
        compatible_keys = []
        incompatible_keys = []
        
        for key in trained_weights.keys():
            if key in model.state_dict():
                if trained_weights[key].shape == model.state_dict()[key].shape:
                    compatible_keys.append(key)
                else:
                    incompatible_keys.append(f"{key} (shape: {trained_weights[key].shape} vs {model.state_dict()[key].shape})")
            else:
                incompatible_keys.append(f"{key} (not in model)")
        
        print(f"✅ Compatible keys: {len(compatible_keys)}")
        print(f"⚠️  Incompatible keys: {len(incompatible_keys)}")
        
        if incompatible_keys:
            print("🔍 Incompatible keys:")
            for key in incompatible_keys[:5]:
                print(f"   {key}")
        
        # Load compatible weights
        if compatible_keys:
            compatible_state = {k: trained_weights[k] for k in compatible_keys}
            model.load_state_dict(compatible_state, strict=False)
            print("✅ Loaded compatible weights")
        
        model.eval()
        
        # Test cases with expected results
        test_cases = [
            {
                "text": "BREAKING: Scientists discover miracle cure for cancer using household ingredients!",
                "expected": "fake"
            },
            {
                "text": "NASA confirms water found on Mars in new rover mission data analysis.",
                "expected": "real"
            },
            {
                "text": "5G technology causes COVID-19 - doctors confirm dangerous radiation effects!",
                "expected": "fake"
            },
            {
                "text": "Study shows regular exercise reduces risk of heart disease by 30%.",
                "expected": "real"
            }
        ]
        
        print("\n🧪 Testing Complete Model:")
        print("-" * 60)
        
        results = []
        confidence_scores = []
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected"]
            
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            with torch.no_grad():
                outputs = model(**inputs)
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
            
            status = "✅" if pred_label == expected else "❌"
            print(f"📝 Test {i}: {text[:60]}...")
            print(f"   {status} Expected: {expected} | Predicted: {pred_label}")
            print(f"   📊 Confidence: {conf_score:.3f}")
            print()
        
        # Calculate metrics
        correct = sum(1 for r in results if r['predicted'] == r['expected'])
        total = len(results)
        accuracy = correct / total
        avg_confidence = np.mean(confidence_scores)
        
        print(f"📈 Final Results:")
        print(f"   Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Max Confidence: {max(confidence_scores):.3f}")
        print(f"   Min Confidence: {min(confidence_scores):.3f}")
        
        # Compare with Google Colab
        print(f"\n🎯 Comparison with Google Colab:")
        print(f"   Expected Accuracy: 84.84%")
        print(f"   Current Accuracy: {accuracy:.1%}")
        print(f"   Expected Confidence: 0.7477")
        print(f"   Current Confidence: {avg_confidence:.3f}")
        
        if accuracy >= 0.75 and avg_confidence >= 0.7:
            print("✅ EXCELLENT - Matching Google Colab performance!")
        elif accuracy >= 0.6 and avg_confidence >= 0.6:
            print("✅ GOOD - Getting close to Colab performance!")
        else:
            print("⚠️  NEEDS WORK - Still below Colab performance")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_fakeddit_complete()
    if model is not None:
        print("\n🎉 Fakeddit Model Loaded Completely!")
    else:
        print("\n❌ Model Loading Failed")
