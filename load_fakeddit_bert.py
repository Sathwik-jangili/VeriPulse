#!/usr/bin/env python3
"""
Load Fakeddit Model with BERT Architecture
Use the correct BERT architecture that matches your training
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
import sys
import os
import json

def load_fakeddit_bert():
    """Load the Fakeddit model with BERT architecture"""
    
    print("🔧 Loading Fakeddit Model with BERT Architecture")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create BERT model (not DistilBERT)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained model has {len(trained_weights)} parameters")
        
        # Check compatibility
        compatible_keys = []
        incompatible_keys = []
        
        for key in trained_weights.keys():
            if key in model.state_dict():
                if trained_weights[key].shape == model.state_dict()[key].shape:
                    compatible_keys.append(key)
                else:
                    incompatible_keys.append(f"{key} (shape mismatch)")
            else:
                incompatible_keys.append(f"{key} (not in model)")
        
        print(f"✅ Compatible keys: {len(compatible_keys)}")
        print(f"⚠️  Incompatible keys: {len(incompatible_keys)}")
        
        # Load compatible weights
        if compatible_keys:
            compatible_state = {k: trained_weights[k] for k in compatible_keys}
            model.load_state_dict(compatible_state, strict=False)
            print("✅ Loaded compatible weights")
        
        model.eval()
        
        # Test cases
        test_cases = [
            "BREAKING: Scientists discover miracle cure for cancer using household ingredients!",
            "NASA confirms water found on Mars in new rover mission data analysis.",
            "5G technology causes COVID-19 - doctors confirm dangerous radiation effects!",
            "Study shows regular exercise reduces risk of heart disease by 30%."
        ]
        
        print("\n🧪 Testing with BERT Architecture:")
        print("-" * 60)
        
        confidence_scores = []
        
        for i, text in enumerate(test_cases, 1):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
                
                pred_label = "fake" if prediction.item() == 1 else "real"
                conf_score = confidence.item()
                
            confidence_scores.append(conf_score)
            
            print(f"📝 Test {i}: {text[:60]}...")
            print(f"   Prediction: {pred_label}")
            print(f"   Confidence: {conf_score:.3f}")
            print()
        
        avg_confidence = np.mean(confidence_scores)
        
        print(f"📈 Results:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Max Confidence: {max(confidence_scores):.3f}")
        print(f"   Min Confidence: {min(confidence_scores):.3f}")
        
        if avg_confidence > 0.7:
            print("✅ HIGH CONFIDENCE - BERT architecture working!")
        elif avg_confidence > 0.6:
            print("✅ MEDIUM CONFIDENCE - Partial success!")
        else:
            print("⚠️  LOW CONFIDENCE - Still issues")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_fakeddit_bert()
    if model is not None:
        print("\n🎉 Fakeddit Model Loaded with BERT!")
    else:
        print("\n❌ Model Loading Failed")
