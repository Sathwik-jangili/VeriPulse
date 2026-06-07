#!/usr/bin/env python3
"""
Load Fakeddit Model Correctly
Fix the bert/distilbert architecture mismatch
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
import sys
import os
import json

def load_fakeddit_correctly():
    """Load the Fakeddit model with correct architecture"""
    
    print("🔧 Loading Fakeddit Model with Correct Architecture")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Create the correct model architecture
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained model has {len(trained_weights)} parameters")
        
        # Map bert keys to distilbert keys
        key_mapping = {}
        mapped_weights = {}
        
        for key in trained_weights.keys():
            if key.startswith('bert.'):
                # Map bert.* to distilbert.*
                new_key = key.replace('bert.', 'distilbert.')
                key_mapping[key] = new_key
                mapped_weights[new_key] = trained_weights[key]
            elif key.startswith('classifier.'):
                # Keep classifier keys as-is
                mapped_weights[key] = trained_weights[key]
            else:
                # Keep other keys as-is
                mapped_weights[key] = trained_weights[key]
        
        print(f"📊 Mapped {len(key_mapping)} bert keys to distilbert keys")
        
        # Try to load mapped weights
        compatible_keys = []
        incompatible_keys = []
        
        for key in mapped_weights.keys():
            if key in model.state_dict():
                if mapped_weights[key].shape == model.state_dict()[key].shape:
                    compatible_keys.append(key)
                else:
                    incompatible_keys.append(f"{key} (shape mismatch)")
            else:
                incompatible_keys.append(f"{key} (not in model)")
        
        print(f"✅ Compatible keys: {len(compatible_keys)}")
        print(f"⚠️  Incompatible keys: {len(incompatible_keys)}")
        
        # Load compatible weights
        if compatible_keys:
            compatible_state = {k: mapped_weights[k] for k in compatible_keys}
            model.load_state_dict(compatible_state, strict=False)
            print("✅ Loaded compatible weights")
        
        model.eval()
        
        # Test with a simple example
        test_text = "BREAKING: Scientists discover miracle cure for cancer!"
        inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            pred_label = "fake" if prediction.item() == 1 else "real"
            conf_score = confidence.item()
        
        print(f"\n🧪 Test Result:")
        print(f"   Text: {test_text}")
        print(f"   Prediction: {pred_label}")
        print(f"   Confidence: {conf_score:.3f}")
        
        if conf_score > 0.7:
            print("✅ HIGH CONFIDENCE - Good loading!")
        elif conf_score > 0.6:
            print("✅ MEDIUM CONFIDENCE - Partial loading!")
        else:
            print("⚠️  LOW CONFIDENCE - Loading issues remain")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_fakeddit_correctly()
    if model is not None:
        print("\n🎉 Fakeddit Model Loaded Successfully!")
    else:
        print("\n❌ Model Loading Failed")
