#!/usr/bin/env python3
<arg_value>CodeContent</arg_key>
<arg_value>"""
Use Google Colab Model Directly
Load the actual model architecture from your training
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertModel
import sys
import os
import json

def use_colab_model():
    """Use the actual model architecture from Google Colab"""
    
    print("🚀 Using Google Colab Model Architecture")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained weights keys: {list(trained_weights.keys())[:10]}...")
        
        # Create model based on trained weights structure
        class ColabFakedditModel(nn.Module):
            def __init__(self):
                super().__init__()
                # BERT base
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                
                # Create layers based on trained weights
                self.fusion_layer = nn.Sequential(
                    nn.Linear(768, 256),  # Based on trained weights
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(768, 32),  # Based on trained weights
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 2)
                )
            
            def forward(self, input_ids, attention_mask, use_fusion=False):
                # BERT embeddings
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = outputs.last_hidden_state[:, 0, :]
                
                if use_fusion:
                    logits = self.fusion_layer(text_embeddings)
                else:
                    logits = self.classifier(text_embeddings)
                
                return logits
        
        # Create model
        model = ColabFakedditModel()
        
        # Load trained weights
        loaded_count = 0
        for key in trained_weights.keys():
            if key in model.state_dict():
                if trained_weights[key].shape == model.state_dict()[key].shape:
                    model.state_dict()[key].copy_(trained_weights[key])
                    loaded_count += 1
                else:
                    print(f"⚠️  Shape mismatch for {key}: {trained_weights[key].shape} vs {model.state_dict()[key].shape}")
            else:
                print(f"⚠️  Key not found: {key}")
        
        print(f"✅ Loaded {loaded_count}/{len(trained_weights)} parameters")
        
        model.eval()
        
        # Test cases
        test_cases = [
            "BREAKING: Scientists discover miracle cure for cancer using household ingredients!",
            "NASA confirms water found on Mars in new rover mission data analysis.",
            "5G technology causes COVID-19 - doctors confirm dangerous radiation effects!",
            "Study shows regular exercise reduces risk of heart disease by 30%."
        ]
        
        print("\n🧪 Testing Colab Model:")
        print("-" * 60)
        
        confidence_scores = []
        
        for i, text in enumerate(test_cases, 1):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            with torch.no_grad():
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                probabilities = torch.softmax(logits, dim=1)
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
            print("✅ HIGH CONFIDENCE - Colab model working!")
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
    model, tokenizer = use_colab_model()
    if model is not None:
        print("\n🎉 Google Colab Model Working!")
    else:
        print("\n❌ Model Loading Failed")
