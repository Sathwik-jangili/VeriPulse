#!/usr/bin/env python3
"""
Load Fakeddit Hybrid Model Complete
Load the complete hybrid architecture with fusion layers
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertModel
import sys
import os
import json

class FakedditHybridModel(nn.Module):
    """Complete Fakeddit Hybrid Model with Fusion Layer"""
    
    def __init__(self):
        super().__init__()
        # BERT base
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion layer (from your training)
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 16, 256),  # text (768) + features (16) -> 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def extract_linguistic_features(self, texts):
        """Extract linguistic features from texts"""
        features = []
        for text in texts:
            # Simple linguistic features
            feature_vector = torch.tensor([
                len(text.split()),  # word count
                text.count('!'),    # exclamation count
                text.count('?'),    # question count
                text.count('.'),    # period count
                text.upper().count('BREAKING'),  # sensational words
                text.upper().count('MIRACLE'),
                text.upper().count('SHOCKING'),
                text.upper().count('ALERT'),
                text.upper().count('URGENT'),
                text.upper().count('WARNING'),
                text.upper().count('SCAM'),
                text.upper().count('FAKE'),
                text.upper().count('HOAX'),
                text.count('http'),  # links
                text.count('@'),    # mentions
                text.count('#'),    # hashtags
                1.0  # bias
            ], dtype=torch.float32)
            features.append(feature_vector)
        return torch.stack(features)
    
    def forward(self, input_ids, attention_mask, use_fusion=True):
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        if use_fusion:
            # Extract linguistic features
            texts = ["dummy"] * input_ids.size(0)  # Will be replaced with actual texts
            linguistic_features = self.extract_linguistic_features(texts)
            
            # Fusion
            combined = torch.cat([text_embeddings, linguistic_features], dim=1)
            logits = self.fusion_layer(combined)
        else:
            # Standard classifier
            logits = self.classifier(text_embeddings)
        
        return logits

def load_fakeddit_hybrid_complete():
    """Load the complete Fakeddit hybrid model"""
    
    print("🔧 Loading Complete Fakeddit Hybrid Model")
    print("=" * 60)
    
    # Model path
    model_path = "models/fakeddit_hybrid_model/advanced_hybrid_fakeddit_trained"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create hybrid model
        model = FakedditHybridModel()
        
        # Load your trained weights
        model_file = os.path.join(model_path, "fakeddit_model.bin")
        trained_weights = torch.load(model_file, map_location='cpu')
        
        print(f"📊 Trained model has {len(trained_weights)} parameters")
        
        # Load weights with mapping
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
        
        print("\n🧪 Testing Complete Hybrid Model:")
        print("-" * 60)
        
        results = []
        confidence_scores = []
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected"]
            
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            # Update linguistic features with actual text
            model.extract_linguistic_features = lambda texts: torch.stack([
                torch.tensor([
                    len(t.split()),
                    t.count('!'),
                    t.count('?'),
                    t.count('.'),
                    t.upper().count('BREAKING'),
                    t.upper().count('MIRACLE'),
                    t.upper().count('SHOCKING'),
                    t.upper().count('ALERT'),
                    t.upper().count('URGENT'),
                    t.upper().count('WARNING'),
                    t.upper().count('SCAM'),
                    t.upper().count('FAKE'),
                    t.upper().count('HOAX'),
                    t.count('http'),
                    t.count('@'),
                    t.count('#'),
                    1.0
                ], dtype=torch.float32) for t in texts
            ])
            
            with torch.no_grad():
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_fusion=True)
                probabilities = torch.softmax(logits, dim=1)
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
        
        print(f"📈 Final Hybrid Results:")
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
    model, tokenizer = load_fakeddit_hybrid_complete()
    if model is not None:
        print("\n🎉 Complete Fakeddit Hybrid Model Loaded!")
    else:
        print("\n❌ Model Loading Failed")
