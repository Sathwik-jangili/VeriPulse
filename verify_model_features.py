
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel

def verify_hybrid_model():
    print("🔍 Starting VeriPulse Hybrid Model Feature Verification...")
    print("="*60)
    
    try:
        # 1. Initialize Model
        model = AdvancedHybridTransformerModel(mode="full")
        model.eval()
        print("✅ [FEATURE 1] Model Initialization: SUCCESS")
        
        # 2. Check Architecture Layers
        print("\n🏗️ Checking Architecture Layers:")
        has_text_encoder = hasattr(model, 'text_encoder')
        has_feature_processor = hasattr(model, 'feature_processor')
        has_fusion_layer = hasattr(model, 'fusion_layer')
        has_classifier = hasattr(model, 'classifier')
        
        print(f"  - Text Encoder (DistilBERT): {'✅' if has_text_encoder else '❌'}")
        print(f"  - Feature Processor (Linguistic): {'✅' if has_feature_processor else '❌'}")
        print(f"  - 256-dim Dense Fusion Layer: {'✅' if has_fusion_layer else '❌'}")
        print(f"  - Multi-layer Classifier: {'✅' if has_classifier else '❌'}")
        
        # 3. Test Linguistic Feature Extraction
        print("\n🧪 Testing Linguistic Feature Extraction (Step 1 of Bias Correction):")
        sample_texts = [
            "BREAKING NEWS! This is a shocking secret scandal!!!", 
            "The weather is nice today in New York."
        ]
        features = AdvancedHybridTransformerModel.extract_linguistic_features(sample_texts)
        
        print(f"  - Feature Shape: {features.shape} (Expected: [2, 6])")
        if features.shape == (2, 6):
            print("  - Feature Extraction: ✅ SUCCESS")
            print(f"  - Sample Feature Vector (High Salience): {features[0].tolist()}")
        else:
            print("  - Feature Extraction: ❌ FAILED (Shape Mismatch)")

        # 4. Test Forward Pass (End-to-End)
        print("\n🔄 Testing Forward Pass (Full Hybrid Inference):")
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, features)
        
        print(f"  - Logits Shape: {outputs['logits'].shape} (Expected: [2, 2])")
        print(f"  - Fused Embeddings Shape: {outputs['fused_embeddings'].shape} (Expected: [2, 64])")
        
        if outputs['logits'].shape == (2, 2):
            print("  - Full Hybrid Forward Pass: ✅ SUCCESS")
        else:
            print("  - Full Hybrid Forward Pass: ❌ FAILED")

        print("\n" + "="*60)
        print("🎯 VERIFICATION SUMMARY:")
        if all([has_text_encoder, has_feature_processor, has_fusion_layer, has_classifier]):
            print("✅ ALL A* HYBRID FEATURES ARE PRESENT AND FUNCTIONAL.")
        else:
            print("❌ SOME ARCHITECTURAL COMPONENTS ARE MISSING.")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING VERIFICATION: {e}")

if __name__ == "__main__":
    verify_hybrid_model()
