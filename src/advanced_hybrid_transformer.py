"""
Advanced Hybrid Transformer with Bias Correction
Implements: Class Weighting, Dense Fusion Layer, Learning Rate Scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from collections import Counter
import math

class AdvancedHybridTransformerModel(nn.Module):
    """Enhanced Hybrid Transformer with bias correction mechanisms"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2, 
                 feature_dim=16, hidden_dim=256, dropout_rate=0.3, mode="full",
                 fusion_text_only=False):
        """
        Advanced Hybrid Transformer with:
        - Class weighting support
        - Dense fusion layer (256-dim)
        - Better regularization

        fusion_text_only: If True, fusion stack takes only CLS embeddings (768-dim).
            Use when loading Colab checkpoints trained with text-only fusion (784 vs 768 mismatch).
        """
        super().__init__()
        
        self.mode = mode
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_text_only = fusion_text_only
        
        # Text branch - DistilBERT
        self.text_encoder = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.text_embedding_dim = 768  # Fixed for DistilBERT
        
        # Enhanced linguistic feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(6, feature_dim * 2),  # 6 -> 32
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim * 2, feature_dim),  # 32 -> 16
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        fusion_in = (
            self.text_embedding_dim if fusion_text_only
            else (self.text_embedding_dim + self.feature_dim)
        )
        # DENSE FUSION LAYER (256-dim) - Your Key Innovation!
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 -> 128
            nn.ReLU(), 
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 128 -> 64
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # Calculate classifier input dimension based on mode
        if mode == "full":
            self.classifier_input_dim = hidden_dim // 4  # 64 from dense fusion
        elif mode == "text_only":
            self.classifier_input_dim = self.text_embedding_dim
        elif mode == "features_only":
            self.classifier_input_dim = self.feature_dim
        
        # Enhanced classifier with better initialization
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, hidden_dim // 8),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 8, num_classes)  # 32 -> 2
        )
        
        # Better weight initialization
        self._init_weights()
    
    @staticmethod
    def infer_fusion_text_only_from_state_dict(state_dict):
        """
        Colab / older exports may use fusion Layer 0 with in_features=768 (text only)
        instead of 784 (CLS + linguistic features).
        """
        if not isinstance(state_dict, dict):
            return False
        for k, v in state_dict.items():
            if k.endswith("fusion_layer.0.weight") and hasattr(v, "shape"):
                return v.shape[1] == 768
        return False

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def extract_linguistic_features(texts):
        """Enhanced linguistic feature extraction"""
        features = []
        salience_keywords = {
            'breaking', 'shocking', 'secret', 'warning', 'urgent', 'exclusive',
            'revealed', 'scandal', 'conspiracy', 'cover-up', 'bombshell', 'sensational',
            'amazing', 'incredible', 'unbelievable', 'miracle', 'revolutionary'
        }
        
        for text in texts:
            # Basic text stats
            words = text.split()
            word_count = len(words)
            
            # Punctuation features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            
            # Uppercase ratio
            uppercase_words = sum(1 for word in words if word.isupper())
            uppercase_ratio = uppercase_words / max(word_count, 1)
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Enhanced salience detection
            text_lower = text.lower()
            salience_count = sum(1 for keyword in salience_keywords if keyword in text_lower)
            salience_ratio = salience_count / max(word_count, 1)
            
            # Normalize features
            features.append([
                min(word_count / 100, 1.0),  # Text length (normalized to 100 words)
                min(exclamation_count / 5, 1.0),  # Exclamation marks (normalized to 5)
                min(question_count / 3, 1.0),  # Question marks (normalized to 3)
                uppercase_ratio,  # Uppercase ratio
                min(avg_word_length / 15, 1.0),  # Avg word length (normalized to 15)
                min(salience_ratio * 3, 1.0)  # Salience keywords (enhanced scaling)
            ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, input_ids, attention_mask, meta_features):
        """Enhanced forward pass with dense fusion"""
        outputs = {}
        
        # Text embeddings
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        outputs['text_embeddings'] = text_embeddings
        
        # Process linguistic features
        feature_embeddings = self.feature_processor(meta_features)
        outputs['feature_embeddings'] = feature_embeddings
        
        if self.mode == "full":
            # DENSE FUSION - Your Key Innovation!
            if self.fusion_text_only:
                fused_embeddings = self.fusion_layer(text_embeddings)
            else:
                combined = torch.cat([text_embeddings, feature_embeddings], dim=-1)
                fused_embeddings = self.fusion_layer(combined)
            outputs['fused_embeddings'] = fused_embeddings
            classifier_input = fused_embeddings
            
        elif self.mode == "text_only":
            classifier_input = text_embeddings
            
        elif self.mode == "features_only":
            classifier_input = feature_embeddings
        
        # Classification
        logits = self.classifier(classifier_input)
        outputs['logits'] = logits
        
        # Attention weights for explainability
        if hasattr(text_outputs, 'attentions'):
            outputs['attention_weights'] = text_outputs.attentions[-1]
        
        return outputs
    
    def predict_with_explanation(self, input_ids, attention_mask, meta_features, tokenizer):
        """Enhanced prediction with detailed explanation"""
        outputs = self.forward(input_ids, attention_mask, meta_features)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1)
        
        # Enhanced explanation
        explanation = {
            'prediction': pred.item(),
            'confidence': confidence[0].item(),
            'probabilities': probs[0].tolist(),
            'text_embeddings': outputs['text_embeddings'],
            'feature_embeddings': outputs['feature_embeddings'],
            'fused_embeddings': outputs.get('fused_embeddings'),
            'attention_weights': outputs.get('attention_weights'),
            'mode': self.mode
        }
        
        return explanation


class ClassWeights:
    """Calculate and manage class weights for balanced training"""
    
    @staticmethod
    def calculate_class_weights(labels):
        """Calculate inverse frequency weights"""
        label_counts = Counter(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        class_weights = {}
        for class_id in sorted(label_counts.keys()):
            frequency = label_counts[class_id] / total
            weight = 1.0 / frequency
            class_weights[class_id] = weight
        
        # Normalize weights
        total_weight = sum(class_weights.values())
        for class_id in class_weights:
            class_weights[class_id] /= total_weight
        
        return class_weights
    
    @staticmethod
    def get_balanced_weights(num_classes=2):
        """Get balanced weights for binary classification"""
        return torch.tensor([1.0, 1.0])  # Start balanced, adjust based on data


class FocalLoss(nn.Module):
    """Focal Loss for better class separation"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LinearWarmupScheduler:
    """Linear warmup scheduler for stable training"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.current_step = 0
        
    def step(self):
        """Update learning rate with linear warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step + 1) / self.warmup_steps
        else:
            # Constant after warmup
            lr = self.max_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr
    
    def get_current_lr(self):
        """Get current learning rate"""
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            return self.max_lr


def create_advanced_hybrid_model(mode="full", class_weights=None):
    """Factory function with class weight support"""
    model = AdvancedHybridTransformerModel(mode=mode)
    
    if class_weights is not None:
        model.class_weights = class_weights
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Advanced Hybrid Transformer with Bias Correction")
    print("=" * 60)
    
    # Test the enhanced model
    model = AdvancedHybridTransformerModel(mode="full")
    
    # Sample inputs
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    meta_features = torch.rand(batch_size, 6)
    
    # Forward pass
    outputs = model(input_ids, attention_mask, meta_features)
    
    print(f"✅ Model initialized successfully")
    print(f"   Text embeddings shape: {outputs['text_embeddings'].shape}")
    print(f"   Feature embeddings shape: {outputs['feature_embeddings'].shape}")
    print(f"   Fused embeddings shape: {outputs['fused_embeddings'].shape}")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Classifier input dim: {model.classifier_input_dim}")
    
    # Test class weights
    sample_labels = [0, 0, 1, 1, 1, 1]  # Imbalanced
    weights = ClassWeights.calculate_class_weights(sample_labels)
    print(f"\n📊 Class weights for {sample_labels}: {weights}")
    
    # Test focal loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(outputs['logits'], torch.tensor([0, 1]))
    print(f"   Focal loss: {loss.item():.4f}")
    
    print(f"\n🎯 Advanced features ready:")
    print(f"   ✅ Dense Fusion Layer (256-dim)")
    print(f"   ✅ Class Weighting Support") 
    print(f"   ✅ Focal Loss for Better Separation")
    print(f"   ✅ Linear Warmup Scheduler")
    print(f"   ✅ Enhanced Feature Processing")
    print(f"   ✅ Better Weight Initialization")
