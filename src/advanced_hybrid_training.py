"""
Advanced Hybrid Training with Class Weighting, Dense Fusion, and Warmup
Implements your three-step bias correction solution
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any
import json
from collections import Counter

try:
    from datasets import Dataset
    from transformers import AutoTokenizer
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install torch transformers datasets")
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.advanced_hybrid_transformer import (
    AdvancedHybridTransformerModel, 
    ClassWeights, 
    FocalLoss, 
    LinearWarmupScheduler
)

class AdvancedHybridTrainer:
    """Advanced trainer with your three-step bias correction"""
    
    def __init__(self, model, tokenizer, device='cpu', class_weights=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.class_weights = class_weights
        
        # Advanced loss function
        if class_weights is not None:
            class_weights_tensor = torch.tensor(
                [class_weights.get(0, 1.0), class_weights.get(1, 1.0)], 
                dtype=torch.float32, device=device
            )
            self.criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        else:
            self.criterion = FocalLoss(gamma=2.0)
        
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
    
    def setup_optimizer_and_scheduler(self, learning_rate=2e-5, total_steps=1000, warmup_steps=100):
        """Setup optimizer with linear warmup scheduler"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = LinearWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, learning_rate
        )
    
    def extract_linguistic_features_batch(self, texts):
        """Extract linguistic features for batch"""
        return AdvancedHybridTransformerModel.extract_linguistic_features(texts)
    
    def train_epoch(self, train_dataloader, epoch):
        """Train one epoch with advanced monitoring"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_predictions = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            meta_features = batch['meta_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, meta_features)
            logits = outputs['logits']
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                current_lr = self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            class_predictions.extend(predictions.cpu().numpy())
            
            # Log progress
            if batch_idx % 50 == 0:
                current_lr = self.scheduler.get_current_lr() if self.scheduler else self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_dataloader)}")
                print(f"  Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total
        
        # Class distribution analysis
        class_dist = Counter(class_predictions)
        class_ratio = class_dist[1] / total if 1 in class_dist else 0
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Class Distribution: {class_dist}")
        print(f"  Unreliable Rate: {class_ratio:.2%}")
        
        # Store training history
        epoch_info = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_distribution': class_dist,
            'unreliable_rate': class_ratio
        }
        self.training_history.append(epoch_info)
        
        return avg_loss, accuracy, class_ratio
    
    def evaluate(self, eval_dataloader):
        """Evaluate model with detailed metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                meta_features = batch['meta_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, meta_features)
                logits = outputs['logits']
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and confidences
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        
        # Confidence analysis
        mean_confidence = np.mean(all_confidences)
        std_confidence = np.std(all_confidences)
        
        # Class distribution
        class_dist = Counter(all_predictions)
        class_ratio = class_dist[1] / len(all_predictions) if 1 in class_dist else 0
        
        print(f"\nEvaluation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Mean Confidence: {mean_confidence:.4f} ± {std_confidence:.4f}")
        print(f"  Class Distribution: {class_dist}")
        print(f"  Unreliable Rate: {class_ratio:.2%}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'class_distribution': class_dist,
            'unreliable_rate': class_ratio
        }


def analyze_dataset_bias(dataset_path, dataset_name):
    """Analyze class distribution in dataset"""
    print(f"\n🔍 Analyzing {dataset_name} Dataset Bias:")
    print("-" * 40)
    
    df = pd.read_csv(dataset_path)
    labels = df['label'].values
    
    # Basic statistics
    label_counts = Counter(labels)
    total = len(labels)
    
    print(f"Total samples: {total}")
    print(f"Class distribution: {dict(label_counts)}")
    
    for class_id in sorted(label_counts.keys()):
        count = label_counts[class_id]
        percentage = count / total * 100
        print(f"  Class {class_id}: {count} ({percentage:.1f}%)")
    
    # Calculate class weights
    class_weights = ClassWeights.calculate_class_weights(labels)
    print(f"\nCalculated class weights: {class_weights}")
    
    # Bias assessment
    majority_ratio = max(label_counts.values()) / total
    if majority_ratio > 0.7:
        print(f"⚠️  HIGH BIAS: Majority class {majority_ratio:.1%}")
    elif majority_ratio > 0.6:
        print(f"⚠️  MODERATE BIAS: Majority class {majority_ratio:.1%}")
    else:
        print(f"✅ BALANCED: Majority class {majority_ratio:.1%}")
    
    return class_weights


def create_advanced_training_script(dataset_type="liar"):
    """Create advanced training script for specific dataset"""
    
    if dataset_type == "liar":
        train_path = "data_processed/liar_train.csv"
        val_path = "data_processed/liar_val.csv"
        test_path = "data_processed/liar_test.csv"
        save_dir = "models/advanced_hybrid_liar_trained"
    else:  # fakeddit
        train_path = "data_processed/fakeddit_train.csv"
        val_path = "data_processed/fakeddit_val.csv"
        test_path = "data_processed/fakeddit_test.csv"
        save_dir = "models/advanced_hybrid_fakeddit_trained"
    
    print(f"🚀 Advanced Hybrid Training: {dataset_type.upper()}")
    print("=" * 60)
    
    # Analyze dataset bias
    class_weights = analyze_dataset_bias(train_path, dataset_type)
    
    # Load datasets
    print(f"\n📁 Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def preprocess_function(examples):
        """Preprocess with linguistic features"""
        # Convert to list if needed
        if isinstance(examples["text"], str):
            texts = [examples["text"]]
            labels = [examples["label"]]
        else:
            texts = examples["text"]
            labels = examples["label"]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors=None,
        )
        
        # Extract linguistic features
        linguistic_features = AdvancedHybridTransformerModel.extract_linguistic_features(texts)
        
        tokenized["meta_features"] = linguistic_features.tolist()
        tokenized["labels"] = labels
        
        return tokenized
    
    # Preprocess datasets
    print(f"🔧 Preprocessing datasets...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function to handle tensors properly"""
        # Convert lists to tensors with proper dtype
        input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
        meta_features = torch.tensor([item['meta_features'] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'meta_features': meta_features,
            'labels': labels
        }
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Initialize model
    print(f"🧠 Initializing Advanced Hybrid Model...")
    model = AdvancedHybridTransformerModel(mode="full")
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = AdvancedHybridTrainer(model, tokenizer, device, class_weights)
    
    # Setup optimizer and scheduler
    total_steps = len(train_dataloader) * 4  # 4 epochs
    warmup_steps = len(train_dataloader)  # 1 epoch warmup
    trainer.setup_optimizer_and_scheduler(
        learning_rate=2e-5, 
        total_steps=total_steps, 
        warmup_steps=warmup_steps
    )
    
    print(f"📊 Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: 16")
    print(f"  Learning rate: 2e-5 (with linear warmup)")
    print(f"  Epochs: 4")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Class weights: {class_weights}")
    print(f"  Loss function: Focal Loss (gamma=2.0)")
    print(f"  Fusion layer: Dense (256-dim)")
    
    # Training loop
    print(f"\n🏃‍♂️ Starting Advanced Training...")
    
    best_f1 = 0
    patience = 2
    patience_counter = 0
    
    for epoch in range(4):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/4")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_unreliable_rate = trainer.train_epoch(train_dataloader, epoch + 1)
        
        # Evaluate
        eval_results = trainer.evaluate(val_dataloader)
        
        # Check for improvement
        if eval_results['f1'] > best_f1:
            best_f1 = eval_results['f1']
            patience_counter = 0
            
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "advanced_hybrid_model.bin"))
            tokenizer.save_pretrained(save_dir)
            
            print(f"✅ New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered")
            break
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation on Test Set:")
    test_results = trainer.evaluate(test_dataloader)
    
    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(save_dir, "advanced_config.json")
    config = {
        "model_type": "AdvancedHybridTransformer",
        "dataset": dataset_type,
        "class_weights": class_weights,
        "fusion_dim": 256,
        "loss_function": "FocalLoss",
        "gamma": 2.0,
        "learning_rate": 2e-5,
        "warmup_steps": warmup_steps,
        "final_test_results": test_results
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎉 Advanced Training Complete!")
    print(f"📁 Model saved to: {save_dir}")
    print(f"📊 Final Test F1: {test_results['f1']:.4f}")
    print(f"🎯 Final Confidence: {test_results['mean_confidence']:.4f} ± {test_results['std_confidence']:.4f}")
    print(f"⚖️  Final Class Balance: {test_results['unreliable_rate']:.2%} unreliable")
    
    return trainer, test_results


if __name__ == "__main__":
    # Test advanced training
    print("🚀 Advanced Hybrid Training System")
    print("Implementing your three-step bias correction solution:")
    print("1. ✅ Class Weighting")
    print("2. ✅ Dense Fusion Layer (256-dim)")
    print("3. ✅ Linear Warmup Scheduler")
    
    # You can choose which dataset to train on
    dataset_type = "liar"  # Change to "fakeddit" for Fakeddit dataset
    
    trainer, results = create_advanced_training_script(dataset_type)
    
    print(f"\n🎯 SOLUTION SUMMARY:")
    print(f"✅ Class Weighting: Balanced Fake vs Real classes")
    print(f"✅ Dense Fusion: 256-dim layer learns feature relationships")
    print(f"✅ Linear Warmup: Stable training with confident patterns")
    print(f"📈 Expected confidence: 0.85+ (vs 0.51 before)")
    print(f"⚖️  Expected balance: ~50/50 (vs biased before)")
