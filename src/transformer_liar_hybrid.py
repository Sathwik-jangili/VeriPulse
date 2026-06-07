"""
Fine-tune Hybrid Transformer on LIAR dataset with linguistic features.
Run from project root: python -m src.transformer_liar_hybrid
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any

try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers import EvalPrediction
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install torch transformers datasets")
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.models.hybrid_transformer import HybridTransformerModel

PROCESSED_DIR = "data_processed"
MODEL_SAVE_DIR = "models/hybrid_liar_trained"
MODEL_NAME = "distilbert-base-uncased"

# --- Hyperparameters for LIAR dataset with Hybrid Model ---
NUM_EPOCHS = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
# Save best checkpoint by validation F1 (class 1)
METRIC_FOR_BEST_MODEL = "f1"
# Full training on all LIAR samples
MAX_TRAIN_SAMPLES = None  # Use all samples
# Hybrid model mode: "full", "text_only", "features_only"
HYBRID_MODE = "full"


def load_datasets():
    """Load LIAR train/val/test CSVs into HuggingFace Dataset with 'text' and 'label'."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_test.csv"))
    
    print(f"LIAR Dataset Loaded:")
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples") 
    print(f"Test: {len(test)} samples")
    
    train_ds = Dataset.from_pandas(train[["text", "label"]].astype({"text": str, "label": int}))
    val_ds = Dataset.from_pandas(val[["text", "label"]].astype({"text": str, "label": int}))
    test_ds = Dataset.from_pandas(test[["text", "label"]].astype({"text": str, "label": int}))
    return train_ds, val_ds, test_ds


def extract_linguistic_features_batch(texts: list) -> torch.Tensor:
    """Extract linguistic features for a batch of texts"""
    return HybridTransformerModel.extract_linguistic_features(texts)


def tokenize_and_extract_features(examples, tokenizer):
    """Tokenize text and extract linguistic features"""
    # Tokenize text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    
    # Extract linguistic features
    linguistic_features = extract_linguistic_features_batch(examples["text"])
    
    # Add features to tokenized output
    tokenized["meta_features"] = linguistic_features.tolist()
    
    return tokenized


class HybridTrainer(Trainer):
    """Custom trainer for hybrid model that handles dual inputs"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for hybrid model"""
        # Extract inputs
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        meta_features = inputs.get("meta_features")
        
        # Convert meta_features to tensor if needed
        if isinstance(meta_features, list):
            meta_features = torch.tensor(meta_features, dtype=torch.float32, device=self.args.device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_features=meta_features
        )
        
        logits = outputs["logits"]
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction):
    """Return accuracy and P/R/F1 for the positive class (label 1)."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    print("=" * 60)
    print(f"LIAR Dataset: Hybrid Transformer Fine-tuning (Mode: {HYBRID_MODE})")
    print("=" * 60)
    
    print("Loading LIAR splits from %s ..." % PROCESSED_DIR)
    train_ds, val_ds, test_ds = load_datasets()
    
    print("Loading tokenizer: %s" % MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing and extracting linguistic features...")
    train_ds = train_ds.map(
        lambda x: tokenize_and_extract_features(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Process train",
    )
    train_ds = train_ds.rename_column("label", "labels")
    
    val_ds = val_ds.map(
        lambda x: tokenize_and_extract_features(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Process val",
    )
    val_ds = val_ds.rename_column("label", "labels")
    
    test_ds = test_ds.map(
        lambda x: tokenize_and_extract_features(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Process test",
    )
    test_ds = test_ds.rename_column("label", "labels")

    print("Loading hybrid model: %s (mode: %s)" % (MODEL_NAME, HYBRID_MODE))
    model = HybridTransformerModel(
        model_name=MODEL_NAME,
        num_classes=2,
        mode=HYBRID_MODE
    )

    # Training configuration for Hybrid Model
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        logging_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
    )

    # Custom trainer for hybrid model
    trainer = HybridTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    print("Starting hybrid training...")
    trainer.train()

    print("\nEvaluating on test set ...")
    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=-1)
    labels = test_out.label_ids
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    
    print("-" * 50)
    print("Hybrid LIAR Test Results:")
    print("Accuracy:  %.4f" % acc)
    print("Precision (class 1): %.4f" % prec)
    print("Recall (class 1):    %.4f" % rec)
    print("F1 (class 1):       %.4f" % f1)
    print("-" * 50)

    # Save model and tokenizer
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    # Save model config
    config = {
        "model_name": MODEL_NAME,
        "mode": HYBRID_MODE,
        "num_classes": 2,
        "feature_dim": 16,
        "hidden_dim": 128,
        "dropout_rate": 0.3,
        "max_length": MAX_LENGTH,
    }
    import json
    with open(os.path.join(MODEL_SAVE_DIR, "hybrid_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("Hybrid model and configuration saved to %s" % MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
